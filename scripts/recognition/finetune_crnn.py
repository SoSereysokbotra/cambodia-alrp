#!/usr/bin/env python3
"""
scripts/recognition/finetune_crnn.py
====================================
Fine-tune the synthetic CRNN on REAL number crops (Phase 4).

Strategy (chosen from the real baseline CER ~95%, which showed a pure DOMAIN gap):
  * start from models/recognition/crnn_best.pth (synthetic, reads perfectly)
  * FULL fine-tune (no frozen layers) at a LOW lr (1e-4)
  * mix real (oversampled + augmented) with some synthetic to avoid overfitting
  * early-stop on a held-out REAL val split (carved from the train labels)

Inputs:
  data/crnn_crops/real_labels.csv   (image_path,plate_text) — needs /train/ rows
  data/synthetic/train_labels.csv   (for the synthetic mix)

Output:
  models/recognition/crnn_finetuned.pth

Run (after labelling train crops):
    python scripts/recognition/finetune_crnn.py
    python scripts/recognition/finetune_crnn.py --epochs 60 --real-oversample 8
    python scripts/recognition/finetune_crnn.py --extra-real-csv data/crnn_crops_augmented/augmented_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src" / "recognition"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from crnn_model import CRNN, CTCDecoder, CHARSET, BLANK   # noqa: E402
from crnn_dataset import collate_fn                        # noqa: E402

IMG_H, IMG_W, N_HIDDEN = 64, 320, 256
REAL_CSV = PROJECT_ROOT / "data" / "crnn_crops" / "real_labels.csv"
SYNTH_CSV = PROJECT_ROOT / "data" / "synthetic" / "train_labels.csv"
BASE = PROJECT_ROOT / "models" / "recognition" / "crnn_best.pth"
OUT = PROJECT_ROOT / "models" / "recognition" / "crnn_finetuned.pth"

CHAR_TO_IDX = {c: i for i, c in enumerate(CHARSET)}


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(preds, tgts):
    e = c = 0
    for p, t in zip(preds, tgts):
        e += levenshtein(p, t)
        c += max(len(t), 1)
    return e / max(c, 1)


def read_csv(path: Path, match: str | None, forbid_test: bool = False):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            p = (r.get("image_path") or "").strip()
            t = (r.get("plate_text") or "").strip().upper()
            t = "".join(ch for ch in t if ch in CHAR_TO_IDX)
            if not p or not t:
                continue
            norm = p.replace("\\", "/")
            if match and match not in norm:
                continue
            if forbid_test and "/test/" in norm:
                raise ValueError(f"refusing test-split row in training CSV: {p}")
            rows.append((p, t))
    return rows


# Probability of flipping a training crop 180 deg (upside-down). 0.0 = original
# behaviour. Set by --rotate180 so the CRNN learns to read upside-down plates
# (the label is unchanged; the model must learn the flipped glyphs + reversed order).
ROTATE180_PROB = 0.0


def augment(img: np.ndarray) -> tuple[np.ndarray, int]:
    """Light appearance augmentation. Returns (image, was_flipped_180)."""
    import cv2
    # ROTATION: teach the CRNN to read upside-down crops (MODEL training approach).
    # The flag is returned because --stn-supervise uses it as a FREE label: we are
    # the ones flipping, so we know exactly what the STN should undo.
    flipped = 0
    if ROTATE180_PROB and random.random() < ROTATE180_PROB:
        img = cv2.rotate(img, cv2.ROTATE_180)
        flipped = 1
    h, w = img.shape
    # brightness / contrast
    if random.random() < 0.7:
        alpha = random.uniform(0.7, 1.3)      # contrast
        beta = random.uniform(-25, 25)        # brightness
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    # blur
    if random.random() < 0.4:
        k = random.choice([3, 3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    # gaussian noise
    if random.random() < 0.5:
        img = np.clip(img.astype(np.float32) +
                      np.random.normal(0, random.uniform(4, 16), img.shape), 0, 255).astype(np.uint8)
    # small rotation
    if random.random() < 0.5:
        ang = random.uniform(-4, 4)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return img, flipped


class RealCropDataset(Dataset):
    def __init__(self, samples, augment_on=True):
        self.samples = samples
        self.aug = augment_on

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        path, text = self.samples[idx]
        full = path if Path(path).is_absolute() else str(PROJECT_ROOT / path)
        img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.full((IMG_H, IMG_W), 127, np.uint8)
        img = cv2.resize(img, (IMG_W, IMG_H))
        flipped = 0
        if self.aug:
            img, flipped = augment(img)
        t = torch.from_numpy(img.astype("float32") / 255.0).unsqueeze(0)
        t = (t - 0.5) / 0.5
        target = torch.tensor([CHAR_TO_IDX[c] for c in text], dtype=torch.long)
        return t, target, len(target), text, flipped


def collate_with_flip(batch):
    """Like crnn_dataset.collate_fn, but keeps the 180-flip flag per sample."""
    images, targets, lengths, texts, flips = zip(*batch)
    return (torch.stack(images, 0),
            torch.cat(targets, 0) if len(targets) else torch.tensor([], dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
            list(texts),
            torch.tensor(flips, dtype=torch.float))


def stn_target_theta(flips: torch.Tensor) -> torch.Tensor:
    """The affine matrix the STN SHOULD predict for each sample.

    flipped -> undo it with an exact 180 rotation [[-1,0,0],[0,-1,0]]
    upright -> leave it alone, identity      [[ 1,0,0],[0, 1,0]]

    Targeting the exact matrix also kills the stray ~9% zoom the unsupervised
    layer drifted into.
    """
    n = flips.shape[0]
    t = torch.zeros(n, 2, 3, device=flips.device)
    s = 1.0 - 2.0 * flips              # +1 upright, -1 flipped
    t[:, 0, 0] = s
    t[:, 1, 1] = s
    return t


@torch.no_grad()
def eval_cer(model, loader, decoder, device):
    model.eval()
    preds, tgts = [], []
    for images, _, _, texts, _ in loader:
        lp = model(images.to(device))
        preds.extend(decoder.decode(lp.cpu()))
        tgts.extend(texts)
    model.train()
    exact = sum(1 for p, t in zip(preds, tgts) if p == t)
    return cer(preds, tgts), exact / max(len(tgts), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--base", type=Path, default=BASE)
    ap.add_argument("--real-csv", type=Path, default=REAL_CSV)
    ap.add_argument("--match", default="/train/", help="Substring selecting REAL train rows.")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--real-oversample", type=int, default=6)
    ap.add_argument("--extra-real-csv", type=Path, default=None,
                    help="Train-only hard-condition rows, e.g. output from augment_real_crops.py.")
    ap.add_argument("--extra-match", default="/train/",
                    help="Substring selecting rows from --extra-real-csv.")
    ap.add_argument("--extra-real-oversample", type=int, default=1)
    ap.add_argument("--synth-n", type=int, default=3000, help="Synthetic samples to mix in.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT,
                    help="where to write the weights. Defaults to the DEPLOYED "
                         "crnn_finetuned.pth — pass a new path to keep the current "
                         "model intact while evaluating a candidate (PLAN_V2 Phase 5).")
    ap.add_argument("--rotate180", type=float, default=0.0,
                    help="probability of flipping a training crop upside-down "
                         "(e.g. 0.5). Teaches the CRNN to read 180-degree plates.")
    ap.add_argument("--stn-supervise", type=float, default=0.0, metavar="W",
                    help="Weight for the STN orientation loss (needs --stn). "
                         "0 = old behaviour (STN learns from CTC alone, and "
                         "measurably never learns to rotate). Try 1.0.")
    ap.add_argument("--stn", action="store_true",
                    help="Way 1: add a learnable straightening layer (Spatial "
                         "Transformer) that turns rotated crops upright before "
                         "reading. Use WITH --rotate180 so it has rotations to learn from.")
    args = ap.parse_args()
    out_path = args.out

    global ROTATE180_PROB
    ROTATE180_PROB = args.rotate180
    if ROTATE180_PROB:
        print(f"[aug] upside-down (180) augmentation ON, p={ROTATE180_PROB}")

    random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not args.base.exists():
        print(f"[X] base weights not found: {args.base}")
        sys.exit(1)

    real = read_csv(args.real_csv, args.match, forbid_test=True)
    if len(real) < 20:
        print(f"[X] only {len(real)} real '{args.match}' labels found in {args.real_csv}.")
        print("    Label train crops first: make_label_sheet.py --split train")
        sys.exit(1)

    random.shuffle(real)
    n_val = max(5, int(len(real) * args.val_frac))
    real_val, real_train = real[:n_val], real[n_val:]

    extra_real = []
    if args.extra_real_csv:
        extra_real = read_csv(args.extra_real_csv, args.extra_match, forbid_test=True)
        if not extra_real:
            print(f"[X] no extra real '{args.extra_match}' rows found in {args.extra_real_csv}.")
            sys.exit(1)

    synth = read_csv(SYNTH_CSV, None)
    random.shuffle(synth)
    synth = synth[:args.synth_n]

    train_samples = (
        real_train * args.real_oversample
        + extra_real * args.extra_real_oversample
        + synth
    )
    random.shuffle(train_samples)

    print("=" * 60)
    print(" FINE-TUNE CRNN ON REAL CROPS")
    print(f"   real train {len(real_train)} (x{args.real_oversample}) + "
          f"extra hard {len(extra_real)} (x{args.extra_real_oversample}) + "
          f"synth {len(synth)} = {len(train_samples)} | real val {len(real_val)}")
    if extra_real:
        print(f"   extra-real CSV is train-only: {args.extra_real_csv}")
    print(f"   lr {args.lr} | epochs {args.epochs} | device {device} | FULL fine-tune")
    print("=" * 60)

    nw = 0 if sys.platform == "win32" else 4
    train_dl = DataLoader(RealCropDataset(train_samples, True), batch_size=args.batch,
                          shuffle=True, collate_fn=collate_with_flip, num_workers=nw, drop_last=True)
    val_dl = DataLoader(RealCropDataset(real_val, False), batch_size=args.batch,
                        shuffle=False, collate_fn=collate_with_flip, num_workers=nw)

    # build model + load synthetic base weights (full fine-tune).
    # Way 1 (--stn): add the learnable straightening layer. The base weights have no
    # STN params, so load them non-strict — the STN stays at its identity init (a
    # no-op) and learns to un-rotate crops from the CTC loss during training.
    model = CRNN(IMG_H, IMG_W, len(CHARSET) + 1, N_HIDDEN, use_stn=args.stn).to(device)
    try:
        state = torch.load(str(args.base), map_location=device, weights_only=True)
    except Exception:
        state = torch.load(str(args.base), map_location=device)
    missing, _ = model.load_state_dict(state, strict=False)
    if args.stn:
        print(f"[stn] straightening layer ON ({sum(1 for m in missing if m.startswith('stn.'))} "
              "new STN params start at identity)")

    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    decoder = CTCDecoder(CHARSET, BLANK)

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    base_cer, _ = eval_cer(model, val_dl, decoder, device)
    print(f"start: real-val CER {base_cer*100:.2f}% (before fine-tuning)")

    best_cer = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        run = n = 0.0
        aux_run = 0.0
        for images, targets, tlen, _, flips in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            images, targets = images.to(device), targets.to(device)
            lp = model(images)
            T, N = lp.size(0), lp.size(1)
            in_len = torch.full((N,), T, dtype=torch.long)
            loss = criterion(lp, targets, in_len, tlen)
            # Supervise the straightening layer directly. We flipped these crops
            # ourselves, so the correct transform is known exactly — no need to
            # hope the CTC loss discovers rotation on its own (it does not).
            if args.stn and args.stn_supervise > 0:
                flips = flips.to(device)
                theta = model.stn.theta(images)
                aux = F.mse_loss(theta, stn_target_theta(flips))
                loss = loss + args.stn_supervise * aux
                aux_run += aux.item()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            run += loss.item(); n += 1
        scheduler.step()
        vcer, vacc = eval_cer(model, val_dl, decoder, device)
        aux_note = (f" | stn {aux_run/max(n,1):.4f}"
                    if args.stn and args.stn_supervise > 0 else "")
        print(f"Epoch {epoch}/{args.epochs} | loss {run/max(n,1):.4f}{aux_note} | "
              f"val CER {vcer*100:.2f}% | val word-acc {vacc*100:.2f}%")
        if vcer < best_cer:
            best_cer = vcer
            torch.save(model.state_dict(), out_path)
            print(f"    [saved] best val CER {vcer*100:.2f}% -> {out_path.name}")

    print("-" * 60)
    print(f"Done. Best real-val CER: {best_cer*100:.2f}% (was {base_cer*100:.2f}%)")
    print(f"Weights -> {out_path}")
    print("\nNow measure on the HELD-OUT test set:")
    print("  python scripts/recognition/evaluate_crnn_on_real.py --split test "
          "--weights models/recognition/crnn_finetuned.pth --tag finetuned")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] fine-tune failed: {exc}")
        raise
