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


def read_csv(path: Path, match: str | None):
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
            if match and match not in p.replace("\\", "/"):
                continue
            rows.append((p, t))
    return rows


def augment(img: np.ndarray) -> np.ndarray:
    """Light appearance augmentation to expand the small real set."""
    import cv2
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
    return img


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
        if self.aug:
            img = augment(img)
        t = torch.from_numpy(img.astype("float32") / 255.0).unsqueeze(0)
        t = (t - 0.5) / 0.5
        target = torch.tensor([CHAR_TO_IDX[c] for c in text], dtype=torch.long)
        return t, target, len(target), text


@torch.no_grad()
def eval_cer(model, loader, decoder, device):
    model.eval()
    preds, tgts = [], []
    for images, _, _, texts in loader:
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
    ap.add_argument("--synth-n", type=int, default=3000, help="Synthetic samples to mix in.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not args.base.exists():
        print(f"[X] base weights not found: {args.base}")
        sys.exit(1)

    real = read_csv(args.real_csv, args.match)
    if len(real) < 20:
        print(f"[X] only {len(real)} real '{args.match}' labels found in {args.real_csv}.")
        print("    Label train crops first: make_label_sheet.py --split train")
        sys.exit(1)

    random.shuffle(real)
    n_val = max(5, int(len(real) * args.val_frac))
    real_val, real_train = real[:n_val], real[n_val:]

    synth = read_csv(SYNTH_CSV, None)
    random.shuffle(synth)
    synth = synth[:args.synth_n]

    train_samples = real_train * args.real_oversample + synth
    random.shuffle(train_samples)

    print("=" * 60)
    print(" FINE-TUNE CRNN ON REAL CROPS")
    print(f"   real train {len(real_train)} (x{args.real_oversample}) + synth {len(synth)} "
          f"= {len(train_samples)} | real val {len(real_val)}")
    print(f"   lr {args.lr} | epochs {args.epochs} | device {device} | FULL fine-tune")
    print("=" * 60)

    nw = 0 if sys.platform == "win32" else 4
    train_dl = DataLoader(RealCropDataset(train_samples, True), batch_size=args.batch,
                          shuffle=True, collate_fn=collate_fn, num_workers=nw, drop_last=True)
    val_dl = DataLoader(RealCropDataset(real_val, False), batch_size=args.batch,
                        shuffle=False, collate_fn=collate_fn, num_workers=nw)

    # build model + load synthetic base weights (full fine-tune)
    model = CRNN(IMG_H, IMG_W, len(CHARSET) + 1, N_HIDDEN).to(device)
    try:
        state = torch.load(str(args.base), map_location=device, weights_only=True)
    except Exception:
        state = torch.load(str(args.base), map_location=device)
    model.load_state_dict(state)

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
        for images, targets, tlen, _ in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            images, targets = images.to(device), targets.to(device)
            lp = model(images)
            T, N = lp.size(0), lp.size(1)
            in_len = torch.full((N,), T, dtype=torch.long)
            loss = criterion(lp, targets, in_len, tlen)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            run += loss.item(); n += 1
        scheduler.step()
        vcer, vacc = eval_cer(model, val_dl, decoder, device)
        print(f"Epoch {epoch}/{args.epochs} | loss {run/max(n,1):.4f} | "
              f"val CER {vcer*100:.2f}% | val word-acc {vacc*100:.2f}%")
        if vcer < best_cer:
            best_cer = vcer
            torch.save(model.state_dict(), OUT)
            print(f"    [saved] best val CER {vcer*100:.2f}% -> {OUT.name}")

    print("-" * 60)
    print(f"Done. Best real-val CER: {best_cer*100:.2f}% (was {base_cer*100:.2f}%)")
    print(f"Weights -> {OUT}")
    print("\nNow measure on the HELD-OUT test set:")
    print("  python scripts/recognition/evaluate_crnn_on_real.py --split test "
          "--weights models/recognition/crnn_finetuned.pth --tag finetuned")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] fine-tune failed: {exc}")
        raise
