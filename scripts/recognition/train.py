#!/usr/bin/env python3
"""
scripts/train_crnn_week5.py
===========================
Train the CRNN plate-number reader on synthetic data with CTC loss.

Prerequisite:
    python scripts/generate_synthetic_plates.py   # creates data/synthetic/*

Outputs:
    models/recognition/crnn_best.pth   (best-CER weights)
    models/recognition/charset.txt     (the character set, for inference)

Run:
    python scripts/train_crnn_week5.py
    python scripts/train_crnn_week5.py --epochs 30 --batch 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src" / "recognition"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from crnn_model import CRNN, CTCDecoder, CHARSET, BLANK   # noqa: E402
from crnn_dataset import PlateDataset, collate_fn          # noqa: E402

# ---- configuration ---- #
IMG_H, IMG_W = 64, 320
N_HIDDEN = 256
DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
OUT_DIR = PROJECT_ROOT / "models" / "recognition"


def levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings (for CER)."""
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
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def calculate_cer(predictions: list[str], targets: list[str]) -> float:
    """Character Error Rate over a set of (pred, target) pairs."""
    total_edits, total_chars = 0, 0
    for pred, tgt in zip(predictions, targets):
        total_edits += levenshtein(pred, tgt)
        total_chars += max(len(tgt), 1)
    return total_edits / max(total_chars, 1)


@torch.no_grad()
def validate(model, loader, decoder, device) -> tuple[float, float]:
    """Return (CER, word_accuracy) over a loader."""
    model.eval()
    preds_all, tgts_all = [], []
    for images, _, _, texts in loader:
        images = images.to(device)
        log_probs = model(images)               # (seq, batch, classes)
        preds = decoder.decode(log_probs.cpu())
        preds_all.extend(preds)
        tgts_all.extend(texts)
    cer = calculate_cer(preds_all, tgts_all)
    exact = sum(1 for p, t in zip(preds_all, tgts_all) if p == t)
    word_acc = exact / max(len(tgts_all), 1)
    model.train()
    return cer, word_acc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None, help="'cuda', 'cpu', or auto.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_csv = DATA_DIR / "train_labels.csv"
    valid_csv = DATA_DIR / "valid_labels.csv"
    if not train_csv.exists():
        print(f"[X] {train_csv} not found. Run generate_synthetic_plates.py first.")
        sys.exit(1)

    print("=" * 60)
    print(" WEEK 5 — TRAIN CRNN (synthetic data, CTC)")
    print(f"   device {device} | epochs {args.epochs} | batch {args.batch} | lr {args.lr}")
    print("=" * 60)

    train_ds = PlateDataset(train_csv, IMG_H, IMG_W, CHARSET)
    valid_ds = PlateDataset(valid_csv, IMG_H, IMG_W, CHARSET)
    print(f"train samples: {len(train_ds)} | valid samples: {len(valid_ds)}")

    # num_workers=0 on Windows avoids loader spawn issues; user can raise it.
    nw = args.workers if sys.platform != "win32" else 0
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=collate_fn, num_workers=nw, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch, shuffle=False,
                          collate_fn=collate_fn, num_workers=nw)

    model = CRNN(IMG_H, IMG_W, len(CHARSET) + 1, N_HIDDEN).to(device)
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    decoder = CTCDecoder(CHARSET, BLANK)

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    best_cer = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for images, targets, target_lengths, _ in tqdm(
                train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            log_probs = model(images)                       # (seq, N, C)
            seq_len, n = log_probs.size(0), log_probs.size(1)
            input_lengths = torch.full((n,), seq_len, dtype=torch.long)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = running / max(n_batches, 1)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f} | LR: {cur_lr:.6f}")

        # validate every 5 epochs (and on the final epoch)
        if epoch % 5 == 0 or epoch == args.epochs:
            cer, word_acc = validate(model, valid_dl, decoder, device)
            print(f"    Valid CER: {cer * 100:.2f}%  |  Word acc: {word_acc * 100:.2f}%")
            if cer < best_cer:
                best_cer = cer
                torch.save(model.state_dict(), OUT_DIR / "crnn_best.pth")
                (OUT_DIR / "charset.txt").write_text(CHARSET, encoding="utf-8")
                print(f"    [saved] new best CER {cer * 100:.2f}% -> crnn_best.pth")

    print("-" * 60)
    print(f"Training complete. Best CER: {best_cer * 100:.2f}%")
    print(f"Weights : {OUT_DIR / 'crnn_best.pth'}")
    print(f"Charset : {OUT_DIR / 'charset.txt'}")
    print("\nNext: python scripts/evaluate_crnn_week5.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] training failed: {exc}")
        raise


