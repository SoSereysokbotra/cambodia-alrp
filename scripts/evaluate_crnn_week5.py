#!/usr/bin/env python3
"""
scripts/evaluate_crnn_week5.py
==============================
Evaluate the trained CRNN on the synthetic TEST set.

Metrics:
    * CER (Character Error Rate) — primary
    * Word accuracy (exact plate match)
    * 10 sample predictions (predicted vs actual)

Saves metrics/crnn_week5_metrics.json.

Run:
    python scripts/evaluate_crnn_week5.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "recognition"))

import torch
from torch.utils.data import DataLoader

from crnn_model import CTCDecoder, CHARSET, BLANK, load_crnn   # noqa: E402
from crnn_dataset import PlateDataset, collate_fn               # noqa: E402
from train_crnn_week5 import calculate_cer                      # reuse CER

IMG_H, IMG_W = 64, 320
DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
MODEL_DIR = PROJECT_ROOT / "models" / "recognition"
WEIGHTS = MODEL_DIR / "crnn_best.pth"
METRICS_JSON = PROJECT_ROOT / "metrics" / "crnn_week5_metrics.json"

CER_TARGET = 0.15       # < 15%
WACC_TARGET = 0.70      # > 70%


def main() -> None:
    if not WEIGHTS.exists():
        print(f"[X] {WEIGHTS} not found. Train first: train_crnn_week5.py")
        sys.exit(1)
    test_csv = DATA_DIR / "test_labels.csv"
    if not test_csv.exists():
        print(f"[X] {test_csv} not found. Generate synthetic data first.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load charset used at train time (fallback to default)
    charset_path = MODEL_DIR / "charset.txt"
    charset = charset_path.read_text(encoding="utf-8") if charset_path.exists() else CHARSET

    print("=" * 60)
    print(" WEEK 5 — EVALUATE CRNN (synthetic test set)")
    print("=" * 60)

    model = load_crnn(WEIGHTS, device=device, charset=charset,
                      img_h=IMG_H, img_w=IMG_W)
    decoder = CTCDecoder(charset, len(charset))

    ds = PlateDataset(test_csv, IMG_H, IMG_W, charset)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    preds_all, tgts_all = [], []
    with torch.no_grad():
        for images, _, _, texts in dl:
            images = images.to(device)
            log_probs = model(images)
            preds_all.extend(decoder.decode(log_probs.cpu()))
            tgts_all.extend(texts)

    cer = calculate_cer(preds_all, tgts_all)
    exact = sum(1 for p, t in zip(preds_all, tgts_all) if p == t)
    word_acc = exact / max(len(tgts_all), 1)

    cer_ok = "✓" if cer < CER_TARGET else "✗"
    wacc_ok = "✓" if word_acc > WACC_TARGET else "✗"

    print("\n=== CRNN EVALUATION ===")
    print(f"Test samples:    {len(tgts_all)}")
    print(f"CER:             {cer * 100:.2f}%  [TARGET: < 15%]  {cer_ok}")
    print(f"Word accuracy:   {word_acc * 100:.2f}%  [TARGET: > 70%]  {wacc_ok}")

    print("\nSample predictions:")
    for p, t in list(zip(preds_all, tgts_all))[:10]:
        mark = "✓" if p == t else "✗"
        print(f'  Actual: "{t}"   Predicted: "{p}"   {mark}')

    metrics = {
        "cer": round(cer, 4),
        "word_accuracy": round(word_acc, 4),
        "test_samples": len(tgts_all),
        "week": 5,
    }
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\n[OK] metrics saved -> {METRICS_JSON}")
    print("\nNext: python scripts/test_crnn_inference_week5.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] evaluation failed: {exc}")
        sys.exit(1)
