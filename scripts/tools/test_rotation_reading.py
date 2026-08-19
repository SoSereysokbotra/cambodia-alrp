#!/usr/bin/env python3
"""
scripts/tools/test_rotation_reading.py
======================================
Measure how well a CRNN reads the NUMBER upright vs upside-down (180 deg).

This is the honest before/after for training the number reader on rotated crops
(finetune_crnn.py --rotate180). Run it on the deployed model and a rotation-trained
candidate and compare — no assumptions, just the numbers on the held-out test crops.

    python scripts/tools/test_rotation_reading.py --weights models/recognition/crnn_finetuned.pth
    python scripts/tools/test_rotation_reading.py --weights models/recognition/crnn_rot.pth
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "recognition"))

REAL_CSV = PROJECT_ROOT / "data" / "crnn_crops" / "real_labels.csv"
DEFAULT_W = PROJECT_ROOT / "models" / "recognition" / "crnn_finetuned.pth"


def norm(s):
    return (s or "").upper().replace(" ", "").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="CRNN upright vs upside-down reading")
    ap.add_argument("--weights", type=Path, default=DEFAULT_W)
    ap.add_argument("--limit", type=int, default=149)
    args = ap.parse_args()

    import cv2
    from crnn_reader import CRNNReader
    reader = CRNNReader(args.weights, PROJECT_ROOT / "models" / "recognition" / "charset.txt")

    rows = []
    with open(REAL_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if "/test/" not in r["image_path"].replace("\\", "/"):
                continue
            p = PROJECT_ROOT / r["image_path"]
            if p.exists():
                rows.append((p, norm(r["plate_text"])))
    rows = rows[:args.limit]

    up_ok = down_ok = n = 0
    for p, gt in rows:
        img = cv2.imread(str(p))
        if img is None:
            continue
        n += 1
        up, _ = reader.read(img)
        down, _ = reader.read(cv2.rotate(img, cv2.ROTATE_180))
        up_ok += (norm(up) == gt)
        down_ok += (norm(down) == gt)

    print("=" * 50)
    print(f" CRNN READING  ({args.weights.name})  n={n} test crops")
    print("=" * 50)
    print(f"  upright        : {100*up_ok/max(n,1):.1f}%  ({up_ok}/{n})")
    print(f"  upside-down    : {100*down_ok/max(n,1):.1f}%  ({down_ok}/{n})")
    print("=" * 50)
    print(" a rotation-trained model should lift upside-down while keeping upright.")


if __name__ == "__main__":
    main()
