#!/usr/bin/env python3
"""
scripts/tools/make_montage.py
=============================
Compose number crops into indexed montage images so they can be transcribed in
bulk (e.g. by a vision model reading each montage). Writes a JSON map from the
global index shown on each cell -> the crop's relative path.

Output:
    results/montages/montage_00.jpg ...
    results/montages/montage_map.json   { "0": "data/crnn_crops/train/xxx.jpg", ... }

Run:
    python scripts/tools/make_montage.py --split train --count 160 --per 20 --cols 2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
CROPS_ROOT = PROJECT_ROOT / "data" / "crnn_crops"
REAL_CSV = CROPS_ROOT / "real_labels.csv"
OUT_DIR = PROJECT_ROOT / "results" / "montages"
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def load_done():
    done = set()
    if REAL_CSV.exists():
        with open(REAL_CSV, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                done.add(r["image_path"])
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "valid", "test"])
    ap.add_argument("--count", type=int, default=160)
    ap.add_argument("--per", type=int, default=20)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--offset", type=int, default=0, help="Skip this many crops first.")
    args = ap.parse_args()

    import cv2
    import numpy as np

    d = CROPS_ROOT / args.split
    crops = sorted(str(p) for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
    done = load_done()
    crops = [c for c in crops
             if Path(c).relative_to(PROJECT_ROOT).as_posix() not in done]
    crops = crops[args.offset: args.offset + args.count]
    if not crops:
        print("[X] no unlabelled crops for this split.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*"):
        old.unlink()

    cols = args.cols
    rows = (args.per + cols - 1) // cols
    cw, ch, lh = 460, 92, 30           # cell width, crop height, label strip
    cell_h = ch + lh
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    mapping = {}
    gi = 0
    montage_idx = 0
    for start in range(0, len(crops), args.per):
        batch = crops[start:start + args.per]
        canvas = np.full((rows * cell_h, cols * cw, 3), 30, np.uint8)
        for j, path in enumerate(batch):
            r, c = j // cols, j % cols
            x, y = c * cw, r * cell_h
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            enh = clahe.apply(gray)
            enh = cv2.resize(enh, (cw - 12, ch), interpolation=cv2.INTER_CUBIC)
            enh = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
            cv2.putText(canvas, f"#{gi}", (x + 6, y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            canvas[y + lh:y + lh + ch, x + 6:x + 6 + (cw - 12)] = enh
            mapping[str(gi)] = Path(path).relative_to(PROJECT_ROOT).as_posix()
            gi += 1
        out = OUT_DIR / f"montage_{montage_idx:02d}.jpg"
        cv2.imwrite(str(out), canvas)
        montage_idx += 1

    (OUT_DIR / "montage_map.json").write_text(
        json.dumps(mapping, indent=0), encoding="utf-8")
    print(f"[OK] {montage_idx} montages, {gi} crops -> {OUT_DIR}")
    print(f"map -> {OUT_DIR / 'montage_map.json'}")


if __name__ == "__main__":
    main()
