#!/usr/bin/env python3
"""
scripts/crop_plates_for_crnn.py
===============================
Extract REAL plate crops from the detection dataset using the trained YOLOv10.

These crops are UNLABELLED (we don't know the number text yet). They are NOT
used to train the CRNN in Week 5 — that uses synthetic data with known text.
Instead these real crops are kept for LATER:
  * a small manually-labelled real test set (measure real-world CER), and
  * fine-tuning the synthetic-trained CRNN on real photos.

Output
------
    data/crnn_crops/train/*.jpg
    data/crnn_crops/valid/*.jpg
    data/crnn_crops/test/*.jpg   (each 320x64 grayscale)

Run
---
    python scripts/crop_plates_for_crnn.py
    python scripts/crop_plates_for_crnn.py --limit 200   # quick sample
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

WEIGHTS = PROJECT_ROOT / "models" / "detection" / "best.pt"
SRC_ROOT = PROJECT_ROOT / "data" / "annotated"
OUT_ROOT = PROJECT_ROOT / "data" / "crnn_crops"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CROP_W, CROP_H = 320, 64


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0,
                    help="Max images per split (0 = all).")
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()

    if not WEIGHTS.exists():
        print(f"[X] detector weights missing: {WEIGHTS}")
        sys.exit(1)

    try:
        import cv2
        from detection.detector import PlateDetector
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x, **k):  # graceful fallback
                return x
    except ImportError as exc:
        print(f"[X] missing dependency: {exc}")
        sys.exit(1)

    print("=" * 60)
    print(" CROP REAL PLATES FOR CRNN (unlabelled)")
    print("=" * 60)
    detector = PlateDetector(WEIGHTS, conf=args.conf)

    totals = {"train": 0, "valid": 0, "test": 0}
    processed = 0
    failed = 0

    for split in ("train", "valid", "test"):
        img_dir = SRC_ROOT / split / "images"
        if not img_dir.is_dir():
            print(f"[skip] {split}: {img_dir} not found")
            continue
        out_dir = OUT_ROOT / split
        out_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if args.limit:
            images = images[:args.limit]

        for img_path in tqdm(images, desc=split):
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    failed += 1
                    continue
                dets = detector.detect(image)
                if not dets:
                    failed += 1
                    continue
                for i, det in enumerate(dets):
                    crop = det["crop"]
                    if crop is None or crop.size == 0:
                        continue
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (CROP_W, CROP_H))
                    out_path = out_dir / f"{img_path.stem}_{i}.jpg"
                    cv2.imwrite(str(out_path), resized)
                    totals[split] += 1
                processed += 1
            except Exception as exc:
                failed += 1
                print(f"  [warn] {img_path.name}: {exc}")

    print("-" * 60)
    print(f"Processed images     : {processed}")
    print(f"Crops saved          : train={totals['train']} "
          f"valid={totals['valid']} test={totals['test']}")
    print(f"Failed (no detection): {failed}")
    print(f"Saved to             : {OUT_ROOT}")
    print("\nNote: these are UNLABELLED. CRNN trains on synthetic data.")
    print("Next: python scripts/generate_synthetic_plates.py (if not done)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] crop failed: {exc}")
        sys.exit(1)
