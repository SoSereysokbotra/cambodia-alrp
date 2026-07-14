#!/usr/bin/env python3
"""
scripts/recognition/crop_numbers.py
===================================
Re-crop the NUMBER line from the real images using the trained number detector
(models/detection/number_best.pt). These crops finally contain ONLY the plate
number (e.g. "2A-0243"), which is what CRNN needs — unlike the old province
crops from crop_plates.py.

Replaces the contents of data/crnn_crops/{train,valid,test}/ with number crops.

Run:
    python scripts/recognition/crop_numbers.py
    python scripts/recognition/crop_numbers.py --conf 0.4 --limit 300
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src"))

WEIGHTS = PROJECT_ROOT / "models" / "detection" / "number_best.pt"
SRC_ROOT = PROJECT_ROOT / "data" / "annotated"
OUT_ROOT = PROJECT_ROOT / "data" / "crnn_crops"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CROP_W, CROP_H = 320, 64


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--limit", type=int, default=0, help="Max images per split (0=all).")
    ap.add_argument("--keep-gray", action="store_true",
                    help="Save grayscale (default). Use --no for BGR.")
    args = ap.parse_args()

    if not WEIGHTS.exists():
        print(f"[X] number detector not found: {WEIGHTS}. Train it first.")
        sys.exit(1)

    import cv2
    from detection.detector import PlateDetector
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    print("=" * 60)
    print(" RE-CROP NUMBER LINES (number_best.pt)")
    print("=" * 60)
    detector = PlateDetector(WEIGHTS, conf=args.conf)

    totals = {"train": 0, "valid": 0, "test": 0}
    processed = failed = 0
    for split in ("train", "valid", "test"):
        img_dir = SRC_ROOT / split / "images"
        if not img_dir.is_dir():
            continue
        out_dir = OUT_ROOT / split
        if out_dir.exists():
            shutil.rmtree(out_dir)   # replace old province crops
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
                # keep the highest-confidence number box
                best = max(dets, key=lambda d: d["confidence"])
                crop = best["crop"]
                if crop is None or crop.size == 0:
                    failed += 1
                    continue
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                out = cv2.resize(gray, (CROP_W, CROP_H))
                cv2.imwrite(str(out_dir / f"{img_path.stem}.jpg"), out)
                totals[split] += 1
                processed += 1
            except Exception as exc:
                failed += 1
                print(f"  [warn] {img_path.name}: {exc}")

    print("-" * 60)
    print(f"Number crops: train={totals['train']} valid={totals['valid']} test={totals['test']}")
    print(f"No number detected (skipped): {failed}")
    print(f"Saved to: {OUT_ROOT}")
    print("\nNext: eyeball a few, then transcribe with label_real_crops.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] crop failed: {exc}")
        sys.exit(1)
