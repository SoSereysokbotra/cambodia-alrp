#!/usr/bin/env python3
"""
scripts/detection/prepare_number_dataset.py
===========================================
Import the Roboflow 'plate-number' export into data/number_detect/ as a clean
YOLO DETECTION dataset (1 class = plate_number).

Roboflow exported the boxes as polygons (class + x1 y1 x2 y2 ...). YOLOv10
detection needs `class cx cy w h`. This converts polygon -> tight bounding box
(and passes through any lines already in 5-value bbox form).

Run:
    python scripts/detection/prepare_number_dataset.py \
        --src "C:/Users/TUF/OneDrive/Pictures/Downloads/cambodian-plate-number.v1i.yolov8"
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
DEST = PROJECT_ROOT / "data" / "number_detect"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def poly_to_bbox(vals: list[float]) -> tuple[float, float, float, float]:
    """vals = [x1,y1,x2,y2,...] normalised -> (cx,cy,w,h) normalised bbox."""
    xs = vals[0::2]
    ys = vals[1::2]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def convert_label(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls = "0"  # force single class
        coords = [float(x) for x in p[1:]]
        if len(coords) == 4:
            cx, cy, w, h = coords                      # already a bbox
        else:
            cx, cy, w, h = poly_to_bbox(coords)        # polygon -> bbox
        # clamp
        cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
        w, h = min(max(w, 0), 1), min(max(h, 0), 1)
        if w <= 0 or h <= 0:
            continue
        out.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="Path to the Roboflow YOLOv8 export folder.")
    ap.add_argument("--clean", action="store_true", help="Wipe dest first.")
    args = ap.parse_args()

    src = args.src
    if not src.is_dir():
        print(f"[X] source not found: {src}")
        sys.exit(1)
    if args.clean and DEST.exists():
        shutil.rmtree(DEST)

    print("=" * 60)
    print(" PREPARE NUMBER-DETECTION DATASET")
    print("=" * 60)
    print(f"src : {src}")
    print(f"dest: {DEST}")

    total_imgs = total_boxes = poly = 0
    for split in ("train", "valid", "test"):
        img_dir = src / split / "images"
        lbl_dir = src / split / "labels"
        if not img_dir.is_dir():
            continue
        dst_img = DEST / split / "images"
        dst_lbl = DEST / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        n = b = 0
        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMG_EXTS:
                continue
            shutil.copy2(img, dst_img / img.name)
            src_txt = lbl_dir / f"{img.stem}.txt"
            lines = convert_label(src_txt.read_text()) if src_txt.exists() else []
            # count polygon conversions for reporting
            if src_txt.exists():
                for raw in src_txt.read_text().splitlines():
                    if len(raw.split()) > 5:
                        poly += 1
            (dst_lbl / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n += 1
            b += len(lines)
        print(f"  [{split}] {n} images, {b} boxes")
        total_imgs += n
        total_boxes += b

    (DEST / "data.yaml").write_text(
        f"path: {DEST.as_posix()}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n\n"
        f"nc: 1\n"
        f"names: ['plate_number']\n",
        encoding="utf-8",
    )

    print("-" * 60)
    print(f"DONE. {total_imgs} images, {total_boxes} boxes "
          f"({poly} polygon labels -> bbox).")
    print(f"data.yaml -> {DEST / 'data.yaml'}")
    print("\nNext: python scripts/detection/train_number_detector.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] prepare failed: {exc}")
        sys.exit(1)
