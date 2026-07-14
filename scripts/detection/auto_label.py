#!/usr/bin/env python3
"""
auto_label.py
=============
Use your trained YOLOv10 detector to AUTO-LABEL your own plate photos.

Instead of drawing 2,000+ boxes by hand, this runs the detector over a folder
of images and writes a YOLO-format label (.txt) for each one. You then only
need to *review and fix* the results (fast) instead of annotating from scratch.

Prerequisite:
    A trained detector at models/detection/best.pt
    (produced by scripts/train_detector.py)

Usage
-----
    python scripts/auto_label.py --images "C:/path/to/your/photos"

    # tune confidence, output location, and copy images alongside labels:
    python scripts/auto_label.py --images data/raw --conf 0.35 --copy-images

Output (default: data/auto_labeled/)
    data/auto_labeled/
      images/           # (only if --copy-images) copies of your photos
      labels/           # one <name>.txt per image, class 0 = license_plate
      preview/          # (only if --save-preview) images with boxes drawn

How to review
-------------
Option 1 (recommended): upload data/auto_labeled/ to a Roboflow project by
selecting the FOLDER (images + labels together) -> Roboflow imports the boxes
-> fix mistakes in the Annotate tab -> Generate -> Export.

Option 2: inspect the --save-preview images to eyeball quality, then hand-fix
only the bad ones.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), Path(__file__).resolve().parents[2])
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "detection" / "best.pt"
DEFAULT_OUT = PROJECT_ROOT / "data" / "auto_labeled"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True,
                        help="Folder containing your photos to auto-label.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help=f"Trained detector weights. Default: {DEFAULT_WEIGHTS}")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output folder. Default: {DEFAULT_OUT}")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold for keeping a detection. Default: 0.35")
    parser.add_argument("--device", default=None, help="'0' for GPU, 'cpu', or blank for auto.")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy the source images next to the labels (for Roboflow re-upload).")
    parser.add_argument("--save-preview", action="store_true",
                        help="Also save images with the predicted boxes drawn, for quick review.")
    args = parser.parse_args()

    if not args.weights.exists():
        raise SystemExit(
            f"ERROR: weights not found: {args.weights}\n"
            "Train the detector first: python scripts/train_detector.py"
        )
    if not args.images.is_dir():
        raise SystemExit(f"ERROR: images folder not found: {args.images}")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed. Run: pip install ultralytics")

    images = [p for p in sorted(args.images.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if not images:
        raise SystemExit(f"No images found under {args.images}")

    lbl_dir = args.out / "labels"
    img_dir = args.out / "images"
    prev_dir = args.out / "preview"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_images:
        img_dir.mkdir(parents=True, exist_ok=True)
    if args.save_preview:
        prev_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading detector: {args.weights}")
    model = YOLO(str(args.weights))

    print(f"Auto-labeling {len(images)} images (conf >= {args.conf}) ...")
    n_boxes_total = 0
    n_with_boxes = 0

    for i, img_path in enumerate(images, 1):
        # single-image inference; verbose off to keep the log clean
        result = model.predict(source=str(img_path), conf=args.conf,
                               device=args.device, verbose=False)[0]

        # Build YOLO label lines. We force class 0 = license_plate.
        lines = []
        boxes = result.boxes
        if boxes is not None and boxes.xywhn is not None:
            for xywhn in boxes.xywhn.tolist():
                cx, cy, w, h = xywhn
                lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        (lbl_dir / f"{img_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        n_boxes_total += len(lines)
        if lines:
            n_with_boxes += 1

        if args.copy_images:
            shutil.copy2(img_path, img_dir / img_path.name)
        if args.save_preview:
            # result.plot() returns a BGR numpy array with boxes drawn
            try:
                import cv2
                cv2.imwrite(str(prev_dir / img_path.name), result.plot())
            except Exception:
                pass

        if i % 100 == 0 or i == len(images):
            print(f"  {i}/{len(images)} done "
                  f"({n_with_boxes} with a plate, {n_boxes_total} boxes total)")

    # A data.yaml so the folder is directly trainable / uploadable.
    (args.out / "data.yaml").write_text(
        f"path: {args.out.as_posix()}\n"
        f"train: images\n"
        f"val: images\n"
        f"nc: 1\n"
        f"names: ['license_plate']\n"
    )

    print("-" * 60)
    print(f"DONE. Labels written to: {lbl_dir}")
    print(f"  images with a detected plate : {n_with_boxes}/{len(images)}")
    print(f"  total boxes                  : {n_boxes_total}")
    no_plate = len(images) - n_with_boxes
    if no_plate:
        print(f"  NOTE: {no_plate} images got NO detection — review these "
              f"(empty label = 'no plate'). Lower --conf if too many.")
    print("\nNext: REVIEW the labels before training on them.")
    print("  - Upload the folder (images + labels) to Roboflow to review/fix, OR")
    print("  - use --save-preview to eyeball the drawn boxes.")


if __name__ == "__main__":
    main()
