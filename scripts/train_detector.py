#!/usr/bin/env python3
"""
train_detector.py
=================
Train a YOLOv10 Cambodian license-plate DETECTOR on the prepared dataset.

Prerequisite:
    python scripts/prepare_detection_dataset.py    # creates data/annotated/data.yaml

Usage
-----
    # sensible defaults (yolov10n, 100 epochs, 640px, auto GPU):
    python scripts/train_detector.py

    # bigger model / more epochs:
    python scripts/train_detector.py --model yolov10s.pt --epochs 150

    # force CPU (slow) or a specific GPU:
    python scripts/train_detector.py --device cpu
    python scripts/train_detector.py --device 0

After training, the best weights are copied to:
    models/detection/best.pt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "data" / "annotated" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "outputs" / "detection_runs"
WEIGHTS_OUT = PROJECT_ROOT / "models" / "detection"
PRETRAINED_DIR = PROJECT_ROOT / "models" / "pretrained"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_YAML,
                        help=f"Path to data.yaml. Default: {DATA_YAML}")
    parser.add_argument("--model", default="yolov10n.pt",
                        help="Base model: yolov10n/s/m/b/l/x .pt. Default: yolov10n.pt")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (-1 = auto based on VRAM).")
    parser.add_argument("--device", default=None,
                        help="'0' for GPU 0, 'cpu', or leave blank for auto.")
    parser.add_argument("--name", default="plate_detector",
                        help="Run name under outputs/detection_runs/.")
    parser.add_argument("--patience", type=int, default=25,
                        help="Early-stopping patience (epochs). Default: 25.")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"ERROR: {args.data} not found.\n"
            "Run scripts/prepare_detection_dataset.py first."
        )

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics not installed. Activate your venv and run:\n"
            "    pip install ultralytics"
        )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_OUT.mkdir(parents=True, exist_ok=True)
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("YOLOv10 plate-detector training")
    print(f"  data   : {args.data}")
    print(f"  model  : {args.model}")
    print(f"  epochs : {args.epochs}   imgsz: {args.imgsz}   batch: {args.batch}")
    print(f"  device : {args.device or 'auto'}")
    print("=" * 60)

    # Ultralytics downloads the base weights automatically if not present.
    model = YOLO(args.model)

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
    )

    # Locate and copy the best weights to models/detection/best.pt
    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else RUNS_DIR / args.name
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        dest = WEIGHTS_OUT / "best.pt"
        shutil.copy2(best, dest)
        print(f"\nBest weights copied to: {dest}")
    else:
        print(f"\nWARNING: best.pt not found under {run_dir}. Check the run folder.")

    print("\nTraining complete.")
    print(f"Full run (plots, metrics): {run_dir}")
    print("\nNext: auto-label your own photos with")
    print("    python scripts/auto_label.py --images <folder-of-your-photos>")


if __name__ == "__main__":
    main()
