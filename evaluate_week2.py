#!/usr/bin/env python3
"""
evaluate_week2.py
=================
STEP 3 of Week 2 — evaluate the trained detector on the TEST set.

The test set (504 images) was NEVER seen during training, so these numbers
are the honest measure of how good the detector is.

Outputs:
    * prints mAP50, mAP50-95, Precision, Recall, inference speed
    * saves metrics/week2_metrics.json

Run (AFTER training):
    python evaluate_week2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "data" / "annotated" / "data.yaml"
WEIGHTS = PROJECT_ROOT / "models" / "detection" / "best.pt"
METRICS_DIR = PROJECT_ROOT / "metrics"
METRICS_JSON = METRICS_DIR / "week2_metrics.json"

TRAIN_IMAGES = 2054
TEST_IMAGES = 504
DATASET_CREDIT = "Plate_v4 (taki-dk0de, CC BY 4.0)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"[X] weights not found: {args.weights}")
        print("    Train first: python train_yolov10_week2.py")
        sys.exit(1)
    if not args.data.exists():
        print(f"[X] data.yaml not found: {args.data}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[X] ultralytics not installed. pip install ultralytics")
        sys.exit(1)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" WEEK 2 — STEP 3: EVALUATE ON TEST SET (504 unseen images)")
    print("=" * 60)

    model = YOLO(str(args.weights))
    # split='test' forces evaluation on the held-out test images.
    m = model.val(data=str(args.data), split="test",
                  device=args.device, verbose=False)

    # ultralytics speed dict is ms/image: preprocess + inference + postprocess
    speed = getattr(m, "speed", {}) or {}
    inference_ms = float(speed.get("inference", 0.0))

    metrics = {
        "mAP50": round(float(m.box.map50), 4),
        "mAP50-95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "inference_ms": round(inference_ms, 1),
        "dataset": DATASET_CREDIT,
        "train_images": TRAIN_IMAGES,
        "test_images": TEST_IMAGES,
        "week": 2,
    }

    print("\n" + "-" * 60)
    print(" TEST-SET METRICS")
    print(f"   mAP50        : {metrics['mAP50']:.4f}")
    print(f"   mAP50-95     : {metrics['mAP50-95']:.4f}")
    print(f"   Precision    : {metrics['precision']:.4f}")
    print(f"   Recall       : {metrics['recall']:.4f}")
    print(f"   Inference    : {metrics['inference_ms']:.1f} ms/image")
    print("-" * 60)

    METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"[OK] metrics saved -> {METRICS_JSON}")

    target = 0.82
    verdict = "✓ meets target" if metrics["mAP50"] >= target else "✗ below target"
    print(f"\nmAP50 {metrics['mAP50']:.4f} vs target {target}: {verdict}")
    print("\nNext: python test_inference_week2.py")


if __name__ == "__main__":
    main()
