#!/usr/bin/env python3
"""
test_inference_week2.py
=======================
STEP 4 of Week 2 — visual sanity check of the detector on real images.

Picks 5 random TEST images, runs detection, draws GREEN boxes with the
confidence score, saves the annotated images, and prints a latency table.

Outputs:
    * results/week2_detections/*.jpg   (annotated images)
    * a per-image latency + confidence table + average latency

Run (AFTER training):
    python test_inference_week2.py
    python test_inference_week2.py --num 10 --device cpu
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), Path(__file__).resolve().parents[2])
WEIGHTS = PROJECT_ROOT / "models" / "detection" / "best.pt"
TEST_IMAGES_DIR = PROJECT_ROOT / "data" / "annotated" / "test" / "images"
OUT_DIR = PROJECT_ROOT / "results" / "week2_detections"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GREEN = (0, 255, 0)          # BGR for OpenCV
TARGET_MS = 120


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--images", type=Path, default=TEST_IMAGES_DIR)
    parser.add_argument("--num", type=int, default=5, help="How many images to test.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"[X] weights not found: {args.weights}. Train first.")
        sys.exit(1)
    if not args.images.is_dir():
        print(f"[X] test images folder not found: {args.images}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
        import cv2
    except ImportError as exc:
        print(f"[X] missing dependency: {exc}. pip install ultralytics opencv-python")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_imgs = [p for p in args.images.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not all_imgs:
        print(f"[X] no images in {args.images}")
        sys.exit(1)

    random.seed(args.seed)
    picks = random.sample(all_imgs, min(args.num, len(all_imgs)))

    print("=" * 66)
    print(" WEEK 2 — STEP 4: INFERENCE TEST (visual check)")
    print("=" * 66)

    model = YOLO(str(args.weights))
    # Warm-up run so the first timing isn't skewed by lazy CUDA init.
    model.predict(source=str(picks[0]), device=args.device,
                  conf=args.conf, verbose=False)

    print(f"\n{'image':<38}{'latency_ms':>12}{'plates':>9}{'conf':>8}")
    print("-" * 66)

    latencies = []
    for img_path in picks:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"{img_path.name:<38}{'READ ERROR':>12}")
            continue

        t0 = time.perf_counter()
        result = model.predict(source=img, device=args.device,
                               conf=args.conf, verbose=False)[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        boxes = result.boxes
        n_plates = 0 if boxes is None else len(boxes)
        best_conf = 0.0

        if boxes is not None:
            for b in boxes:
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                conf = float(b.conf[0])
                best_conf = max(best_conf, conf)
                cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)
                label = f"plate {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), GREEN, -1)
                cv2.putText(img, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        out_path = OUT_DIR / f"det_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), img)

        conf_str = f"{best_conf:.2f}" if n_plates else "-"
        name = img_path.name if len(img_path.name) <= 36 else img_path.name[:33] + "..."
        print(f"{name:<38}{latency_ms:>11.1f}{n_plates:>9}{conf_str:>8}")

    print("-" * 66)
    if latencies:
        avg = sum(latencies) / len(latencies)
        verdict = "✓ under target" if avg < TARGET_MS else "✗ over target"
        print(f"Average latency: {avg:.1f} ms  (target < {TARGET_MS} ms)  {verdict}")
    print(f"Annotated images saved to: {OUT_DIR}")
    print("\nNext: python summary_week2.py")


if __name__ == "__main__":
    main()
