#!/usr/bin/env python3
"""
scripts/test_crnn_inference_week5.py
====================================
Latency + sanity check for the CRNN reader on a few synthetic test crops.

Target: < 50 ms per crop.

Run:
    python scripts/test_crnn_inference_week5.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src" / "recognition"))

MODEL_DIR = PROJECT_ROOT / "models" / "recognition"
WEIGHTS = MODEL_DIR / "crnn_best.pth"
CHARSET_TXT = MODEL_DIR / "charset.txt"
TEST_DIR = PROJECT_ROOT / "data" / "synthetic" / "test"
IMG_EXTS = {".jpg", ".jpeg", ".png"}
TARGET_MS = 50


def main() -> None:
    if not WEIGHTS.exists():
        print(f"[X] {WEIGHTS} not found. Train first.")
        sys.exit(1)
    if not TEST_DIR.is_dir():
        print(f"[X] {TEST_DIR} not found. Generate synthetic data first.")
        sys.exit(1)

    import cv2
    from crnn_reader import CRNNReader

    reader = CRNNReader(WEIGHTS, CHARSET_TXT)

    images = [p for p in TEST_DIR.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not images:
        print(f"[X] no crops in {TEST_DIR}")
        sys.exit(1)
    random.seed(0)
    picks = random.sample(images, min(5, len(images)))

    print("=" * 60)
    print(" WEEK 5 — CRNN INFERENCE TEST")
    print("=" * 60)
    print(f"\n{'crop':<26}{'predicted':<14}{'latency_ms':>12}")
    print("-" * 52)

    latencies = []
    for img_path in picks:
        crop = cv2.imread(str(img_path))
        if crop is None:
            continue
        text, conf = reader.read(crop)
        lat = reader.get_avg_latency_ms()  # last-call avg; per-call below
        latencies.append(reader._latencies[-1] if reader._latencies else 0.0)
        name = img_path.name[:24]
        print(f"{name:<26}{text:<14}{latencies[-1]:>11.1f}")

    if latencies:
        avg = sum(latencies) / len(latencies)
        verdict = "✓ under target" if avg < TARGET_MS else "✗ over target"
        print("-" * 52)
        print(f"Average latency: {avg:.1f} ms  (target < {TARGET_MS} ms)  {verdict}")
    print("\nNext: python scripts/alpr_pipeline_week5.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] inference test failed: {exc}")
        sys.exit(1)
