#!/usr/bin/env python3
"""
scripts/tools/test_detector_rotation.py
=======================================
MODEL_IMPROVEMENT_PLAN Step 4 validation — the ROTATION-ROBUSTNESS test.

Rotates each test frame by a range of angles (0 .. 180, both directions), runs the
plate detector, and reports the detection rate at each angle. A rotation-invariant
detector should stay high at every angle; the original (upright-only) one falls off
a cliff past ~+-20deg and is blind at +-90deg.

Run on the current detector and a candidate, compare:
    python scripts/tools/test_detector_rotation.py                                  # best.pt
    python scripts/tools/test_detector_rotation.py --weights models/detection/best_rot.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TEST_IMAGES = PROJECT_ROOT / "data" / "annotated" / "test" / "images"
DEFAULT_W = PROJECT_ROOT / "models" / "detection" / "best.pt"
ANGLES = [0, 10, 20, 30, 45, 90, 135, 180, -10, -20, -30, -45, -90]


def rotate_expand(img, deg):
    """Rotate keeping the whole plate in frame (canvas grows, no crop)."""
    import cv2
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - cx
    M[1, 2] += nh / 2 - cy
    return cv2.warpAffine(img, M, (nw, nh), borderValue=(114, 114, 114))


def main() -> None:
    ap = argparse.ArgumentParser(description="detector rotation-robustness test")
    ap.add_argument("--weights", type=Path, default=DEFAULT_W)
    ap.add_argument("--limit", type=int, default=60, help="test frames to use")
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()

    import cv2
    from detection.detector import PlateDetector
    det = PlateDetector(args.weights, conf=args.conf)

    imgs = [p for p in sorted(TEST_IMAGES.iterdir())
            if p.suffix.lower() == ".jpg"][:args.limit]
    hits = {a: 0 for a in ANGLES}
    n = 0
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        n += 1
        for a in ANGLES:
            frame = rotate_expand(img, a) if a else img
            hits[a] += (len(det.detect(frame)) > 0)

    print("=" * 46)
    print(f" DETECTOR ROTATION ROBUSTNESS  ({args.weights.name})")
    print(f" {n} test frames")
    print("=" * 46)
    print(f" {'angle':>6}  {'detect%':>7}")
    for a in sorted(ANGLES, key=lambda x: (abs(x), x)):
        pct = 100 * hits[a] / max(n, 1)
        bar = "#" * int(pct / 100 * 26)
        print(f" {a:>4}°  {pct:>6.0f}%  {bar}")
    worst = min(100 * hits[a] / max(n, 1) for a in ANGLES)
    print("-" * 46)
    print(f" worst angle detection rate : {worst:.0f}%   (higher = more rotation-robust)")


if __name__ == "__main__":
    main()
