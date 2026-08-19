#!/usr/bin/env python3
"""
scripts/tools/test_orientation_search.py
========================================
MODEL_IMPROVEMENT_PLAN Step 5 — prove end-to-end any-angle reading.

Rotates each test frame to a "camera orientation" (0/90/180/270), then reads it two
ways and compares NUMBER accuracy vs ground truth:
  * NAIVE          — detect + read once (what the live pipeline does today).
  * ORIENTATION    — try the frame at 0/90/180/270, read each, keep the read with the
                     best (crnn_conf x format-validity). No new training; the readers
                     themselves judge which orientation is upright.

If the approach works, NAIVE collapses at 90/180/270 while ORIENTATION stays high.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "recognition"))

TEST_IMAGES = PROJECT_ROOT / "data" / "annotated" / "test" / "images"
REAL_CSV = PROJECT_ROOT / "data" / "crnn_crops" / "real_labels.csv"
INVALID_PENALTY = 0.25          # weight a format-invalid read this much less


def norm(s):
    return (s or "").upper().replace(" ", "").strip()


def rot90k(img, k):
    import numpy as np
    return np.ascontiguousarray(np.rot90(img, k))     # k*90deg counter-clockwise


def read_naive(frame, det, reader):
    ds = det.detect(frame)
    if not ds:
        return ""
    best = max(ds, key=lambda d: d["confidence"])
    num, _ = reader.read(best["crop"])
    return norm(num)


def read_oriented(frame, det, reader, is_valid):
    """Try 4 orientations; return the read that scores best (conf x validity)."""
    best_txt, best_score = "", -1.0
    for k in range(4):                      # 0, 90, 180, 270
        rf = rot90k(frame, k)
        ds = det.detect(rf)
        for d in ds:
            num, conf = reader.read(d["crop"])
            num = norm(num)
            if not num:
                continue
            score = conf * (1.0 if is_valid(num) else INVALID_PENALTY)
            if score > best_score:
                best_score, best_txt = score, num
    return best_txt


def main() -> None:
    ap = argparse.ArgumentParser(description="any-angle reading proof (Step 5)")
    ap.add_argument("--weights", type=Path,
                    default=PROJECT_ROOT / "models" / "detection" / "best.pt")
    ap.add_argument("--limit", type=int, default=149)
    args = ap.parse_args()

    import cv2
    from detection.detector import PlateDetector
    from recognition.crnn_reader import CRNNReader
    from plate_format import is_valid

    det = PlateDetector(args.weights, conf=0.5)
    reader = CRNNReader(PROJECT_ROOT / "models" / "recognition" / "crnn_finetuned.pth",
                        PROJECT_ROOT / "models" / "recognition" / "charset.txt")

    rows = []
    with open(REAL_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if "/test/" not in r["image_path"].replace("\\", "/"):
                continue
            fn = Path(r["image_path"]).name
            img = TEST_IMAGES / fn
            if img.exists():
                rows.append((img, norm(r["plate_text"])))
    rows = rows[:args.limit]

    cam_rots = {"0°": 0, "90°": 1, "180°": 2, "270°": 3}   # simulated camera orientation
    naive = {k: 0 for k in cam_rots}
    orient = {k: 0 for k in cam_rots}
    n = 0
    for img_path, gt in rows:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        n += 1
        for name, k in cam_rots.items():
            frame = rot90k(img, k)
            naive[name] += (read_naive(frame, det, reader) == gt)
            orient[name] += (read_oriented(frame, det, reader, is_valid) == gt)

    print("=" * 58)
    print(f" ANY-ANGLE READING  (n={n} test frames)  [{args.weights.name}]")
    print("=" * 58)
    print(f" {'camera rotation':<18}{'NAIVE':>10}{'ORIENTATION':>14}")
    for name in cam_rots:
        print(f" {name:<18}{100*naive[name]/n:>9.1f}%{100*orient[name]/n:>13.1f}%")
    print("-" * 58)
    print(f" {'mean':<18}{100*sum(naive.values())/(4*n):>9.1f}%"
          f"{100*sum(orient.values())/(4*n):>13.1f}%")
    print("=" * 58)
    print(" NAIVE reads once (today's pipeline); ORIENTATION searches 4 rotations.")


if __name__ == "__main__":
    main()
