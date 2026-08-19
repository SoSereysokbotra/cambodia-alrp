#!/usr/bin/env python3
"""
scripts/tools/test_province_stability.py
========================================
MODEL_IMPROVEMENT_PLAN Step 2b / Step 3 validation — the FRAMING-STABILITY test.

Measures the flicker directly: keep each plate identical, jitter the detector's box
(shift +-10%, scale +-8% — the amount a detector wobbles frame-to-frame on live
video), and see how often the province prediction changes. A framing-INVARIANT model
should give the same province regardless of the box, so:
    - flip rate LOW,
    - "plates with >1 province under jitter" LOW,
    - while base (well-framed) accuracy stays high.

Run on the deployed model and a candidate, and compare:
    python scripts/tools/test_province_stability.py                       # deployed
    python scripts/tools/test_province_stability.py --weights models/recognition/province_classifier_framing.pth --config models/recognition/province_classifier_framing_config.json
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
PROV_GT = PROJECT_ROOT / "data" / "crnn_crops" / "province_test_labels.csv"
DEFAULT_W = PROJECT_ROOT / "models" / "recognition" / "province_classifier_best.pth"


def jitter_boxes(x1, y1, x2, y2, W, H):
    """The detector-wobble grid used in Step 2b (~45 boxes per plate)."""
    w, h = x2 - x1, y2 - y1
    out = []
    for dx in (-0.10, -0.05, 0, 0.05, 0.10):
        for dy in (-0.08, 0, 0.08):
            for s in (0.92, 1.0, 1.08):
                nx1 = x1 + dx * w - (s - 1) * w / 2
                ny1 = y1 + dy * h - (s - 1) * h / 2
                nx2 = x2 + dx * w + (s - 1) * w / 2
                ny2 = y2 + dy * h + (s - 1) * h / 2
                a, b = max(0, int(nx1)), max(0, int(ny1))
                c, e = min(W, int(nx2)), min(H, int(ny2))
                if c - a > 4 and e - b > 4:
                    out.append((a, b, c, e))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="province framing-stability test")
    ap.add_argument("--weights", type=Path, default=DEFAULT_W)
    ap.add_argument("--config", type=Path, default=None,
                    help="config json (defaults to the one beside --weights)")
    args = ap.parse_args()

    import cv2
    import numpy as np
    from detection.detector import PlateDetector
    from recognition.province_classifier import ProvinceClassifier

    det = PlateDetector(PROJECT_ROOT / "models" / "detection" / "best.pt", conf=0.5)
    clf = ProvinceClassifier(args.weights, config_path=args.config)

    gt = {}
    with open(PROV_GT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("note") or "").upper().startswith("UNVERIFIABLE"):
                continue
            gt[r["image"]] = int(r["province_class"])

    n = base_ok = flips = total_jit = jit_ok = unstable = 0
    distinct = []
    for name, g in gt.items():
        img = cv2.imread(str(TEST_IMAGES / name))
        if img is None:
            continue
        d = det.detect(img)
        if not d:
            continue
        n += 1
        H, W = img.shape[:2]
        x1, y1, x2, y2 = d[0]["bbox"]
        base, _ = clf.predict(img[y1:y2, x1:x2])
        base_ok += (base == g)
        preds = []
        for a, b, c, e in jitter_boxes(x1, y1, x2, y2, W, H):
            p, _ = clf.predict(img[b:e, a:c])
            preds.append(p)
            total_jit += 1
            flips += (p != base)
            jit_ok += (p == g)
        distinct.append(len(set(preds)))
        unstable += (len(set(preds)) > 1)

    print("=" * 56)
    print(f" PROVINCE FRAMING-STABILITY  ({args.weights.name})")
    print("=" * 56)
    print(f" plates tested                       : {n}")
    print(f" base (well-framed) accuracy         : {100*base_ok/max(n,1):.1f}%")
    print(f" province FLIP rate under box jitter : {100*flips/max(total_jit,1):.1f}%")
    print(f" jittered-crop accuracy vs GT        : {100*jit_ok/max(total_jit,1):.1f}%")
    print(f" plates giving >1 province (FLICKER) : {unstable}/{n} "
          f"({100*unstable/max(n,1):.0f}%)")
    print(f" mean distinct provinces / plate     : {np.mean(distinct):.2f}  "
          f"(max {max(distinct) if distinct else 0})")
    print("=" * 56)
    print(" lower flip / flicker / distinct = more framing-invariant = less live flicker")


if __name__ == "__main__":
    main()
