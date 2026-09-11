#!/usr/bin/env python3
"""
scripts/tools/test_province_rotation.py
=======================================
Measure how well a province classifier reads the Khmer line upright vs
upside-down (180 deg) — the province twin of test_rotation_reading.py.

Scored on the 567 held-out province test crops, with the same
`province_crop_trim` the live pipeline applies, so the number is comparable to
what the system actually does.

    python scripts/tools/test_province_rotation.py --weights models/recognition/province_classifier_rot.pth
    python scripts/tools/test_province_rotation.py --weights models/recognition/province_classifier_best.pth

Measured 2026-08-23 for reference:
    province_classifier_best.pth   upright 96.1%   upside-down 19.4%
"""
from __future__ import annotations

import argparse
import glob
import os
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

TEST_DIR = PROJECT_ROOT / "data" / "province_crops" / "test"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None,
                    help="matching *_config.json (defaults to <weights>_config.json)")
    ap.add_argument("--trim", type=float, default=None,
                    help="province_crop_trim (default: read from system_config.yaml)")
    args = ap.parse_args()

    import cv2
    import yaml
    from province_classifier import ProvinceClassifier
    from core.alpr_system import ALPRSystem

    trim = args.trim
    if trim is None:
        cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "system_config.yaml")
                             .read_text(encoding="utf-8"))
        trim = float(cfg.get("gate", {}).get("province_crop_trim", 0.125))

    clf = ProvinceClassifier(args.weights, args.config)

    items = []
    for d in sorted(glob.glob(str(TEST_DIR / "*"))):
        if not os.path.isdir(d):
            continue
        cls = int(os.path.basename(d))
        for f in sorted(glob.glob(os.path.join(d, "*.jpg"))):
            im = cv2.imread(f)
            if im is not None:
                items.append((im, cls))
    if not items:
        print(f"[X] no crops under {TEST_DIR}")
        sys.exit(1)

    up = dn = 0
    for im, cls in items:
        t = ALPRSystem._tighten_crop(im, trim)
        up += clf.predict(t)[0] == cls
        dn += clf.predict(cv2.rotate(t, cv2.ROTATE_180))[0] == cls

    n = len(items)
    print("=" * 56)
    print(f" PROVINCE READING  ({args.weights.name})  n={n} test crops")
    print(f" province_crop_trim={trim}")
    print("=" * 56)
    print(f"  upright        : {100*up/n:5.1f}%  ({up}/{n})")
    print(f"  upside-down    : {100*dn/n:5.1f}%  ({dn}/{n})")
    print("=" * 56)
    print(" A rotation-trained model should lift upside-down while keeping upright.")
    print(" Baseline for comparison: best.pth = 96.1% upright / 19.4% upside-down.")


if __name__ == "__main__":
    main()
