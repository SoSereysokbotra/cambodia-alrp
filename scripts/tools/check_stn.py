#!/usr/bin/env python3
"""
scripts/tools/check_stn.py
==========================
Does the STN "straightening layer" actually straighten anything?

Background (measured 2026-08-20): trained with the CTC loss alone, it does NOT.
On `crnn_stn2/4/5` it predicts a near-identity transform for flipped and upright
input alike:

    theta = [[+1.09, 0.00, +0.05],
             [ 0.00,+1.05, -0.01]]        (a 180 flip would be [[-1,0,0],[0,-1,0]])

so the reader was learning flipped text directly and the STN contributed nothing.
`finetune_crnn.py --stn-supervise W` adds a loss that teaches it the transform
directly. This script is how you check whether that worked.

    python scripts/tools/check_stn.py --weights models/recognition/crnn_stn6.pth

PASS = theta[0][0] is near -1 on flipped input, and STN(flipped) resembles the
upright image more than the flipped one.
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
BS = chr(92)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=149)
    args = ap.parse_args()

    import numpy as np
    import torch
    import cv2
    from crnn_reader import CRNNReader

    reader = CRNNReader(args.weights, PROJECT_ROOT / "models" / "recognition" / "charset.txt")
    stn = getattr(reader.model, "stn", None)
    if stn is None:
        print(f"[X] {args.weights.name} has no STN layer.")
        sys.exit(1)

    csv_path = PROJECT_ROOT / "data" / "crnn_crops" / "real_labels.csv"
    rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8"))
            if "/test/" in r["image_path"].replace(BS, "/")][:args.limit]

    up_t, dn_t, to_up, to_dn = [], [], [], []
    for r in rows:
        img = cv2.imread(str(PROJECT_ROOT / r["image_path"].replace(BS, "/")))
        if img is None:
            continue
        x_up = reader._preprocess(img).to(reader.device)
        x_dn = reader._preprocess(cv2.rotate(img, cv2.ROTATE_180)).to(reader.device)
        with torch.no_grad():
            up_t.append(stn.theta(x_up)[0].cpu().numpy())
            dn_t.append(stn.theta(x_dn)[0].cpu().numpy())
            out = stn(x_dn).cpu().numpy()[0, 0]
        to_up.append(((out - x_up.cpu().numpy()[0, 0]) ** 2).mean())
        to_dn.append(((out - x_dn.cpu().numpy()[0, 0]) ** 2).mean())

    up_a, dn_a = np.stack(up_t), np.stack(dn_t)
    print("=" * 68)
    print(f" STN CHECK — {args.weights.name}   (n={len(up_a)} test crops)")
    print("=" * 68)
    for name, a in (("upright input", up_a), ("flipped input", dn_a)):
        print(f"  {name:<16} theta = [[{a[:,0,0].mean():+.3f} {a[:,0,1].mean():+.3f} "
              f"{a[:,0,2].mean():+.3f}] [{a[:,1,0].mean():+.3f} {a[:,1,1].mean():+.3f} "
              f"{a[:,1,2].mean():+.3f}]]")
    print()
    print(f"  wanted — upright: [[+1 0 0] [0 +1 0]]     flipped: [[-1 0 0] [0 -1 0]]")
    print()
    print(f"  MSE( STN(flipped), UPRIGHT ) = {np.mean(to_up):.4f}   <- want LOW")
    print(f"  MSE( STN(flipped), FLIPPED ) = {np.mean(to_dn):.4f}   <- want HIGH")
    print()

    rotates = dn_a[:, 0, 0].mean() < -0.5
    keeps = up_a[:, 0, 0].mean() > 0.5
    looks_upright = np.mean(to_up) < np.mean(to_dn)
    frac = float((dn_a[:, 0, 0] < -0.5).mean())

    print(f"  flips a flipped crop      : {'YES' if rotates else 'NO'} "
          f"({100*frac:.0f}% of crops individually)")
    print(f"  leaves an upright crop    : {'YES' if keeps else 'NO'}")
    print(f"  output resembles upright  : {'YES' if looks_upright else 'NO'}")
    print("=" * 68)
    if rotates and keeps and looks_upright:
        print(" PASS — the straightening layer is doing real work.")
    else:
        print(" FAIL — the STN is not straightening. It is decorative;")
        print("        do not claim it as the mechanism in the write-up.")


if __name__ == "__main__":
    main()
