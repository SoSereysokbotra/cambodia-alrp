#!/usr/bin/env python3
"""
scripts/tools/analyze_province_classifier.py
============================================
Diagnostics for the trained province classifier on the TEST set:
  * overall accuracy (with AND without class 25 'other')
  * per-class accuracy (sorted worst-first)
  * top confusions (are errors CONCENTRATED in a few class pairs, or SPREAD?)

Run:
    python scripts/tools/analyze_province_classifier.py
"""

from __future__ import annotations

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

DATA_DIR = PROJECT_ROOT / "data" / "province_crops"
MODEL = PROJECT_ROOT / "models" / "recognition" / "province_classifier_best.pth"
OTHER_ID = 25


def main() -> None:
    if not MODEL.exists():
        print("[X] classifier not trained. Run train_province_classifier.py first.")
        sys.exit(1)

    import numpy as np
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from province_classifier import ProvinceClassifier
    from province_map import province_latin

    clf = ProvinceClassifier(MODEL)
    idx_to_class = clf.idx_to_class or list(range(clf.n_classes))

    tf = transforms.Compose([
        transforms.Resize((clf.img_size, clf.img_size)),
        transforms.ToTensor(), transforms.Normalize(clf.mean, clf.std)])
    ds = datasets.ImageFolder(str(DATA_DIR / "test"), transform=tf)
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    folder_to_trueid = [int(c) for c in ds.classes]   # test folder idx -> true id

    n = clf.n_classes
    conf = np.zeros((max(idx_to_class) + 1, max(idx_to_class) + 1), dtype=int)

    with torch.no_grad():
        for x, y in dl:
            raw = clf.model(x.to(clf.device)).argmax(1).cpu().numpy()
            for r, t in zip(raw, y.numpy()):
                pred_id = idx_to_class[r] if r < len(idx_to_class) else OTHER_ID
                true_id = folder_to_trueid[t]
                conf[true_id, pred_id] += 1

    total = conf.sum()
    correct = np.trace(conf)
    acc = correct / max(total, 1)

    # accuracy excluding class 25 (rows/cols where TRUE label == 25 removed)
    mask = [i for i in range(conf.shape[0]) if i != OTHER_ID]
    sub = conf[np.ix_(mask, mask)]
    # note: predictions INTO 25 still count as wrong for non-25 rows
    row_tot_ex = conf[mask, :].sum()
    correct_ex = sum(conf[i, i] for i in mask)
    acc_ex = correct_ex / max(row_tot_ex, 1)

    print("=" * 60)
    print(" PROVINCE CLASSIFIER — TEST DIAGNOSTICS")
    print("=" * 60)
    print(f"Test crops        : {total}")
    print(f"Overall accuracy  : {acc*100:.2f}%   [TARGET >= 95%]  "
          f"{'PASS' if acc >= 0.95 else 'FAIL'}")
    print(f"Excluding class 25: {acc_ex*100:.2f}%   (class 25 = {conf[OTHER_ID].sum()} crops)")

    # per-class accuracy, worst first
    print("\nPer-class accuracy (worst first):")
    rows = []
    for i in range(conf.shape[0]):
        tot = conf[i].sum()
        if tot == 0:
            continue
        rows.append((conf[i, i] / tot, i, int(tot)))
    rows.sort()
    for a, i, tot in rows[:10]:
        name = "OTHER" if i == OTHER_ID else province_latin(i)
        print(f"  class {i:<2} {name:<18} {a*100:6.1f}%  ({conf[i,i]}/{tot})")

    # top confusions (off-diagonal)
    print("\nTop confusions (true -> predicted):")
    pairs = []
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            if i != j and conf[i, j] > 0:
                pairs.append((conf[i, j], i, j))
    pairs.sort(reverse=True)
    off_total = sum(p[0] for p in pairs)
    top5 = sum(p[0] for p in pairs[:5])
    for cnt, i, j in pairs[:8]:
        ni = "OTHER" if i == OTHER_ID else province_latin(i)
        nj = "OTHER" if j == OTHER_ID else province_latin(j)
        print(f"  {cnt:>3}x  {ni} -> {nj}")

    concentration = (top5 / off_total * 100) if off_total else 0
    print(f"\nErrors: {off_total} total | top-5 pairs = {concentration:.0f}% of them")
    print("  -> " + ("CONCENTRATED (specific look-alike classes)"
                     if concentration >= 50 else
                     "SPREAD (data/augmentation-driven, not a few pairs)"))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] analysis failed: {exc}")
        sys.exit(1)
