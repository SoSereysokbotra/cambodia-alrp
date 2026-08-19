#!/usr/bin/env python3
"""
scripts/tools/make_colab_bundle.py
==================================
Zip ONLY the files needed to train the models on Google Colab, so you upload a
small bundle instead of the whole project (no photos/, .venv/, runs/, .git/).

    python scripts/tools/make_colab_bundle.py

-> creates  alpr_colab_bundle.zip  in the project root. Upload it to Google Drive
   and run the Colab notebook (colab_train.ipynb).
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
OUT = PROJECT_ROOT / "alpr_colab_bundle.zip"

# Everything the CRNN + province + detector training needs — and nothing else.
INCLUDE_DIRS = [
    "src",                       # the model/reader modules
    "scripts/recognition",       # finetune_crnn.py, train_province_classifier.py
    "scripts/detection",         # train.py (detector)
    "scripts/tools",             # measurement tools
    "configs",
    "data/crnn_crops",           # CRNN number crops + real_labels.csv
    "data/province_crops",       # province classifier crops
    "data/synthetic",            # synthetic mix for the CRNN
    "data/annotated",            # detector dataset (drop this line if not training the detector)
]
INCLUDE_FILES = [
    "models/recognition/crnn_best.pth",            # CRNN fine-tune BASE
    "models/recognition/crnn_finetuned.pth",       # current deployed (to compare)
    "models/recognition/charset.txt",
    "models/recognition/province_classifier_best.pth",
    "models/recognition/province_classifier_config.json",
    "requirements.txt",
]
SKIP_SUFFIX = {".pyc"}


def main() -> None:
    n = 0
    total = 0
    OUT.unlink(missing_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for d in INCLUDE_DIRS:
            base = PROJECT_ROOT / d
            if not base.exists():
                print(f"  [skip] {d} (not found)")
                continue
            for p in base.rglob("*"):
                if p.is_file() and p.suffix not in SKIP_SUFFIX and "__pycache__" not in p.parts:
                    z.write(p, p.relative_to(PROJECT_ROOT).as_posix())
                    n += 1
                    total += p.stat().st_size
        for f in INCLUDE_FILES:
            p = PROJECT_ROOT / f
            if p.exists():
                z.write(p, p.relative_to(PROJECT_ROOT).as_posix())
                n += 1
                total += p.stat().st_size
            else:
                print(f"  [skip] {f} (not found)")

    size_mb = OUT.stat().st_size / 1e6
    print(f"\n[ok] {OUT.name}: {n} files, {total/1e6:.0f} MB raw -> {size_mb:.0f} MB zipped")
    print(f"     {OUT}")
    print("\nNext: upload this zip to Google Drive, then run colab_train.ipynb in Colab.")


if __name__ == "__main__":
    main()
