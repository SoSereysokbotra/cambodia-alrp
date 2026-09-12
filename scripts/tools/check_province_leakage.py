#!/usr/bin/env python3
"""
scripts/tools/check_province_leakage.py
=======================================
Leakage audit for the province-classifier split (data/province_crops/).

check_leakage.py audits the CRNN split by plate NUMBER. Province crops have no
number label, so this audit uses the two signals that ARE available:

  1. source image  - every crop name is  <roboflow-source>_jpg.rf.<hash>_<n>.jpg.
                     Roboflow's augmented copies of one photo share <source>. If a
                     source appears in train AND test, the test crop is a (possibly
                     augmented) copy of a training photo -> memorisation risk.
  2. pixel hash    - byte-identical files across splits (the crudest leak).

    python scripts/tools/check_province_leakage.py            # audit only
    python scripts/tools/check_province_leakage.py --log      # + experiment_log row

Nothing is modified. Use the printed numbers in the study's limitations section.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
CROPS = PROJECT_ROOT / "data" / "province_crops"
SPLITS = ("train", "val", "test")
_SRC = re.compile(r"^(.*?)_jpg\.rf\.[0-9a-f]{32}(?:_\d+)?\.jpg$", re.I)


def source_of(name: str) -> str:
    m = _SRC.match(name)
    return m.group(1) if m else name


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", action="store_true", help="append to metrics/experiment_log.csv")
    args = ap.parse_args()

    files = {s: sorted(CROPS.glob(f"{s}/*/*.jpg")) for s in SPLITS}
    for s in SPLITS:
        if not files[s]:
            sys.exit(f"[X] no crops under {CROPS/s}")
    src = {s: collections.defaultdict(list) for s in SPLITS}
    for s in SPLITS:
        for p in files[s]:
            src[s][source_of(p.name)].append(p)

    train_val_sources = set(src["train"]) | set(src["val"])
    leaked_sources = sorted(set(src["test"]) & train_val_sources)
    leaked_test = [p for k in leaked_sources for p in src["test"][k]]
    n_test = len(files["test"])

    print("=" * 62)
    print(" PROVINCE SPLIT LEAKAGE AUDIT")
    print("=" * 62)
    for s in SPLITS:
        print(f"  {s:<6} {len(files[s]):5d} crops from {len(src[s]):5d} source photos")
    print("-" * 62)
    print(f"  test sources also in train/val   : {len(leaked_sources)} of {len(src['test'])}")
    print(f"  CONTAMINATED test crops           : {len(leaked_test)} of {n_test} "
          f"({100*len(leaked_test)/max(n_test,1):.1f}%)")

    # A source photo can hold several cars, so its crops may legitimately carry
    # different provinces. Count those separately: they are still a leak (same
    # photo, same lighting/angle in both splits) but not necessarily label noise.
    mixed = 0
    for k in leaked_sources:
        classes = {p.parent.name for s in SPLITS for p in src[s].get(k, [])}
        if len(classes) > 1:
            mixed += 1
    print(f"  ...of which multi-plate photos    : {mixed} sources (crops span >1 class)")
    print("  root cause: build_province_dataset.py split at CROP level, not photo level")

    hashes = {s: {md5(p): p for p in files[s]} for s in SPLITS}
    dup = set(hashes["test"]) & (set(hashes["train"]) | set(hashes["val"]))
    print(f"  byte-identical test/train copies  : {len(dup)}")
    print("-" * 62)
    if leaked_test:
        print("  first contaminated test crops:")
        for p in leaked_test[:8]:
            print(f"    {p.relative_to(CROPS).as_posix()}")
    print("=" * 62)
    verdict = ("CLEAN - no source photo is shared between test and train/val"
               if not leaked_test else
               "LEAK - report the contaminated fraction in the limitations section")
    print(f"  {verdict}")

    # The list lets make_study_figures.py report a CLEAN-subset accuracy for
    # every approach without changing the split the runs were trained on.
    out_list = PROJECT_ROOT / "results" / "province_study" / "leaked_test_crops.txt"
    out_list.parent.mkdir(parents=True, exist_ok=True)
    out_list.write_text("\n".join(p.relative_to(PROJECT_ROOT).as_posix() for p in leaked_test)
                        + ("\n" if leaked_test else ""), encoding="utf-8")
    print(f"  leaked-crop list -> {out_list.relative_to(PROJECT_ROOT)}")

    if args.log:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tools"))
        from experiment_log import log_metric
        log_metric("dataset", "province_test_leaked_crop_fraction",
                   f"{len(leaked_test)/max(n_test,1):.4f}", split="province-test",
                   notes=f"{len(leaked_test)}/{n_test} crops share a source photo with train/val; "
                         f"{len(dup)} byte-identical")
        print("  [logged to metrics/experiment_log.csv]")


if __name__ == "__main__":
    main()
