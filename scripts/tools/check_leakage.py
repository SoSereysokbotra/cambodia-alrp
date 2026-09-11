#!/usr/bin/env python3
"""
scripts/tools/check_leakage.py
==============================
Plate-level leakage audit for the CRNN test set — and the fix.

Why this exists
---------------
`finetune_crnn.py` refuses any row whose PATH contains `/test/`. That guard is
necessary but not sufficient: the crops come from a scraped dataset where the
same physical plate was photographed several times, so the same plate NUMBER can
sit in both `train/` and `test/` under different filenames. The model can then
memorise "this car -> this number" and score on the test set without reading
anything.

Measured on 2026-08-20 this affected **52 of the 149 test crops (34.9%)**, and it
inflated the headline upside-down score of `crnn_stn4` from a true 56.7% to 69.1%.

    # audit only — how bad is it, and how does a model score on each bucket?
    python scripts/tools/check_leakage.py --audit
    python scripts/tools/check_leakage.py --audit --weights models/recognition/crnn_stn4.pth

    # write a de-leaked training CSV (test rows kept, contaminated train/valid dropped)
    python scripts/tools/check_leakage.py --write-clean

The de-leaked file is written to `data/crnn_crops/real_labels_deleaked.csv`.
`real_labels.csv` is never modified. Train against the new file with:

    --real-csv data/crnn_crops/real_labels_deleaked.csv
    --extra-real-csv data/crnn_crops/real_labels_deleaked.csv --extra-match /valid/
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
LABELS = PROJECT_ROOT / "data" / "crnn_crops" / "real_labels.csv"
CLEAN_OUT = PROJECT_ROOT / "data" / "crnn_crops" / "real_labels_deleaked.csv"
BS = chr(92)


def load():
    rows = []
    with open(LABELS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = r["image_path"].replace(BS, "/")
            rows.append((p, p.split("/")[2], r["plate_text"].strip().upper()))
    return rows


def file_hash(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def audit(weights: Path | None, do_log: bool = False) -> None:
    rows = load()
    by = collections.defaultdict(list)
    for p, s, t in rows:
        by[s].append((p, t))

    test_plates = {t for _, t in by["test"]}
    tv_plates = {t for s in ("train", "valid") for _, t in by[s]}
    leaked_plates = sorted(test_plates & tv_plates)
    leaked_crops = [(p, t) for p, t in by["test"] if t in tv_plates]

    print("=" * 66)
    print(" PLATE-LEVEL LEAKAGE AUDIT")
    print("=" * 66)
    for s in ("train", "valid", "test"):
        print(f"  {s:<6} {len(by[s]):>5} crops   "
              f"{len({t for _, t in by[s]}):>5} distinct plate numbers")
    print()
    print(f"  test plates also in train/valid : {len(leaked_plates)} of {len(test_plates)}")
    print(f"  CONTAMINATED test crops         : {len(leaked_crops)} of {len(by['test'])}"
          f"  ({100 * len(leaked_crops) / max(1, len(by['test'])):.1f}%)")

    tv_hashes: dict[str, str] = {}
    for s in ("train", "valid"):
        for p, _ in by[s]:
            fp = PROJECT_ROOT / p
            if fp.exists():
                tv_hashes.setdefault(file_hash(fp), p)
    dups = [(p, tv_hashes[file_hash(PROJECT_ROOT / p)])
            for p, _ in by["test"]
            if (PROJECT_ROOT / p).exists() and file_hash(PROJECT_ROOT / p) in tv_hashes]
    print(f"  byte-identical duplicate crops  : {len(dups)}")
    for a, b in dups[:10]:
        print(f"      {Path(a).name}  ==  {Path(b).name}")

    n_drop_tr = sum(1 for p, t in by["train"] if t in test_plates)
    n_drop_va = sum(1 for p, t in by["valid"] if t in test_plates)
    print()
    print(f"  --write-clean would drop {n_drop_tr} train + {n_drop_va} valid crops")
    print(f"  leaving train {len(by['train']) - n_drop_tr}, valid {len(by['valid']) - n_drop_va}")

    if not weights:
        print("\n  (pass --weights to score a model on each bucket)")
        return

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "recognition"))
    import cv2
    from crnn_reader import CRNNReader
    reader = CRNNReader(weights, PROJECT_ROOT / "models" / "recognition" / "charset.txt")

    def norm(s):
        return (s or "").upper().replace(" ", "").strip()

    buckets = {"CLEAN": [0, 0, 0], "LEAKED": [0, 0, 0]}
    for p, gt in by["test"][:149]:
        img = cv2.imread(str(PROJECT_ROOT / p))
        if img is None:
            continue
        key = "LEAKED" if gt in tv_plates else "CLEAN"
        up, _ = reader.read(img)
        down, _ = reader.read(cv2.rotate(img, cv2.ROTATE_180))
        buckets[key][0] += 1
        buckets[key][1] += norm(up) == gt
        buckets[key][2] += norm(down) == gt

    print()
    print("=" * 66)
    print(f" {weights.name}  —  scored per bucket")
    print("=" * 66)
    print(f" {'bucket':<8}{'n':>6}{'upright':>20}{'upside-down':>20}")
    tot = [0, 0, 0]
    for k in ("CLEAN", "LEAKED"):
        n, u, d = buckets[k]
        tot = [tot[i] + buckets[k][i] for i in range(3)]
        print(f" {k:<8}{n:>6}{100 * u / max(n, 1):>13.1f}% ({u:>3}){100 * d / max(n, 1):>13.1f}% ({d:>3})")
    n, u, d = tot
    print(f" {'ALL':<8}{n:>6}{100 * u / max(n, 1):>13.1f}% ({u:>3}){100 * d / max(n, 1):>13.1f}% ({d:>3})")
    print("=" * 66)
    print(" CLEAN is the honest number. A large CLEAN/LEAKED gap = memorisation.")

    if do_log:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tools"))
        from experiment_log import log_metric
        wname = weights.name
        # Per-bucket accuracy rows
        for k in ("CLEAN", "LEAKED"):
            n, u, d = buckets[k]
            log_metric("crnn", "word_accuracy_upright",
                       f"{u / max(n, 1):.4f}",
                       split=f"real-test-{k.lower()}",
                       notes=f"{wname}, n={n}")
            log_metric("crnn", "word_accuracy_upsidedown",
                       f"{d / max(n, 1):.4f}",
                       split=f"real-test-{k.lower()}",
                       notes=f"{wname}, n={n}")
        # ALL bucket
        n, u, d = tot
        log_metric("crnn", "word_accuracy_upright",
                   f"{u / max(n, 1):.4f}",
                   split="real-test-all",
                   notes=f"{wname}, n={n}")
        log_metric("crnn", "word_accuracy_upsidedown",
                   f"{d / max(n, 1):.4f}",
                   split="real-test-all",
                   notes=f"{wname}, n={n}")
        # Contamination level
        leak_frac = len(leaked_crops) / max(len(by["test"]), 1)
        log_metric("dataset", "test_leaked_crop_fraction",
                   f"{leak_frac:.4f}", split="real-test")
        logged = 2 * 3 + 1  # 2 metrics * 3 buckets + 1 leak fraction
        print(f"\n  [log] {logged} rows appended to experiment_log.csv")


def write_clean() -> None:
    rows = load()
    test_plates = {t for _, s, t in rows if s == "test"}
    kept, dropped = [], collections.Counter()
    for p, s, t in rows:
        if s in ("train", "valid") and t in test_plates:
            dropped[s] += 1
            continue
        kept.append({"image_path": p, "plate_text": t})

    with open(CLEAN_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "plate_text"])
        w.writeheader()
        w.writerows(kept)

    c = collections.Counter(r["image_path"].split("/")[2] for r in kept)
    print(f"[ok] wrote {CLEAN_OUT.relative_to(PROJECT_ROOT)}")
    print(f"     dropped {dropped['train']} train + {dropped['valid']} valid contaminated crops")
    print(f"     kept: {dict(c)}")
    assert c["test"] == 149, "test split must stay at 149"

    rem = {t for r in kept for t in [r["plate_text"]]
           if r["image_path"].split("/")[2] in ("train", "valid")}
    assert not (rem & test_plates), "leakage remains — refusing to claim success"
    print("     verified: zero plate overlap between train/valid and test")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", action="store_true", help="report leakage (and score a model)")
    ap.add_argument("--weights", type=Path, default=None, help="model to score per bucket")
    ap.add_argument("--write-clean", action="store_true", help="write the de-leaked CSV")
    ap.add_argument("--log", action="store_true",
                    help="Append leakage/accuracy metrics to metrics/experiment_log.csv.")
    args = ap.parse_args()

    if not args.audit and not args.write_clean:
        ap.error("pass --audit and/or --write-clean")
    if args.audit:
        audit(args.weights, do_log=args.log)
    if args.write_clean:
        print()
        write_clean()


if __name__ == "__main__":
    main()
