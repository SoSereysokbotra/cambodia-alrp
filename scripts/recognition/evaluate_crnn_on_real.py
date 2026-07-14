#!/usr/bin/env python3
"""
scripts/recognition/evaluate_crnn_on_real.py
============================================
Measure the CURRENT CRNN's performance on REAL, hand-labelled plate crops.
This is the honest real-world baseline (and, later, the yard-stick that tells
us whether Phase-4 fine-tuning actually helped).

Input CSV (produced by scripts/tools/label_real_crops.py):
    image_path,plate_text
    data/crnn_crops/test/xxxx.jpg,2B-4445

Reports: CER, word accuracy, mean confidence, % below the REVIEW_REQUIRED gate,
and sample predictions. Saves metrics/crnn_real_baseline.json.

Run (after labelling the test split):
    python scripts/recognition/evaluate_crnn_on_real.py
    python scripts/recognition/evaluate_crnn_on_real.py --split all
    python scripts/recognition/evaluate_crnn_on_real.py --weights models/recognition/crnn_finetuned.pth --tag finetuned
"""

from __future__ import annotations

import argparse
import csv
import json
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

DEFAULT_CSV = PROJECT_ROOT / "data" / "crnn_crops" / "real_labels.csv"
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "recognition" / "crnn_best.pth"
CHARSET_TXT = PROJECT_ROOT / "models" / "recognition" / "charset.txt"
METRICS_DIR = PROJECT_ROOT / "metrics"

CER_TARGET = 0.15          # SRS-ish target for a PASS marker
GATE = 0.70                # REC-005 REVIEW_REQUIRED threshold


# --- self-contained CER (no cross-script import) --- #
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def calculate_cer(preds: list[str], tgts: list[str]) -> float:
    edits = chars = 0
    for p, t in zip(preds, tgts):
        edits += levenshtein(p, t)
        chars += max(len(t), 1)
    return edits / max(chars, 1)


def load_rows(csv_path: Path, split: str) -> list[tuple[str, str]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            path = (r.get("image_path") or "").strip()
            text = (r.get("plate_text") or "").strip().upper()
            if not path or not text:
                continue
            if split != "all" and f"/{split}/" not in path.replace("\\", "/"):
                continue
            rows.append((path, text))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--split", default="test", choices=["train", "valid", "test", "all"])
    ap.add_argument("--tag", default="baseline", help="Label for the saved JSON.")
    ap.add_argument("--samples", type=int, default=15)
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"[X] labels CSV not found: {args.csv}")
        print("    Label some crops first: python scripts/tools/label_real_crops.py --split test")
        sys.exit(1)
    if not args.weights.exists():
        print(f"[X] CRNN weights not found: {args.weights}")
        sys.exit(1)

    rows = load_rows(args.csv, args.split)
    if not rows:
        print(f"[!] no labelled rows for split='{args.split}' in {args.csv} yet.")
        print("    Keep labelling, then re-run. (Nothing to evaluate.)")
        sys.exit(0)

    import cv2
    from crnn_reader import CRNNReader
    reader = CRNNReader(args.weights, CHARSET_TXT)

    print("=" * 64)
    print(f" CRNN ON REAL CROPS — split='{args.split}', tag='{args.tag}'")
    print(f" weights: {args.weights.name}")
    print("=" * 64)

    preds, tgts, confs = [], [], []
    missing = 0
    for path, text in rows:
        full = path if Path(path).is_absolute() else str(PROJECT_ROOT / path)
        img = cv2.imread(full)
        if img is None:
            missing += 1
            continue
        pred, conf = reader.read(img)
        preds.append(pred)
        tgts.append(text)
        confs.append(conf)

    n = len(tgts)
    if n == 0:
        print("[X] none of the labelled crops could be read from disk.")
        sys.exit(1)

    cer = calculate_cer(preds, tgts)
    exact = sum(1 for p, t in zip(preds, tgts) if p == t)
    word_acc = exact / n
    mean_conf = sum(confs) / n
    below_gate = sum(1 for c in confs if c < GATE)

    print(f"\nSamples (predicted vs actual):")
    print(f"  {'actual':<14}{'predicted':<14}{'conf':>6}  ok")
    print("  " + "-" * 40)
    for p, t, c in list(zip(preds, tgts, confs))[:args.samples]:
        mark = "Y" if p == t else "."
        print(f"  {t:<14}{(p or '(empty)'):<14}{c:>6.2f}   {mark}")

    print("\n" + "-" * 64)
    print(f" Labelled crops evaluated : {n}" + (f"  ({missing} unreadable files)" if missing else ""))
    print(f" CER                      : {cer*100:.2f}%   [target < {CER_TARGET*100:.0f}%]  "
          f"{'PASS' if cer < CER_TARGET else 'FAIL'}")
    print(f" Word accuracy (exact)    : {word_acc*100:.2f}%")
    print(f" Mean confidence          : {mean_conf:.2f}")
    print(f" Below {GATE:.2f} gate (REVIEW) : {below_gate}/{n} ({below_gate/n*100:.0f}%)")
    print("-" * 64)

    out = {
        "tag": args.tag,
        "split": args.split,
        "weights": args.weights.name,
        "n": n,
        "cer": round(cer, 4),
        "word_accuracy": round(word_acc, 4),
        "mean_confidence": round(mean_conf, 4),
        "below_gate_pct": round(below_gate / n, 4),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / f"crnn_real_{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] saved -> {out_path}")
    if args.tag == "baseline":
        print("\nThis is the pre-fine-tune baseline. After Phase-4 fine-tuning,")
        print("re-run with --weights <finetuned> --tag finetuned to compare.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] evaluation failed: {exc}")
        sys.exit(1)
