#!/usr/bin/env python3
"""
scripts/tools/tune_province.py
==============================
Hyperparameter grid for the province classifier: learning rate x weight decay,
on ONE approach (default: C, ImageNet fine-tune). Every run is a normal
train_province_classifier.py invocation with a fixed seed, so each leaves its
own results/province_study/tune_<lr>_<wd>/ folder, and this script collects
them into results/province_study/tuning.csv.

Selection is by VALIDATION accuracy. Test accuracy is recorded for the report
but is never used to choose - choosing on test would leak the test set into
the model-selection decision.

Run:
    python scripts/tools/tune_province.py                       # 3 lr x 2 wd = 6 runs
    python scripts/tools/tune_province.py --lrs 1e-4 3e-4 --wds 0 --epochs 20
    python scripts/tools/tune_province.py --collect             # only rebuild tuning.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
TRAIN = PROJECT_ROOT / "scripts" / "recognition" / "train_province_classifier.py"
STUDY = PROJECT_ROOT / "results" / "province_study"
OUT = STUDY / "tuning.csv"


def run_name(lr: float, wd: float) -> str:
    return f"tune_lr{lr:g}_wd{wd:g}"


def collect() -> list[dict]:
    rows = []
    for rj in sorted(STUDY.glob("tune_*/run.json")):
        r = json.loads(rj.read_text(encoding="utf-8"))
        rows.append({k: r[k] for k in ("run_name", "arch", "mode", "lr", "weight_decay",
                                       "best_epoch", "best_val_acc", "test_acc",
                                       "test_macro_f1", "train_wall_sec", "seed")})
    if rows:
        with open(OUT, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        best = max(rows, key=lambda r: r["best_val_acc"])
        print(f"\n{'run':<22} {'lr':>7} {'wd':>7} {'val_acc':>8} {'test_acc':>9}")
        print("-" * 58)
        for r in sorted(rows, key=lambda r: -r["best_val_acc"]):
            mark = "  <- selected (val)" if r is best else ""
            print(f"{r['run_name']:<22} {r['lr']:>7g} {r['weight_decay']:>7g} "
                  f"{100*r['best_val_acc']:>7.2f}% {100*r['test_acc']:>8.2f}%{mark}")
        print(f"\ntuning.csv -> {OUT.relative_to(PROJECT_ROOT)}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    ap.add_argument("--wds", type=float, nargs="+", default=[0.0, 1e-4])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="tune approach A (scratch) instead of C (fine-tune)")
    ap.add_argument("--collect", action="store_true", help="skip training, only rebuild tuning.csv")
    args = ap.parse_args()

    if not args.collect:
        for lr in args.lrs:
            for wd in args.wds:
                name = run_name(lr, wd)
                if (STUDY / name / "run.json").exists():
                    print(f"[skip] {name} already done")
                    continue
                cmd = [sys.executable, str(TRAIN), "--arch", args.arch,
                       "--epochs", str(args.epochs), "--seed", str(args.seed),
                       "--lr", f"{lr:g}", "--weight-decay", f"{wd:g}",
                       "--run-name", name,
                       "--out", str(PROJECT_ROOT / "models" / "recognition" / f"{name}.pth")]
                if not args.no_pretrained:
                    cmd.append("--pretrained")
                print("\n" + "=" * 70 + f"\n[tune] {name}\n" + "=" * 70)
                rc = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode
                if rc != 0:
                    print(f"[X] {name} failed (exit {rc}) - continuing with the rest")
    collect()


if __name__ == "__main__":
    main()
