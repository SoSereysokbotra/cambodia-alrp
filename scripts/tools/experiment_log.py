#!/usr/bin/env python3
"""
scripts/tools/experiment_log.py
===============================
IMPROVEMENT_ROADMAP item 3.2 — lightweight experiment tracking.

One append-only CSV (`metrics/experiment_log.csv`) is the single source of truth
for "what did we measure, when, at which commit." No W&B/MLflow, no new deps —
just a versioned file you can diff, grep, and open in Excel. Prevents regressions
(like the DET-005 padding one) from being silently rediscovered.

Columns: timestamp, git_commit, component, metric, value, split, notes

Use from the shell:
    python scripts/tools/experiment_log.py --component crnn --metric cer \
        --value 0.1021 --split test --notes "fine-tuned on 473 real"
    python scripts/tools/experiment_log.py --show                 # print the log
    python scripts/tools/experiment_log.py --backfill             # seed history once

Use from Python (e.g. inside a benchmark / training script):
    from experiment_log import log_metric
    log_metric("pipeline", "number_accuracy", 0.6779, split="real-test",
               notes="benchmark_composed.py")
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
LOG_CSV = PROJECT_ROOT / "metrics" / "experiment_log.csv"
FIELDS = ["timestamp", "git_commit", "component", "metric", "value", "split", "notes"]


def git_commit() -> str:
    """Short current commit hash (or 'nogit' outside a repo)."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=str(PROJECT_ROOT), capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def log_metric(component: str, metric: str, value, split: str = "",
               notes: str = "", commit: str | None = None,
               csv_path: Path = LOG_CSV) -> None:
    """Append one measurement row to the experiment log (creates it if absent)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit or git_commit(),
        "component": component,
        "metric": metric,
        "value": value,
        "split": split,
        "notes": notes,
    }
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def show(csv_path: Path = LOG_CSV) -> None:
    if not csv_path.exists():
        print(f"(no log yet at {csv_path.relative_to(PROJECT_ROOT)})")
        return
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows:
        print("(log is empty)")
        return
    w = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in FIELDS}
    print(" | ".join(c.ljust(w[c]) for c in FIELDS))
    print("-+-".join("-" * w[c] for c in FIELDS))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(w[c]) for c in FIELDS))
    print(f"\n{len(rows)} rows  ->  {csv_path.relative_to(PROJECT_ROOT)}")


# --------------------------------------------------------------------------- #
# One-time backfill of the project's documented history (from docs/HANDOFF.md),
# so the log starts with the baseline story instead of an empty file. Idempotent
# guard: refuses to run if rows already exist (use --force to append anyway).
# --------------------------------------------------------------------------- #
_HISTORY = [
    # component, metric, value, split, notes
    ("yolo_plate_detector", "map50", 0.9664, "test", "best.pt (Plate_v4)"),
    ("yolo_number_detector", "map50", 0.943, "test", "number_best.pt"),
    ("province_classifier", "accuracy", 0.9718, "test", "ResNet18 26-class"),
    ("crnn", "cer", 0.9489, "real-test", "synthetic-only baseline"),
    ("crnn", "cer", 0.2593, "real-test", "fine-tuned on 143 real"),
    ("crnn", "cer", 0.2032, "real-test", "fine-tuned on 324 real"),
    ("crnn", "cer", 0.1021, "real-test", "fine-tuned on 473 real (SRS target met)"),
    ("crnn", "word_accuracy", 0.7248, "real-test", "fine-tuned on 473 real"),
    ("pipeline", "number_e2e_accuracy", 0.706, "real-test", "two-detector, 101/143 (recorded)"),
    ("pipeline", "latency_ms", 51.0, "real", "two-detector avg (~19.6 FPS)"),
]


def backfill(force: bool = False) -> None:
    if LOG_CSV.exists() and not force:
        existing = list(csv.DictReader(open(LOG_CSV, encoding="utf-8")))
        if existing:
            print(f"[skip] {LOG_CSV.relative_to(PROJECT_ROOT)} already has "
                  f"{len(existing)} rows. Use --force to append history anyway.")
            return
    commit = git_commit()
    for comp, metric, val, split, notes in _HISTORY:
        log_metric(comp, metric, val, split=split,
                   notes=f"[history] {notes}", commit=commit)
    print(f"[ok] backfilled {len(_HISTORY)} historical rows into "
          f"{LOG_CSV.relative_to(PROJECT_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="append-only experiment log (roadmap 3.2)")
    ap.add_argument("--component"); ap.add_argument("--metric")
    ap.add_argument("--value"); ap.add_argument("--split", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.backfill:
        backfill(force=args.force)
        return
    if args.component and args.metric and args.value is not None:
        log_metric(args.component, args.metric, args.value,
                   split=args.split, notes=args.notes)
        print(f"[ok] logged {args.component}/{args.metric}={args.value} "
              f"({args.split or 'no-split'})")
        return
    show()      # default action: print the log


if __name__ == "__main__":
    main()
