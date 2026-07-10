#!/usr/bin/env python3
"""
summary_week2.py
================
STEP 5 of Week 2 — print a clean, human-readable Week 2 summary and save it.

Reads:
    metrics/week2_metrics.json   (from evaluate_week2.py)
    runs/detect/                 (training run folder, for context)
    models/detection/best.pt     (for model size on disk)

Outputs:
    * prints the === WEEK 2 SUMMARY ===
    * saves metrics/week2_summary.txt

Run (LAST):
    python summary_week2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
METRICS_JSON = PROJECT_ROOT / "metrics" / "week2_metrics.json"
SUMMARY_TXT = PROJECT_ROOT / "metrics" / "week2_summary.txt"
WEIGHTS = PROJECT_ROOT / "models" / "detection" / "best.pt"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"

MAP_TARGET = 0.82
LATENCY_TARGET = 120  # ms


def latest_run_dir() -> Path | None:
    """Newest training run folder under runs/detect/, if any."""
    if not RUNS_DIR.is_dir():
        return None
    runs = [d for d in RUNS_DIR.iterdir() if d.is_dir() and (d / "weights").exists()]
    if not runs:
        runs = [d for d in RUNS_DIR.iterdir() if d.is_dir()]
    return max(runs, key=lambda d: d.stat().st_mtime) if runs else None


def main() -> None:
    if not METRICS_JSON.exists():
        print(f"[X] {METRICS_JSON} not found. Run evaluate_week2.py first.")
        sys.exit(1)

    m = json.loads(METRICS_JSON.read_text())

    # model size on disk
    size_mb = WEIGHTS.stat().st_size / (1024 * 1024) if WEIGHTS.exists() else 0.0

    # training run context
    run_dir = latest_run_dir()
    run_note = run_dir.name if run_dir else "not found"

    map50 = m.get("mAP50", 0.0)
    latency = m.get("inference_ms", 0.0)

    map_ok = map50 >= MAP_TARGET
    lat_ok = 0 < latency < LATENCY_TARGET
    ready = map_ok and lat_ok

    map_mark = "✓" if map_ok else "✗"
    lat_mark = "✓" if lat_ok else "✗"
    status = "READY FOR WEEK 3" if ready else "NEEDS IMPROVEMENT"

    lines = [
        "=" * 46,
        " WEEK 2 SUMMARY",
        "=" * 46,
        f" Dataset:      Plate_v4 (3,299 Cambodian plates)",
        f" Credit:       {m.get('dataset', 'taki-dk0de, CC BY 4.0')}",
        f" Model:        YOLOv10 nano",
        f" Training run: {run_note}",
        "-" * 46,
        f" mAP50:        {map50:.4f}  [TARGET >= {MAP_TARGET}]  {map_mark}",
        f" mAP50-95:     {m.get('mAP50-95', 0.0):.4f}",
        f" Precision:    {m.get('precision', 0.0):.4f}",
        f" Recall:       {m.get('recall', 0.0):.4f}",
        f" Latency:      {latency:.1f} ms  [TARGET < {LATENCY_TARGET} ms]  {lat_mark}",
        f" Model size:   {size_mb:.1f} MB",
        f" Model path:   models/detection/best.pt",
        "-" * 46,
        f" Status:       {status}",
        "=" * 46,
    ]

    if not ready:
        lines.append("")
        if not map_ok:
            lines.append(" Tip: mAP50 below target -> retrain with --epochs 150,")
            lines.append("      or add augmentation / a larger model (yolov10s).")
        if not lat_ok:
            lines.append(" Tip: latency high -> ensure GPU is used (device 0),")
            lines.append("      or keep the nano model / lower imgsz.")

    report = "\n".join(lines)
    print(report)

    SUMMARY_TXT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_TXT.write_text(report + "\n", encoding="utf-8")
    print(f"\n[OK] summary saved -> {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
