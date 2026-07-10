#!/usr/bin/env python3
"""
scripts/demo_week3.py
=====================
STEP 3 of Week 3 — the TEACHER DEMO.

Runs the whole pipeline on 5 test images and prints a clean, sectioned report
proving the end-to-end integration works: detect -> lookup -> decide -> log.

Run:
    python scripts/demo_week3.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.database import PlateDatabase          # noqa: E402

WEIGHTS = PROJECT_ROOT / "models" / "detection" / "best.pt"
DB_PATH = PROJECT_ROOT / "plates.db"
TEST_DIR = PROJECT_ROOT / "data" / "annotated" / "test" / "images"
METRICS_JSON = PROJECT_ROOT / "metrics" / "week2_metrics.json"
IMG_EXTS = {".jpg", ".jpeg", ".png"}

OK = "✓"
NO = "✗"


def banner() -> None:
    print("==========================================")
    print("  CAMBODIAN ALPR SYSTEM - WEEK 3 DEMO")
    print("  End-to-End Pipeline Demonstration")
    print("==========================================")


def load_week2_metrics() -> dict:
    try:
        return json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    banner()

    # ---------------- SECTION 1 — System check ---------------- #
    print("\n[SECTION 1] System Check")
    print("-" * 42)
    ok = True
    if WEIGHTS.exists():
        print(f"  {OK} detector weights found: models/detection/best.pt")
    else:
        print(f"  {NO} MISSING detector weights: {WEIGHTS}")
        ok = False
    if DB_PATH.exists():
        print(f"  {OK} database found: plates.db")
    else:
        print(f"  {NO} MISSING plates.db — run setup_database_week3.py")
        ok = False
    if not ok:
        print("\nSystem check failed. Fix the above and re-run.")
        sys.exit(1)

    # Import the pipeline only after the checks pass.
    from alpr_pipeline_week3 import ALPRPipeline

    m = load_week2_metrics()
    map50 = m.get("mAP50", "?")
    lat = m.get("inference_ms", "?")

    pipe = ALPRPipeline(WEIGHTS, DB_PATH)
    n_reg = pipe.db.get_stats()["total_registered"]
    print(f"  {OK} YOLOv10 detector loaded (mAP50: {map50}, {lat}ms latency)")
    print(f"  {OK} Database connected ({n_reg} registered plates)")

    # ---------------- SECTION 2 — Process 5 images ---------------- #
    print("\n[SECTION 2] Process 5 Test Images")
    print("-" * 42)
    images = sorted(p for p in TEST_DIR.iterdir()
                    if p.suffix.lower() in IMG_EXTS)[:5] if TEST_DIR.is_dir() else []
    if not images:
        print(f"  {NO} no test images in {TEST_DIR}")
        pipe.close()
        sys.exit(1)

    for img in images:
        res = pipe.process_image(img)
        if res["error"]:
            print(f"  [{NO}] {img.name[:28]:<28} | ERROR: {res['error']}")
            continue
        n = res["plates_detected"]
        if n == 0:
            print(f"  [{NO}] {img.name[:28]:<28} | Plates: 0 | NO_PLATE")
            continue
        for d in res["detections"]:
            mark = OK if d["action"] == "ENTRY_ALLOWED" else NO
            name = img.name[:28]
            print(f"  [{mark}] {name:<28} | Plates: {n} | "
                  f"Conf: {d['confidence']:.2f} | {d['action']:<13} | "
                  f"{res['detection_latency_ms']:.0f}ms")

    # ---------------- SECTION 3 — Database log ---------------- #
    print("\n[SECTION 3] Database Audit Log (last 5)")
    print("-" * 42)
    print(f"  {'Time':<10} | {'Plate':<18} | {'Action':<13} | Conf")
    print("  " + "-" * 52)
    for r in pipe.db.get_recent_reads(limit=5):
        ts = r.get("timestamp", "")
        # keep only HH:MM:SS if it's a full timestamp
        t = ts.split(" ")[1] if isinstance(ts, str) and " " in ts else ts
        print(f"  {str(t):<10} | {str(r['detected_plate'])[:18]:<18} | "
              f"{str(r['action']):<13} | {r['yolo_confidence']:.2f}")

    # ---------------- SECTION 4 — Statistics ---------------- #
    print("\n[SECTION 4] Statistics")
    print("-" * 42)
    stats = pipe.db.get_stats()
    sess = pipe.get_session_stats()
    print(f"  Total processed today : {stats['allowed_today'] + stats['denied_today']}")
    print(f"  Entry allowed         : {stats['allowed_today']}")
    print(f"  Entry denied          : {stats['denied_today']}")
    print(f"  Avg detection latency : {sess['avg_detection_latency_ms']} ms")

    # ---------------- SECTION 5 — Week 2 metrics ---------------- #
    print("\n[SECTION 5] Detector Performance (Week 2)")
    print("-" * 42)
    map_ok = OK if isinstance(map50, (int, float)) and map50 >= 0.82 else "?"
    lat_ok = OK if isinstance(lat, (int, float)) and lat < 120 else "?"
    print(f"  YOLOv10 mAP50    : {map50}  {map_ok}")
    print(f"  YOLOv10 latency  : {lat}ms  {lat_ok}")
    print("  Pipeline status  : Stage 1 complete, CRNN pending (Week 9)")

    print("\n  NOTE: plate text is a placeholder (CRNN not built yet), so every")
    print("  plate is treated as UNKNOWN and DENIED — the fail-safe default.")
    print("  Week 9 adds CRNN text reading to enable real ENTRY_ALLOWED matches.")

    pipe.close()
    print("\nDemo complete. Ready for Week 9 (CRNN text recognition).")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] demo failed: {exc}")
        sys.exit(1)
