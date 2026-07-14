#!/usr/bin/env python3
"""
scripts/latency_profiler_week11.py
==================================
Profile per-stage latency over 100 frames and report avg/min/max/p95.

Run:
    python scripts/latency_profiler_week11.py
    python scripts/latency_profiler_week11.py --frames 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.alpr_system import ALPRSystem   # noqa: E402

CONFIG = PROJECT_ROOT / "configs" / "system_config.yaml"
TEST_DIR = PROJECT_ROOT / "data" / "annotated" / "test" / "images"
METRICS_JSON = PROJECT_ROOT / "metrics" / "latency_profile_week11.json"
IMG_EXTS = {".jpg", ".jpeg", ".png"}

TOTAL_TARGET_MS = 100
FPS_TARGET = 10


def stats(values: list[float]) -> dict:
    if not values:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0}
    s = sorted(values)
    p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
    return {"avg": sum(s) / len(s), "min": s[0], "max": s[-1], "p95": p95}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=100)
    args = ap.parse_args()

    if not TEST_DIR.is_dir():
        print(f"[X] test images not found: {TEST_DIR}")
        sys.exit(1)

    import cv2
    system = ALPRSystem(str(CONFIG))

    images = sorted(p for p in TEST_DIR.iterdir() if p.suffix.lower() in IMG_EXTS)
    images = images[:args.frames]
    if not images:
        print("[X] no test images.")
        sys.exit(1)

    load_t, yolo_t, crnn_t, db_t, total_t = [], [], [], [], []

    # Warm-up (cold-start cuDNN init) — not counted in the profile.
    warm = cv2.imread(str(images[0]))
    if warm is not None:
        system.process_frame(warm)

    print(f"Profiling {len(images)} frames ...")
    for i, img_path in enumerate(images, 1):
        t_load = time.perf_counter()
        frame = cv2.imread(str(img_path))
        load_ms = (time.perf_counter() - t_load) * 1000
        if frame is None:
            continue
        res = system.process_frame(frame)

        load_t.append(load_ms)
        yolo_t.append(res["yolo_ms"])
        total_t.append(res["total_ms"] + load_ms)
        # per-plate stages: sum across plates in the frame (0 if none)
        crnn_t.append(sum(p["crnn_ms"] for p in res["plates"]))
        db_t.append(sum(p["db_ms"] for p in res["plates"]))
        if i % 25 == 0:
            print(f"  {i}/{len(images)}")

    s_load, s_yolo = stats(load_t), stats(yolo_t)
    s_crnn, s_db, s_total = stats(crnn_t), stats(db_t), stats(total_t)
    avg_total = s_total["avg"]
    throughput = 1000.0 / avg_total if avg_total > 0 else 0.0

    def row(name, s):
        return f"{name:<16}{s['avg']:>6.1f}ms{s['min']:>7.1f}ms{s['max']:>8.1f}ms{s['p95']:>8.1f}ms"

    print(f"\n=== LATENCY PROFILE ({len(total_t)} frames) ===")
    print(f"{'Stage':<16}{'Avg':>8}{'Min':>9}{'Max':>10}{'P95':>10}")
    print("-" * 53)
    print(row("Frame load", s_load))
    print(row("YOLOv10", s_yolo))
    print(row("CRNN read", s_crnn))
    print(row("DB lookup", s_db))
    print("-" * 53)
    total_ok = "PASS" if avg_total < TOTAL_TARGET_MS else "FAIL"
    fps_ok = "PASS" if throughput > FPS_TARGET else "FAIL"
    print(f"{'TOTAL':<16}{avg_total:>6.1f}ms   [TARGET <{TOTAL_TARGET_MS}ms] {total_ok}")
    print(f"{'Throughput':<16}{throughput:>6.1f} FPS [TARGET >{FPS_TARGET}FPS] {fps_ok}")

    # flag slow stages
    for name, s in [("YOLOv10", s_yolo), ("CRNN read", s_crnn), ("DB lookup", s_db)]:
        if s["p95"] > 100:
            print(f"WARNING: {name} P95 is high ({s['p95']:.1f}ms)")

    out = {
        "frames": len(total_t),
        "frame_load_ms": s_load, "yolo_ms": s_yolo, "crnn_ms": s_crnn,
        "db_ms": s_db, "total_ms": s_total,
        "throughput_fps": round(throughput, 1),
        "total_pass": total_ok == "PASS", "fps_pass": fps_ok == "PASS",
    }
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[OK] saved -> {METRICS_JSON}")
    system.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] profiler failed: {exc}")
        sys.exit(1)
