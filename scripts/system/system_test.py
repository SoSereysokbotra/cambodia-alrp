#!/usr/bin/env python3
"""
scripts/system_test_week12.py
=============================
End-to-end system readiness test: checks all 6 components, runs a 10-frame
pipeline test, and prints a PASS/FAIL verdict.

Run:
    python scripts/system_test_week12.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CONFIG = PROJECT_ROOT / "configs" / "system_config.yaml"
TEST_DIR = PROJECT_ROOT / "data" / "annotated" / "test" / "images"
YOLO_METRICS = PROJECT_ROOT / "metrics" / "week2_metrics.json"
CRNN_METRICS = PROJECT_ROOT / "metrics" / "crnn_week5_metrics.json"
OUT_TXT = PROJECT_ROOT / "metrics" / "system_test_week12.txt"
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    lines: list[str] = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 46)
    out(" CAMBODIAN ALPR - SYSTEM TEST (Week 12)")
    out("=" * 46)

    failures: list[str] = []

    # Import system (also validates the src packages load)
    try:
        from core.alpr_system import ALPRSystem
        system = ALPRSystem(str(CONFIG))
    except Exception as exc:
        out(f"[FATAL] could not initialise ALPRSystem: {exc}")
        OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
        OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        sys.exit(1)

    health = system.health_check()

    # [1/6] GPU
    gpu = health.get("gpu", "?")
    if gpu == "OK":
        out("[1/6] GPU/CUDA ............ OK: CUDA available")
    else:
        out(f"[1/6] GPU/CUDA ............ WARNING: {gpu}")  # not fatal

    # [2/6] YOLO
    ym = load_json(YOLO_METRICS)
    if health.get("yolo") == "OK":
        out(f"[2/6] YOLOv10 model ....... OK: mAP50={ym.get('mAP50','?')}, "
            f"{ym.get('inference_ms','?')}ms")
    else:
        out(f"[2/6] YOLOv10 model ....... FAIL: {health.get('yolo')}")
        failures.append("yolo")

    # [3/6] CRNN
    cm = load_json(CRNN_METRICS)
    charset_n = len(system.reader.charset) if hasattr(system.reader, "charset") else 0
    if health.get("crnn") == "OK":
        cer = cm.get("cer")
        cer_s = f"{cer*100:.2f}%" if isinstance(cer, (int, float)) else "?"
        out(f"[3/6] CRNN model .......... OK: CER={cer_s}, charset={charset_n}")
    else:
        out(f"[3/6] CRNN model .......... FAIL: {health.get('crnn')}")
        failures.append("crnn")

    # [4/6] Database
    db_status = health.get("database", "")
    if db_status.startswith("OK"):
        out(f"[4/6] Database ............ {db_status}")
    else:
        out(f"[4/6] Database ............ FAIL: {db_status}")
        failures.append("database")

    # [5/6] Camera/source
    n_imgs = system.camera.frame_count()
    if system.camera.mode == "folder" and n_imgs > 0:
        out(f"[5/6] Camera/source ....... OK: image folder ({n_imgs} images found)")
    elif system.camera.mode in ("webcam", "stream", "video"):
        out(f"[5/6] Camera/source ....... OK: {system.camera.mode} configured")
    else:
        out("[5/6] Camera/source ....... FAIL: no frames available")
        failures.append("camera")

    # [6/6] MQTT gate
    gate_status = health.get("mqtt", "error")
    if gate_status in ("connected", "mock"):
        label = "connected" if gate_status == "connected" else "MOCK mode"
        out(f"[6/6] MQTT gate ........... OK: {label}")
    else:
        out(f"[6/6] MQTT gate ........... FAIL: {gate_status}")
        failures.append("mqtt")

    # ---- 10-frame pipeline test ---- #
    out("")
    out("Running 10-frame pipeline test ...")
    import cv2
    images = sorted(p for p in TEST_DIR.iterdir()
                    if p.suffix.lower() in IMG_EXTS)[:11] if TEST_DIR.is_dir() else []
    # Warm-up frame (cold-start cuDNN init) — not timed, mirrors real continuous run.
    if images:
        warm = cv2.imread(str(images[0]))
        if warm is not None:
            system.process_frame(warm)
        images = images[1:]

    detections_hit = 0
    max_total = 0.0
    max_db = 0.0
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        res = system.process_frame(frame)
        if res["plates_count"] >= 1:
            detections_hit += 1
        max_total = max(max_total, res["total_ms"])
        for p in res["plates"]:
            max_db = max(max_db, p["db_ms"])

    n = len(images)
    out(f"  detections: {detections_hit}/{n} frames had >=1 plate")
    out(f"  max DB lookup: {max_db:.2f} ms")
    out(f"  max total latency: {max_total:.1f} ms")

    if n and detections_hit < (n // 2 + 1):
        failures.append("detection-rate")
        out("  WARNING: fewer than half the frames had a detection")
    if max_db >= 5:
        failures.append("db-latency")
        out("  WARNING: DB lookup exceeded 5 ms")
    if max_total >= 200:
        failures.append("total-latency")
        out("  WARNING: total latency exceeded 200 ms")

    # ---- verdict ---- #
    out("")
    out("=" * 46)
    if not failures:
        out(" SYSTEM TEST: PASS")
        out(" Ready for pilot deployment: YES")
    else:
        out(" SYSTEM TEST: FAIL")
        out(f" Failed checks: {failures}")
    out("=" * 46)

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[OK] saved -> {OUT_TXT}")
    system.close()
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] system test failed: {exc}")
        sys.exit(1)
