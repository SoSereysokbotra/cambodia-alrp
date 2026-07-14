#!/usr/bin/env python3
"""
scripts/run_demo_week12.py
==========================
MAIN teacher demo — the whole integrated system on any source.

Examples:
    python scripts/run_demo_week12.py                     # image folder (default)
    python scripts/run_demo_week12.py --source data/synthetic/test --limit 12
    python scripts/run_demo_week12.py --source 0          # webcam
    python scripts/run_demo_week12.py --source clip.mp4 --save-video
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.alpr_system import ALPRSystem   # noqa: E402

CONFIG = PROJECT_ROOT / "configs" / "system_config.yaml"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_folder_source(source: str) -> bool:
    s = str(source)
    if s.isdigit() or s.startswith(("rtsp://", "http://", "https://")):
        return False
    p = Path(s)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.is_dir()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="data/annotated/test/images/")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    system = ALPRSystem(str(CONFIG))
    # Synthetic inputs are already-cropped plates -> crop mode so ENTRY_ALLOWED
    # can be demonstrated (real photos use full YOLO detection).
    if "synthetic" in str(args.source).lower():
        system.assume_crop = True

    gate_mode = system.gate.get_status().upper()
    n_reg = system.database.get_stats()["total_registered"]

    print("==========================================")
    print("  CAMBODIAN ALPR - WEEK 12 DEMO")
    print("==========================================")
    print(f"Source: {args.source}")
    print(f"Gate:   {gate_mode}")
    print(f"DB:     {n_reg} registered plates")
    print(f"Mode:   {'crop (pre-cropped plates)' if system.assume_crop else 'full YOLO->CRNN'}")
    print("-" * 42)

    if is_folder_source(args.source):
        system.run_on_images(args.source, limit=args.limit, save=True)
    else:
        print("Live source — press 'q' in the window to stop.")
        system.run_video(show_display=not args.no_display,
                         save_video=args.save_video)

    # ---- session summary ---- #
    stats = system.get_session_stats()
    avg_lat = "n/a"
    summary = [
        "==========================================",
        "  SESSION SUMMARY",
        "==========================================",
        f"Frames processed : {stats['processed']}",
        f"Plates allowed   : {stats['allowed']}",
        f"Plates denied    : {stats['denied']}",
        f"Uptime (s)       : {stats['uptime_sec']}",
        f"Gate status      : {stats['gate_status']}",
        f"Output saved to  : {system.session_dir}",
        "==========================================",
    ]
    print("\n" + "\n".join(summary))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = system.log_dir / f"session_{ts}.txt"
    try:
        log_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
        print(f"[OK] summary saved -> {log_path}")
    except Exception as exc:
        print(f"[warn] could not write session log: {exc}")

    system.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] demo failed: {exc}")
        sys.exit(1)
