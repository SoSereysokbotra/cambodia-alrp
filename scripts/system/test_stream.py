#!/usr/bin/env python3
"""
scripts/system/test_stream.py
=============================
Quick connectivity test for a phone/IP camera stream BEFORE running the full
ALPR pipeline. Opens the source, grabs frames, reports resolution + measured
FPS, and saves a snapshot to results/stream_snapshot.jpg.

Examples (Android "IP Webcam" app — use the /video MJPEG URL it serves):
    python scripts/system/test_stream.py --source http://192.168.1.5:8080/video
    python scripts/system/test_stream.py                 # uses camera_source from config
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
CONFIG = PROJECT_ROOT / "configs" / "system_config.yaml"
OUT = PROJECT_ROOT / "results" / "stream_snapshot.jpg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None,
                    help="Stream URL / webcam index. Default: camera_source from config.")
    ap.add_argument("--seconds", type=float, default=5.0, help="How long to sample.")
    args = ap.parse_args()

    import cv2

    source = args.source
    if source is None:
        import yaml
        source = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["camera_source"]
    src = int(source) if str(source).isdigit() else str(source)

    print(f"Opening: {src}")
    t0 = time.time()
    cap = cv2.VideoCapture(src)
    # give network streams a moment to negotiate
    while not cap.isOpened() and time.time() - t0 < 8:
        time.sleep(0.3)
        cap.open(src)

    if not cap.isOpened():
        print("\n[FAIL] could not open the stream. Checklist:")
        print("  - Phone and PC on the SAME Wi-Fi network")
        print("  - IP Webcam app is running ('Start server' tapped)")
        print("  - URL is the /video MJPEG endpoint, e.g. http://<phone-ip>:8080/video")
        print("  - Open http://<phone-ip>:8080 in your PC browser first to confirm it loads")
        print("  - Firewall isn't blocking Python/OpenCV")
        sys.exit(1)

    n, first = 0, None
    t1 = time.time()
    while time.time() - t1 < args.seconds:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if first is None:
            first = frame
        n += 1
    dur = max(time.time() - t1, 1e-6)
    cap.release()

    if first is None:
        print("[FAIL] opened the source but received NO frames. Try the other IP Webcam "
              "URL (http://<phone-ip>:8080/videofeed) or lower the app's resolution.")
        sys.exit(1)

    h, w = first.shape[:2]
    fps = n / dur
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), first)
    print(f"\n[OK] connected. {w}x{h} | ~{fps:.1f} FPS over {dur:.1f}s | {n} frames")
    print(f"     snapshot saved -> {OUT}")
    print(f"     (SRS VID targets: >=15 FPS decode, >=20 FPS for the F1 2h run)")
    print("\nNext:")
    print(f'  1) set  camera_source: "{source}"  in configs/system_config.yaml')
    print(f"  2) run  python scripts/system/dashboard.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] stream test failed: {exc}")
        sys.exit(1)
