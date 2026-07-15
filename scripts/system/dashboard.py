#!/usr/bin/env python3
"""
scripts/system/dashboard.py
===========================
Live control dashboard (SRS Phase 10 — UI-001, USAB-001/002). One screen shows
the live annotated video, gate status, FPS/latency/GPU, session counters, the
last events, and clickable MANUAL-OPEN / E-STOP controls (also keys o / e / q).

Runs on any source the pipeline accepts (image folder loops, video, webcam, RTSP):
    python scripts/system/dashboard.py                          # config source
    python scripts/system/dashboard.py --source data/annotated/test/images/
    python scripts/system/dashboard.py --source rtsp://192.168.1.5:8080/h264
    python scripts/system/dashboard.py --headless 30            # no window; save a preview PNG
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.alpr_system import ALPRSystem          # noqa: E402
from utils.rtsp_reader import RTSPReader          # noqa: E402

CONFIG = PROJECT_ROOT / "configs" / "system_config.yaml"

# ---- layout (fixed so drawing + mouse hit-testing agree) ------------------ #
VID_W, VID_H = 720, 540
PANEL_W = 380
W, H = VID_W + PANEL_W, 640                        # panel taller than the video
X0 = VID_W                                        # panel left edge
BTN_OPEN = (X0 + 20, H - 120, X0 + PANEL_W - 20, H - 80)
BTN_ESTOP = (X0 + 20, H - 70, X0 + PANEL_W - 20, H - 30)
EVENTS_TOP = 394                                  # first event row y
EVENTS_MAX = max(1, (BTN_OPEN[1] - 20 - EVENTS_TOP) // 20)

# BGR colors
BG = (28, 28, 30)
WHITE = (238, 238, 238)
GREY = (120, 120, 120)
GREEN = (60, 200, 90)
RED = (60, 60, 225)
AMBER = (40, 180, 255)
ACTION_COLOR = {
    "ENTRY_ALLOWED": GREEN, "ENTRY_DENIED": RED,
    "REVIEW_REQUIRED": AMBER, "MANUAL_OVERRIDE": GREEN, "ERROR": GREY,
}


def _letterbox(img, w, h):
    import cv2
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh))
    import numpy as np
    canvas = np.full((h, w, 3), 0, np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def _text(img, s, org, scale=0.6, color=WHITE, thick=1):
    import cv2
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _button(img, rect, label, color, active=False):
    import cv2
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1 if active else 2)
    tc = (20, 20, 20) if active else color
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    _text(img, label, (x1 + (x2 - x1 - tw) // 2, y2 - 12), 0.7, tc, 2)


def build_frame(system, video, fps, latency_ms, gpu_mb, gate_open, events):
    import numpy as np
    stats = system.get_session_stats()
    canvas = np.full((H, W, 3), BG, np.uint8)
    canvas[0:VID_H, 0:VID_W] = _letterbox(video, VID_W, VID_H)

    # ---- right panel ---- #
    _text(canvas, "ALPR CONTROL DASHBOARD", (X0 + 16, 34), 0.7, WHITE, 2)
    _text(canvas, time.strftime("%Y-%m-%d %H:%M:%S"), (X0 + 16, 58), 0.5, GREY)

    # gate status banner
    if system.estop_active:
        gstate, gcol = "EMERGENCY STOP", RED
    elif gate_open:
        gstate, gcol = "GATE OPEN", GREEN
    else:
        gstate, gcol = "GATE CLOSED", GREY
    import cv2
    cv2.rectangle(canvas, (X0 + 16, 74), (W - 16, 116), gcol, -1)
    _text(canvas, gstate, (X0 + 28, 103), 0.9, (20, 20, 20), 2)

    # metrics
    y = 150
    for label, val in (("FPS", f"{fps:5.1f}"),
                       ("Latency", f"{latency_ms:5.0f} ms"),
                       ("GPU mem", f"{gpu_mb:6.0f} MB"),
                       ("Uptime", f"{stats['uptime_sec']:6.0f} s")):
        _text(canvas, f"{label:<9}: {val}", (X0 + 16, y), 0.55, WHITE)
        y += 26

    # counters
    y += 6
    for label, val, col in (("Allowed", stats["allowed"], GREEN),
                            ("Denied", stats["denied"], RED),
                            ("Review", stats["review"], AMBER),
                            ("Override", stats["manual_override"], WHITE)):
        _text(canvas, f"{label:<9}: {val}", (X0 + 16, y), 0.55, col)
        y += 26

    # recent events (capped so the list never overlaps the control buttons)
    _text(canvas, "RECENT EVENTS", (X0 + 16, EVENTS_TOP - 22), 0.55, WHITE, 1)
    y = EVENTS_TOP
    for ev in list(events)[:EVENTS_MAX]:
        col = ACTION_COLOR.get(ev["action"], WHITE)
        _text(canvas, f"{ev['t']}  {ev['num']:<9} {ev['action']}",
              (X0 + 16, y), 0.44, col)
        y += 20

    # controls
    _button(canvas, BTN_OPEN, "MANUAL OPEN  [o]", GREEN)
    _button(canvas, BTN_ESTOP, "E-STOP  [e]", RED, active=system.estop_active)
    # footer hint under the video column
    _text(canvas, "[o] manual open    [e] emergency stop    [q] quit",
          (16, VID_H + 40), 0.6, GREEN, 1)
    return canvas


def _num_of(plate_text: str) -> str:
    """Latin number part (last token) — Khmer province won't render in cv2."""
    return plate_text.split(" ")[-1] if plate_text else "?"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=None,
                    help="Override camera_source (folder / video / webcam idx / rtsp url).")
    ap.add_argument("--headless", type=int, default=0,
                    help="No window: process N frames and save a preview PNG.")
    args = ap.parse_args()

    import cv2
    system = ALPRSystem(str(CONFIG))
    source = args.source if args.source is not None else system.config["camera_source"]
    if isinstance(source, str) and not source.isdigit() \
            and not source.startswith(("rtsp://", "http://", "https://")):
        p = Path(source)
        source = str(p if p.is_absolute() else PROJECT_ROOT / p)

    events: deque = deque(maxlen=20)
    lat_hist: deque = deque(maxlen=30)
    last_open = 0.0

    def record(res):
        nonlocal last_open
        lat_hist.append(res["total_ms"])
        for pl in res["plates"]:
            events.appendleft({"t": time.strftime("%H:%M:%S"),
                               "num": _num_of(pl["plate_text"]),
                               "action": pl["action"]})
            if pl["action"] == "ENTRY_ALLOWED":
                last_open = time.time()

    def metrics():
        lat = sum(lat_hist) / len(lat_hist) if lat_hist else 0.0
        fps = 1000.0 / max(lat, 1e-6) if lat_hist else 0.0
        return fps, lat, system._gpu_memory_mb()

    # ---------------- headless preview (no display needed) ---------------- #
    if args.headless:
        imgs = sorted(p for p in Path(source).iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) \
            if Path(source).is_dir() else []
        # seed with the enrolled plate so the preview shows a real ENTRY_ALLOWED
        enrolled = next((p for p in imgs if p.stem.startswith("12cb732b")), None)
        seq = ([enrolled] if enrolled else []) + [p for p in imgs if p != enrolled]
        video = None
        for ip in seq[:max(args.headless, 1)]:
            frame = cv2.imread(str(ip))
            if frame is None:
                continue
            res = system.process_frame(frame)
            record(res)
            video = res["frame"]
        fps, lat, gpu = metrics()
        gate_open = (time.time() - last_open) < 3.0
        canvas = build_frame(system, video if video is not None else
                             __import__("numpy").zeros((VID_H, VID_W, 3), "uint8"),
                             fps, lat, gpu, gate_open, events)
        out = PROJECT_ROOT / "results" / "dashboard_preview.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), canvas)
        print(f"[OK] dashboard preview -> {out}")
        system.close()
        return

    # ---------------- live interactive dashboard -------------------------- #
    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if BTN_OPEN[0] <= x <= BTN_OPEN[2] and BTN_OPEN[1] <= y <= BTN_OPEN[3]:
            system.manual_override()
        elif BTN_ESTOP[0] <= x <= BTN_ESTOP[2] and BTN_ESTOP[1] <= y <= BTN_ESTOP[3]:
            system.emergency_stop()

    win = "ALPR Control Dashboard"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    reader = RTSPReader(source).start()
    frame_i = 0
    print("Dashboard running — keys: [o] manual open  [e] e-stop  [q] quit")
    try:
        while True:
            frame = reader.get_frame()
            if frame is None:
                if not reader.is_connected():
                    print("[dashboard] source ended / disconnected.")
                    break
                continue
            res = system.process_frame(frame)
            record(res)
            frame_i += 1
            if frame_i % system.metrics_every == 0:
                system.log_metrics_sample()
            fps, lat, gpu = metrics()
            gate_open = (time.time() - last_open) < system.open_duration
            canvas = build_frame(system, res["frame"], fps, lat, gpu, gate_open, events)
            cv2.imshow(win, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("o"):
                system.manual_override()
                last_open = time.time()
            elif key == ord("e"):
                system.emergency_stop()
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        system.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] dashboard failed: {exc}")
        raise
