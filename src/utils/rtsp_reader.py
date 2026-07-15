"""
src/utils/rtsp_reader.py
========================
Threaded video-source reader that works with ANY source:

    RTSPReader(0)                              -> webcam
    RTSPReader("rtsp://user:pass@ip:554/...")  -> IP camera stream
    RTSPReader("clip.mp4")                     -> video file
    RTSPReader("data/annotated/test/images/")  -> image folder (loops)

A background thread reads frames continuously; get_frame() returns the LATEST
frame (so processing never falls behind the stream). Designed to never crash
the pipeline — connection problems return None / trigger reconnect.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


class RTSPReader:
    def __init__(self, source, queue_size: int = 5, log_interval: int = 30,
                 reconnect_interval: float = 5.0,
                 max_reconnect_attempts: int = 10) -> None:
        self.source_raw = source
        self.mode = self._detect_mode(source)
        self.log_interval = log_interval
        # SRS VID-001: reconnect every `reconnect_interval` s, up to `max` attempts.
        self.reconnect_interval = float(reconnect_interval)
        self.max_reconnect_attempts = int(max_reconnect_attempts)

        self._latest = None
        self._latest_ts = 0.0                       # VID-002: capture time (epoch s)
        self.frame_timestamp_ms = 0.0               # capture time of last get_frame()
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap = None
        self._fps = 0.0
        self._connected = False

        self._images: list[Path] = []
        self._idx = 0
        if self.mode == "folder":
            self._images = sorted(p for p in Path(str(source)).iterdir()
                                  if p.suffix.lower() in IMG_EXTS)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _detect_mode(source) -> str:
        if isinstance(source, int):
            return "webcam"
        s = str(source)
        if s.isdigit():
            return "webcam"
        if s.startswith(("rtsp://", "http://", "https://")):
            return "stream"
        p = Path(s)
        if p.is_dir():
            return "folder"
        if p.suffix.lower() in VIDEO_EXTS:
            return "video"
        return "video"  # default: treat as a media path

    def _open_capture(self) -> bool:
        import cv2
        src = int(self.source_raw) if self.mode == "webcam" else str(self.source_raw)
        self._cap = cv2.VideoCapture(src)
        self._connected = bool(self._cap and self._cap.isOpened())
        return self._connected

    # ------------------------------------------------------------------ #
    def start(self) -> "RTSPReader":
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        import cv2
        last_log = time.time()
        frame_times: list[float] = []

        if self.mode in ("webcam", "stream", "video"):
            if not self._open_capture() and not self.reconnect():
                print(f"[RTSPReader] could not open source: {self.source_raw}")
                self._connected = False
                return

        while not self._stop.is_set():
            t0 = time.time()
            if self.mode == "folder":
                if not self._images:
                    print("[RTSPReader] image folder is empty")
                    break
                frame = cv2.imread(str(self._images[self._idx]))
                self._idx = (self._idx + 1) % len(self._images)  # loop
                self._connected = True
                if frame is None:
                    continue
            else:
                ok, frame = self._cap.read()
                if not ok:
                    if self.mode == "video":
                        self._connected = False
                        break  # end of file
                    if not self.reconnect():
                        break
                    continue

            with self._lock:
                self._latest = frame
                self._latest_ts = time.time()       # VID-002: acquisition timestamp
                self._new.set()

            dt = time.time() - t0
            frame_times.append(dt)
            frame_times = frame_times[-30:]
            total = sum(frame_times)
            if total > 0:
                self._fps = len(frame_times) / total

            if time.time() - last_log >= self.log_interval:
                status = "connected" if self._connected else "disconnected"
                print(f"[RTSPReader] status={status} fps={self._fps:.1f} mode={self.mode}")
                last_log = time.time()

            # throttle file/folder playback toward ~30 FPS
            if self.mode in ("folder", "video"):
                time.sleep(max(0.0, (1 / 30) - (time.time() - t0)))

        self._connected = False

    # ------------------------------------------------------------------ #
    def get_frame(self, timeout: float = 2.0):
        """Return the latest frame, or None after `timeout` seconds. Also records
        that frame's acquisition time in `self.frame_timestamp_ms` (VID-002)."""
        if self._new.wait(timeout=timeout):
            with self._lock:
                frame = None if self._latest is None else self._latest.copy()
                self.frame_timestamp_ms = self._latest_ts * 1000.0
            self._new.clear()
            return frame
        return None

    def get_frame_ts(self, timeout: float = 2.0):
        """Like get_frame() but returns (frame, capture_time_ms) (VID-002)."""
        frame = self.get_frame(timeout=timeout)
        return frame, self.frame_timestamp_ms

    def get_fps(self) -> float:
        return round(self._fps, 1)

    def is_connected(self) -> bool:
        return self._connected

    def reconnect(self, retries: int | None = None) -> bool:
        """SRS VID-001: retry the source every `reconnect_interval` s, up to
        `max_reconnect_attempts` times (defaults from config)."""
        retries = self.max_reconnect_attempts if retries is None else retries
        for i in range(retries):
            print(f"[RTSPReader] reconnect attempt {i + 1}/{retries} "
                  f"(every {self.reconnect_interval:.0f}s) ...")
            if self._open_capture():
                print("[RTSPReader] reconnected.")
                return True
            if self._stop.is_set():
                break
            time.sleep(self.reconnect_interval)
        return False

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._connected = False

    def frame_count(self) -> int:
        """Number of images (folder mode) — handy for bounded test loops."""
        return len(self._images)
