"""
src/recognition/crnn_reader.py
==============================
Inference-time wrapper: turn a plate CROP (numpy BGR) into a text string.

Used by the Week-5 pipeline to replace the Week-3 placeholder:
    Week 3: plate_text = f"PLATE_{i+1}_DETECTED"
    Week 5: plate_text = reader.read(detection["crop"])
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from crnn_model import CTCDecoder, CHARSET, load_crnn


class CRNNReader:
    """Loads the trained CRNN once and reads plate text from crops."""

    def __init__(self, weights_path: str | Path,
                 charset_path: str | Path | None = None,
                 img_h: int = 64, img_w: int = 320,
                 device: str | None = None) -> None:
        self.img_h = img_h
        self.img_w = img_w
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # charset must match training; fall back to the default CHARSET
        charset = CHARSET
        if charset_path and Path(charset_path).exists():
            charset = Path(charset_path).read_text(encoding="utf-8")
        self.charset = charset

        self.model = load_crnn(weights_path, device=self.device, charset=charset,
                               img_h=img_h, img_w=img_w)
        self.decoder = CTCDecoder(charset, len(charset))
        self._latencies: list[float] = []
        self._warmup()

    def _warmup(self) -> None:
        try:
            import numpy as np
            dummy = np.zeros((self.img_h, self.img_w), dtype="uint8")
            self._infer(dummy)
        except Exception:
            pass
        self._latencies.clear()

    def _preprocess(self, crop):
        """BGR/gray numpy crop -> (1,1,H,W) float tensor in [-1,1]."""
        import cv2
        import numpy as np
        if crop.ndim == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(crop, (self.img_w, self.img_h))
        arr = resized.astype("float32") / 255.0
        arr = (arr - 0.5) / 0.5
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        return t.to(self.device)

    def _infer(self, crop) -> str:
        t = self._preprocess(crop)
        with torch.no_grad():
            log_probs = self.model(t)          # (seq, 1, classes)
        return self.decoder.decode(log_probs.cpu().squeeze(1))

    def read(self, crop) -> str:
        """Read plate text from a crop. Returns '' on failure (never raises)."""
        if crop is None or getattr(crop, "size", 0) == 0:
            return ""
        try:
            t0 = time.perf_counter()
            text = self._infer(crop)
            self._latencies.append((time.perf_counter() - t0) * 1000.0)
            return text
        except Exception as exc:
            print(f"[CRNNReader] read() error: {exc}")
            return ""

    def get_avg_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)
