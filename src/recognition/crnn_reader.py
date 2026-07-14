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
        # A plausible Cambodian plate number has ~5+ characters; shorter reads
        # get their confidence scaled down (used by the REC-005 gate).
        self.min_plausible_len = 5
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

    def _infer(self, crop) -> tuple[str, float]:
        """Return (text, confidence). Confidence = mean of the per-timestep max
        softmax probabilities over the non-blank (character-emitting) timesteps —
        i.e. the average per-character confidence (SRS REC-004)."""
        t = self._preprocess(crop)
        with torch.no_grad():
            log_probs = self.model(t)              # (seq, 1, classes), log-softmax
        seq = log_probs.cpu().squeeze(1)           # (seq, classes)
        probs = seq.exp()                          # back to probabilities
        maxp, argmax = probs.max(dim=1)            # (seq,), (seq,)
        text = self.decoder._collapse(argmax)      # greedy CTC collapse
        mask = argmax != self.decoder.blank        # non-blank timesteps
        raw_conf = float(maxp[mask].mean()) if bool(mask.any()) else 0.0
        # Length plausibility: a real Cambodian plate number is ~5-9 chars.
        # An implausibly short read (e.g. a single spurious char on a blank/
        # non-plate crop) is penalised so it falls below the REVIEW threshold.
        n = len(text.replace(" ", "").replace("-", ""))
        length_factor = min(1.0, n / self.min_plausible_len)
        return text, raw_conf * length_factor

    def read(self, crop) -> tuple[str, float]:
        """Read plate text from a crop.

        Returns (text, confidence). On failure returns ("", 0.0) — never raises.
        NOTE: as of the SRS-alignment work this returns a TUPLE, not just text.
        """
        if crop is None or getattr(crop, "size", 0) == 0:
            return "", 0.0
        try:
            t0 = time.perf_counter()
            text, conf = self._infer(crop)
            self._latencies.append((time.perf_counter() - t0) * 1000.0)
            return text, conf
        except Exception as exc:
            print(f"[CRNNReader] read() error: {exc}")
            return "", 0.0

    def get_avg_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)
