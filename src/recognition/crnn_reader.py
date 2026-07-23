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
from plate_format import is_valid as _format_is_valid


class CRNNReader:
    """Loads the trained CRNN once and reads plate text from crops."""

    def __init__(self, weights_path: str | Path,
                 charset_path: str | Path | None = None,
                 img_h: int = 64, img_w: int = 320,
                 device: str | None = None,
                 format_validation: bool = False,
                 confidence_mode: str = "mean") -> None:
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
        # PLAN_V2 Phase 1 (level 1 — reject only): when on, a read whose shape no
        # real Cambodian plate has gets confidence 0, so the REC-005 gate routes it
        # to REVIEW_REQUIRED. This only ever LOWERS a confidence, so it cannot turn
        # a DENY into an ALLOW — no false-accept risk. See plate_format.py.
        self.format_validation = bool(format_validation)
        # PLAN_V2 Phase 2 — how a whole-string confidence is reduced from the
        # per-character ones. An exact whitelist match dies on ONE wrong character,
        # but a mean over ~7 characters hides it (72% of 7,341 logged reads sit
        # above 0.9, so the 0.70 gate barely discriminates). "min" scores the
        # weakest link instead. Since min <= mean always, switching can only LOWER
        # a confidence -> strictly fewer auto-opens -> no new false-accept risk.
        if confidence_mode not in ("mean", "min", "geometric"):
            raise ValueError(f"unknown confidence_mode: {confidence_mode!r}")
        self.confidence_mode = confidence_mode
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

    def _char_confidences(self, maxp, argmax) -> list[float]:
        """Per-EMITTED-character confidences, aligned 1:1 with the decoded text.

        CTC emits a character at the first timestep of each non-blank run (the same
        rule as `CTCDecoder._collapse`), so the probability at that timestep is the
        model's confidence in that character. This differs from the legacy
        whole-string mean, which averaged over every non-blank timestep including
        the repeated ones.
        """
        out, prev = [], -1
        blank, n_chars = self.decoder.blank, len(self.charset)
        for i, idx in enumerate(argmax.tolist()):
            if idx != blank and idx != prev and 0 <= idx < n_chars:
                out.append(float(maxp[i]))
            prev = idx
        return out

    def _infer(self, crop) -> tuple[str, float, list[float]]:
        """Return (text, confidence, per_char_confidences).

        Confidence is reduced from the per-character probabilities according to
        `confidence_mode` (SRS REC-004):
          * "mean"      — legacy: mean over every non-blank TIMESTEP (bit-identical
                          to the pre-Phase-2 behaviour, kept as the default).
          * "min"       — the weakest character, because one wrong character is
                          enough to break an exact whitelist match.
          * "geometric" — exp(mean(log p)), a middle ground that still punishes a
                          single weak character but less absolutely than min.
        """
        t = self._preprocess(crop)
        with torch.no_grad():
            log_probs = self.model(t)              # (seq, 1, classes), log-softmax
        seq = log_probs.cpu().squeeze(1)           # (seq, classes)
        probs = seq.exp()                          # back to probabilities
        maxp, argmax = probs.max(dim=1)            # (seq,), (seq,)
        text = self.decoder._collapse(argmax)      # greedy CTC collapse
        char_confs = self._char_confidences(maxp, argmax)

        if self.confidence_mode == "mean":
            mask = argmax != self.decoder.blank     # non-blank timesteps (legacy)
            raw_conf = float(maxp[mask].mean()) if bool(mask.any()) else 0.0
        elif not char_confs:
            raw_conf = 0.0
        elif self.confidence_mode == "min":
            raw_conf = min(char_confs)
        else:                                       # geometric
            import math
            raw_conf = math.exp(
                sum(math.log(max(c, 1e-12)) for c in char_confs) / len(char_confs))
        # Length plausibility: a real Cambodian plate number is ~5-9 chars.
        # An implausibly short read (e.g. a single spurious char on a blank/
        # non-plate crop) is penalised so it falls below the REVIEW threshold.
        n = len(text.replace(" ", "").replace("-", ""))
        length_factor = min(1.0, n / self.min_plausible_len)
        conf = raw_conf * length_factor
        # Phase 1: an impossible plate shape is not a low-confidence read, it is a
        # wrong one — drop it to 0 so the confidence gate sends it for review.
        # (Kept AFTER length_factor so the two signals compose rather than mask.)
        if self.format_validation and not _format_is_valid(text):
            conf = 0.0
        return text, conf, char_confs

    def read(self, crop) -> tuple[str, float]:
        """Read plate text from a crop.

        Returns (text, confidence). On failure returns ("", 0.0) — never raises.
        NOTE: as of the SRS-alignment work this returns a TUPLE, not just text.
        Use `read_detailed()` when the per-character confidences are wanted too.
        """
        text, conf, _ = self.read_detailed(crop)
        return text, conf

    def read_detailed(self, crop) -> tuple[str, float, list[float]]:
        """Like `read()`, plus the per-character confidences aligned to the text.

        Lets a caller show *which* character is doubtful (PLAN_V2 Phase 2) instead
        of only a single opaque number. On failure returns ("", 0.0, []).
        """
        if crop is None or getattr(crop, "size", 0) == 0:
            return "", 0.0, []
        try:
            t0 = time.perf_counter()
            text, conf, char_confs = self._infer(crop)
            self._latencies.append((time.perf_counter() - t0) * 1000.0)
            return text, conf, char_confs
        except Exception as exc:
            print(f"[CRNNReader] read() error: {exc}")
            return "", 0.0, []

    @staticmethod
    def weakest_char(text: str, char_confs: list[float]) -> tuple[str, int, float] | None:
        """(character, index, confidence) of the least-confident character, or None.

        The character a human reviewer should look at first.
        """
        if not text or not char_confs or len(char_confs) != len(text):
            return None
        i = min(range(len(char_confs)), key=lambda k: char_confs[k])
        return text[i], i, char_confs[i]

    def get_avg_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)
