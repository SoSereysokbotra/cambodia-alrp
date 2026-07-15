"""
src/detection/detector.py
=========================
Reusable YOLOv10 license-plate DETECTION module.

Loads the trained detector (models/detection/best.pt) ONCE and exposes a clean
interface the rest of the pipeline (and Week 9's CRNN) can build on.

Example
-------
    import cv2
    from detection.detector import PlateDetector

    det = PlateDetector("models/detection/best.pt")
    frame = cv2.imread("photo.jpg")
    plates = det.detect(frame)              # [{"bbox", "confidence", "crop"}]
    annotated = det.draw_boxes(frame, plates, ["ENTRY_ALLOWED"])
    print(det.get_avg_latency_ms())
"""

from __future__ import annotations

import time
from pathlib import Path

# BGR colours for OpenCV (action -> colour).
COLOR_ALLOWED = (0, 255, 0)     # green
COLOR_DENIED = (0, 0, 255)      # red
COLOR_UNKNOWN = (0, 255, 255)   # yellow

ACTION_COLORS = {
    "ENTRY_ALLOWED": COLOR_ALLOWED,
    "ENTRY_DENIED": COLOR_DENIED,
    "UNKNOWN": COLOR_UNKNOWN,
}


class PlateDetector:
    """Wraps a trained YOLOv10 model for plate detection + visualisation."""

    def __init__(self, weights_path: str | Path = "models/detection/best.pt",
                 conf: float = 0.5, device: str | None = None) -> None:
        """
        Parameters
        ----------
        weights_path : path to the trained best.pt (loaded once, here).
        conf         : confidence threshold for a valid detection (default 0.5).
        device       : '0' GPU, 'cpu', or None to auto-select.
        """
        self.weights_path = Path(weights_path)
        self.conf = conf
        self.device = device
        self._latencies: list[float] = []

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Detector weights not found: {self.weights_path}\n"
                "Train the model first (Week 2) so best.pt exists."
            )

        # Import here so importing this module is cheap and doesn't require
        # ultralytics unless a detector is actually created.
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ultralytics is required. Activate the venv and: "
                "pip install ultralytics"
            ) from exc

        # Load ONCE. Every detect() call reuses this in-memory model.
        self.model = YOLO(str(self.weights_path))
        self._warmup()

    def _warmup(self) -> None:
        """Run one throwaway inference so the first REAL detect() isn't skewed
        by lazy CUDA/kernel initialisation. Not counted in latency stats."""
        try:
            import numpy as np
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(source=dummy, conf=self.conf,
                               device=self.device, verbose=False)
        except Exception:
            pass  # warm-up is best-effort only
        self._latencies.clear()

    # ------------------------------------------------------------------ #
    # Core inference
    # ------------------------------------------------------------------ #
    def detect(self, image, pad: float = 0.0) -> list[dict]:
        """Run detection on a single BGR numpy image.

        Parameters
        ----------
        pad : fraction to expand each bbox by before cropping (SRS DET-005,
              e.g. 0.10 = 10% margin of context), clamped to the image bounds.
              The returned "bbox" stays tight (for drawing); only "crop" is padded.

        Returns a list of dicts:
            [{"bbox": (x1, y1, x2, y2),
              "confidence": float,
              "crop": numpy BGR array of the plate region}]
        Returns [] if nothing is found or on a bad image (never raises).
        """
        if image is None or getattr(image, "size", 0) == 0:
            return []

        detections: list[dict] = []
        try:
            t0 = time.perf_counter()
            results = self.model.predict(
                source=image, conf=self.conf,
                device=self.device, verbose=False,
            )
            self._latencies.append((time.perf_counter() - t0) * 1000.0)

            result = results[0]
            boxes = result.boxes
            if boxes is None:
                return []

            h, w = image.shape[:2]
            for b in boxes:
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                # Clamp to image bounds so crops never go out of range.
                x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
                y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))
                if x2 <= x1 or y2 <= y1:
                    continue  # degenerate box, skip
                # DET-005: expand the crop by `pad` on each side (clamped),
                # keeping the reported bbox tight for visualisation.
                if pad > 0:
                    dx, dy = int((x2 - x1) * pad), int((y2 - y1) * pad)
                    cx1, cy1 = max(0, x1 - dx), max(0, y1 - dy)
                    cx2, cy2 = min(w, x2 + dx), min(h, y2 + dy)
                else:
                    cx1, cy1, cx2, cy2 = x1, y1, x2, y2
                crop = image[cy1:cy2, cx1:cx2].copy()
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(b.conf[0]),
                    "crop": crop,
                })
        except Exception as exc:  # never let a bad frame crash the pipeline
            print(f"[PlateDetector] detect() error: {exc}")
            return detections

        return detections

    # ------------------------------------------------------------------ #
    # Visualisation
    # ------------------------------------------------------------------ #
    def draw_boxes(self, image, detections: list[dict],
                   actions: list[str] | None = None):
        """Return an annotated COPY of image (original is never modified).

        actions[i] colours detections[i]:
            ENTRY_ALLOWED -> green, ENTRY_DENIED -> red, UNKNOWN -> yellow.
        """
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise ImportError("opencv-python is required for draw_boxes().") from exc

        out = image.copy()
        for i, det in enumerate(detections):
            action = actions[i] if actions and i < len(actions) else "UNKNOWN"
            color = ACTION_COLORS.get(action, COLOR_UNKNOWN)
            x1, y1, x2, y2 = det["bbox"]
            conf = det.get("confidence", 0.0)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{action} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            # filled label background above the box (kept inside the frame)
            ly = max(y1, th + 6)
            cv2.rectangle(out, (x1, ly - th - 6), (x1 + tw + 4, ly), color, -1)
            cv2.putText(out, label, (x1 + 2, ly - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return out

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def get_avg_latency_ms(self) -> float:
        """Average detect() latency (ms) over all calls so far."""
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    def reset_latency(self) -> None:
        self._latencies.clear()
