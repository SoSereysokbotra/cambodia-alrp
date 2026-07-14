"""
src/recognition/province_classifier.py
=======================================
ResNet18-based province classifier (Phase 3, Option A).

Reads a plate CROP (numpy BGR) and predicts which of the 26 classes
(25 provinces + 'other') it belongs to.

    clf = ProvinceClassifier("models/recognition/province_classifier_best.pth")
    province_id, confidence = clf.predict(crop_bgr)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch


def build_resnet18(n_classes: int):
    """ResNet18 with a fresh n_classes head (no torchvision weights required)."""
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
    return model


class ProvinceClassifier:
    def __init__(self, weights_path: str | Path,
                 config_path: str | Path | None = None,
                 device: str | None = None) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"classifier weights not found: {self.weights_path}")

        # config sits next to the weights by default
        if config_path is None:
            config_path = self.weights_path.parent / "province_classifier_config.json"
        cfg = {}
        if Path(config_path).exists():
            cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.n_classes = int(cfg.get("n_classes", 26))
        # maps model output index -> true province class id (handles ImageFolder's
        # lexicographic folder ordering). Identity if absent.
        self.idx_to_class = cfg.get("idx_to_class")
        self.img_size = int(cfg.get("img_size", 128))
        self.mean = cfg.get("mean", [0.485, 0.456, 0.406])
        self.std = cfg.get("std", [0.229, 0.224, 0.225])

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_resnet18(self.n_classes)
        try:
            state = torch.load(str(self.weights_path), map_location=self.device,
                               weights_only=True)
        except Exception:
            state = torch.load(str(self.weights_path), map_location=self.device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self._latencies: list[float] = []
        self._warmup()

    def _warmup(self) -> None:
        try:
            import numpy as np
            self.predict(np.zeros((self.img_size, self.img_size, 3), dtype="uint8"))
        except Exception:
            pass
        self._latencies.clear()

    def _preprocess(self, crop):
        import cv2
        import numpy as np
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size)).astype("float32") / 255.0
        rgb = (rgb - np.array(self.mean, dtype="float32")) / np.array(self.std, dtype="float32")
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
        return t.to(self.device)

    def predict(self, crop) -> tuple[int, float]:
        """Return (province_id, confidence). (OTHER_CLASS, 0.0) on failure."""
        if crop is None or getattr(crop, "size", 0) == 0:
            return 25, 0.0
        try:
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = self.model(self._preprocess(crop))
                probs = torch.softmax(logits, dim=1)
                conf, idx = probs.max(dim=1)
            self._latencies.append((time.perf_counter() - t0) * 1000.0)
            raw = int(idx.item())
            # translate model output index -> true province class id
            if self.idx_to_class and 0 <= raw < len(self.idx_to_class):
                raw = int(self.idx_to_class[raw])
            return raw, float(conf.item())
        except Exception as exc:
            print(f"[ProvinceClassifier] predict() error: {exc}")
            return 25, 0.0

    def get_avg_latency_ms(self) -> float:
        return sum(self._latencies) / len(self._latencies) if self._latencies else 0.0
