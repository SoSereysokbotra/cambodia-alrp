"""
src/recognition/crnn_dataset.py
===============================
Dataset + collate for CRNN training.

Expects a CSV with a header and two columns:
    image_path,plate_text
    data/synthetic/train/000001.jpg,1AB-2345

Each item is returned as:
    (image_tensor (1,H,W), target_indices (L,), target_length, raw_text)
so training can use the indices for CTC and validation can compare raw_text.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from crnn_model import CHARSET   # noqa: E402  (src/recognition is on sys.path)

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise ImportError("Pillow is required: pip install pillow") from exc


class PlateDataset(Dataset):
    def __init__(self, csv_path, img_h: int = 64, img_w: int = 320,
                 charset: str = CHARSET, max_len: int = 40) -> None:
        self.img_h = img_h
        self.img_w = img_w
        self.charset = charset
        self.char_to_idx = {c: i for i, c in enumerate(charset)}
        self.max_len = max_len

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"labels CSV not found: {csv_path}")

        self.samples: list[tuple[str, str]] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = (row.get("image_path") or "").strip()
                text = (row.get("plate_text") or "").strip()
                if not path or not text:
                    continue
                # keep only labels fully representable by the charset
                enc = self.encode(text)
                if 1 <= len(enc) <= max_len and Path(path).exists():
                    self.samples.append((path, text))

        if not self.samples:
            raise ValueError(
                f"No usable samples in {csv_path}. Check image paths / charset."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def encode(self, text: str) -> list[int]:
        """Text -> list of char indices (chars not in charset are dropped)."""
        return [self.char_to_idx[c] for c in text.upper() if c in self.char_to_idx]

    def __getitem__(self, idx: int):
        path, text = self.samples[idx]
        try:
            img = Image.open(path).convert("L")             # grayscale
        except Exception:
            # Return a blank image on read failure (never crash the loader).
            img = Image.new("L", (self.img_w, self.img_h), color=127)
        img = img.resize((self.img_w, self.img_h))
        # to tensor, normalise to [-1, 1]
        t = torch.from_numpy(_to_float_array(img)).unsqueeze(0)  # (1,H,W)
        t = (t - 0.5) / 0.5
        target = torch.tensor(self.encode(text), dtype=torch.long)
        return t, target, len(target), text


def _to_float_array(pil_img):
    import numpy as np
    return (np.asarray(pil_img, dtype="float32") / 255.0)


def collate_fn(batch):
    """Stack images; concatenate targets for CTC; keep raw texts for CER."""
    images, targets, lengths, texts = zip(*batch)
    images = torch.stack(images, 0)                          # (N,1,H,W)
    target_lengths = torch.tensor(lengths, dtype=torch.long)
    targets_cat = torch.cat(targets, 0) if len(targets) else torch.tensor([], dtype=torch.long)
    return images, targets_cat, target_lengths, list(texts)
