"""
src/recognition/crnn_model.py
=============================
CRNN (CNN + BiLSTM + CTC) for reading the plate NUMBER from a crop.

Design note
-----------
This CRNN reads the alphanumeric plate number (e.g. "1AB-2345"), NOT the Khmer
province name. Rationale: the number is the identifying part, and it renders
reliably for synthetic training (Khmer needs complex-script shaping that basic
image libs get wrong). The charset is therefore digits + Latin + separators.
The province can be added later via Plate_v4's 29-class detector or a wider
charset once properly-shaped Khmer training data exists.

Character set
-------------
    index 0 .. N-1 : the visible characters (CHARSET)
    index N        : the CTC 'blank' token (BLANK)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Character set
# --------------------------------------------------------------------------- #
DIGITS = "0123456789"
LATIN = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SEPARATORS = "- "                 # dash and space
CHARSET = DIGITS + LATIN + SEPARATORS
BLANK = len(CHARSET)              # CTC blank = last index
N_CLASSES = len(CHARSET) + 1      # visible chars + blank

CHAR_TO_IDX = {c: i for i, c in enumerate(CHARSET)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARSET)}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network for sequence text recognition."""

    def __init__(self, img_h: int = 64, img_w: int = 320,
                 n_classes: int = N_CLASSES, n_hidden: int = 256) -> None:
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.n_classes = n_classes

        # CNN backbone. Asymmetric pooling preserves width (the "time" axis)
        # while shrinking height toward 1.
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                                    # H/2  W/2
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                                    # H/4  W/4
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),                  # H/8  W/4+1
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),                  # H/16 W/4+2
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),
        )

        # Two stacked BiLSTM layers.
        self.rnn = nn.LSTM(512, n_hidden, num_layers=2,
                           bidirectional=True, batch_first=False)
        self.fc = nn.Linear(n_hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, 1, img_h, img_w)
        returns log-probs of shape (seq_len, batch, n_classes) for CTC.
        """
        conv = self.cnn(x)                      # (b, c, h, w)
        b, c, h, w = conv.size()
        if h != 1:
            # Safety: collapse any residual height to 1 (keeps this robust to
            # img_h=64 or 32). Averages the remaining vertical band.
            conv = F.adaptive_avg_pool2d(conv, (1, w))
        conv = conv.squeeze(2)                  # (b, c, w)
        conv = conv.permute(2, 0, 1)            # (w, b, c) = (seq, batch, feat)

        rnn_out, _ = self.rnn(conv)             # (seq, batch, 2*hidden)
        logits = self.fc(rnn_out)               # (seq, batch, n_classes)
        return F.log_softmax(logits, dim=2)


# --------------------------------------------------------------------------- #
# Greedy CTC decoder
# --------------------------------------------------------------------------- #
class CTCDecoder:
    """Greedy decode: argmax -> collapse repeats -> drop blanks."""

    def __init__(self, charset: str = CHARSET, blank: int = BLANK) -> None:
        self.charset = charset
        self.blank = blank

    def decode(self, log_probs: torch.Tensor):
        """
        log_probs : (seq, batch, n_classes) -> list[str]
                    (seq, n_classes)        -> str
        """
        if log_probs.dim() == 3:
            best = log_probs.argmax(2)          # (seq, batch)
            best = best.permute(1, 0)           # (batch, seq)
            return [self._collapse(row) for row in best]
        best = log_probs.argmax(1)              # (seq,)
        return self._collapse(best)

    def _collapse(self, seq: torch.Tensor) -> str:
        out, prev = [], -1
        for idx in seq.tolist():
            if idx != self.blank and idx != prev and 0 <= idx < len(self.charset):
                out.append(self.charset[idx])
            prev = idx
        return "".join(out)


# --------------------------------------------------------------------------- #
# Loading helper
# --------------------------------------------------------------------------- #
def load_crnn(weights_path: str | Path, device: str = "cpu",
              charset: str = CHARSET, img_h: int = 64, img_w: int = 320,
              n_hidden: int = 256) -> CRNN:
    """Instantiate a CRNN and load trained weights (eval mode)."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"CRNN weights not found: {weights_path}")
    model = CRNN(img_h=img_h, img_w=img_w,
                 n_classes=len(charset) + 1, n_hidden=n_hidden)
    # weights_only=True is safe here (we save a pure state_dict) and silences
    # the torch pickle warning.
    try:
        state = torch.load(str(weights_path), map_location=device, weights_only=True)
    except Exception:
        state = torch.load(str(weights_path), map_location=device)
    # accept either a raw state_dict or a checkpoint dict
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model
