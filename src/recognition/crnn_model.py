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
class STN(nn.Module):
    """Spatial Transformer Network — the 'straightening layer' (Way 1).

    A small CNN looks at the crop and predicts an affine transform (which includes
    rotation), then warps the crop before the reader sees it. It is initialised to
    the IDENTITY transform, so it starts as a no-op and *learns* the correction
    purely from the CTC reading loss — i.e. the model teaches itself to turn a
    rotated/upside-down plate upright before reading. No angle labels needed.
    """

    def __init__(self, img_h: int = 64, img_w: int = 320) -> None:
        super().__init__()
        self.loc = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.fc = nn.Sequential(nn.Linear(64 * 4 * 8, 128), nn.ReLU(True),
                                nn.Linear(128, 6))
        # start as identity so the reader is unchanged before any training
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def theta(self, x: torch.Tensor) -> torch.Tensor:
        """The predicted affine matrix, (N,2,3).

        Exposed so training can SUPERVISE it directly. Left to the CTC loss alone
        the layer never learns to rotate — measured 2026-08-20, it sits at
        [[+1.09,0,0.05],[0,+1.05,-0.01]] (near-identity) for flipped and upright
        input alike, i.e. it straightens nothing. See --stn-supervise in
        scripts/recognition/finetune_crnn.py.
        """
        return self.fc(self.loc(x).flatten(1)).view(-1, 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = F.affine_grid(self.theta(x), x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)


class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network for sequence text recognition."""

    def __init__(self, img_h: int = 64, img_w: int = 320,
                 n_classes: int = N_CLASSES, n_hidden: int = 256,
                 use_stn: bool = False) -> None:
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.n_classes = n_classes
        # Way 1: optional learnable straightening layer in front of the reader.
        self.use_stn = use_stn
        self.stn = STN(img_h, img_w) if use_stn else None

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
        if self.stn is not None:                # Way 1: straighten first
            x = self.stn(x)
        conv = self.cnn(x)                      # (b, c, h, w)
        b, c, h, w = conv.size()
        if h != 1:
            # Safety: collapse any residual height to 1 (keeps this robust to
            # img_h=64 or 32). Averaging the vertical band via mean(dim=2) is
            # mathematically identical to adaptive_avg_pool2d(., (1, w)) but is
            # ONNX-exportable (adaptive pooling with a dynamic size is not).
            conv = conv.mean(dim=2, keepdim=True)
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
    # weights_only=True is safe here (we save a pure state_dict) and silences
    # the torch pickle warning.
    try:
        state = torch.load(str(weights_path), map_location=device, weights_only=True)
    except Exception:
        state = torch.load(str(weights_path), map_location=device)
    # accept either a raw state_dict or a checkpoint dict
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    # Auto-detect a straightening layer (Way 1): if the weights include STN params,
    # build the model WITH the STN so the shapes match — no flag needed anywhere.
    use_stn = any(k.startswith("stn.") for k in state.keys())
    model = CRNN(img_h=img_h, img_w=img_w,
                 n_classes=len(charset) + 1, n_hidden=n_hidden, use_stn=use_stn)
    model.load_state_dict(state)
    model.to(device).eval()
    return model
