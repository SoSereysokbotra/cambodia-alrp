"""Recognition package — CRNN plate-text reader (Stage 2)."""
from .crnn_model import CRNN, CTCDecoder, CHARSET, BLANK, load_crnn

__all__ = ["CRNN", "CTCDecoder", "CHARSET", "BLANK", "load_crnn"]
