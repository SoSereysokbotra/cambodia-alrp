"""
src/utils/logger.py
===================
Structured application logging for the ALPR system (SRS LOG-003).

get_logger() returns a process-wide logger that writes timestamped, levelled
records to both the console and a rotating file (logs/alpr.log). UTF-8 so Khmer
plate text logs cleanly on Windows (cp1252 consoles would otherwise crash).
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGER: logging.Logger | None = None
_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(log_dir: str | Path = "logs", name: str = "alpr",
               level: int = logging.INFO) -> logging.Logger:
    """Return the shared ALPR logger, configuring handlers once (idempotent)."""
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter(_FMT)

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_path / "alpr.log", maxBytes=2_000_000, backupCount=5,
            encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)          # console: warnings+ only (keep stdout clean)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    _LOGGER = logger
    return logger
