"""Structured logging utility for NLP Projects workspace.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started", extra={"epoch": 1, "lr": 2e-5})
"""

import logging
import sys
from pathlib import Path


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Return a configured logger with rich-style console output.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__``).
    level : int
        Logging level (default: INFO).
    log_file : str | Path | None
        Optional path to a log file.  If provided the logger writes there too.
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers twice when the module is re-imported
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Optional file handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
