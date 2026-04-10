"""
Centralized logging configuration using loguru.

Usage:
    from utils.logger import get_logger
    logger = get_logger("my_module")
    logger.info("Training started")
    logger.warning("Low GPU memory")
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _loguru_logger

from utils.paths import LOGS_DIR

# Remove loguru's default stderr handler so we control output
_loguru_logger.remove()

# ── Format ──────────────────────────────────────────────────
_FMT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[module]}</cyan> | "
    "{message}"
)

# ── Console handler (always active) ────────────────────────
_loguru_logger.add(
    sys.stderr,
    format=_FMT,
    level="INFO",
    colorize=True,
    filter=lambda record: "module" in record["extra"],
)

# ── File handler (rotated, always active) ───────────────────
LOGS_DIR.mkdir(parents=True, exist_ok=True)
_loguru_logger.add(
    str(LOGS_DIR / "cv_projects.log"),
    format=_FMT,
    level="DEBUG",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
    filter=lambda record: "module" in record["extra"],
)


def get_logger(module_name: str = "root") -> _loguru_logger.__class__:
    """Return a logger bound to *module_name*.

    Parameters
    ----------
    module_name : str
        A short label identifying the calling module (e.g. ``"train"``,
        ``"dataloader"``).  Appears in every log line.

    Returns
    -------
    loguru.Logger
        Bound logger instance.
    """
    return _loguru_logger.bind(module=module_name)


if __name__ == "__main__":
    log = get_logger("logger_test")
    log.debug("This is a DEBUG message (file only)")
    log.info("This is an INFO message")
    log.warning("This is a WARNING message")
    log.error("This is an ERROR message")
    print(f"\nLog file: {LOGS_DIR / 'cv_projects.log'}")
