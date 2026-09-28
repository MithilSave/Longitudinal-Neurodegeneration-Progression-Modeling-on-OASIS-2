"""Logging setup for the oasis2 package."""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str = "oasis2",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configure and return a named logger that writes to stdout (and optionally a file)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s  %(name)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
