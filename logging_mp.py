"""A very small multi-process safe logging shim used by the project.

This provides a minimal `get_logger(name)` and common level constants so
other modules can import `logging_mp` without pulling external deps.
"""
import logging
from typing import Optional


def basic_config(level: int = logging.INFO) -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=level)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


# Export level constants
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL
