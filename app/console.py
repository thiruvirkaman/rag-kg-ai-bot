"""Console helpers shared across the app."""
from __future__ import annotations

import logging
import sys
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

from .config import get_settings

_console: Console | None = None


def console() -> Console:
    global _console
    if _console is None:
        _console = Console(stderr=True, highlight=False)
    return _console


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    settings = get_settings()
    handler = RichHandler(console=console(), show_time=True, show_path=False)
    handler.setLevel(settings.log_level)
    logger.setLevel(settings.log_level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def out(msg: Any = "") -> None:
    console().print(msg)


def die(msg: Any, code: int = 1) -> None:
    console().print(f"[red]ERROR:[/red] {msg}")
    sys.exit(code)
