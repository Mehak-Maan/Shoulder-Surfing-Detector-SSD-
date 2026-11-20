"""Process-level plumbing for the headless service: logging, single-instance guard, signals."""

from __future__ import annotations

import logging
import logging.handlers
import os
import signal
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any

from config import RuntimeConfig

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(cfg: RuntimeConfig, level: str, console: bool) -> Path | None:
    """Configure root logging: rotating file (if writable) and optionally the console.

    Returns the log file path, or ``None`` if the file could not be opened.
    """
    handlers: list[logging.Handler] = []
    log_path: Path | None = Path(cfg.log_file).expanduser()
    try:
        assert log_path is not None
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                log_path, maxBytes=cfg.log_max_bytes, backupCount=cfg.log_backups, encoding="utf-8"
            )
        )
    except OSError as exc:
        log_path = None
        console = True
        sys.stderr.write(f"Cannot open log file {cfg.log_file}: {exc}; logging to the console only\n")
    if console:
        handlers.append(logging.StreamHandler())
    for handler in handlers:
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.basicConfig(level=level, handlers=handlers, force=True)
    return log_path


class SingleInstance:
    """Cross-platform exclusive lock on a file; a second process cannot acquire it."""

    def __init__(self, path: Path) -> None:
        self._path = path.expanduser()
        self._handle: Any = None

    def acquire(self) -> bool:
        """Try to take the lock; False if another instance already holds it."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self._path, "a+")  # noqa: SIM115 - kept open for the life of the lock
        try:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        self._handle = handle
        return True

    def release(self) -> None:
        """Release the lock (also happens automatically when the process exits)."""
        if self._handle is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        self._handle.close()
        self._handle = None


def install_signal_handlers(stop: threading.Event) -> Callable[[], None]:
    """Make SIGINT/SIGTERM (and SIGBREAK on Windows) set ``stop``; return a restore function."""
    previous: dict[int, Any] = {}

    def handler(signum: int, _frame: FrameType | None) -> None:
        logger.info("Received signal %d; shutting down", signum)
        stop.set()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            previous[number] = signal.signal(number, handler)
        except (ValueError, OSError):  # not the main thread, or unsupported on this platform
            continue

    def restore() -> None:
        for number, old in previous.items():
            signal.signal(number, old)

    return restore
