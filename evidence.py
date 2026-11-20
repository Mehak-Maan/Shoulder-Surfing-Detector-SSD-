"""Optional encrypted evidence snapshots (opt-in, local only).

A JPEG of the frame that triggered ALERT/LOCKED is encrypted with Fernet (AES-128-CBC +
HMAC) using a key stored next to it. Nothing is ever written unless
``privacy.evidence_enabled`` is true.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2

from config import PrivacyConfig
from detector import Image

logger = logging.getLogger(__name__)

SUFFIX = ".jpg.enc"


def get_fernet(key_file: Path, create: bool) -> Any:
    """Load the Fernet key (creating it with mode 0600 when ``create``) and return the cipher."""
    from cryptography.fernet import Fernet

    if not key_file.exists():
        if not create:
            raise FileNotFoundError(f"Key file not found: {key_file}")
        key_file.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(Fernet.generate_key())
        logger.info("Created evidence key %s", key_file)
    return Fernet(key_file.read_bytes().strip())


class EvidenceStore:
    """Writes encrypted snapshots and keeps at most ``evidence_max_files`` of them."""

    def __init__(self, cfg: PrivacyConfig, clock: Callable[[], float] = time.time) -> None:
        self._cfg = cfg
        self._dir = Path(cfg.evidence_dir).expanduser()
        self._key_file = Path(cfg.key_file).expanduser()
        self._clock = clock
        self._fernet = get_fernet(self._key_file, create=True)

    def save(self, image: Image, label: str) -> Path | None:
        """Encrypt and store ``image``; return the file path or ``None`` on failure."""
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self._cfg.jpeg_quality])
        if not ok:
            logger.error("Evidence JPEG encoding failed")
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self._clock()))
        safe = "".join(c for c in label if c.isalnum() or c in "-_") or "event"
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            path = self._dir / f"{stamp}-{safe}{SUFFIX}"
            path.write_bytes(self._fernet.encrypt(jpeg.tobytes()))
            self._prune()
        except OSError as exc:
            logger.error("Could not write evidence: %s", exc)
            return None
        logger.warning("Encrypted evidence snapshot saved: %s", path)
        return path

    def _prune(self) -> None:
        files = sorted(self._dir.glob(f"*{SUFFIX}"), key=lambda p: p.stat().st_mtime)
        for old in files[: max(0, len(files) - self._cfg.evidence_max_files)]:
            old.unlink(missing_ok=True)


def decrypt_file(key_file: Path, encrypted: Path) -> bytes:
    """Return the JPEG bytes stored in ``encrypted`` (raises if the key is wrong)."""
    fernet = get_fernet(key_file.expanduser(), create=False)
    return bytes(fernet.decrypt(encrypted.read_bytes()))
