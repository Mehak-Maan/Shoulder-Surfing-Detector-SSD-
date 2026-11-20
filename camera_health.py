"""Camera health: detects a covered/dark lens and a frozen stream (time based)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from config import HealthConfig
from detector import Image

_SIZE = (64, 36)


@dataclass(frozen=True)
class HealthStatus:
    """``ok`` is False once a problem has lasted long enough; ``reason`` names it."""

    ok: bool
    reason: str = ""


class CameraHealth:
    """Looks at tiny grayscale copies of the frames, so it costs well under a millisecond.

    * covered/dark: mean brightness and contrast both very low for ``covered_s``. A truly
      dark room looks the same as a covered lens and is treated the same (fail closed:
      no face could be detected there either).
    * frozen: consecutive frames are (almost) identical for ``frozen_s``. A live sensor
      always has noise; a hung driver repeats the same buffer.
    """

    def __init__(self, cfg: HealthConfig) -> None:
        self._cfg = cfg
        self._previous: NDArray[np.int16] | None = None
        self._dark_since: float | None = None
        self._frozen_since: float | None = None

    def reset(self) -> None:
        """Forget history (use after a reconnect or a pause)."""
        self._previous = None
        self._dark_since = None
        self._frozen_since = None

    def update(self, image: Image, now: float) -> HealthStatus:
        """Evaluate one frame captured at ``now`` (seconds)."""
        cfg = self._cfg
        if not cfg.enabled:
            return HealthStatus(True)
        gray = cv2.cvtColor(cv2.resize(image, _SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        small = gray.astype(np.int16)
        dark = float(gray.mean()) <= cfg.dark_luma_max and float(gray.std()) <= cfg.dark_std_max
        frozen = self._previous is not None and float(np.abs(small - self._previous).mean()) <= cfg.frozen_diff_max
        self._previous = small
        self._dark_since = (self._dark_since if self._dark_since is not None else now) if dark else None
        self._frozen_since = (self._frozen_since if self._frozen_since is not None else now) if frozen else None
        if self._dark_since is not None and now - self._dark_since >= cfg.covered_s:
            return HealthStatus(False, "covered_or_dark")
        if self._frozen_since is not None and now - self._frozen_since >= cfg.frozen_s:
            return HealthStatus(False, "frozen")
        return HealthStatus(True)
