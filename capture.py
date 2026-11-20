"""Frame sources.

* :class:`CameraSource` - webcam capture in its own thread; only the latest frame is
  kept, so a slow consumer never builds up latency. Disconnects are detected and the
  camera is re-opened automatically.
* :class:`VideoSource` - deterministic replay of a video file using a *virtual*
  clock (frame index / video FPS), so state-machine timing is reproducible.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from config import CameraConfig
from detector import as_image

logger = logging.getLogger(__name__)

Image = NDArray[np.uint8]


@dataclass(frozen=True)
class FramePacket:
    """One tick of input: a frame (or ``None`` if the camera is unusable) and its time."""

    image: Image | None
    timestamp: float
    camera_ok: bool
    seq: int


class FrameSource(Protocol):
    """Producer of :class:`FramePacket` objects."""

    def read(self) -> FramePacket | None:
        """Return the next packet, or ``None`` when the stream has ended."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class CameraSource:
    """Threaded webcam reader that keeps only the newest frame."""

    _POLL_S = 0.25

    def __init__(self, cfg: CameraConfig, clock: Callable[[], float] = time.monotonic) -> None:
        self._cfg = cfg
        self._clock = clock
        self._cond = threading.Condition()
        self._image: Image | None = None
        self._timestamp = 0.0
        self._seq = 0
        self._ok = False
        self._delivered = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)
        self._thread.start()

    def read(self) -> FramePacket:
        """Wait up to 250 ms for a new frame; on timeout return an image-less (not ok) packet."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != self._delivered, timeout=self._POLL_S)
            if self._seq != self._delivered:
                self._delivered = self._seq
                image = self._image
                return FramePacket(image, self._timestamp, self._ok and image is not None, self._seq)
            return FramePacket(None, self._clock(), False, self._seq)

    def close(self) -> None:
        """Stop the capture thread and release the camera."""
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _publish(self, image: Image | None, ok: bool) -> None:
        with self._cond:
            self._image = image
            self._timestamp = self._clock()
            self._ok = ok
            self._seq += 1
            self._cond.notify_all()

    def _open(self) -> cv2.VideoCapture | None:
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self._cfg.index, backend)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self._cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logger.info(
            "Camera %d opened at %dx%d",
            self._cfg.index,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        return cap

    def _run(self) -> None:
        cap: cv2.VideoCapture | None = None
        failures = 0
        next_attempt = 0.0
        while not self._stop.is_set():
            if cap is None:
                now = self._clock()
                if now < next_attempt:
                    self._stop.wait(0.1)
                    continue
                cap = self._open()
                if cap is None:
                    logger.warning("Camera %d unavailable; retrying", self._cfg.index)
                    next_attempt = now + self._cfg.reconnect_interval_s
                    self._publish(None, False)
                    continue
                failures = 0
            ok, frame = cap.read()
            if ok and frame is not None:
                failures = 0
                self._publish(as_image(frame), True)
                continue
            failures += 1
            if failures >= self._cfg.read_failure_limit:
                logger.warning("Camera lost after %d failed reads; reconnecting", failures)
                cap.release()
                cap = None
                next_attempt = self._clock() + self._cfg.reconnect_interval_s
                self._publish(None, False)
            else:
                self._stop.wait(0.02)
        if cap is not None:
            cap.release()


class VideoSource:
    """Replay a video file with timestamps ``frame_index / fps`` (deterministic)."""

    def __init__(self, path: Path, realtime: bool = False) -> None:
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            self._cap.release()
            raise FileNotFoundError(f"Cannot open video file: {path}")
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._fps = fps if fps > 1.0 else 30.0
        self._realtime = realtime
        self._index = 0
        self._started = time.perf_counter()

    def read(self) -> FramePacket | None:
        """Return the next frame with its virtual timestamp, or ``None`` at the end."""
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        timestamp = self._index / self._fps
        self._index += 1
        if self._realtime:
            delay = self._started + timestamp - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        return FramePacket(as_image(frame), timestamp, True, self._index)

    def close(self) -> None:
        """Release the video file."""
        self._cap.release()
