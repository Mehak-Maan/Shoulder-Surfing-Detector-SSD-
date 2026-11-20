"""Mode B: headless service. Low-rate processing, tray control, graceful shutdown."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from capture import FrameSource
from config import RuntimeConfig
from pipeline import Pipeline
from security_actions import ResponseHandler
from state_machine import State
from tray import Tray

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_ERRORS = 10


class HeadlessRunner:
    """Runs the pipeline at ``runtime.headless_fps`` without any window.

    * ``paced`` frames: sleep between frames; the camera thread keeps only the newest frame,
      so the 8 FPS loop always sees a fresh image and never a backlog.
    * Pause (tray): detection and reactions stop, the overlay is removed and the state
      machine is reset; the camera stays open.
    * Errors: an exception in one iteration is logged and the loop continues; after
      ``MAX_CONSECUTIVE_ERRORS`` in a row the service stops with exit code 1 instead of
      pretending to protect the screen.
    """

    def __init__(
        self,
        cfg: RuntimeConfig,
        source: FrameSource,
        pipeline: Pipeline,
        handler: ResponseHandler,
        tray: Tray,
        stop: threading.Event | None = None,
        paced: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = cfg
        self._source = source
        self._pipeline = pipeline
        self._handler = handler
        self._tray = tray
        self._stop = stop if stop is not None else threading.Event()
        self._paused = threading.Event()
        self._paced = paced
        self._clock = clock

    @property
    def paused(self) -> bool:
        """True while detection is paused."""
        return self._paused.is_set()

    def toggle_pause(self) -> None:
        """Pause or resume detection (called from the tray thread)."""
        if self._paused.is_set():
            self._paused.clear()
            logger.info("Resumed")
        else:
            self._paused.set()
            logger.info("Paused")

    def request_stop(self) -> None:
        """Ask the loop to finish (signal handlers and the tray call this)."""
        self._stop.set()

    def run(self) -> int:
        """Run until stopped, the source ends, or too many consecutive errors; return exit code."""
        interval = 1.0 / self._cfg.headless_fps
        next_tick = self._clock()
        was_paused = False
        last_state: State | None = None
        errors = 0
        self._tray.start(self.toggle_pause, self.request_stop)
        logger.info("Headless service started at %.1f FPS", self._cfg.headless_fps)
        try:
            while not self._stop.is_set():
                if self._paced:
                    wait = next_tick - self._clock()
                    if wait > 0 and self._stop.wait(wait):
                        break
                    next_tick = max(next_tick + interval, self._clock())
                packet = self._source.read()
                if packet is None:
                    logger.info("Source ended")
                    break
                try:
                    paused = self._paused.is_set()
                    if paused != was_paused:
                        self._handler.reset()
                        self._pipeline.reset(packet.timestamp)
                        was_paused = paused
                        last_state = State.SECURE
                        self._tray.set_status(State.SECURE, paused)
                    if paused:
                        continue
                    result = self._pipeline.process(packet)
                    self._handler(packet, result)
                    if result.snapshot.state is not last_state:
                        last_state = result.snapshot.state
                        self._tray.set_status(last_state, False)
                    errors = 0
                except Exception:  # noqa: BLE001 - keep the service alive, but never silently
                    errors += 1
                    logger.exception("Processing error (%d in a row)", errors)
                    if errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.critical("Too many consecutive errors; stopping the service")
                        return 1
        finally:
            self._handler.reset()
            self._tray.stop()
            logger.info("Headless service stopped")
        return 0
