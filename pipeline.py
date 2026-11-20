"""Glue: frame -> camera health -> faces -> tracks -> roles -> security state."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from camera_health import CameraHealth
from capture import FramePacket
from config import AppConfig
from detector import (
    DetectionPass,
    FaceBackend,
    FaceFinder,
    FrameAnalysis,
    IdentityVerifier,
    IntruderClassifier,
    IoUTracker,
)
from state_machine import SecurityStateMachine, StateSnapshot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    """Output of one :meth:`Pipeline.process` call."""

    timestamp: float
    analysis: FrameAnalysis
    snapshot: StateSnapshot
    detection: DetectionPass | None
    latency_ms: float
    camera_issue: str = ""


class Pipeline:
    """Owns the detector, tracker, classifier, health check and state machine (no global state)."""

    def __init__(self, cfg: AppConfig, backend: FaceBackend, verifier: IdentityVerifier | None = None) -> None:
        self._finder = FaceFinder(cfg.detector, backend)
        self._tracker = IoUTracker(cfg.tracker)
        self._classifier = IntruderClassifier(cfg.detector, cfg.tracker, cfg.owner, cfg.gaze, verifier)
        self._health = CameraHealth(cfg.health)
        self._machine = SecurityStateMachine(cfg.state)
        self._frame_width = 1

    @property
    def machine(self) -> SecurityStateMachine:
        """The security state machine (read-only use intended)."""
        return self._machine

    def reset(self, now: float) -> None:
        """Return to SECURE and forget faces, owner and camera history (used when pausing)."""
        self._tracker.reset()
        self._classifier.reset()
        self._health.reset()
        self._machine.reset(now)

    def process(self, packet: FramePacket) -> PipelineResult:
        """Analyse one packet. An unusable camera (missing, covered, frozen) counts as a fault."""
        started = time.perf_counter()
        now = packet.timestamp
        detection: DetectionPass | None = None
        issue = ""
        if packet.image is None or not packet.camera_ok:
            issue = "disconnected"
        else:
            status = self._health.update(packet.image, now)
            if not status.ok:
                issue = status.reason
        if issue or packet.image is None:
            self._tracker.update([], now)
            analysis = FrameAnalysis()
            snapshot = self._machine.update(now, intruder_present=False, camera_ok=False)
        else:
            self._frame_width = packet.image.shape[1]
            detection = self._finder.find(packet.image, now)
            tracks = self._tracker.update(detection.faces, now)
            analysis = self._classifier.classify(tracks, now, self._frame_width, packet.image)
            snapshot = self._machine.update(now, analysis.intruder_present, camera_ok=True)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return PipelineResult(now, analysis, snapshot, detection, latency_ms, issue)

    def close(self) -> None:
        """Release the detector."""
        self._finder.close()
