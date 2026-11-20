"""Owner enrollment: collect face embeddings from the live camera (no image is ever kept)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from capture import FrameSource
from config import AppConfig, DetectorConfig, IdentityConfig
from detector import FaceBackend, FaceFinder, Image, RawFace
from gaze import estimate_yaw_deg
from identity import FaceEmbedder, OwnerProfile, ProfileError, Vector, align_face

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnrollmentProgress:
    """What the enrollment loop tells the user after each frame."""

    collected: int
    target: int
    message: str
    done: bool


class EnrollmentSession:
    """Accepts a sample only when exactly one big, frontal, confident face is in view.

    Samples are spaced ``enroll_interval_s`` apart so slow head movements give varied views.
    Each accepted frame is embedded immediately and then dropped.
    """

    def __init__(self, cfg: IdentityConfig, det_cfg: DetectorConfig, embedder: FaceEmbedder) -> None:
        self._cfg = cfg
        self._det = det_cfg
        self._embedder = embedder
        self._vectors: list[Vector] = []
        self._last_sample = -1e9

    @property
    def collected(self) -> int:
        """Number of embeddings gathered so far."""
        return len(self._vectors)

    def feed(self, frame: Image, faces: Sequence[RawFace], now: float) -> EnrollmentProgress:
        """Consider one frame with its detected ``faces``."""
        cfg = self._cfg
        done = len(self._vectors) >= cfg.enroll_samples

        def progress(message: str) -> EnrollmentProgress:
            return EnrollmentProgress(len(self._vectors), cfg.enroll_samples, message, done)

        if done:
            return progress("Enough samples collected")
        if not faces:
            return progress("No face found - look at the screen")
        if len(faces) > 1:
            return progress("More than one face in view - only the owner should be visible")
        face = faces[0]
        face_px = face.box.w * frame.shape[1]
        if face_px < cfg.enroll_min_face_px:
            return progress("Move closer to the camera")
        yaw = estimate_yaw_deg(face.keypoints, face_px, self._det)
        if face.keypoints is None or yaw is None or yaw > cfg.enroll_max_yaw_deg:
            return progress("Face the screen")
        if now - self._last_sample < cfg.enroll_interval_s:
            return progress("Hold on - slowly turn your head a little between samples")
        patch = align_face(frame, face.box, face.keypoints, self._embedder.spec)
        if patch is None:
            return progress("Face not usable, move a little")
        self._vectors.append(self._embedder.embed(patch))
        self._last_sample = now
        done = len(self._vectors) >= cfg.enroll_samples
        return progress("Sample accepted" if not done else "Enough samples collected")

    def build_profile(self) -> OwnerProfile:
        """Create the profile from the collected samples."""
        return OwnerProfile.build(self._vectors, self._embedder.name, self._cfg)


def run_enrollment(
    cfg: AppConfig,
    source: FrameSource,
    backend: FaceBackend,
    embedder: FaceEmbedder,
    report: Callable[[str], None],
    clock: Callable[[], float] = time.monotonic,
) -> OwnerProfile:
    """Run enrollment on ``source`` until enough samples exist or ``enroll_timeout_s`` passes."""
    finder = FaceFinder(cfg.detector, backend)
    session = EnrollmentSession(cfg.identity, cfg.detector, embedder)
    started = clock()
    last_message = ""
    try:
        while clock() - started < cfg.identity.enroll_timeout_s:
            packet = source.read()
            if packet is None:
                break
            if packet.image is None:
                message = "Camera not available"
            else:
                faces = finder.find(packet.image, packet.timestamp).faces
                progress = session.feed(packet.image, faces, packet.timestamp)
                message = f"[{progress.collected}/{progress.target}] {progress.message}"
            if message != last_message:
                report(message)
                last_message = message
            if packet.image is not None and session.collected >= cfg.identity.enroll_samples:
                return session.build_profile()
    finally:
        finder.close()
    raise ProfileError(f"Enrollment incomplete: {session.collected}/{cfg.identity.enroll_samples} samples")
