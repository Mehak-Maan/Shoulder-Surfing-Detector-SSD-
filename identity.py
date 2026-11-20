"""Owner identity: face alignment, embeddings, the encrypted owner profile and live verification.

Privacy: only embedding vectors are kept. Enrollment frames are embedded and dropped
immediately; the profile file is encrypted with the same local key as the evidence snapshots.
An embedding is not an image, but it is still biometric data: treat the profile like a password.

Quality: the built-in ``lbp`` embedder is a classic texture descriptor. It is cheap and needs
no model file, but it is far weaker than a trained face-recognition network and is sensitive to
lighting. For serious use point ``identity.embedder: onnx`` at a face-embedding model.
"""

from __future__ import annotations

import io
import logging
import math
import os
from collections import deque
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from config import IdentityConfig
from detector import Box, IdentityStatus, Image, Keypoints, Track, as_image
from evidence import get_fernet

logger = logging.getLogger(__name__)

Vector = NDArray[np.float32]
PROFILE_VERSION = 1


class ProfileError(Exception):
    """The owner profile is missing, unreadable or made by a different embedder."""


# ------------------------------------------------------------------ alignment
@dataclass(frozen=True)
class AlignSpec:
    """Where the eyes go in the aligned square patch (fractions of its size)."""

    size: int
    eye_y: float
    eye_span: float


def align_face(frame: Image, box: Box, keypoints: Keypoints | None, spec: AlignSpec) -> Image | None:
    """Return an upright ``spec.size`` square patch of the face, or ``None`` if it is too small.

    With keypoints the eyes are rotated level and scaled to a fixed distance, which removes most
    of the head-roll and distance variation. Without them the padded box is just resized.
    """
    height, width = frame.shape[:2]
    pad = 0.5 * box.w * width
    x0, y0 = max(0, int(box.x1 * width - pad)), max(0, int(box.y1 * height - pad))
    x1, y1 = min(width, int(box.x2 * width + pad)), min(height, int(box.y2 * height + pad))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    roi = frame[y0:y1, x0:x1]
    size = spec.size
    if keypoints is None or len(keypoints) < 2:
        crop = frame[
            max(0, int(box.y1 * height)) : min(height, int(box.y2 * height)),
            max(0, int(box.x1 * width)) : min(width, int(box.x2 * width)),
        ]
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            return None
        return as_image(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA))
    left, right = sorted(keypoints[:2], key=lambda p: p[0])
    lx, ly = left[0] * width - x0, left[1] * height - y0
    rx, ry = right[0] * width - x0, right[1] * height - y0
    eye_dist = math.hypot(rx - lx, ry - ly)
    if eye_dist < 4.0:
        return None
    scale = spec.eye_span * size / eye_dist
    if scale < 1.0:  # anti-alias before shrinking so near and far faces look alike
        sigma = 0.6 / scale
        roi = as_image(cv2.GaussianBlur(roi, (0, 0), sigma))
    angle = math.degrees(math.atan2(ry - ly, rx - lx))
    center = ((lx + rx) / 2.0, (ly + ry) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += size / 2.0 - center[0]
    matrix[1, 2] += spec.eye_y * size - center[1]
    return as_image(cv2.warpAffine(roi, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))


# ------------------------------------------------------------------ embedders
class FaceEmbedder(Protocol):
    """Turns an aligned face patch into a unit-length vector."""

    name: str
    spec: AlignSpec

    def embed(self, face: Image) -> Vector:
        """Return an L2-normalised embedding of ``face``."""
        ...


def _unit(vector: NDArray[Any]) -> Vector:
    norm = float(np.linalg.norm(vector))
    out = np.asarray(vector, dtype=np.float32)
    return out if norm < 1e-9 else np.asarray(out / norm, dtype=np.float32)


def _uniform_lut() -> NDArray[np.uint8]:
    """Map the 256 LBP codes to 59 'uniform' classes (<= 2 circular bit transitions) plus one catch-all."""
    lut = np.zeros(256, dtype=np.uint8)
    next_index = 0
    for code in range(256):
        bits = [(code >> i) & 1 for i in range(8)]
        transitions = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
        if transitions <= 2:
            lut[code] = next_index
            next_index += 1
        else:
            lut[code] = 58
    return lut


_LUT = _uniform_lut()
_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))


class LbpEmbedder:
    """4x4 grid of uniform-LBP histograms on an equalised 64x64 patch (Hellinger-normalised)."""

    name = "lbp-grid4x4-v1"
    spec = AlignSpec(size=64, eye_y=0.38, eye_span=0.40)

    def embed(self, face: Image) -> Vector:
        """Return the 944-dimensional descriptor of ``face``."""
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if face.ndim == 3 else face
        gray = cv2.equalizeHist(cv2.resize(gray, (self.spec.size, self.spec.size), interpolation=cv2.INTER_AREA))
        g = gray.astype(np.int16)
        h, w = g.shape
        centre = g[1:-1, 1:-1]
        code = np.zeros(centre.shape, dtype=np.uint8)
        for bit, (dy, dx) in enumerate(_OFFSETS):
            neighbour = g[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx]
            code |= (neighbour >= centre).astype(np.uint8) << bit
        mapped = _LUT[code][:60, :60]
        cells = mapped.reshape(4, 15, 4, 15).swapaxes(1, 2).reshape(16, 225)
        hist = np.stack([np.bincount(cell, minlength=59) for cell in cells]).astype(np.float32)
        hist /= hist.sum(axis=1, keepdims=True)
        return _unit(np.sqrt(hist).ravel())


class OnnxEmbedder:
    """Any ArcFace/MobileFaceNet style ONNX model run through OpenCV's DNN module (CPU).

    Expects a 112x112 RGB input scaled to [-1, 1] and a single embedding output. Not exercised
    by the automated tests (no model file is shipped); verify it with ``--enroll`` and the GUI
    similarity read-out before relying on it.
    """

    spec = AlignSpec(size=112, eye_y=0.46, eye_span=0.315)

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Face embedding model not found: {model_path}")
        self.name = f"onnx-{model_path.name}"
        self._net: Any = cv2.dnn.readNetFromONNX(str(model_path))

    def embed(self, face: Image) -> Vector:
        """Return the model's embedding of ``face``."""
        blob = cv2.dnn.blobFromImage(face, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        self._net.setInput(blob)
        return _unit(np.asarray(self._net.forward()).ravel())


def create_embedder(cfg: IdentityConfig, base_dir: Path) -> FaceEmbedder:
    """Build the embedder selected in the configuration."""
    if cfg.embedder == "onnx":
        path = Path(cfg.onnx_model).expanduser()
        return OnnxEmbedder(path if path.is_absolute() else base_dir / path)
    return LbpEmbedder()


# ------------------------------------------------------------------ profile
@dataclass(frozen=True)
class OwnerProfile:
    """Enrolled embeddings plus the decision threshold calibrated on them."""

    embedder: str
    vectors: NDArray[np.float32]
    threshold: float
    top_k: int

    def score(self, vector: Vector) -> float:
        """Mean of the ``top_k`` highest cosine similarities to the enrolled vectors."""
        sims = np.sort(self.vectors @ vector)
        return float(sims[-min(self.top_k, len(sims)) :].mean())

    @staticmethod
    def build(vectors: Sequence[Vector], embedder: str, cfg: IdentityConfig) -> OwnerProfile:
        """Create a profile; the threshold is ``clip(mean leave-one-out similarity - margin)``."""
        if len(vectors) < 5:
            raise ProfileError("At least 5 samples are needed to build a profile")
        matrix = np.stack([_unit(v) for v in vectors]).astype(np.float32)
        loo: list[float] = []
        for i in range(len(matrix)):
            others = np.delete(matrix, i, axis=0)
            sims = np.sort(others @ matrix[i])
            loo.append(float(sims[-min(cfg.top_k, len(sims)) :].mean()))
        raw = float(np.mean(loo)) - cfg.threshold_margin
        threshold = min(cfg.threshold_ceiling, max(cfg.threshold_floor, raw))
        return OwnerProfile(embedder, matrix, threshold, cfg.top_k)

    @property
    def loo_hint(self) -> str:
        """Human-readable summary for the enrollment output."""
        return f"{len(self.vectors)} samples, threshold {self.threshold:.3f}"


def save_profile(profile: OwnerProfile, path: Path, key_file: Path) -> None:
    """Write the profile encrypted with the local key (file mode 0600)."""
    buffer = io.BytesIO()
    np.savez(
        buffer,
        version=np.array(PROFILE_VERSION),
        embedder=np.array(profile.embedder),
        vectors=profile.vectors,
        threshold=np.array(profile.threshold),
        top_k=np.array(profile.top_k),
    )
    token = get_fernet(key_file.expanduser(), create=True).encrypt(buffer.getvalue())
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(token)


def load_profile(path: Path, key_file: Path, embedder: str) -> OwnerProfile:
    """Read and decrypt the profile; ``ProfileError`` explains what is wrong."""
    path = path.expanduser()
    if not path.exists():
        raise ProfileError(f"No owner profile at {path}. Run: python main.py --enroll")
    try:
        raw = get_fernet(key_file.expanduser(), create=False).decrypt(path.read_bytes())
        data = np.load(io.BytesIO(raw), allow_pickle=False)
        if int(data["version"]) != PROFILE_VERSION:
            raise ProfileError("Owner profile has an unsupported version; run --enroll again")
        stored = str(data["embedder"])
        profile = OwnerProfile(stored, data["vectors"].astype(np.float32), float(data["threshold"]), int(data["top_k"]))
    except ProfileError:
        raise
    except Exception as exc:  # noqa: BLE001 - wrong key, corrupt file, missing key: all mean "cannot use this profile"
        raise ProfileError(f"Cannot read owner profile {path}: {exc}") from exc
    if profile.embedder != embedder:
        raise ProfileError(f"Profile was made with '{profile.embedder}' but '{embedder}' is configured; run --enroll")
    return profile


# ------------------------------------------------------------------ live verification
@dataclass
class _TrackIdentity:
    sims: deque[float] = field(default_factory=deque)
    last_check: float = -math.inf
    failures: int = 0
    is_owner: bool = False
    score: float | None = None


class OwnerVerifier:
    """Compares tracked faces with the enrolled owner, cheaply and with a per-frame budget.

    A track is verified a few times (``min_samples``), then re-checked every ``verify_period_s``.
    Decisions use the median of the last ``window`` similarities, and an accepted owner has to
    drop ``stay_margin`` below the threshold to be rejected (hysteresis).
    """

    def __init__(self, cfg: IdentityConfig, profile: OwnerProfile, embedder: FaceEmbedder) -> None:
        self._cfg = cfg
        self._profile = profile
        self._embedder = embedder
        self._tracks: dict[int, _TrackIdentity] = {}

    def reset(self) -> None:
        """Forget all per-track decisions."""
        self._tracks.clear()

    def update(self, tracks: Sequence[Track], frame: Image | None, now: float) -> None:
        """Embed the faces that are due (largest first, at most ``max_embeds_per_frame``)."""
        self._prune({t.id for t in tracks})
        if frame is None:
            return
        cfg = self._cfg
        width = frame.shape[1]
        budget = cfg.max_embeds_per_frame
        for track in sorted(tracks, key=lambda t: t.box.area, reverse=True):
            if budget <= 0:
                break
            if track.box.w * width < cfg.min_embed_px:
                continue
            state = self._tracks.setdefault(track.id, _TrackIdentity(deque(maxlen=cfg.window)))
            period = cfg.verify_period_s if len(state.sims) >= cfg.min_samples else cfg.verify_period_s / 4.0
            if now - state.last_check < period:
                continue
            state.last_check = now
            budget -= 1
            patch = align_face(frame, track.box, track.keypoints, self._embedder.spec)
            if patch is None:
                state.failures += 1
                continue
            state.sims.append(self._profile.score(self._embedder.embed(patch)))
            if len(state.sims) >= cfg.min_samples:
                state.score = float(np.median(state.sims))
                limit = self._profile.threshold - (cfg.stay_margin if state.is_owner else 0.0)
                state.is_owner = state.score >= limit

    def status(self, track_id: int, face_px: float) -> IdentityStatus:
        """Verification state of a track."""
        cfg = self._cfg
        if face_px < cfg.min_embed_px:
            return IdentityStatus.UNVERIFIABLE
        state = self._tracks.get(track_id)
        if state is None or (len(state.sims) < cfg.min_samples and state.failures < cfg.min_samples):
            return IdentityStatus.PENDING
        if len(state.sims) < cfg.min_samples:
            return IdentityStatus.OTHER  # alignment keeps failing: do not stay "pending" (fail closed)
        return IdentityStatus.OWNER if state.is_owner else IdentityStatus.OTHER

    def score(self, track_id: int) -> float | None:
        """Latest similarity of the track to the owner (``None`` until verified)."""
        state = self._tracks.get(track_id)
        return None if state is None else state.score

    def _prune(self, alive: Collection[int]) -> None:
        for track_id in [t for t in self._tracks if t not in alive]:
            del self._tracks[track_id]
