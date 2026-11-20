"""Face detection (MediaPipe Tasks), IoU tracking and owner/intruder classification.

Coordinates everywhere in this module are *normalized* to the full camera
frame (0..1), so the logic is independent of capture resolution.

The MediaPipe legacy ``mp.solutions`` API is removed in current releases; this
module uses the supported Tasks API (``mediapipe.tasks.python.vision.FaceDetector``).
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from config import DetectorConfig, GazeConfig, OwnerConfig, TrackerConfig
from gaze import GazeFilter
from gaze import estimate_yaw_deg as estimate_yaw_deg

logger = logging.getLogger(__name__)

Image = NDArray[np.uint8]
Point = tuple[float, float]
Keypoints = tuple[Point, ...]


def as_image(array: NDArray[Any]) -> Image:
    """Narrow an OpenCV/NumPy result to the uint8 image type used throughout the project."""
    return cast(Image, array)


# --------------------------------------------------------------------------- geometry
@dataclass(frozen=True)
class Box:
    """Axis-aligned box in normalized frame coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def intersection(self, other: Box) -> float:
        """Return the overlap area with ``other``."""
        iw = min(self.x2, other.x2) - max(self.x1, other.x1)
        ih = min(self.y2, other.y2) - max(self.y1, other.y1)
        return max(0.0, iw) * max(0.0, ih)

    def iou(self, other: Box) -> float:
        """Return intersection over union with ``other``."""
        inter = self.intersection(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def overlap_of_smaller(self, other: Box) -> float:
        """Return intersection divided by the smaller box area (containment measure)."""
        smaller = min(self.area, other.area)
        return self.intersection(other) / smaller if smaller > 0 else 0.0

    def clipped(self) -> Box:
        """Return the box clipped to the unit square."""
        return Box(
            min(max(self.x1, 0.0), 1.0),
            min(max(self.y1, 0.0), 1.0),
            min(max(self.x2, 0.0), 1.0),
            min(max(self.y2, 0.0), 1.0),
        )

    def from_tile(self, tile: Box) -> Box:
        """Map a box that is normalized to ``tile`` into full-frame coordinates."""
        return Box(
            tile.x1 + self.x1 * tile.w,
            tile.y1 + self.y1 * tile.h,
            tile.x1 + self.x2 * tile.w,
            tile.y1 + self.y2 * tile.h,
        )

    def blend(self, other: Box, weight: float) -> Box:
        """Return ``weight * other + (1 - weight) * self`` (exponential smoothing)."""
        keep = 1.0 - weight
        return Box(
            keep * self.x1 + weight * other.x1,
            keep * self.y1 + weight * other.y1,
            keep * self.x2 + weight * other.x2,
            keep * self.y2 + weight * other.y2,
        )


@dataclass(frozen=True)
class RawFace:
    """One detection as returned by a backend (normalized to the image it saw)."""

    box: Box
    score: float
    keypoints: Keypoints | None = None

    def from_tile(self, tile: Box) -> RawFace:
        """Map this detection from tile coordinates into full-frame coordinates."""
        kps = (
            tuple((tile.x1 + x * tile.w, tile.y1 + y * tile.h) for x, y in self.keypoints)
            if self.keypoints is not None
            else None
        )
        return RawFace(self.box.from_tile(tile), self.score, kps)


def nms(faces: Sequence[RawFace], iou_threshold: float, containment: float = 0.7) -> list[RawFace]:
    """Greedy non-maximum suppression that also drops boxes mostly inside a better one."""
    kept: list[RawFace] = []
    for face in sorted(faces, key=lambda f: f.score, reverse=True):
        if all(
            face.box.iou(k.box) < iou_threshold and face.box.overlap_of_smaller(k.box) < containment
            for k in kept
        ):
            kept.append(face)
    return kept


def plan_tiles(
    roi: tuple[float, float, float, float], cols: int, rows: int, overlap: float
) -> list[Box]:
    """Split ``roi`` into a ``cols`` x ``rows`` grid of overlapping tiles (clipped to the ROI)."""
    x0, y0, x1, y1 = roi
    cell_w = (x1 - x0) / cols
    cell_h = (y1 - y0) / rows
    tiles: list[Box] = []
    for row in range(rows):
        for col in range(cols):
            cx = x0 + (col + 0.5) * cell_w
            cy = y0 + (row + 0.5) * cell_h
            half_w = cell_w * (1.0 + overlap) / 2.0
            half_h = cell_h * (1.0 + overlap) / 2.0
            tiles.append(
                Box(max(x0, cx - half_w), max(y0, cy - half_h), min(x1, cx + half_w), min(y1, cy + half_h))
            )
    return tiles


# --------------------------------------------------------------------------- backend
class FaceBackend(Protocol):
    """Anything that can find faces in an RGB image."""

    def detect(self, rgb: Image) -> list[RawFace]:
        """Return faces normalized to ``rgb``."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class ModelNotFoundError(FileNotFoundError):
    """Raised when the MediaPipe model file is missing."""


class MediaPipeBackend:
    """BlazeFace short-range detector through the MediaPipe Tasks API (CPU, IMAGE mode)."""

    def __init__(self, cfg: DetectorConfig) -> None:
        path = Path(cfg.model_path)
        if not path.is_file():
            raise ModelNotFoundError(
                f"Model file not found: {path}. Download blaze_face_short_range.tflite once "
                "(see README, 'Setup') and place it there; the app itself never uses the network."
            )
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=cfg.min_confidence,
            min_suppression_threshold=cfg.nms_iou,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def detect(self, rgb: Image) -> list[RawFace]:
        """Run the detector on an RGB uint8 image."""
        height, width = rgb.shape[:2]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._detector.detect(mp_image)
        faces: list[RawFace] = []
        for det in result.detections:
            bb = det.bounding_box
            box = Box(
                bb.origin_x / width,
                bb.origin_y / height,
                (bb.origin_x + bb.width) / width,
                (bb.origin_y + bb.height) / height,
            ).clipped()
            score = float(det.categories[0].score) if det.categories else 0.0
            kps = tuple((float(k.x), float(k.y)) for k in det.keypoints) if det.keypoints else None
            faces.append(RawFace(box, score, kps))
        return faces

    def close(self) -> None:
        """Release the underlying MediaPipe task."""
        self._detector.close()


# --------------------------------------------------------------------------- finder
@dataclass(frozen=True)
class DetectionPass:
    """Everything one call of :meth:`FaceFinder.find` produced."""

    faces: list[RawFace]
    tiles_run: tuple[Box, ...]
    all_tiles: tuple[Box, ...]


class FaceFinder:
    """Near-field pass on the whole frame plus scheduled far-field tile passes."""

    _EDGE_MARGIN = 0.02

    def __init__(self, cfg: DetectorConfig, backend: FaceBackend) -> None:
        self._cfg = cfg
        self._backend = backend
        self._tiles: list[Box] = (
            plan_tiles(cfg.far_field_roi, cfg.far_field_cols, cfg.far_field_rows, cfg.far_field_overlap)
            if cfg.far_field_enabled
            else []
        )
        self._last_run: list[float | None] = [None] * len(self._tiles)

    def find(self, frame_bgr: Image, now: float) -> DetectionPass:
        """Detect faces in a BGR frame captured at time ``now`` (seconds)."""
        height, width = frame_bgr.shape[:2]
        scale = min(1.0, self._cfg.process_width / width)
        small = (
            cv2.resize(frame_bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else frame_bgr
        )
        faces = list(self._backend.detect(as_image(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))))

        ran: list[Box] = []
        for index in self._due_tiles(now):
            tile = self._tiles[index]
            self._last_run[index] = now
            ran.append(tile)
            x1, y1 = int(tile.x1 * width), int(tile.y1 * height)
            x2, y2 = math.ceil(tile.x2 * width), math.ceil(tile.y2 * height)
            if x2 - x1 < 32 or y2 - y1 < 32:
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            for face in self._backend.detect(as_image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))):
                if not self._cut_by_tile(face.box, tile):
                    faces.append(face.from_tile(tile))
        return DetectionPass(nms(faces, self._cfg.nms_iou), tuple(ran), tuple(self._tiles))

    def _due_tiles(self, now: float) -> list[int]:
        period = self._cfg.far_field_period_s
        last_runs = self._last_run
        due = [
            (float("-inf") if last is None else last, i)
            for i, last in enumerate(last_runs)
            if last is None or now - last >= period
        ]
        due.sort()
        return [i for _, i in due[: self._cfg.far_field_max_tiles_per_frame]]

    def _cut_by_tile(self, local: Box, tile: Box) -> bool:
        """True if a tile-local box touches an interior tile edge (likely a partial face)."""
        m = self._EDGE_MARGIN
        return (
            (local.x1 < m and tile.x1 > 1e-6)
            or (local.x2 > 1 - m and tile.x2 < 1 - 1e-6)
            or (local.y1 < m and tile.y1 > 1e-6)
            or (local.y2 > 1 - m and tile.y2 < 1 - 1e-6)
        )

    def close(self) -> None:
        """Close the backend."""
        self._backend.close()


# --------------------------------------------------------------------------- tracker
@dataclass
class Track:
    """A face followed over time under a stable ID."""

    id: int
    box: Box
    score: float
    keypoints: Keypoints | None
    first_seen: float
    last_seen: float
    hits: int = 1
    visible: bool = True


class IoUTracker:
    """Greedy IoU matcher with a centroid fallback and time-based track expiry."""

    def __init__(self, cfg: TrackerConfig) -> None:
        self._cfg = cfg
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        """Forget all tracks (IDs keep increasing)."""
        self._tracks.clear()

    def update(self, faces: Sequence[RawFace], now: float) -> list[Track]:
        """Associate ``faces`` with tracks; return every live track (visible or coasting)."""
        cfg = self._cfg
        for tid in [t for t, tr in self._tracks.items() if now - tr.last_seen > cfg.ttl_s]:
            del self._tracks[tid]

        pairs: list[tuple[float, int, int]] = []
        for track in self._tracks.values():
            for di, face in enumerate(faces):
                affinity = self._affinity(track.box, face.box)
                if affinity > 0.0:
                    pairs.append((affinity, track.id, di))
        pairs.sort(key=lambda p: p[0], reverse=True)

        used_tracks: set[int] = set()
        used_faces: set[int] = set()
        for _, tid, di in pairs:
            if tid in used_tracks or di in used_faces:
                continue
            used_tracks.add(tid)
            used_faces.add(di)
            track = self._tracks[tid]
            face = faces[di]
            track.box = track.box.blend(face.box, cfg.smoothing)
            track.score = face.score
            track.keypoints = face.keypoints
            track.last_seen = now
            track.hits += 1

        for track in self._tracks.values():
            track.visible = track.id in used_tracks
        for di, face in enumerate(faces):
            if di not in used_faces:
                self._tracks[self._next_id] = Track(
                    self._next_id, face.box, face.score, face.keypoints, first_seen=now, last_seen=now
                )
                self._next_id += 1
        return list(self._tracks.values())

    def _affinity(self, track_box: Box, det_box: Box) -> float:
        iou = track_box.iou(det_box)
        if iou >= self._cfg.iou_threshold:
            return 1.0 + iou
        dx = abs(track_box.cx - det_box.cx) / max(track_box.w, det_box.w, 1e-6)
        dy = abs(track_box.cy - det_box.cy) / max(track_box.h, det_box.h, 1e-6)
        dist = math.hypot(dx, dy)
        limit = self._cfg.max_center_distance
        return 0.5 * (1.0 - dist / limit) + 1e-6 if dist < limit else 0.0


# --------------------------------------------------------------------------- classification
class Role(Enum):
    """What a tracked face means for security."""

    OWNER = "owner"
    INTRUDER = "intruder"
    IGNORED = "ignored"


class IdentityStatus(Enum):
    """Result of comparing a track with the enrolled owner."""

    OWNER = "owner"
    OTHER = "other"
    PENDING = "pending"  # not enough samples yet
    UNVERIFIABLE = "unverifiable"  # face too small to embed: it cannot be the owner


class IdentityVerifier(Protocol):
    """Owner verification used by the classifier when an owner is enrolled (see ``identity.py``)."""

    def update(self, tracks: Sequence[Track], frame: Image | None, now: float) -> None:
        """Embed the tracks that are due."""
        ...

    def status(self, track_id: int, face_px: float) -> IdentityStatus:
        """Verification state of a track."""
        ...

    def score(self, track_id: int) -> float | None:
        """Latest similarity to the owner, if any."""
        ...

    def reset(self) -> None:
        """Forget all decisions."""
        ...


@dataclass(frozen=True)
class ClassifiedFace:
    """A track snapshot with its role and the reason for that role."""

    track: Track
    role: Role
    reason: str
    yaw_deg: float | None
    area_ratio: float
    similarity: float | None = None


@dataclass(frozen=True)
class FrameAnalysis:
    """Result of classifying all live tracks of one frame."""

    faces: tuple[ClassifiedFace, ...] = ()
    owner_id: int | None = None

    @property
    def intruders(self) -> tuple[ClassifiedFace, ...]:
        """All faces currently classified as intruders."""
        return tuple(f for f in self.faces if f.role is Role.INTRUDER)

    @property
    def intruder_present(self) -> bool:
        """True when at least one intruder is present."""
        return any(f.role is Role.INTRUDER for f in self.faces)


class IntruderClassifier:
    """Decides which tracked face is the owner and which faces are intruders.

    Without enrollment the owner is the largest confirmed face and stays the owner by track ID
    ("sticky"). With an enrolled owner (``verifier``) the owner is whoever matches the stored
    embeddings, so a stranger who sits down is an intruder and the returning owner is recognised.
    A face that is not turned towards the screen (gaze filter) is ignored either way.
    """

    def __init__(
        self,
        det_cfg: DetectorConfig,
        trk_cfg: TrackerConfig,
        owner_cfg: OwnerConfig,
        gaze_cfg: GazeConfig | None = None,
        verifier: IdentityVerifier | None = None,
    ) -> None:
        self._det = det_cfg
        self._trk = trk_cfg
        self._owner_cfg = owner_cfg
        self._gaze = GazeFilter(det_cfg, gaze_cfg if gaze_cfg is not None else GazeConfig())
        self._verifier = verifier
        self._owner_id: int | None = None
        self._owner_box: Box | None = None
        self._owner_lost_at: float | None = None

    def reset(self) -> None:
        """Forget the current owner, gaze history and identity decisions."""
        self._owner_id = None
        self._owner_box = None
        self._owner_lost_at = None
        self._gaze.reset()
        if self._verifier is not None:
            self._verifier.reset()

    def classify(
        self, tracks: Sequence[Track], now: float, frame_width: int, frame: Image | None = None
    ) -> FrameAnalysis:
        """Classify ``tracks`` (all live tracks) at time ``now``; ``frame`` is needed for identity checks."""
        confirmed = [t for t in tracks if t.hits >= self._trk.min_hits]
        alive = {t.id: t for t in tracks}
        self._gaze.prune(alive)
        gaze = {t.id: self._gaze.update(t.id, t.keypoints, t.box.w * frame_width) for t in tracks}
        owner_ids = self._resolve_owners(alive, confirmed, now, frame_width, frame)
        current_owner = alive.get(self._owner_id) if self._owner_id is not None else None
        owner_box = current_owner.box if current_owner is not None else None
        out: list[ClassifiedFace] = []
        for track in tracks:
            area = track.box.area
            face_px = track.box.w * frame_width
            similarity = self._verifier.score(track.id) if self._verifier is not None else None
            if track.id in owner_ids:
                role, reason = Role.OWNER, "owner"
            elif track.hits < self._trk.min_hits:
                role, reason = Role.IGNORED, "unconfirmed"
            elif now - track.last_seen > self._trk.presence_s:
                role, reason = Role.IGNORED, "coasting"
            elif self._verifier is not None and self._verifier.status(track.id, face_px) is IdentityStatus.PENDING:
                role, reason = Role.IGNORED, "verifying"
            elif owner_box is not None and (
                track.box.iou(owner_box) >= self._owner_cfg.dedupe_iou
                or track.box.overlap_of_smaller(owner_box) >= 0.7
            ):
                role, reason = Role.IGNORED, "duplicate_of_owner"
            elif area < self._det.min_face_area_ratio:
                role, reason = Role.IGNORED, "too_small"
            elif not gaze[track.id].facing:
                role, reason = Role.IGNORED, "not_facing"
            else:
                role, reason = Role.INTRUDER, "intruder"
            snapshot = dataclasses.replace(track)
            out.append(ClassifiedFace(snapshot, role, reason, gaze[track.id].yaw_deg, area, similarity))
        return FrameAnalysis(tuple(out), self._owner_id)

    def _resolve_owners(
        self, alive: dict[int, Track], confirmed: Sequence[Track], now: float, frame_width: int, frame: Image | None
    ) -> set[int]:
        if self._verifier is None:
            self._update_owner(alive, confirmed, now)
            return set() if self._owner_id is None or self._owner_id not in alive else {self._owner_id}
        self._verifier.update(confirmed, frame, now)
        verifier = self._verifier
        matches = [t for t in confirmed if verifier.status(t.id, t.box.w * frame_width) is IdentityStatus.OWNER]
        if not matches:
            self._owner_id = None
            self._owner_box = None
            return set()
        best = max(matches, key=lambda t: verifier.score(t.id) or 0.0)
        if best.id != self._owner_id:
            logger.info("Owner recognised as track #%d", best.id)
        self._owner_id = best.id
        self._owner_box = best.box
        return {t.id for t in matches}

    def _update_owner(self, alive: dict[int, Track], confirmed: Sequence[Track], now: float) -> None:
        owner = alive.get(self._owner_id) if self._owner_id is not None else None
        if owner is not None:
            self._owner_box = owner.box
            self._owner_lost_at = None
            return
        if self._owner_id is not None and self._owner_lost_at is None:
            self._owner_lost_at = now
        lost_for = float("inf") if self._owner_lost_at is None else now - self._owner_lost_at

        if self._owner_box is not None and lost_for <= self._owner_cfg.continuity_window_s:
            heirs = [
                (t.box.iou(self._owner_box), t)
                for t in confirmed
                if t.box.iou(self._owner_box) >= self._owner_cfg.continuity_iou
            ]
            if heirs:
                heir = max(heirs, key=lambda pair: pair[0])[1]
                self._adopt(heir)
                return
        if (self._owner_id is None or lost_for >= self._owner_cfg.reacquire_s) and confirmed:
            self._adopt(max(confirmed, key=lambda t: t.box.area))

    def _adopt(self, track: Track) -> None:
        logger.info("Owner is now track #%d", track.id)
        self._owner_id = track.id
        self._owner_box = track.box
        self._owner_lost_at = None
