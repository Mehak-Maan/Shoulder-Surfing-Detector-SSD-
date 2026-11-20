"""Gaze / attention filter: is a face turned towards the screen?

Works on the six BlazeFace keypoints (right eye, left eye, nose, mouth, two ears), so it is
only a coarse head-pose heuristic, not eye tracking. Values are smoothed per track and use
hysteresis so a face hovering around the limit does not flicker between roles.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from config import DetectorConfig, GazeConfig

Point = tuple[float, float]


def estimate_yaw_deg(keypoints: Sequence[Point] | None, face_px: float, cfg: DetectorConfig) -> float | None:
    """Estimate the absolute head yaw (degrees) from BlazeFace keypoints.

    Uses the horizontal nose offset from the eye midpoint relative to the eye
    distance. Keypoint order is: right eye, left eye, nose tip, mouth, ears.
    Returns ``None`` when the estimate would be unreliable (no keypoints or a
    face narrower than ``cfg.yaw_min_face_px``); callers must then assume the
    face is looking at the screen.
    """
    if keypoints is None or len(keypoints) < 3 or face_px < cfg.yaw_min_face_px:
        return None
    (rx, _), (lx, _), (nx, _) = keypoints[0], keypoints[1], keypoints[2]
    eye_dist = abs(lx - rx)
    if eye_dist < 1e-4:
        return 90.0
    ratio = abs(nx - (rx + lx) / 2.0) / eye_dist
    return math.degrees(math.asin(min(1.0, ratio / cfg.nose_depth_ratio)))


def estimate_pitch_deg(
    keypoints: Sequence[Point] | None, face_px: float, det_cfg: DetectorConfig, cfg: GazeConfig
) -> float | None:
    """Estimate the absolute head pitch (degrees) from the nose position between eyes and mouth."""
    if keypoints is None or len(keypoints) < 4 or face_px < det_cfg.yaw_min_face_px:
        return None
    eyes_y = (keypoints[0][1] + keypoints[1][1]) / 2.0
    span = keypoints[3][1] - eyes_y
    if span < 1e-4:
        return None
    ratio = (keypoints[2][1] - eyes_y) / span
    offset = max(-1.0, min(1.0, (ratio - cfg.pitch_neutral_ratio) / cfg.pitch_range))
    return abs(math.degrees(math.asin(offset)))


@dataclass(frozen=True)
class GazeResult:
    """Smoothed head pose of one track and whether it counts as facing the screen."""

    facing: bool
    yaw_deg: float | None
    pitch_deg: float | None


@dataclass
class _TrackGaze:
    yaw: float | None = None
    pitch: float | None = None
    facing: bool = True


def _smooth(previous: float | None, raw: float | None, alpha: float) -> float | None:
    if raw is None:
        return previous
    if previous is None:
        return raw
    return alpha * raw + (1.0 - alpha) * previous


class GazeFilter:
    """Per-track smoothing and hysteresis. Unknown pose keeps the previous verdict (default: facing)."""

    def __init__(self, det_cfg: DetectorConfig, cfg: GazeConfig) -> None:
        self._det = det_cfg
        self._cfg = cfg
        self._tracks: dict[int, _TrackGaze] = {}

    def reset(self) -> None:
        """Forget every track."""
        self._tracks.clear()

    def prune(self, alive: Collection[int]) -> None:
        """Drop the state of tracks that no longer exist."""
        for track_id in [t for t in self._tracks if t not in alive]:
            del self._tracks[track_id]

    def update(self, track_id: int, keypoints: Sequence[Point] | None, face_px: float) -> GazeResult:
        """Feed one observation of ``track_id`` and return its verdict."""
        cfg = self._cfg
        state = self._tracks.setdefault(track_id, _TrackGaze())
        state.yaw = _smooth(state.yaw, estimate_yaw_deg(keypoints, face_px, self._det), cfg.smoothing_alpha)
        if cfg.use_pitch:
            raw_pitch = estimate_pitch_deg(keypoints, face_px, self._det, cfg)
            state.pitch = _smooth(state.pitch, raw_pitch, cfg.smoothing_alpha)
        if cfg.enabled:
            # While facing, the limit applies; once turned away, the face must come back by the hysteresis.
            slack = 0.0 if state.facing else cfg.hysteresis_deg
            turned = state.yaw is not None and state.yaw > self._det.max_yaw_deg - slack
            if cfg.use_pitch:
                turned = turned or (state.pitch is not None and state.pitch > cfg.max_pitch_deg - slack)
            known = state.yaw is not None or (cfg.use_pitch and state.pitch is not None)
            if known:
                state.facing = not turned
        else:
            state.facing = True
        return GazeResult(state.facing, state.yaw, state.pitch)
