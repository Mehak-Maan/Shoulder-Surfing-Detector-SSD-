"""Synthetic scenes for identity tests: textured 'people' rendered at any position, size and roll."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from capture import FramePacket
from detector import Box, Image, RawFace, as_image

W, H = 1280, 720
_BASE = 200
_POINTS = np.array(
    [[70, 80], [130, 80], [100, 115], [100, 150], [30, 100], [170, 100]], dtype=np.float32
)  # eyes, nose, mouth, ears in base coordinates


_BG: list[Image] = []


def _background() -> Image:
    if not _BG:
        _BG.append(np.clip(np.random.default_rng(0).normal(90, 3, (H, W, 3)), 0, 255).astype(np.uint8))
    return _BG[0]


def texture(seed: int) -> Image:
    """A person's 'face': 10x10 colour blocks smoothly upsampled, with two dark eyes."""
    rng = np.random.default_rng(seed)
    small = rng.integers(20, 235, (10, 10, 3)).astype(np.uint8)
    img = as_image(cv2.resize(small, (_BASE, _BASE), interpolation=cv2.INTER_CUBIC))
    for x, y in _POINTS[:2]:
        cv2.circle(img, (int(x), int(y)), 9, (15, 15, 15), -1)
    return img


@dataclass(frozen=True)
class Person:
    """One rendered face: identity ``seed``, centre (frame fractions), pixel width and head roll."""

    seed: int
    cx: float
    cy: float
    width_px: float
    roll_deg: float = 0.0
    nose_shift: float = 0.0  # 0 = frontal; larger = head turned (as in helpers.make_face)


def render(frame: Image, person: Person) -> RawFace:
    """Draw ``person`` into ``frame`` and return the matching detector output."""
    scale = person.width_px / _BASE
    matrix = cv2.getRotationMatrix2D((_BASE / 2, _BASE / 2), -person.roll_deg, scale)
    matrix[0, 2] += person.cx * W - _BASE / 2
    matrix[1, 2] += person.cy * H - _BASE / 2
    layer = cv2.warpAffine(texture(person.seed), matrix, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(np.full((_BASE, _BASE), 255, np.uint8), matrix, (W, H), flags=cv2.INTER_NEAREST)
    frame[mask > 0] = layer[mask > 0]
    pts = cv2.transform(_POINTS.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    if person.nose_shift:
        eye = float(np.linalg.norm(pts[1] - pts[0]))
        pts[2] = pts[2] + np.array([person.nose_shift * eye, 0.0], dtype=np.float32)
    corners = cv2.transform(np.array([[[0, 0], [_BASE, 0], [_BASE, _BASE], [0, _BASE]]], np.float32), matrix)[0]
    box = Box(
        float(corners[:, 0].min() / W), float(corners[:, 1].min() / H),
        float(corners[:, 0].max() / W), float(corners[:, 1].max() / H),
    )
    return RawFace(box, 0.95, tuple((float(x) / W, float(y) / H) for x, y in pts))


def frame_with(people: Sequence[Person], rng: np.random.Generator | None = None) -> tuple[Image, list[RawFace]]:
    """A grey noisy background with ``people`` drawn on it, plus their detections."""
    del rng  # kept for call compatibility; the background is a fixed noisy grey (cheap to copy)
    frame = _background().copy()
    return frame, [render(frame, p) for p in people]


def build_scene(
    spec: Callable[[float], Sequence[Person]], seconds: float, fps: float = 10.0
) -> tuple[list[FramePacket], list[list[RawFace]]]:
    """Packets and per-frame detections for ``spec(t) -> people at time t``."""
    packets: list[FramePacket] = []
    faces: list[list[RawFace]] = []
    for i in range(round(seconds * fps)):
        t = i / fps
        frame, detections = frame_with(spec(t))
        packets.append(FramePacket(frame, t, True, i + 1))
        faces.append(detections)
    return packets, faces


def roll_for(t: float) -> float:
    """Gentle head sway used when a person is 'sitting'."""
    return 4.0 * math.sin(t * 1.3)
