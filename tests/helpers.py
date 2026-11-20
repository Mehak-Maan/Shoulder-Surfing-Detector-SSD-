"""Shared test helpers: fake backend, synthetic faces and list-based frame sources."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from capture import FramePacket
from detector import Box, Image, Keypoints, RawFace

FRAME_W, FRAME_H = 1280, 720


def make_face(
    cx: float,
    cy: float,
    w: float,
    nose_shift: float = 0.0,
    score: float = 0.9,
    keypoints: bool = True,
) -> RawFace:
    """A frame-normalized face; ``h`` follows ``w`` so it is square in pixels on a 16:9 frame.

    ``nose_shift`` is the nose offset from the eye midpoint in eye distances (0 = frontal).
    """
    h = w * FRAME_W / FRAME_H
    box = Box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    kps: Keypoints | None = None
    if keypoints:
        eye = 0.2 * w
        kps = (
            (cx - eye, cy - 0.15 * h),
            (cx + eye, cy - 0.15 * h),
            (cx + nose_shift * 2 * eye, cy),
            (cx, cy + 0.2 * h),
            (cx - 0.45 * w, cy),
            (cx + 0.45 * w, cy),
        )
    return RawFace(box, score, kps)


class ScriptedBackend:
    """Fake :class:`FaceBackend` whose output depends only on the call index."""

    def __init__(self, script: Callable[[int], list[RawFace]]) -> None:
        self._script = script
        self.calls = 0
        self.closed = False

    def detect(self, rgb: Image) -> list[RawFace]:
        faces = self._script(self.calls)
        self.calls += 1
        return faces

    def close(self) -> None:
        self.closed = True


class ListSource:
    """Frame source that replays a prepared list of packets."""

    def __init__(self, packets: Iterable[FramePacket]) -> None:
        self._iter = iter(packets)

    def read(self) -> FramePacket | None:
        return next(self._iter, None)

    def close(self) -> None:
        return None


def blank(width: int = FRAME_W, height: int = FRAME_H) -> Image:
    """A black BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def packets(count: int, fps: float = 30.0, ok: bool = True) -> list[FramePacket]:
    """``count`` packets with virtual timestamps ``i / fps``."""
    return [FramePacket(blank() if ok else None, i / fps, ok, i + 1) for i in range(count)]
