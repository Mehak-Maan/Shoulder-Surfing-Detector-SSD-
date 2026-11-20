"""Mode A: OpenCV debug window (green = secure/owner, red = intruder)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from capture import FramePacket, FrameSource
from config import GuiConfig
from detector import Role, as_image
from pipeline import Pipeline, PipelineResult
from state_machine import State

logger = logging.getLogger(__name__)

Image = NDArray[np.uint8]
Color = tuple[int, int, int]

GREEN: Color = (60, 200, 60)
RED: Color = (40, 40, 240)
GREY: Color = (160, 160, 160)
CYAN: Color = (220, 200, 0)
WHITE: Color = (255, 255, 255)
STATE_COLORS: dict[State, Color] = {
    State.SECURE: GREEN,
    State.SUSPECT: (0, 215, 255),
    State.WARNING: (0, 140, 255),
    State.ALERT: RED,
    State.LOCKED: (180, 0, 180),
    State.RECOVERY: (255, 160, 0),
}
HELP_LINES = (
    "q / Esc : quit",
    "d : debug (ignored faces, timers, yaw)",
    "k : keypoints",
    "t : far-field tiles",
    "h : this help",
)


@dataclass
class OverlayOptions:
    """Runtime-toggleable drawing options."""

    debug: bool = False
    keypoints: bool = False
    tiles: bool = False
    help: bool = False


def blank_frame(width: int = 640, height: int = 360) -> Image:
    """Return a black canvas used while the camera is unavailable."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def _text(img: Image, text: str, org: tuple[int, int], color: Color, scale: float = 0.55) -> None:
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 2, y - height - 3), (x + width + 2, y + baseline), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_overlay(
    frame: Image, result: PipelineResult, fps: float, options: OverlayOptions, max_width: int
) -> Image:
    """Return a copy of ``frame`` annotated with boxes, state, FPS and (optionally) debug data."""
    if frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        canvas = as_image(cv2.resize(frame, (max_width, round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA))
    else:
        canvas = frame.copy()
    h, w = canvas.shape[:2]
    snap = result.snapshot
    state_color = STATE_COLORS[snap.state]

    if options.tiles and result.detection is not None:
        for tile in result.detection.all_tiles:
            color = CYAN if tile in result.detection.tiles_run else (90, 90, 90)
            cv2.rectangle(canvas, (int(tile.x1 * w), int(tile.y1 * h)), (int(tile.x2 * w), int(tile.y2 * h)), color, 1)

    for face in result.analysis.faces:
        if face.role is Role.IGNORED and not options.debug:
            continue
        box = face.track.box
        p1 = (int(box.x1 * w), int(box.y1 * h))
        p2 = (int(box.x2 * w), int(box.y2 * h))
        color = {Role.OWNER: GREEN, Role.INTRUDER: RED, Role.IGNORED: GREY}[face.role]
        cv2.rectangle(canvas, p1, p2, color, 3 if face.role is Role.INTRUDER else 2)
        label = f"{face.role.value.upper()} #{face.track.id}"
        if options.debug:
            yaw = "" if face.yaw_deg is None else f" yaw {face.yaw_deg:.0f}"
            sim = "" if face.similarity is None else f" sim {face.similarity:.2f}"
            label += f" [{face.reason}] a={face.area_ratio * 100:.2f}%{yaw}{sim}"
        _text(canvas, label, (p1[0], max(14, p1[1] - 6)), color, 0.5)
        if options.keypoints and face.track.keypoints:
            for kx, ky in face.track.keypoints:
                cv2.circle(canvas, (int(kx * w), int(ky * h)), 3, CYAN, -1)

    if snap.state.is_escalated:
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), state_color, 6)
    cv2.rectangle(canvas, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.circle(canvas, (16, 15), 8, state_color, -1)
    _text(canvas, snap.state.value, (32, 21), state_color, 0.65)
    info = f"FPS {fps:5.1f}  lat {result.latency_ms:4.1f} ms  faces {len(result.analysis.faces)}"
    _text(canvas, info, (170, 21), WHITE, 0.5)
    if options.debug:
        timers = (
            f"intruder {snap.intruder_for_s:3.1f}s  clear {snap.clear_for_s:3.1f}s  "
            f"in-state {snap.time_in_state_s:4.1f}s"
        )
        _text(canvas, timers, (w - 330, 21), WHITE, 0.45)
    if snap.camera_fault:
        issue = result.camera_issue.replace("_", " ").upper() or "UNAVAILABLE"
        _text(canvas, f"CAMERA {issue}", (w // 2 - 140, h // 2), RED, 0.8)
    _text(canvas, "q quit  d debug  k keypoints  t tiles  h help", (10, h - 10), GREY, 0.45)
    if options.help:
        for i, line in enumerate(HELP_LINES):
            _text(canvas, line, (20, 60 + 22 * i), WHITE, 0.55)
    return canvas


def run_gui(
    gui_cfg: GuiConfig,
    source: FrameSource,
    pipeline: Pipeline,
    on_result: Callable[[FramePacket, PipelineResult], object] | None = None,
) -> int:
    """Show the live debug window until the user quits or the source ends.

    ``on_result`` is called with every packet and its result (used to run security actions).
    """
    options = OverlayOptions()
    cv2.namedWindow(gui_cfg.window_name, cv2.WINDOW_NORMAL)
    fps = 0.0
    last = time.perf_counter()
    try:
        while True:
            packet = source.read()
            if packet is None:
                logger.info("End of video")
                break
            result = pipeline.process(packet)
            if on_result is not None:
                on_result(packet, result)
            now = time.perf_counter()
            instant = 1.0 / max(now - last, 1e-6)
            fps = instant if fps == 0.0 else 0.9 * fps + 0.1 * instant
            last = now
            image = packet.image if packet.image is not None else blank_frame()
            cv2.imshow(gui_cfg.window_name, draw_overlay(image, result, fps, options, gui_cfg.max_display_width))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                options.debug = not options.debug
            elif key == ord("k"):
                options.keypoints = not options.keypoints
            elif key == ord("t"):
                options.tiles = not options.tiles
            elif key == ord("h"):
                options.help = not options.help
            if cv2.getWindowProperty(gui_cfg.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()
    return 0
