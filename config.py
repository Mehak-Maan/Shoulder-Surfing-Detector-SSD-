"""Typed configuration for the Shoulder Surfing Detector.

All thresholds live in frozen dataclasses with defaults. ``config.yaml`` may
override any subset of them; unknown keys, wrong types and out-of-range values
raise :class:`ConfigError` so that a typo never silently weakens security.
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml

CameraFaultPolicy = Literal["warn", "blur", "lock"]

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when the configuration file is malformed or out of range."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


@dataclass(frozen=True)
class CameraConfig:
    """Webcam capture settings."""

    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    reconnect_interval_s: float = 2.0
    read_failure_limit: int = 15

    def __post_init__(self) -> None:
        _require(self.index >= 0, "camera.index must be >= 0")
        _require(self.width >= 160 and self.height >= 120, "camera size too small")
        _require(self.fps >= 1, "camera.fps must be >= 1")
        _require(self.reconnect_interval_s > 0, "camera.reconnect_interval_s must be > 0")
        _require(self.read_failure_limit >= 1, "camera.read_failure_limit must be >= 1")


@dataclass(frozen=True)
class DetectorConfig:
    """Face detection and intruder qualification settings."""

    model_path: str = "models/blaze_face_short_range.tflite"
    min_confidence: float = 0.5
    nms_iou: float = 0.3
    process_width: int = 640
    min_face_area_ratio: float = 0.0025
    max_yaw_deg: float = 55.0
    yaw_min_face_px: int = 40
    nose_depth_ratio: float = 0.6
    far_field_enabled: bool = True
    far_field_cols: int = 3
    far_field_rows: int = 2
    far_field_overlap: float = 0.25
    far_field_roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.7)
    far_field_period_s: float = 0.3
    far_field_max_tiles_per_frame: int = 2

    def __post_init__(self) -> None:
        _require(0.0 < self.min_confidence < 1.0, "detector.min_confidence must be in (0, 1)")
        _require(0.0 < self.nms_iou < 1.0, "detector.nms_iou must be in (0, 1)")
        _require(self.process_width >= 160, "detector.process_width must be >= 160")
        _require(0.0 < self.min_face_area_ratio < 0.5, "detector.min_face_area_ratio out of range")
        _require(10.0 <= self.max_yaw_deg <= 90.0, "detector.max_yaw_deg must be in [10, 90]")
        _require(self.yaw_min_face_px >= 1, "detector.yaw_min_face_px must be >= 1")
        _require(0.1 <= self.nose_depth_ratio <= 2.0, "detector.nose_depth_ratio out of range")
        _require(self.far_field_cols >= 1 and self.far_field_rows >= 1, "far_field grid must be >= 1x1")
        _require(0.0 <= self.far_field_overlap < 1.0, "detector.far_field_overlap must be in [0, 1)")
        x0, y0, x1, y1 = self.far_field_roi
        _require(0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0, "detector.far_field_roi invalid")
        _require(self.far_field_period_s > 0, "detector.far_field_period_s must be > 0")
        _require(self.far_field_max_tiles_per_frame >= 1, "far_field_max_tiles_per_frame must be >= 1")


@dataclass(frozen=True)
class TrackerConfig:
    """Centroid/IoU tracker settings (time based, so FPS independent)."""

    iou_threshold: float = 0.2
    max_center_distance: float = 1.0
    ttl_s: float = 0.6
    presence_s: float = 0.15
    min_hits: int = 2
    smoothing: float = 0.5

    def __post_init__(self) -> None:
        _require(0.0 < self.iou_threshold < 1.0, "tracker.iou_threshold must be in (0, 1)")
        _require(self.max_center_distance > 0, "tracker.max_center_distance must be > 0")
        _require(self.ttl_s > 0, "tracker.ttl_s must be > 0")
        _require(0.0 <= self.presence_s <= self.ttl_s, "tracker.presence_s must be in [0, ttl_s]")
        _require(self.min_hits >= 1, "tracker.min_hits must be >= 1")
        _require(0.0 < self.smoothing <= 1.0, "tracker.smoothing must be in (0, 1]")


@dataclass(frozen=True)
class OwnerConfig:
    """Owner selection without enrollment (largest, sticky face)."""

    reacquire_s: float = 10.0
    continuity_window_s: float = 1.0
    continuity_iou: float = 0.25
    dedupe_iou: float = 0.4

    def __post_init__(self) -> None:
        _require(self.reacquire_s > 0, "owner.reacquire_s must be > 0")
        _require(self.continuity_window_s >= 0, "owner.continuity_window_s must be >= 0")
        _require(0.0 < self.continuity_iou < 1.0, "owner.continuity_iou must be in (0, 1)")
        _require(0.0 < self.dedupe_iou < 1.0, "owner.dedupe_iou must be in (0, 1)")


@dataclass(frozen=True)
class StateConfig:
    """Time-based debounce, hysteresis and cooldown of the state machine."""

    confirm_s: float = 0.7
    gap_tolerance_s: float = 0.4
    warning_to_alert_s: float = 2.0
    alert_to_lock_s: float = 4.0
    lock_enabled: bool = True
    clear_s: float = 2.0
    recovery_s: float = 1.0
    cooldown_s: float = 10.0
    camera_fault_grace_s: float = 3.0
    camera_fault_policy: CameraFaultPolicy = "blur"

    def __post_init__(self) -> None:
        _require(self.confirm_s > 0, "state.confirm_s must be > 0")
        _require(self.gap_tolerance_s >= 0, "state.gap_tolerance_s must be >= 0")
        _require(self.warning_to_alert_s > 0, "state.warning_to_alert_s must be > 0")
        _require(self.alert_to_lock_s > 0, "state.alert_to_lock_s must be > 0")
        _require(self.clear_s >= self.confirm_s, "state.clear_s must be >= state.confirm_s (hysteresis)")
        _require(self.recovery_s >= 0, "state.recovery_s must be >= 0")
        _require(self.cooldown_s >= 0, "state.cooldown_s must be >= 0")
        _require(self.camera_fault_grace_s > 0, "state.camera_fault_grace_s must be > 0")
        _require(
            self.camera_fault_policy in ("warn", "blur", "lock"),
            "state.camera_fault_policy must be warn, blur or lock",
        )


@dataclass(frozen=True)
class ActionsConfig:
    """What the escalation ladder does at each level."""

    toast_enabled: bool = True
    blur_enabled: bool = True
    relock_min_interval_s: float = 30.0
    toast_title: str = "Privacy warning"
    toast_message: str = "Someone is looking at your screen."
    fault_message: str = "The camera is unavailable, so screen privacy cannot be checked."

    def __post_init__(self) -> None:
        _require(self.relock_min_interval_s >= 0, "actions.relock_min_interval_s must be >= 0")
        _require(bool(self.toast_title.strip()), "actions.toast_title must not be empty")


@dataclass(frozen=True)
class OverlayConfig:
    """Fullscreen blur overlay settings."""

    blur_downscale: int = 4
    blur_kernel: int = 31
    darken: float = 0.3
    message: str = "Privacy protection active"

    def __post_init__(self) -> None:
        _require(1 <= self.blur_downscale <= 16, "overlay.blur_downscale must be in [1, 16]")
        _require(self.blur_kernel >= 3 and self.blur_kernel % 2 == 1, "overlay.blur_kernel must be odd and >= 3")
        _require(0.0 <= self.darken <= 0.9, "overlay.darken must be in [0, 0.9]")


@dataclass(frozen=True)
class HealthConfig:
    """Camera health: covered/dark and frozen-stream detection (time based)."""

    enabled: bool = True
    dark_luma_max: float = 12.0
    dark_std_max: float = 6.0
    frozen_diff_max: float = 0.05
    covered_s: float = 2.0
    frozen_s: float = 3.0

    def __post_init__(self) -> None:
        _require(0.0 <= self.dark_luma_max <= 60.0, "health.dark_luma_max must be in [0, 60]")
        _require(0.0 <= self.dark_std_max <= 30.0, "health.dark_std_max must be in [0, 30]")
        _require(0.0 <= self.frozen_diff_max <= 5.0, "health.frozen_diff_max must be in [0, 5]")
        _require(self.covered_s > 0 and self.frozen_s > 0, "health.covered_s and frozen_s must be > 0")


@dataclass(frozen=True)
class PrivacyConfig:
    """Evidence snapshots are OFF by default; frames are never stored unless enabled."""

    evidence_enabled: bool = False
    evidence_dir: str = "~/.shoulder_surfing_detector/evidence"
    key_file: str = "~/.shoulder_surfing_detector/evidence.key"
    evidence_max_files: int = 20
    jpeg_quality: int = 80

    def __post_init__(self) -> None:
        _require(self.evidence_max_files >= 1, "privacy.evidence_max_files must be >= 1")
        _require(30 <= self.jpeg_quality <= 100, "privacy.jpeg_quality must be in [30, 100]")


@dataclass(frozen=True)
class RuntimeConfig:
    """Headless service settings."""

    headless_fps: float = 8.0
    tray_enabled: bool = True
    log_file: str = "~/.shoulder_surfing_detector/ssd.log"
    log_max_bytes: int = 1_000_000
    log_backups: int = 3
    lock_file: str = "~/.shoulder_surfing_detector/ssd.lock"

    def __post_init__(self) -> None:
        _require(1.0 <= self.headless_fps <= 30.0, "runtime.headless_fps must be in [1, 30]")
        _require(self.log_max_bytes >= 10_000, "runtime.log_max_bytes must be >= 10000")
        _require(self.log_backups >= 0, "runtime.log_backups must be >= 0")


@dataclass(frozen=True)
class GuiConfig:
    """OpenCV debug window settings."""

    window_name: str = "Shoulder Surfing Detector"
    max_display_width: int = 960

    def __post_init__(self) -> None:
        _require(self.max_display_width >= 320, "gui.max_display_width must be >= 320")


@dataclass(frozen=True)
class GazeConfig:
    """Attention filter: only faces turned towards the screen count as watchers.

    The yaw limit itself is ``detector.max_yaw_deg``; this section adds smoothing,
    hysteresis (no flicker around the limit) and an optional pitch check.
    """

    enabled: bool = True
    smoothing_alpha: float = 0.4
    hysteresis_deg: float = 8.0
    use_pitch: bool = False
    max_pitch_deg: float = 50.0
    pitch_neutral_ratio: float = 0.45
    pitch_range: float = 0.3

    def __post_init__(self) -> None:
        _require(0.0 < self.smoothing_alpha <= 1.0, "gaze.smoothing_alpha must be in (0, 1]")
        _require(0.0 <= self.hysteresis_deg <= 40.0, "gaze.hysteresis_deg must be in [0, 40]")
        _require(10.0 <= self.max_pitch_deg <= 90.0, "gaze.max_pitch_deg must be in [10, 90]")
        _require(0.1 <= self.pitch_neutral_ratio <= 0.9, "gaze.pitch_neutral_ratio out of range")
        _require(0.05 <= self.pitch_range <= 1.0, "gaze.pitch_range out of range")


@dataclass(frozen=True)
class IdentityConfig:
    """Owner enrollment. Only embedding vectors are stored (encrypted), never images."""

    enabled: bool = False
    embedder: Literal["lbp", "onnx"] = "lbp"
    onnx_model: str = "models/face_embedder.onnx"
    profile_file: str = "~/.shoulder_surfing_detector/owner.profile.enc"
    enroll_samples: int = 20
    enroll_interval_s: float = 0.5
    enroll_timeout_s: float = 90.0
    enroll_max_yaw_deg: float = 25.0
    enroll_min_face_px: int = 80
    min_embed_px: int = 48
    verify_period_s: float = 1.0
    min_samples: int = 3
    window: int = 5
    max_embeds_per_frame: int = 2
    top_k: int = 3
    threshold_floor: float = 0.5
    threshold_ceiling: float = 0.95
    threshold_margin: float = 0.1
    stay_margin: float = 0.05

    def __post_init__(self) -> None:
        _require(self.embedder in ("lbp", "onnx"), "identity.embedder must be 'lbp' or 'onnx'")
        _require(self.enroll_samples >= 5, "identity.enroll_samples must be >= 5")
        _require(self.enroll_interval_s > 0, "identity.enroll_interval_s must be > 0")
        _require(self.enroll_timeout_s > 0, "identity.enroll_timeout_s must be > 0")
        _require(5.0 <= self.enroll_max_yaw_deg <= 60.0, "identity.enroll_max_yaw_deg must be in [5, 60]")
        _require(self.enroll_min_face_px >= self.min_embed_px, "identity.enroll_min_face_px must be >= min_embed_px")
        _require(self.min_embed_px >= 16, "identity.min_embed_px must be >= 16")
        _require(self.verify_period_s > 0, "identity.verify_period_s must be > 0")
        _require(1 <= self.min_samples <= self.window, "identity.min_samples must be in [1, window]")
        _require(self.max_embeds_per_frame >= 1, "identity.max_embeds_per_frame must be >= 1")
        _require(self.top_k >= 1, "identity.top_k must be >= 1")
        _require(
            0.0 < self.threshold_floor <= self.threshold_ceiling < 1.0,
            "identity.threshold_floor/ceiling must satisfy 0 < floor <= ceiling < 1",
        )
        _require(0.0 <= self.threshold_margin < 0.5, "identity.threshold_margin must be in [0, 0.5)")
        _require(0.0 <= self.stay_margin < 0.3, "identity.stay_margin must be in [0, 0.3)")


@dataclass(frozen=True)
class AppConfig:
    """Complete application configuration."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    owner: OwnerConfig = field(default_factory=OwnerConfig)
    state: StateConfig = field(default_factory=StateConfig)
    actions: ActionsConfig = field(default_factory=ActionsConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)


def _coerce(value: Any, hint: Any, where: str) -> Any:
    """Validate ``value`` against the type ``hint`` and return the typed value."""
    origin = typing.get_origin(hint)
    if dataclasses.is_dataclass(hint) and isinstance(hint, type):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected a mapping")
        return _from_mapping(hint, value, where)
    if origin is tuple:
        args = typing.get_args(hint)
        if not isinstance(value, (list, tuple)) or len(value) != len(args):
            raise ConfigError(f"{where}: expected a list of {len(args)} values")
        return tuple(_coerce(v, a, f"{where}[{i}]") for i, (v, a) in enumerate(zip(value, args, strict=True)))
    if origin is Literal:
        if value not in typing.get_args(hint):
            raise ConfigError(f"{where}: must be one of {typing.get_args(hint)}")
        return value
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string")
        return value
    raise ConfigError(f"{where}: unsupported type {hint!r}")


def _from_mapping(cls: type[T], data: Mapping[str, Any], where: str) -> T:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - names)
    if unknown:
        raise ConfigError(f"{where or 'config'}: unknown key(s) {unknown}")
    kwargs = {k: _coerce(v, hints[k], f"{where}.{k}" if where else k) for k, v in data.items()}
    try:
        return cls(**kwargs)
    except ConfigError as exc:
        raise ConfigError(f"{where or 'config'}: {exc}") from exc


def load_config(path: Path | None) -> AppConfig:
    """Load ``path`` (YAML) on top of the defaults; ``None`` returns the defaults."""
    if path is None:
        return AppConfig()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if raw is None:
        return AppConfig()
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path}: top level must be a mapping")
    return _from_mapping(AppConfig, raw, "")
