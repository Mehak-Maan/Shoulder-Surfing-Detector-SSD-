"""Configuration loading and validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import AppConfig, ConfigError, load_config

ROOT = Path(__file__).resolve().parent.parent


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "c.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_shipped_yaml_equals_dataclass_defaults() -> None:
    assert load_config(ROOT / "config.yaml") == AppConfig()


def test_none_and_empty_file_give_defaults(tmp_path: Path) -> None:
    assert load_config(None) == AppConfig()
    assert load_config(_write(tmp_path, "")) == AppConfig()


def test_partial_override(tmp_path: Path) -> None:
    text = "state:\n  confirm_s: 1.5\n  camera_fault_policy: lock\ndetector:\n  far_field_roi: [0, 0, 1, 0.5]\n"
    cfg = load_config(_write(tmp_path, text))
    assert cfg.state.confirm_s == 1.5 and cfg.state.camera_fault_policy == "lock"
    assert cfg.detector.far_field_roi == (0.0, 0.0, 1.0, 0.5)
    assert cfg.camera == AppConfig().camera


@pytest.mark.parametrize(
    "text",
    [
        "state:\n  confirm_sec: 1\n",  # unknown key (typo)
        "bogus: 1\n",  # unknown section
        "state:\n  confirm_s: fast\n",  # wrong type
        "camera:\n  index: true\n",  # bool is not an int
        "state:\n  camera_fault_policy: ignore\n",  # not an allowed literal
        "detector:\n  min_confidence: 1.5\n",  # out of range
        "detector:\n  far_field_roi: [0, 0, 1]\n",  # wrong length
        "state:\n  confirm_s: 1.0\n  clear_s: 0.5\n",  # hysteresis violated
        "- just\n- a list\n",
        "state: [1, 2]\n",
    ],
)
def test_invalid_configs_are_rejected(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, text))


def test_missing_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")
