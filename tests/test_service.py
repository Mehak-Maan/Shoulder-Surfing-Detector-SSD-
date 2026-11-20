"""Phase 3 tests: camera health, evidence, headless runner, runtime plumbing, tray helpers, CLI."""

from __future__ import annotations

import dataclasses
import logging
import signal
import threading
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest
from helpers import ListSource, ScriptedBackend, blank, make_face, packets

import main as main_module
from camera_health import CameraHealth
from capture import FramePacket
from config import AppConfig, ConfigError, HealthConfig, PrivacyConfig, RuntimeConfig, load_config
from detector import Image, RawFace
from evidence import SUFFIX, EvidenceStore, decrypt_file
from headless import MAX_CONSECUTIVE_ERRORS, HeadlessRunner
from pipeline import Pipeline
from runtime import SingleInstance, install_signal_handlers, setup_logging
from security_actions import ActionExecutor, ResponseHandler
from state_machine import State
from tray import NullTray, create_tray, icon_color, make_icon

OWNER = make_face(0.45, 0.55, 0.22)
INTRUDER = make_face(0.82, 0.4, 0.09, nose_shift=0.2)


# ------------------------------------------------------------------ camera health
def _noisy(rng: np.random.Generator, level: float = 100.0) -> Image:
    return np.clip(rng.normal(level, 12, (720, 1280, 3)), 0, 255).astype(np.uint8)


def _run_health(frames: Callable[[int], Image], seconds: float, fps: float = 10.0) -> list[tuple[float, bool, str]]:
    health = CameraHealth(HealthConfig())
    out = []
    for i in range(round(seconds * fps)):
        t = i / fps
        status = health.update(frames(i), t)
        out.append((t, status.ok, status.reason))
    return out


def test_live_noisy_camera_is_healthy() -> None:
    rng = np.random.default_rng(0)
    assert all(ok for _, ok, _ in _run_health(lambda _i: _noisy(rng), 8.0))


def test_covered_lens_is_detected_after_grace_period() -> None:
    rng = np.random.default_rng(0)
    dark = lambda i: (rng.normal(3, 1, (720, 1280, 3)).clip(0, 255).astype(np.uint8))  # noqa: E731
    result = _run_health(dark, 5.0)
    first_bad = next(t for t, ok, _ in result if not ok)
    assert 1.9 <= first_bad <= 2.2 and result[-1][2] == "covered_or_dark"


def test_frozen_stream_is_detected_and_recovers() -> None:
    rng = np.random.default_rng(0)
    still = _noisy(rng)
    live = [_noisy(rng) for _ in range(80)]
    result = _run_health(lambda i: live[i] if i < 20 or i >= 60 else still, 8.0)
    bad = [t for t, ok, reason in result if not ok and reason == "frozen"]
    assert bad and 4.9 <= bad[0] <= 5.3  # frozen from t=2.0 (+3 s grace)
    assert result[-1][1]  # live frames again -> healthy


def test_health_can_be_disabled_and_reset() -> None:
    health = CameraHealth(dataclasses.replace(HealthConfig(), enabled=False))
    assert all(health.update(blank(), t / 10).ok for t in range(100))
    health = CameraHealth(HealthConfig())
    for t in range(40):
        health.update(blank(), t / 10)
    assert not health.update(blank(), 4.0).ok
    health.reset()
    assert health.update(blank(), 4.1).ok


def test_pipeline_treats_covered_camera_as_fault() -> None:
    base = AppConfig()
    cfg = dataclasses.replace(base, detector=dataclasses.replace(base.detector, far_field_enabled=False))
    pipeline = Pipeline(cfg, ScriptedBackend(lambda _i: [OWNER]))
    results = [pipeline.process(p) for p in packets(150, 10.0)]  # 15 s of black frames
    assert results[-1].camera_issue in ("covered_or_dark", "frozen") and results[-1].snapshot.camera_fault
    assert results[-1].snapshot.state is State.ALERT  # default camera_fault_policy = blur


# ------------------------------------------------------------------ evidence
def test_evidence_is_encrypted_roundtrips_and_is_pruned(tmp_path: Path) -> None:
    cfg = PrivacyConfig(
        evidence_enabled=True,
        evidence_dir=str(tmp_path / "ev"),
        key_file=str(tmp_path / "k.key"),
        evidence_max_files=2,
    )
    ticks = iter(range(1_000_000_000, 1_000_000_100))
    store = EvidenceStore(cfg, clock=lambda: float(next(ticks)))
    frame = np.random.default_rng(1).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    paths = [store.save(frame, "alert") for _ in range(3)]
    assert all(p is not None for p in paths)
    files = sorted((tmp_path / "ev").glob(f"*{SUFFIX}"))
    assert len(files) == 2 and not paths[0].exists()  # type: ignore[union-attr]
    raw = files[-1].read_bytes()
    assert not raw.startswith(b"\xff\xd8")  # not a plain JPEG
    jpeg = decrypt_file(tmp_path / "k.key", files[-1])
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape == frame.shape
    other_key = tmp_path / "other.key"
    EvidenceStore(dataclasses.replace(cfg, key_file=str(other_key)))
    with pytest.raises(Exception):  # noqa: B017, PT011 - cryptography raises InvalidToken
        decrypt_file(other_key, files[-1])


def test_response_handler_saves_evidence_only_on_fresh_alert_or_lock() -> None:
    class Sink:
        def __init__(self) -> None:
            self.labels: list[str] = []

        def save(self, image: Image, label: str) -> object:
            self.labels.append(label)
            return None

    class Recorder:
        visible = False

        def show(self) -> None: ...
        def hide(self) -> None: ...
        def close(self) -> None: ...

    class Plat:
        def notify(self, title: str, message: str) -> bool:
            return True

        def lock_screen(self) -> bool:
            return True

    cfg = dataclasses.replace(
        AppConfig(), detector=dataclasses.replace(AppConfig().detector, far_field_enabled=False),
        health=dataclasses.replace(AppConfig().health, enabled=False),
    )
    pipeline = Pipeline(cfg, ScriptedBackend(lambda i: [OWNER, INTRUDER] if i >= 30 else [OWNER]))
    sink = Sink()
    handler = ResponseHandler(ActionExecutor(cfg.actions, Plat(), Recorder()), sink)
    for packet in packets(400, 30.0):
        handler(packet, pipeline.process(packet))
    assert sink.labels == ["alert", "locked"]


# ------------------------------------------------------------------ headless runner
class FakeTray:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.statuses: list[tuple[State, bool]] = []

    def start(self, on_toggle_pause: Callable[[], None], on_quit: Callable[[], None]) -> None:
        self.events.append("start")

    def set_status(self, state: State, paused: bool) -> None:
        self.statuses.append((state, paused))

    def stop(self) -> None:
        self.events.append("stop")


def _cfg() -> AppConfig:
    base = AppConfig()
    return dataclasses.replace(
        base,
        detector=dataclasses.replace(base.detector, far_field_enabled=False),
        health=dataclasses.replace(base.health, enabled=False),
    )


class HookedSource(ListSource):
    """ListSource that calls ``hook(index)`` before returning each packet."""

    def __init__(self, items: list[FramePacket], hook: Callable[[int], None]) -> None:
        super().__init__(items)
        self._hook = hook
        self._i = 0

    def read(self) -> FramePacket | None:
        self._hook(self._i)
        self._i += 1
        return super().read()


def _runner(source: ListSource, script: Callable[[int], list[RawFace]], stop: threading.Event | None = None) -> tuple[
    HeadlessRunner, FakeTray, ResponseHandler
]:
    class Overlay:
        visible = False

        def show(self) -> None:
            self.visible = True

        def hide(self) -> None:
            self.visible = False

        def close(self) -> None: ...

    class Plat:
        def __init__(self) -> None:
            self.locks = 0

        def notify(self, title: str, message: str) -> bool:
            return True

        def lock_screen(self) -> bool:
            self.locks += 1
            return True

    cfg = _cfg()
    handler = ResponseHandler(ActionExecutor(cfg.actions, Plat(), Overlay()))
    tray = FakeTray()
    pipeline = Pipeline(cfg, ScriptedBackend(script))
    runner = HeadlessRunner(cfg.runtime, source, pipeline, handler, tray, stop, paced=False)
    return runner, tray, handler


def test_headless_runs_to_end_of_source_and_updates_tray() -> None:
    runner, tray, _ = _runner(ListSource(packets(300, 30.0)), lambda i: [OWNER, INTRUDER] if 60 <= i < 200 else [OWNER])
    assert runner.run() == 0
    assert tray.events == ["start", "stop"]
    states = [s for s, paused in tray.statuses if not paused]
    assert State.WARNING in states and State.ALERT in states and states[-1] in (State.RECOVERY, State.SECURE)


def test_headless_stop_event_ends_loop_and_cleans_up() -> None:
    stop = threading.Event()
    source = HookedSource(packets(1000, 30.0), lambda i: stop.set() if i == 20 else None)
    runner, tray, _ = _runner(source, lambda _i: [OWNER], stop)
    assert runner.run() == 0 and tray.events == ["start", "stop"]


def test_headless_pause_resets_and_ignores_intruder_until_resume() -> None:
    holder: dict[str, HeadlessRunner] = {}
    frame = {"i": 0}

    def hook(i: int) -> None:
        frame["i"] = i
        if i in (50, 250):
            holder["r"].toggle_pause()

    source = HookedSource(packets(400, 30.0), hook)
    runner, tray, _ = _runner(source, lambda _calls: [OWNER, INTRUDER] if 60 <= frame["i"] < 240 else [OWNER])
    holder["r"] = runner
    runner.run()
    assert (State.SECURE, True) in tray.statuses and (State.SECURE, False) in tray.statuses
    assert not any(s in (State.WARNING, State.ALERT) for s, _ in tray.statuses)  # intruder only while paused


def test_headless_survives_errors_but_gives_up_after_too_many() -> None:
    raised = {"done": False}

    def failing(_calls: int) -> list[RawFace]:
        if not raised["done"]:
            raised["done"] = True
            raise RuntimeError("detector glitch")
        return [OWNER]

    runner, _, _ = _runner(ListSource(packets(50, 30.0)), failing)
    assert runner.run() == 0  # one error: logged, loop continues

    def always(_i: int) -> list[RawFace]:
        raise RuntimeError("broken")

    runner, tray, _ = _runner(ListSource(packets(MAX_CONSECUTIVE_ERRORS + 5, 30.0)), always)
    assert runner.run() == 1 and tray.events[-1] == "stop"


# ------------------------------------------------------------------ runtime plumbing
def test_single_instance_guard(tmp_path: Path) -> None:
    lock = tmp_path / "x" / "ssd.lock"
    first, second = SingleInstance(lock), SingleInstance(lock)
    assert first.acquire()
    assert not second.acquire()
    first.release()
    assert second.acquire()
    second.release()
    assert lock.read_text() == str(__import__("os").getpid())


def test_signal_handlers_set_stop_and_restore() -> None:
    stop = threading.Event()
    old = signal.getsignal(signal.SIGTERM)
    restore = install_signal_handlers(stop)
    signal.raise_signal(signal.SIGTERM)
    assert stop.is_set()
    restore()
    assert signal.getsignal(signal.SIGTERM) == old


def test_rotating_log_file(tmp_path: Path) -> None:
    cfg = RuntimeConfig(log_file=str(tmp_path / "logs" / "ssd.log"), log_max_bytes=10_000, log_backups=2)
    path = setup_logging(cfg, "INFO", console=False)
    try:
        log = logging.getLogger("rotation-test")
        for i in range(400):
            log.info("line %04d %s", i, "x" * 80)
    finally:
        logging.getLogger().handlers.clear()
    assert path is not None and path.exists()
    assert (tmp_path / "logs" / "ssd.log.1").exists() and not (tmp_path / "logs" / "ssd.log.3").exists()


def test_log_setup_falls_back_when_path_is_unwritable(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    cfg = RuntimeConfig(log_file=str(blocker / "sub" / "ssd.log"))
    try:
        assert setup_logging(cfg, "INFO", console=False) is None
    finally:
        logging.getLogger().handlers.clear()


# ------------------------------------------------------------------ tray helpers, config, CLI
def test_tray_icon_helpers() -> None:
    assert icon_color(State.ALERT, False) != icon_color(State.SECURE, False)
    assert icon_color(State.ALERT, True) == icon_color(State.SECURE, True)  # paused is grey for every state
    img = make_icon((10, 20, 30))
    assert img.size == (64, 64) and img.convert("RGB").getpixel((32, 8)) == (10, 20, 30)
    assert isinstance(create_tray(False), NullTray)
    NullTray().start(lambda: None, lambda: None)


def test_phase3_config_sections(tmp_path: Path) -> None:
    cfg = AppConfig()
    assert cfg.runtime.headless_fps == 8.0 and not cfg.privacy.evidence_enabled and cfg.health.enabled
    bad: list[Callable[[], object]] = [
        lambda: RuntimeConfig(headless_fps=0.5),
        lambda: PrivacyConfig(jpeg_quality=10),
        lambda: HealthConfig(covered_s=0),
    ]
    for build in bad:
        with pytest.raises(ConfigError):
            build()
    path = tmp_path / "c.yaml"
    path.write_text("runtime:\n  headless_fps: 4\nprivacy:\n  evidence_enabled: true\n", encoding="utf-8")
    loaded = load_config(path)
    assert loaded.runtime.headless_fps == 4.0 and loaded.privacy.evidence_enabled


def test_cli_headless_refuses_second_instance_and_decrypts(tmp_path: Path) -> None:
    lock = tmp_path / "ssd.lock"
    key = tmp_path / "k.key"
    conf = tmp_path / "c.yaml"
    conf.write_text(
        f"runtime:\n  lock_file: {lock}\n  log_file: {tmp_path / 'l.log'}\nprivacy:\n  key_file: {key}\n"
        f"  evidence_dir: {tmp_path / 'ev'}\n",
        encoding="utf-8",
    )
    holder = SingleInstance(lock)
    assert holder.acquire()
    try:
        assert main_module.main(["--mode", "headless", "--config", str(conf)]) == 3
    finally:
        holder.release()
        logging.getLogger().handlers.clear()
    store = EvidenceStore(PrivacyConfig(evidence_dir=str(tmp_path / "ev"), key_file=str(key)))
    saved = store.save(blank(160, 120), "alert")
    assert saved is not None
    out = tmp_path / "out.jpg"
    assert main_module.main(["--config", str(conf), "--decrypt-evidence", str(saved), "--out", str(out)]) == 0
    assert out.read_bytes()[:2] == b"\xff\xd8"
    assert main_module.main(["--config", str(conf), "--decrypt-evidence", str(saved)]) == 2
