"""Escalation ladder, platform commands and blur overlay tests (no real desktop is touched)."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import pytest

from config import ActionsConfig, AppConfig, ConfigError, OverlayConfig, StateConfig
from detector import Image
from overlay import BlurOverlay, NullOverlay, Screenshot, make_blurred
from security_actions import (
    ActionExecutor,
    DryRunPlatform,
    SystemPlatform,
    lock_commands,
    toast_command,
)
from state_machine import SecurityStateMachine, State, StateSnapshot


class FakePlatform:
    def __init__(self, lock_ok: bool = True) -> None:
        self.toasts: list[tuple[str, str]] = []
        self.locks = 0
        self._lock_ok = lock_ok

    def notify(self, title: str, message: str) -> bool:
        self.toasts.append((title, message))
        return True

    def lock_screen(self) -> bool:
        self.locks += 1
        return self._lock_ok


class FakeOverlay:
    def __init__(self) -> None:
        self.visible = False
        self.shown = 0
        self.hidden = 0

    def show(self) -> None:
        self.visible = True
        self.shown += 1

    def hide(self) -> None:
        self.visible = False
        self.hidden += 1

    def close(self) -> None:
        self.visible = False


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class Rig:
    """State machine -> executor, stepped at 0.1 s."""

    def __init__(self, state_cfg: StateConfig | None = None, actions: ActionsConfig | None = None) -> None:
        self.sm = SecurityStateMachine(state_cfg or StateConfig())
        self.platform = FakePlatform()
        self.overlay = FakeOverlay()
        self.clock = Clock()
        self.executor = ActionExecutor(actions or ActionsConfig(), self.platform, self.overlay, self.clock)
        self.t = 0.0
        self.done: list[str] = []

    def run(self, seconds: float, intruder: bool, camera_ok: bool = True) -> None:
        for _ in range(round(seconds / 0.1)):
            self.t = round(self.t + 0.1, 6)
            self.clock.now = self.t
            snap = self.sm.update(self.t, intruder, camera_ok)
            self.done += self.executor.handle(snap)


def test_full_ladder_toast_then_blur_then_lock_then_release() -> None:
    rig = Rig()
    rig.run(1.0, intruder=True)
    assert rig.done == ["toast"] and rig.overlay.shown == 0
    rig.run(2.5, intruder=True)
    assert rig.done == ["toast", "overlay_shown"] and rig.overlay.visible and rig.platform.locks == 0
    rig.run(4.5, intruder=True)
    assert rig.done[-1] == "lock" and rig.platform.locks == 1 and rig.executor.locked
    rig.run(10.0, intruder=True)
    assert rig.platform.locks == 1  # never re-locked while the intruder stays
    rig.run(2.5, intruder=False)
    assert not rig.overlay.visible and not rig.executor.locked and rig.done[-1] == "overlay_hidden"


def test_toast_cooldown_and_relock_interval() -> None:
    rig = Rig()
    rig.run(1.0, intruder=True)
    rig.run(4.0, intruder=False)  # back to SECURE
    rig.run(1.0, intruder=True)  # second WARNING inside the 10 s cooldown
    assert len(rig.platform.toasts) == 1
    rig.run(7.0, intruder=True)  # ALERT, then LOCKED (first lock)
    assert rig.platform.locks == 1
    rig.run(3.0, intruder=False)
    rig.run(1.2, intruder=True)  # new episode, still < 30 s after the lock
    rig.run(7.0, intruder=True)
    assert rig.sm.state is State.LOCKED and rig.platform.locks == 1  # suppressed by relock_min_interval_s


def test_lock_not_repeated_for_two_fresh_locked_snapshots() -> None:
    executor = ActionExecutor(ActionsConfig(), FakePlatform(), FakeOverlay(), Clock())
    snap = StateSnapshot(State.LOCKED, State.ALERT, True, "intruder_persists", 0, 5, 0, False, True)
    assert "lock" in executor.handle(snap)
    assert "lock" not in executor.handle(snap)


def test_failed_lock_is_retried_on_next_locked_entry() -> None:
    platform = FakePlatform(lock_ok=False)
    executor = ActionExecutor(ActionsConfig(), platform, FakeOverlay(), Clock())
    snap = StateSnapshot(State.LOCKED, State.ALERT, True, "intruder_persists", 0, 5, 0, False, True)
    assert "lock" not in executor.handle(snap) and not executor.locked
    executor.handle(snap)
    assert platform.locks == 2


@pytest.mark.parametrize(
    ("policy", "toasts", "overlay", "locks"),
    [("warn", 1, False, 0), ("blur", 1, True, 0), ("lock", 0, True, 1)],
)
def test_camera_fault_policies(policy: str, toasts: int, overlay: bool, locks: int) -> None:
    cfg = dataclasses.replace(StateConfig(), camera_fault_policy=policy)  # type: ignore[arg-type]
    rig = Rig(cfg)
    rig.run(3.5, intruder=False, camera_ok=False)
    assert len(rig.platform.toasts) == toasts and rig.overlay.visible is overlay and rig.platform.locks == locks
    if toasts:
        assert rig.platform.toasts[0][1] == ActionsConfig().fault_message


def test_actions_can_be_disabled_individually() -> None:
    rig = Rig(actions=ActionsConfig(toast_enabled=False, blur_enabled=False))
    rig.run(8.0, intruder=True)
    assert rig.platform.toasts == [] and rig.overlay.shown == 0 and rig.platform.locks == 1


def test_dry_run_objects_only_record() -> None:
    overlay = NullOverlay()
    executor = ActionExecutor(ActionsConfig(), DryRunPlatform(), overlay, Clock())
    sm = SecurityStateMachine(StateConfig())
    t = 0.0
    for _ in range(80):
        t = round(t + 0.1, 6)
        executor.handle(sm.update(t, True))
    assert overlay.visible and executor.locked


# ------------------------------------------------------------------ platform commands
def test_toast_commands_per_os() -> None:
    win = toast_command("win32", "T", "it's me")
    assert win is not None and win[0] == "powershell" and "'it''s me'" in win[-1]
    mac = toast_command("darwin", 'a "b"', "m")
    assert mac is not None and mac[0] == "osascript" and '\\"b\\"' in mac[-1]
    assert toast_command("linux", "T", "M") == ["notify-send", "--urgency=critical", "T", "M"]
    assert toast_command("freebsd", "T", "M") is None


def test_lock_commands_per_os() -> None:
    assert lock_commands("linux") == [["loginctl", "lock-session"], ["xdg-screensaver", "lock"]]
    assert lock_commands("darwin") == [["pmset", "displaysleepnow"]]
    assert lock_commands("win32") == []


def test_system_platform_lock_fallback_and_windows_path() -> None:
    calls: list[Sequence[str]] = []

    def runner(cmd: Sequence[str]) -> bool:
        calls.append(cmd)
        return cmd[0] == "xdg-screensaver"

    linux = SystemPlatform("linux", runner)
    assert linux.lock_screen() and [c[0] for c in calls] == ["loginctl", "xdg-screensaver"]
    assert not SystemPlatform("linux", lambda _c: False).lock_screen()
    assert SystemPlatform("win32", runner, lambda: True).lock_screen()

    def broken() -> bool:
        raise OSError("no user32")

    assert not SystemPlatform("win32", runner, broken).lock_screen()
    assert not SystemPlatform("freebsd", runner).notify("t", "m")


# ------------------------------------------------------------------ overlay
def _checker() -> Image:
    pattern = (np.indices((120, 160)).sum(axis=0) % 2) * 255
    return np.dstack([pattern] * 3).astype(np.uint8)


def test_make_blurred_destroys_detail_and_darkens() -> None:
    src = _checker()
    out = make_blurred(src, OverlayConfig(message=""))
    assert out.shape == src.shape and out.dtype == np.uint8
    assert out[:40].std() < 5.0 < src.std()  # text-like high-frequency detail is gone
    assert out[:40].mean() < src.mean()  # darkened
    with_text = make_blurred(src, OverlayConfig(message="X" * 10))
    assert not np.array_equal(with_text, out)  # the message is drawn


class FakeGrabber:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def grab(self) -> Screenshot:
        self._log.append("grab")
        return Screenshot(_checker(), -1920, 0)


class FakeWindow:
    def __init__(self, log: list[str]) -> None:
        self._log = log
        self.args: tuple[int, int] | None = None

    def show(self, image: Image, left: int, top: int) -> None:
        self._log.append("show")
        self.args = (left, top)

    def hide(self) -> None:
        self._log.append("hide")

    def close(self) -> None:
        self._log.append("close")


def test_blur_overlay_grabs_before_showing_and_is_idempotent() -> None:
    log: list[str] = []
    window = FakeWindow(log)
    overlay = BlurOverlay(OverlayConfig(), FakeGrabber(log), window)
    overlay.show()
    overlay.show()
    assert log == ["grab", "show"] and overlay.visible and window.args == (-1920, 0)
    overlay.hide()
    overlay.hide()
    overlay.close()
    assert log == ["grab", "show", "hide", "close"] and not overlay.visible


def test_new_config_sections_validate() -> None:
    assert AppConfig().actions == ActionsConfig() and AppConfig().overlay == OverlayConfig()
    bad_builders = (
        lambda: OverlayConfig(blur_kernel=30),
        lambda: OverlayConfig(darken=1.0),
        lambda: ActionsConfig(toast_title=" "),
    )
    for build in bad_builders:
        with pytest.raises(ConfigError):
            build()
