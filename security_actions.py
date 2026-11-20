"""Escalation ladder: toast warning -> blur overlay -> OS lock (cross-platform).

:class:`ActionExecutor` reacts to state-machine snapshots. Side effects live behind two
small protocols (:class:`Platform`, :class:`Overlay`) so the ladder is fully testable and
a *dry-run* mode can log what would happen without touching the desktop.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from typing import Protocol

from capture import FramePacket
from config import ActionsConfig
from detector import Image
from pipeline import PipelineResult
from state_machine import State, StateSnapshot

logger = logging.getLogger(__name__)

Runner = Callable[[Sequence[str]], bool]


class Platform(Protocol):
    """Operating-system side effects."""

    def notify(self, title: str, message: str) -> bool:
        """Show a desktop notification; return success."""
        ...

    def lock_screen(self) -> bool:
        """Lock the session; return success."""
        ...


class Overlay(Protocol):
    """Something that can cover the screen."""

    visible: bool

    def show(self) -> None:
        """Cover the screen."""
        ...

    def hide(self) -> None:
        """Uncover the screen."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


def _ps_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def toast_command(os_name: str, title: str, message: str) -> list[str] | None:
    """Build the command that shows a notification on ``os_name`` (``sys.platform`` value)."""
    if os_name == "win32":
        script = (
            "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
            "ContentType = WindowsRuntime] | Out-Null;"
            "$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
            "[Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
            "$n = $t.GetElementsByTagName('text');"
            f"$n.Item(0).AppendChild($t.CreateTextNode({_ps_quote(title)})) | Out-Null;"
            f"$n.Item(1).AppendChild($t.CreateTextNode({_ps_quote(message)})) | Out-Null;"
            "$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe';"
            "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app)"
            ".Show([Windows.UI.Notifications.ToastNotification]::new($t))"
        )
        return ["powershell", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", script]
    if os_name == "darwin":
        script = f"display notification {_applescript(message)} with title {_applescript(title)}"
        return ["osascript", "-e", script]
    if os_name.startswith("linux"):
        return ["notify-send", "--urgency=critical", title, message]
    return None


def _applescript(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def lock_commands(os_name: str) -> list[list[str]]:
    """Commands to try, in order, to lock the session (Windows uses ctypes instead)."""
    if os_name == "darwin":
        return [["pmset", "displaysleepnow"]]
    if os_name.startswith("linux"):
        return [["loginctl", "lock-session"], ["xdg-screensaver", "lock"]]
    return []


def run_command(command: Sequence[str]) -> bool:
    """Run ``command`` quietly with a timeout; False if missing, failing or hanging."""
    if shutil.which(command[0]) is None:
        return False
    try:
        no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        flags = no_window if sys.platform == "win32" else 0
        done = subprocess.run(
            list(command), check=False, timeout=8, capture_output=True, creationflags=flags
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def windows_lock() -> bool:
    """Lock the Windows session with ``LockWorkStation``."""
    import ctypes

    return bool(ctypes.windll.user32.LockWorkStation())  # type: ignore[attr-defined,unused-ignore]


class SystemPlatform:
    """Real implementation of :class:`Platform` for Windows, Linux and macOS."""

    def __init__(
        self,
        os_name: str = sys.platform,
        runner: Runner = run_command,
        win_lock: Callable[[], bool] = windows_lock,
    ) -> None:
        self._os = os_name
        self._run = runner
        self._win_lock = win_lock

    def notify(self, title: str, message: str) -> bool:
        """Show a notification; on unsupported systems only log it."""
        command = toast_command(self._os, title, message)
        if command is None:
            logger.warning("No notification support on %s: %s - %s", self._os, title, message)
            return False
        ok = self._run(command)
        if not ok:
            logger.warning("Notification command failed or is unavailable (%s)", command[0])
        return ok

    def lock_screen(self) -> bool:
        """Lock the session using the first method that works."""
        if self._os == "win32":
            try:
                return self._win_lock()
            except (OSError, AttributeError) as exc:
                logger.error("LockWorkStation failed: %s", exc)
                return False
        for command in lock_commands(self._os):
            if self._run(command):
                return True
        logger.error("No working screen-lock command on %s", self._os)
        return False


class DryRunPlatform:
    """Logs what would happen; never touches the desktop."""

    def notify(self, title: str, message: str) -> bool:
        """Log the notification."""
        logger.info("[dry-run] notification: %s - %s", title, message)
        return True

    def lock_screen(self) -> bool:
        """Log the lock."""
        logger.info("[dry-run] OS screen lock would be triggered")
        return True


class ActionExecutor:
    """Maps state transitions to the escalation ladder.

    * WARNING  -> toast (only when the state machine says the action is *fresh*, i.e. not
      inside the toast cooldown)
    * ALERT    -> blur overlay
    * LOCKED   -> OS lock, at most once per lock episode and at most every
      ``relock_min_interval_s`` seconds
    * RECOVERY -> overlay removed, lock episode ended
    * camera fault -> a notification that explains the reaction
    """

    def __init__(
        self,
        cfg: ActionsConfig,
        platform: Platform,
        overlay: Overlay,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = cfg
        self._platform = platform
        self._overlay = overlay
        self._clock = clock
        self._locked = False
        self._last_lock_at: float | None = None

    def reset(self) -> None:
        """Remove the overlay and forget the lock episode (used when the service is paused)."""
        if self._overlay.visible:
            self._overlay.hide()
        self._locked = False

    @property
    def locked(self) -> bool:
        """True while a lock triggered by this executor is considered active."""
        return self._locked

    def handle(self, snap: StateSnapshot) -> list[str]:
        """React to ``snap``; return the names of the actions that were performed."""
        if not snap.changed:
            return []
        done: list[str] = []
        state = snap.state
        if state is State.RECOVERY or state is State.SECURE:
            if self._overlay.visible:
                self._overlay.hide()
                done.append("overlay_hidden")
            self._locked = False
            return done
        if not snap.fresh_action:
            return done
        fault = snap.reason == "camera_fault"
        if state is State.WARNING or (fault and state is State.ALERT):
            if self._cfg.toast_enabled:
                message = self._cfg.fault_message if fault else self._cfg.toast_message
                if self._platform.notify(self._cfg.toast_title, message):
                    done.append("toast")
        if state in (State.ALERT, State.LOCKED) and self._cfg.blur_enabled and not self._overlay.visible:
            self._overlay.show()
            done.append("overlay_shown")
        if state is State.LOCKED and self._may_lock():
            if self._platform.lock_screen():
                self._locked = True
                self._last_lock_at = self._clock()
                done.append("lock")
        return done

    def _may_lock(self) -> bool:
        if self._locked:
            logger.info("Session already locked by us; not locking again")
            return False
        if self._last_lock_at is not None:
            age = self._clock() - self._last_lock_at
            if age < self._cfg.relock_min_interval_s:
                logger.info("Lock suppressed: last lock was %.0f s ago", age)
                return False
        return True


class EvidenceSink(Protocol):
    """Stores a snapshot of the frame that triggered an alarm."""

    def save(self, image: Image, label: str) -> object:
        """Store ``image`` under ``label``."""
        ...


class ResponseHandler:
    """Runs the escalation ladder and, if enabled, stores evidence for every fresh ALERT/LOCKED."""

    def __init__(self, executor: ActionExecutor | None, evidence: EvidenceSink | None = None) -> None:
        self._executor = executor
        self._evidence = evidence

    def reset(self) -> None:
        """Undo visible reactions (overlay) and forget the lock episode."""
        if self._executor is not None:
            self._executor.reset()

    def __call__(self, packet: FramePacket, result: PipelineResult) -> list[str]:
        """Handle one pipeline result; return the actions performed."""
        snap = result.snapshot
        done = self._executor.handle(snap) if self._executor is not None else []
        if (
            self._evidence is not None
            and snap.changed
            and snap.fresh_action
            and snap.state in (State.ALERT, State.LOCKED)
            and packet.image is not None
        ):
            self._evidence.save(packet.image, snap.state.value.lower())
            done.append("evidence")
        return done
