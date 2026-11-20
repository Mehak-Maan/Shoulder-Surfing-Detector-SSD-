"""Time-based security state machine.

States::

    SECURE -> SUSPECT -> WARNING -> ALERT -> LOCKED -> RECOVERY -> SECURE

Every decision is based on elapsed *seconds* passed in by the caller (never on
frame counts), so behaviour is identical at 8 FPS and 30 FPS and a recorded
video replays deterministically.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum

from config import StateConfig

logger = logging.getLogger(__name__)


class State(Enum):
    """Security states, ordered by escalation (RECOVERY sits outside the ladder)."""

    SECURE = "SECURE"
    SUSPECT = "SUSPECT"
    WARNING = "WARNING"
    ALERT = "ALERT"
    LOCKED = "LOCKED"
    RECOVERY = "RECOVERY"

    @property
    def level(self) -> int:
        """Escalation level: 0 secure/recovery, 1 suspect, 2 warning, 3 alert, 4 locked."""
        return _LEVELS[self]

    @property
    def is_escalated(self) -> bool:
        """True for WARNING, ALERT and LOCKED (a user-visible reaction is active)."""
        return self.level >= 2


_LEVELS = {
    State.SECURE: 0,
    State.RECOVERY: 0,
    State.SUSPECT: 1,
    State.WARNING: 2,
    State.ALERT: 3,
    State.LOCKED: 4,
}
_POLICY_TARGET = {"warn": State.WARNING, "blur": State.ALERT, "lock": State.LOCKED}


@dataclass(frozen=True)
class Transition:
    """One state change, kept for logging and evaluation."""

    time: float
    previous: State
    state: State
    reason: str


@dataclass(frozen=True)
class StateSnapshot:
    """What the caller learns from one :meth:`SecurityStateMachine.update`."""

    state: State
    previous: State
    changed: bool
    reason: str
    time_in_state_s: float
    intruder_for_s: float
    clear_for_s: float
    camera_fault: bool
    fresh_action: bool


class SecurityStateMachine:
    """Debounced, hysteretic escalation ladder.

    * Debounce: an intruder streak must last ``confirm_s`` before WARNING. Detection
      dropouts shorter than ``gap_tolerance_s`` do not break a streak (flicker).
    * Hysteresis: escalations are released only after ``clear_s`` of continuous,
      camera-valid absence; a returning intruder restarts that clock.
    * Cooldown: ``fresh_action`` is False for a WARNING entered within
      ``cooldown_s`` of the previous WARNING, so toasts are not spammed.
      ALERT and LOCKED are never suppressed.
    * Camera fault: an unusable camera for ``camera_fault_grace_s`` jumps straight
      to the state named by ``camera_fault_policy`` (an extra edge on the ladder).
    """

    def __init__(self, cfg: StateConfig, start_time: float = 0.0) -> None:
        self._cfg = cfg
        self._start_time = start_time
        self.history: deque[Transition] = deque(maxlen=512)
        self.reset(start_time)

    def reset(self, now: float | None = None) -> None:
        """Return to SECURE and forget all timers."""
        t = self._start_time if now is None else now
        self._state = State.SECURE
        self._entered_at = t
        self._last_now = t
        self._streak_start: float | None = None
        self._last_seen: float | None = None
        self._clear_anchor = t
        self._fault_since: float | None = None
        self._last_warning_at: float | None = None

    @property
    def state(self) -> State:
        """The current state."""
        return self._state

    def update(self, now: float, intruder_present: bool, camera_ok: bool = True) -> StateSnapshot:
        """Advance the machine to time ``now`` (seconds, non-decreasing)."""
        if now < self._last_now:
            raise ValueError(f"time went backwards: {now} < {self._last_now}")
        self._last_now = now
        cfg = self._cfg

        if not camera_ok:
            if self._fault_since is None:
                self._fault_since = now
            self._streak_start = None
            self._last_seen = None
        else:
            if self._fault_since is not None:
                self._fault_since = None
                self._clear_anchor = now
            if intruder_present:
                if self._streak_start is None:
                    self._streak_start = now
                self._last_seen = now
                self._clear_anchor = now
            elif (
                self._streak_start is not None
                and self._last_seen is not None
                and now - self._last_seen > cfg.gap_tolerance_s
            ):
                self._streak_start = None

        # Observed span of the streak: bridged dropouts must not inflate the debounce timer.
        intruder_for = (
            0.0
            if self._streak_start is None or self._last_seen is None
            else self._last_seen - self._streak_start
        )
        active = self._streak_start is not None
        clear_for = now - self._clear_anchor if camera_ok and not intruder_present else 0.0
        fault_for = 0.0 if self._fault_since is None else now - self._fault_since
        fault_active = fault_for >= cfg.camera_fault_grace_s
        in_state = now - self._entered_at

        target, reason = self._next_state(in_state, intruder_present, active, intruder_for, clear_for, fault_active)
        previous = self._state
        fresh = False
        if target is not None:
            fresh = self._enter(target, now, reason)
        return StateSnapshot(
            state=self._state,
            previous=previous,
            changed=target is not None,
            reason=reason if target is not None else "",
            time_in_state_s=now - self._entered_at,
            intruder_for_s=intruder_for,
            clear_for_s=clear_for,
            camera_fault=fault_active,
            fresh_action=fresh,
        )

    def _next_state(
        self,
        in_state: float,
        intruder_present: bool,
        active: bool,
        intruder_for: float,
        clear_for: float,
        fault_active: bool,
    ) -> tuple[State | None, str]:
        cfg = self._cfg
        state = self._state

        if fault_active:
            wanted = _POLICY_TARGET[cfg.camera_fault_policy]
            if state.level < wanted.level:
                return wanted, "camera_fault"
            return None, ""

        if state is State.SECURE:
            if intruder_present:
                return State.SUSPECT, "intruder_seen"
        elif state is State.SUSPECT:
            if not active:
                return State.SECURE, "flicker_cleared"
            if intruder_for >= cfg.confirm_s:
                return State.WARNING, "intruder_confirmed"
        elif state is State.WARNING:
            if clear_for >= cfg.clear_s:
                return State.RECOVERY, "clear"
            if active and in_state >= cfg.warning_to_alert_s:
                return State.ALERT, "intruder_persists"
        elif state is State.ALERT:
            if clear_for >= cfg.clear_s:
                return State.RECOVERY, "clear"
            if cfg.lock_enabled and active and in_state >= cfg.alert_to_lock_s:
                return State.LOCKED, "intruder_persists"
        elif state is State.LOCKED:
            if clear_for >= cfg.clear_s:
                return State.RECOVERY, "clear"
        elif state is State.RECOVERY:
            if intruder_present:
                return State.SUSPECT, "intruder_returned"
            if in_state >= cfg.recovery_s:
                return State.SECURE, "recovered"
        return None, ""

    def _enter(self, target: State, now: float, reason: str) -> bool:
        """Switch state; return whether the entry's side effect should fire."""
        previous = self._state
        self._state = target
        self._entered_at = now
        self.history.append(Transition(now, previous, target, reason))
        logger.info("%s -> %s (%s) at t=%.2f", previous.value, target.value, reason, now)
        if target is State.WARNING:
            fresh = self._last_warning_at is None or now - self._last_warning_at >= self._cfg.cooldown_s
            if fresh:
                self._last_warning_at = now
            return fresh
        return target in (State.ALERT, State.LOCKED)
