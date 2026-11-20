"""State machine tests on synthetic detection sequences (time based, deterministic)."""

from __future__ import annotations

import dataclasses

import pytest

from config import StateConfig
from state_machine import SecurityStateMachine, State, StateSnapshot


class Sim:
    """Drives a state machine in fixed 0.1 s steps."""

    def __init__(self, cfg: StateConfig | None = None, dt: float = 0.1) -> None:
        self.sm = SecurityStateMachine(cfg or StateConfig())
        self.dt = dt
        self.t = 0.0
        self.snaps: list[StateSnapshot] = []

    def run(self, seconds: float, intruder: bool, camera_ok: bool = True) -> StateSnapshot:
        snap = self.snaps[-1] if self.snaps else self.sm.update(self.t, False)
        for _ in range(round(seconds / self.dt)):
            self.t = round(self.t + self.dt, 6)
            snap = self.sm.update(self.t, intruder, camera_ok)
            self.snaps.append(snap)
        return snap

    def current(self) -> State:
        return self.sm.state

    def entries(self, state: State) -> list[float]:
        return [tr.time for tr in self.sm.history if tr.state is state]

    def visited(self) -> list[State]:
        return [tr.state for tr in self.sm.history]


def test_sustained_intruder_escalates_on_schedule() -> None:
    sim = Sim()
    sim.run(15.0, intruder=True)
    assert sim.visited() == [State.SUSPECT, State.WARNING, State.ALERT, State.LOCKED]
    assert sim.entries(State.SUSPECT)[0] == pytest.approx(0.1)
    assert sim.entries(State.WARNING)[0] == pytest.approx(0.8, abs=0.11)  # 0.7 s debounce
    assert sim.entries(State.ALERT)[0] - sim.entries(State.WARNING)[0] == pytest.approx(2.0, abs=0.11)
    assert sim.entries(State.LOCKED)[0] - sim.entries(State.ALERT)[0] == pytest.approx(4.0, abs=0.11)


def test_brief_pass_by_never_warns() -> None:
    sim = Sim()
    sim.run(0.5, intruder=True)
    sim.run(5.0, intruder=False)
    assert State.WARNING not in sim.visited()
    assert sim.current() is State.SECURE
    assert sim.visited() == [State.SUSPECT, State.SECURE]


def test_flicker_with_short_gaps_still_confirms() -> None:
    sim = Sim()
    for _ in range(10):  # 0.1 s on, 0.2 s off: gaps are below the 0.4 s tolerance
        sim.run(0.1, intruder=True)
        sim.run(0.2, intruder=False)
    assert State.WARNING in sim.visited()


def test_flicker_with_long_gaps_never_confirms() -> None:
    sim = Sim()
    for _ in range(10):  # 0.2 s on, 0.6 s off: each streak restarts
        sim.run(0.2, intruder=True)
        sim.run(0.6, intruder=False)
    assert State.WARNING not in sim.visited()
    assert sim.current() in (State.SECURE, State.SUSPECT)


def test_hysteresis_requires_two_seconds_clear() -> None:
    sim = Sim()
    sim.run(1.0, intruder=True)
    assert sim.current() is State.WARNING
    sim.run(1.0, intruder=False)
    assert sim.current() is State.WARNING
    sim.run(0.1, intruder=True)  # returns briefly: the clear clock restarts
    sim.run(1.9, intruder=False)
    assert sim.current() is State.WARNING  # 1.9 s clear is not enough
    sim.run(0.3, intruder=False)
    assert sim.current() is State.RECOVERY
    sim.run(1.2, intruder=False)
    assert sim.current() is State.SECURE


def test_recovery_interrupted_by_returning_intruder() -> None:
    sim = Sim()
    sim.run(1.0, intruder=True)
    sim.run(2.2, intruder=False)
    assert sim.current() is State.RECOVERY
    sim.run(0.1, intruder=True)
    assert sim.current() is State.SUSPECT
    sim.run(1.0, intruder=True)
    assert sim.current() is State.WARNING


def test_owner_leaves_and_nobody_else_appears() -> None:
    sim = Sim()
    sim.run(30.0, intruder=False)
    assert sim.current() is State.SECURE
    assert not sim.sm.history


def test_warning_cooldown_suppresses_repeat_toasts_but_not_alerts() -> None:
    sim = Sim()
    sim.run(1.0, intruder=True)
    first = [s for s in sim.snaps if s.state is State.WARNING and s.changed]
    assert first[0].fresh_action
    sim.run(4.0, intruder=False)  # RECOVERY then SECURE
    sim.run(1.0, intruder=True)  # second WARNING ~6 s after the first
    second = [s for s in sim.snaps if s.state is State.WARNING and s.changed]
    assert len(second) == 2 and not second[1].fresh_action
    sim.run(4.0, intruder=True)  # ALERT is never suppressed
    alerts = [s for s in sim.snaps if s.state is State.ALERT and s.changed]
    assert alerts[-1].fresh_action
    sim.run(4.0, intruder=False)
    sim.run(20.0, intruder=False)
    sim.run(1.0, intruder=True)  # well past cooldown
    third = [s for s in sim.snaps if s.state is State.WARNING and s.changed]
    assert third[-1].fresh_action


def test_lock_can_be_disabled() -> None:
    sim = Sim(dataclasses.replace(StateConfig(), lock_enabled=False))
    sim.run(30.0, intruder=True)
    assert sim.current() is State.ALERT
    assert State.LOCKED not in sim.visited()


def test_alert_held_when_intruder_briefly_disappears() -> None:
    sim = Sim()
    sim.run(5.0, intruder=True)
    assert sim.current() is State.ALERT
    sim.run(1.5, intruder=False)
    assert sim.current() is State.ALERT


@pytest.mark.parametrize(
    ("policy", "expected"),
    [("warn", State.WARNING), ("blur", State.ALERT), ("lock", State.LOCKED)],
)
def test_camera_fault_policy_after_grace(policy: str, expected: State) -> None:
    cfg = dataclasses.replace(StateConfig(), camera_fault_policy=policy)  # type: ignore[arg-type]
    sim = Sim(cfg)
    sim.run(2.8, intruder=False, camera_ok=False)
    assert sim.current() is State.SECURE  # still within the 3 s grace period
    sim.run(0.4, intruder=False, camera_ok=False)
    assert sim.current() is expected
    assert sim.sm.history[-1].reason == "camera_fault"
    sim.run(1.9, intruder=False, camera_ok=True)  # camera is back: 2 s of valid clear needed
    assert sim.current() is expected
    sim.run(0.3, intruder=False, camera_ok=True)
    assert sim.current() is State.RECOVERY


def test_short_camera_glitch_is_ignored() -> None:
    sim = Sim()
    sim.run(2.0, intruder=False, camera_ok=False)
    sim.run(10.0, intruder=False, camera_ok=True)
    assert not sim.sm.history


def test_time_must_not_go_backwards() -> None:
    sm = SecurityStateMachine(StateConfig())
    sm.update(1.0, False)
    with pytest.raises(ValueError, match="backwards"):
        sm.update(0.5, False)


def test_config_rejects_clear_shorter_than_confirm() -> None:
    from config import ConfigError

    with pytest.raises(ConfigError):
        StateConfig(confirm_s=1.0, clear_s=0.5)
