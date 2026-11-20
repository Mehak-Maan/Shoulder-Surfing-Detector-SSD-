"""Deterministic replay and quality metrics (false alarms/hour, alert latency)."""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from capture import FrameSource
from pipeline import Pipeline
from state_machine import Transition


@dataclass(frozen=True)
class Window:
    """A labelled interval (seconds) during which a real intruder was present."""

    start: float
    end: float


@dataclass(frozen=True)
class ReplayRun:
    """Raw outcome of replaying a source through the pipeline."""

    transitions: tuple[Transition, ...]
    duration_s: float
    frames: int


@dataclass(frozen=True)
class EvaluationReport:
    """Quality metrics of one replay."""

    duration_s: float
    alarms: int
    false_alarms: int
    false_alarms_per_hour: float
    fault_alarms: int
    windows: int
    detected_windows: int
    latencies_s: tuple[float, ...]

    @property
    def missed_windows(self) -> int:
        """Labelled intruder windows that never produced an alarm."""
        return self.windows - self.detected_windows

    @property
    def median_latency_s(self) -> float | None:
        """Median alert latency over detected windows."""
        return statistics.median(self.latencies_s) if self.latencies_s else None


def load_windows(path: Path) -> list[Window]:
    """Read ``{"intruder_windows": [[start, end], ...]}`` from a JSON labels file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Window(float(s), float(e)) for s, e in data.get("intruder_windows", [])]


def replay(source: FrameSource, pipeline: Pipeline) -> ReplayRun:
    """Push every packet of ``source`` through ``pipeline`` and collect transitions."""
    transitions: list[Transition] = []
    first: float | None = None
    last = 0.0
    frames = 0
    while (packet := source.read()) is not None:
        result = pipeline.process(packet)
        first = packet.timestamp if first is None else first
        last = packet.timestamp
        frames += 1
        snap = result.snapshot
        if snap.changed:
            transitions.append(Transition(packet.timestamp, snap.previous, snap.state, snap.reason))
    return ReplayRun(tuple(transitions), 0.0 if first is None else last - first, frames)


def evaluate(
    transitions: Sequence[Transition],
    duration_s: float,
    windows: Sequence[Window],
    grace_s: float = 1.0,
) -> EvaluationReport:
    """Score alarms against labelled intruder windows.

    An *alarm* is a transition from a non-escalated state into WARNING/ALERT/LOCKED.
    An alarm is true if it starts inside ``[window.start, window.end + grace_s]``;
    alarms caused by a camera fault are counted separately and never as false alarms.
    """
    alarms = [t for t in transitions if t.state.is_escalated and not t.previous.is_escalated]
    fault = [t for t in alarms if t.reason == "camera_fault"]
    detection_alarms = [t for t in alarms if t.reason != "camera_fault"]

    def inside(t: float, w: Window) -> bool:
        return w.start <= t <= w.end + grace_s

    false_alarms = [a for a in detection_alarms if not any(inside(a.time, w) for w in windows)]
    latencies: list[float] = []
    for w in windows:
        hits = [a.time for a in detection_alarms if inside(a.time, w)]
        if hits:
            latencies.append(min(hits) - w.start)
    hours = duration_s / 3600.0
    return EvaluationReport(
        duration_s=duration_s,
        alarms=len(alarms),
        false_alarms=len(false_alarms),
        false_alarms_per_hour=len(false_alarms) / hours if hours > 0 else 0.0,
        fault_alarms=len(fault),
        windows=len(windows),
        detected_windows=len(latencies),
        latencies_s=tuple(latencies),
    )


def format_report(report: EvaluationReport) -> str:
    """Render ``report`` as plain text."""
    median = report.median_latency_s
    lines = [
        f"Replay duration        : {report.duration_s:.1f} s",
        f"Alarms (total)         : {report.alarms} (camera-fault alarms: {report.fault_alarms})",
        f"False alarms           : {report.false_alarms}  ({report.false_alarms_per_hour:.2f} per hour)",
        f"Intruder windows       : {report.windows} "
        f"(detected {report.detected_windows}, missed {report.missed_windows})",
        "Alert latency (median) : " + ("n/a" if median is None else f"{median:.2f} s"),
    ]
    return "\n".join(lines)
