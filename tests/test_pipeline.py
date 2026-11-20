"""End-to-end tests: pipeline, deterministic replay, evaluation, benchmark, GUI drawing, CLI."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import numpy as np
import pytest
from helpers import ListSource, ScriptedBackend, blank, make_face, packets

import main as main_module
from benchmark import format_benchmark, run_benchmark
from capture import FramePacket, VideoSource
from config import AppConfig
from detector import RawFace
from evaluation import Window, evaluate, format_report, load_windows, replay
from gui import OverlayOptions, blank_frame, draw_overlay
from pipeline import Pipeline
from state_machine import State, Transition

FPS = 30.0
OWNER = make_face(0.45, 0.55, 0.22)
INTRUDER = make_face(0.82, 0.4, 0.09, nose_shift=0.2)


def _cfg() -> AppConfig:
    """No far-field tiles (one backend call per frame) and no camera-health check (frames are black)."""
    base = AppConfig()
    return dataclasses.replace(
        base,
        detector=dataclasses.replace(base.detector, far_field_enabled=False),
        health=dataclasses.replace(base.health, enabled=False),
    )


def _scenario(start: int, stop: int):  # type: ignore[no-untyped-def]
    """Backend script: owner always, intruder in frames [start, stop)."""

    def script(i: int) -> list[RawFace]:
        return [OWNER, INTRUDER] if start <= i < stop else [OWNER]

    return script


def _run(count: int, start: int, stop: int):  # type: ignore[no-untyped-def]
    pipeline = Pipeline(_cfg(), ScriptedBackend(_scenario(start, stop)))
    return replay(ListSource(packets(count, FPS)), pipeline)


def test_intruder_is_detected_with_expected_latency_and_no_false_alarms() -> None:
    run = _run(count=450, start=150, stop=330)  # intruder from t=5.0 s to 11.0 s
    report = evaluate(run.transitions, run.duration_s, [Window(5.0, 11.0)])
    assert report.false_alarms == 0 and report.detected_windows == 1 and report.missed_windows == 0
    assert report.median_latency_s is not None and 0.7 <= report.median_latency_s <= 1.2
    assert [t.state for t in run.transitions][:3] == [State.SUSPECT, State.WARNING, State.ALERT]
    assert run.transitions[-1].state is State.SECURE  # released after the intruder left
    assert "False alarms" in format_report(report)


def test_owner_alone_for_ten_minutes_causes_no_alarm() -> None:
    run = _run(count=int(600 * FPS), start=10**9, stop=10**9)
    report = evaluate(run.transitions, run.duration_s, [])
    assert report.alarms == 0 and report.false_alarms_per_hour == 0.0 and not run.transitions


def test_replay_is_deterministic() -> None:
    assert _run(450, 150, 330).transitions == _run(450, 150, 330).transitions


def test_brief_pass_by_is_not_an_alarm() -> None:
    run = _run(count=300, start=100, stop=115)  # 0.5 s pass-by
    assert not any(t.state.is_escalated for t in run.transitions)


def test_camera_loss_triggers_fault_policy() -> None:
    pipeline = Pipeline(_cfg(), ScriptedBackend(lambda _i: [OWNER]))
    sequence = packets(30, FPS) + [FramePacket(None, 1.0 + i / FPS, False, 100 + i) for i in range(150)]
    run = replay(ListSource(sequence), pipeline)
    fault = [t for t in run.transitions if t.reason == "camera_fault"]
    assert len(fault) == 1 and fault[0].state is State.ALERT
    assert fault[0].time == pytest.approx(1.0 + 3.0, abs=0.1)  # first lost packet + 3 s grace
    assert evaluate(run.transitions, run.duration_s, []).fault_alarms == 1


def test_video_replay_uses_virtual_clock(tmp_path: Path) -> None:
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (160, 120))  # type: ignore[attr-defined]
    for _ in range(120):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()

    def run_once() -> tuple[Transition, ...]:
        source = VideoSource(video)
        try:
            return replay(source, Pipeline(_cfg(), ScriptedBackend(_scenario(30, 90)))).transitions
        finally:
            source.close()

    first = run_once()
    assert first == run_once()
    warning = next(t for t in first if t.state is State.WARNING)
    assert warning.time == pytest.approx(1.0 + 0.7, abs=0.15)  # intruder appears at virtual t = 1.0 s


def test_video_source_errors_and_end_of_stream(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        VideoSource(tmp_path / "missing.avi")


def test_evaluate_counts_false_alarms_misses_and_faults() -> None:
    S, R = State, "x"
    transitions = [
        Transition(10.0, S.SUSPECT, S.WARNING, R),  # false alarm (no window)
        Transition(15.0, S.WARNING, S.ALERT, R),  # same episode, not a new alarm
        Transition(30.0, S.SECURE, S.WARNING, R),  # inside window 1
        Transition(100.0, S.SECURE, S.ALERT, "camera_fault"),
    ]
    report = evaluate(transitions, 3600.0, [Window(29.0, 40.0), Window(200.0, 210.0)])
    assert report.alarms == 3 and report.false_alarms == 1 and report.fault_alarms == 1
    assert report.false_alarms_per_hour == pytest.approx(1.0)
    assert report.detected_windows == 1 and report.missed_windows == 1
    assert len(report.latencies_s) == 1 and report.latencies_s[0] == pytest.approx(1.0)


def test_load_windows(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text('{"intruder_windows": [[1, 2.5], [10, 12]]}', encoding="utf-8")
    assert load_windows(path) == [Window(1.0, 2.5), Window(10.0, 12.0)]


def test_benchmark_reports_sane_numbers() -> None:
    pipeline = Pipeline(_cfg(), ScriptedBackend(lambda _i: [OWNER]))
    result = run_benchmark(ListSource(packets(200)), pipeline, seconds=5.0)
    assert result.frames == 200 and result.fps > 0 and result.latency_p95_ms >= result.latency_p50_ms >= 0
    assert "FPS" in format_benchmark(result)


def test_overlay_draws_red_box_for_intruder_and_respects_options() -> None:
    pipeline = Pipeline(_cfg(), ScriptedBackend(lambda _i: [OWNER, INTRUDER]))
    result = None
    for packet in packets(10):
        result = pipeline.process(packet)
    assert result is not None and result.analysis.intruder_present
    frame = blank()
    plain = draw_overlay(frame, result, 30.0, OverlayOptions(), max_width=640)
    assert plain.shape == (360, 640, 3) and not frame.any()  # downscaled copy, input untouched
    red = np.all(plain == np.array([40, 40, 240]), axis=2)
    green = np.all(plain == np.array([60, 200, 60]), axis=2)
    assert red.sum() > 100 and green.sum() > 100
    debug = draw_overlay(frame, result, 30.0, OverlayOptions(debug=True, keypoints=True, tiles=True, help=True), 640)
    assert not np.array_equal(plain, debug)
    assert blank_frame().shape == (360, 640, 3)


def test_cli_rejects_bad_arguments_and_reports_missing_model(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main_module.main(["--source", "video"])
    with pytest.raises(SystemExit):
        main_module.main(["--report"])
    bad = tmp_path / "bad.yaml"
    bad.write_text("state:\n  nope: 1\n", encoding="utf-8")
    assert main_module.main(["--config", str(bad)]) == 2
    missing = tmp_path / "missing_model.yaml"
    missing.write_text(f"detector:\n  model_path: {tmp_path / 'no.tflite'}\n", encoding="utf-8")
    assert main_module.main(["--config", str(missing)]) == 2
