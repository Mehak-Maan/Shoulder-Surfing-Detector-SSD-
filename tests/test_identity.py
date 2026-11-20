"""Phase 4 tests: gaze filter, embeddings, owner profile, verification, enrollment."""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
from helpers import ScriptedBackend, make_face
from scene import W, Person, build_scene, frame_with, roll_for

import main as main_module
from capture import FramePacket
from config import AppConfig, ConfigError, DetectorConfig, GazeConfig, IdentityConfig, load_config
from detector import IdentityStatus, Image, RawFace, Role
from enrollment import EnrollmentSession, run_enrollment
from gaze import GazeFilter, estimate_pitch_deg
from identity import (
    LbpEmbedder,
    OwnerProfile,
    OwnerVerifier,
    ProfileError,
    Vector,
    align_face,
    create_embedder,
    load_profile,
    save_profile,
)
from pipeline import Pipeline
from state_machine import State

EMB = LbpEmbedder()
ID_CFG = IdentityConfig(enabled=True)
OWNER, STRANGER = 1, 2


# ------------------------------------------------------------------ gaze
def test_gaze_smoothing_and_hysteresis_prevent_flicker() -> None:
    det, gaze = DetectorConfig(), GazeConfig()
    f = GazeFilter(det, gaze)
    frontal = make_face(0.5, 0.5, 0.2).keypoints
    assert f.update(1, frontal, 128).facing
    # nose_shift 0.5 -> about 56 degrees (> 55): one frame is smoothed away, a sustained turn is not
    turned = make_face(0.5, 0.5, 0.2, nose_shift=0.55).keypoints
    assert f.update(1, turned, 128).facing
    verdicts = [f.update(1, turned, 128).facing for _ in range(6)]
    assert verdicts[-1] is False
    # hovering just under the limit does not flip back (hysteresis), a clear return does
    almost = make_face(0.5, 0.5, 0.2, nose_shift=0.46).keypoints
    assert [f.update(1, almost, 128).facing for _ in range(10)][-1] is False
    back = make_face(0.5, 0.5, 0.2).keypoints
    assert [f.update(1, back, 128).facing for _ in range(15)][-1] is True


def test_gaze_unknown_pose_keeps_verdict_and_disabled_filter_always_faces() -> None:
    det = DetectorConfig()
    f = GazeFilter(det, GazeConfig())
    turned = make_face(0.5, 0.5, 0.2, nose_shift=0.55).keypoints
    for _ in range(8):
        f.update(7, turned, 128)
    assert f.update(7, None, 128).facing is False  # no keypoints this frame: previous verdict stays
    off = GazeFilter(det, dataclasses.replace(GazeConfig(), enabled=False))
    assert all(off.update(1, turned, 128).facing for _ in range(10))
    f.prune({1})
    assert f.update(7, None, 128).facing is True  # state of pruned tracks is gone


def test_pitch_estimate_and_optional_pitch_filter() -> None:
    det, gaze = DetectorConfig(), dataclasses.replace(GazeConfig(), use_pitch=True)
    frontal = make_face(0.5, 0.5, 0.2).keypoints
    assert frontal is not None
    assert estimate_pitch_deg(frontal, 128, det, gaze) is not None
    assert estimate_pitch_deg(None, 128, det, gaze) is None
    assert estimate_pitch_deg(frontal, 10, det, gaze) is None
    assert estimate_pitch_deg(frontal[:2], 128, det, gaze) is None
    assert estimate_pitch_deg(frontal, 128, det, gaze) == pytest.approx(
        abs(np.degrees(np.arcsin((0.15 / 0.35 - 0.45) / 0.3))), abs=1e-3
    )
    assert GazeFilter(det, gaze).update(1, frontal, 128).pitch_deg is not None


def test_new_config_sections_validate_and_load(tmp_path: Path) -> None:
    assert not AppConfig().identity.enabled and AppConfig().gaze.enabled
    bad: list[Callable[[], object]] = [
        lambda: GazeConfig(smoothing_alpha=0),
        lambda: IdentityConfig(enroll_samples=2),
        lambda: IdentityConfig(min_samples=9, window=5),
        lambda: IdentityConfig(threshold_floor=0.9, threshold_ceiling=0.8),
        lambda: IdentityConfig(enroll_min_face_px=20, min_embed_px=48),
    ]
    for build in bad:
        with pytest.raises(ConfigError):
            build()
    path = tmp_path / "c.yaml"
    path.write_text("identity:\n  enabled: true\n  embedder: onnx\ngaze:\n  use_pitch: true\n", encoding="utf-8")
    loaded = load_config(path)
    assert loaded.identity.enabled and loaded.identity.embedder == "onnx" and loaded.gaze.use_pitch
    path.write_text("identity:\n  embedder: resnet\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


# ------------------------------------------------------------------ embeddings
def _vec(person: Person) -> Vector:
    frame, faces = frame_with([person])
    patch = align_face(frame, faces[0].box, faces[0].keypoints, EMB.spec)
    assert patch is not None and patch.shape[:2] == (64, 64)
    return EMB.embed(patch)


def test_lbp_embedding_is_unit_length_deterministic_and_discriminative() -> None:
    a = _vec(Person(OWNER, 0.5, 0.55, 250))
    assert a.shape == (944,) and np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)
    assert np.array_equal(a, _vec(Person(OWNER, 0.5, 0.55, 250)))
    same = _vec(Person(OWNER, 0.4, 0.45, 140, roll_deg=9))  # other place, size and head roll
    other = _vec(Person(STRANGER, 0.5, 0.55, 250))
    assert float(a @ same) > 0.95 > float(a @ other) + 0.2


def test_lbp_ignores_brightness_and_contrast_changes() -> None:
    frame, faces = frame_with([Person(OWNER, 0.5, 0.55, 250)])
    darker = np.clip(frame.astype(np.float32) * 0.55 + 10, 0, 255).astype(np.uint8)
    patches = [align_face(img, faces[0].box, faces[0].keypoints, EMB.spec) for img in (frame, darker)]
    assert patches[0] is not None and patches[1] is not None
    base, dim = EMB.embed(patches[0]), EMB.embed(patches[1])
    assert float(base @ dim) > 0.95


def test_align_rejects_tiny_faces_and_works_without_keypoints() -> None:
    frame, faces = frame_with([Person(OWNER, 0.5, 0.5, 250)])
    tiny = dataclasses.replace(faces[0], box=dataclasses.replace(faces[0].box, x2=faces[0].box.x1 + 0.001))
    assert align_face(frame, tiny.box, None, EMB.spec) is None
    assert align_face(frame, faces[0].box, None, EMB.spec) is not None
    squeezed = ((0.5, 0.5), (0.5001, 0.5), (0.5, 0.52), (0.5, 0.55), (0.4, 0.5), (0.6, 0.5))
    assert align_face(frame, faces[0].box, squeezed, EMB.spec) is None  # eyes closer than 4 px


def test_create_embedder_and_missing_onnx_model(tmp_path: Path) -> None:
    assert isinstance(create_embedder(IdentityConfig(), tmp_path), LbpEmbedder)
    with pytest.raises(FileNotFoundError):
        create_embedder(IdentityConfig(embedder="onnx", onnx_model="nope.onnx"), tmp_path)


# ------------------------------------------------------------------ profile
def _owner_vectors(n: int = 12) -> list[Vector]:
    return [
        _vec(Person(OWNER, 0.5 + 0.01 * (i % 4), 0.55, 220 + 8 * i, roll_deg=3.0 * (i % 5) - 6)) for i in range(n)
    ]


def test_profile_threshold_is_calibrated_and_clipped() -> None:
    vectors = _owner_vectors()
    profile = OwnerProfile.build(vectors, EMB.name, ID_CFG)
    assert 0.5 <= profile.threshold <= 0.95 and profile.threshold > 0.8
    assert profile.score(vectors[0]) > profile.threshold > profile.score(_vec(Person(STRANGER, 0.5, 0.55, 250)))
    strict = OwnerProfile.build(vectors, EMB.name, dataclasses.replace(ID_CFG, threshold_margin=0.0))
    assert strict.threshold <= ID_CFG.threshold_ceiling
    with pytest.raises(ProfileError):
        OwnerProfile.build(vectors[:3], EMB.name, ID_CFG)


def test_profile_roundtrip_is_encrypted_and_validated(tmp_path: Path) -> None:
    key, path = tmp_path / "k.key", tmp_path / "owner.enc"
    profile = OwnerProfile.build(_owner_vectors(), EMB.name, ID_CFG)
    save_profile(profile, path, key)
    assert b"NUMPY" not in path.read_bytes()
    if sys.platform != "win32":
        assert path.stat().st_mode & 0o077 == 0  # owner-only permissions
    loaded = load_profile(path, key, EMB.name)
    assert np.allclose(loaded.vectors, profile.vectors) and loaded.threshold == pytest.approx(profile.threshold)
    with pytest.raises(ProfileError, match="different|configured"):
        load_profile(path, key, "onnx-other")
    with pytest.raises(ProfileError, match="--enroll"):
        load_profile(tmp_path / "missing.enc", key, EMB.name)
    other_key = tmp_path / "other.key"
    save_profile(profile, tmp_path / "x.enc", other_key)
    with pytest.raises(ProfileError):
        load_profile(path, other_key, EMB.name)
    path.write_bytes(b"garbage")
    with pytest.raises(ProfileError):
        load_profile(path, key, EMB.name)


# ------------------------------------------------------------------ verifier
def _verifier(cfg: IdentityConfig = ID_CFG) -> OwnerVerifier:
    return OwnerVerifier(cfg, OwnerProfile.build(_owner_vectors(), EMB.name, cfg), EMB)


def test_verifier_status_flow_owner_other_pending_and_small() -> None:
    from detector import IoUTracker

    cfg = ID_CFG
    verifier = _verifier(cfg)
    tracker = IoUTracker(AppConfig().tracker)
    owner, stranger = Person(OWNER, 0.35, 0.55, 250), Person(STRANGER, 0.7, 0.45, 200)
    small = Person(3, 0.15, 0.2, 40)
    statuses: dict[int, IdentityStatus] = {}
    for i in range(60):
        t = i / 15.0
        frame, faces = frame_with([owner, stranger, small], np.random.default_rng(i))
        tracks = tracker.update(faces, t)
        verifier.update(tracks, frame, t)
        statuses = {tr.id: verifier.status(tr.id, tr.box.w * W) for tr in tracks}
        if i == 0:
            assert set(statuses.values()) == {IdentityStatus.PENDING, IdentityStatus.UNVERIFIABLE}
    assert sorted(s.value for s in statuses.values()) == ["other", "owner", "unverifiable"]
    owner_id = next(i for i, s in statuses.items() if s is IdentityStatus.OWNER)
    assert (verifier.score(owner_id) or 0) > ID_CFG.threshold_floor
    verifier.reset()
    assert verifier.score(owner_id) is None


def test_verifier_does_not_stay_pending_when_alignment_keeps_failing() -> None:
    from detector import Track

    verifier = _verifier()
    frame, faces = frame_with([Person(OWNER, 0.5, 0.5, 250)])
    bad_kps = ((0.5, 0.5), (0.5001, 0.5), (0.5, 0.52), (0.5, 0.55), (0.4, 0.5), (0.6, 0.5))
    track = Track(id=5, box=faces[0].box, score=0.9, keypoints=bad_kps, first_seen=0.0, last_seen=0.0, hits=5)
    for i in range(30):
        verifier.update([dataclasses.replace(track, last_seen=i / 15)], frame, i / 15.0)
    assert verifier.status(5, 250) is IdentityStatus.OTHER


# ------------------------------------------------------------------ pipeline with identity
def _cfg() -> AppConfig:
    base = AppConfig()
    return dataclasses.replace(
        base,
        detector=dataclasses.replace(base.detector, far_field_enabled=False),
        health=dataclasses.replace(base.health, enabled=False),
        identity=ID_CFG,
    )


def _run(
    spec_fn: Callable[[float], Sequence[Person]], seconds: float, enrolled: bool
) -> tuple[list[State], list[str], list[tuple[Role, str]]]:
    packets, faces = build_scene(spec_fn, seconds)
    cfg = _cfg()
    backend = ScriptedBackend(lambda i: faces[i])
    pipeline = Pipeline(cfg, backend, _verifier() if enrolled else None)
    states: list[State] = []
    reasons: list[str] = []
    roles: list[tuple[Role, str]] = []
    for packet in packets:
        result = pipeline.process(packet)
        states.append(result.snapshot.state)
        reasons.extend(f.reason for f in result.analysis.faces)
        roles.extend((f.role, f.reason) for f in result.analysis.faces)
    return states, reasons, roles


def _sitting(seed: int, t: float) -> Person:
    return Person(seed, 0.5, 0.55, 250, roll_deg=roll_for(t))


def test_enrolled_owner_alone_is_never_an_intruder() -> None:
    states, _, roles = _run(lambda t: [_sitting(OWNER, t)], 8.0, enrolled=True)
    assert set(states) == {State.SECURE}
    assert (Role.OWNER, "owner") in roles and not any(r is Role.INTRUDER for r, _ in roles)


def test_stranger_taking_the_seat_is_caught_only_with_enrollment() -> None:
    # owner for 4 s, away for 12 s (> owner.reacquire_s), then a stranger sits in the same place
    def spec(t: float) -> list[Person]:
        return [_sitting(OWNER, t)] if t < 3 else ([] if t < 14 else [_sitting(STRANGER, t)])

    with_id, reasons, _ = _run(spec, 20.0, enrolled=True)
    assert State.WARNING in with_id and "verifying" in reasons
    without, _, _ = _run(spec, 20.0, enrolled=False)
    assert set(without) == {State.SECURE}  # the Phase 1-3 weakness: the stranger is adopted as the owner


def test_owner_returning_after_a_long_absence_does_not_alarm() -> None:
    def spec(t: float) -> list[Person]:
        return [] if 3 <= t < 14 else [_sitting(OWNER, t)]

    states, _, _ = _run(spec, 20.0, enrolled=True)
    assert set(states) == {State.SECURE}
    without, _, _ = _run(spec, 20.0, enrolled=False)
    assert set(without) == {State.SECURE}  # (without enrollment the owner is re-adopted after reacquire_s)


def test_second_person_behind_enrolled_owner_escalates() -> None:
    def spec(t: float) -> list[Person]:
        people = [_sitting(OWNER, t)]
        if t >= 3:
            people.append(Person(STRANGER, 0.82, 0.4, 110))
        return people

    states, _, roles = _run(spec, 10.0, enrolled=True)
    assert State.WARNING in states and (Role.INTRUDER, "intruder") in roles
    assert (Role.IGNORED, "verifying") in roles  # a new face is not an intruder before it was checked


def test_stranger_turned_away_is_ignored_by_the_gaze_filter() -> None:
    def spec(t: float) -> list[Person]:
        people = [_sitting(OWNER, t)]
        if t >= 3:
            people.append(dataclasses.replace(Person(STRANGER, 0.82, 0.4, 110), nose_shift=0.55))
        return people

    states, _, roles = _run(spec, 9.0, enrolled=True)
    assert set(states) == {State.SECURE} and (Role.IGNORED, "not_facing") in roles


# ------------------------------------------------------------------ enrollment
def _face(person: Person) -> tuple[Image, list[RawFace]]:
    return frame_with([person])


def test_enrollment_session_gives_guidance_and_builds_a_profile() -> None:
    cfg = dataclasses.replace(ID_CFG, enroll_samples=6, enroll_interval_s=0.5)
    session = EnrollmentSession(cfg, DetectorConfig(), EMB)
    frame, faces = _face(Person(OWNER, 0.5, 0.55, 250))
    assert "No face" in session.feed(frame, [], 0.0).message
    assert "More than one" in session.feed(frame, faces + faces, 0.0).message
    far_frame, far_faces = _face(Person(OWNER, 0.5, 0.55, 60))
    assert "closer" in session.feed(far_frame, far_faces, 0.0).message
    side_frame, side_faces = _face(dataclasses.replace(Person(OWNER, 0.5, 0.55, 250), nose_shift=0.5))
    assert "Face the screen" in session.feed(side_frame, side_faces, 0.0).message
    no_kps = [dataclasses.replace(faces[0], keypoints=None)]
    assert "Face the screen" in session.feed(frame, no_kps, 0.0).message
    assert session.feed(frame, faces, 0.0).message == "Sample accepted"
    assert "Hold on" in session.feed(frame, faces, 0.2).message  # spacing between samples
    t = 0.6
    progress = session.feed(frame, faces, t)
    while not progress.done:
        t += 0.6
        progress = session.feed(frame, faces, t)
    assert progress.collected == 6 and "Enough" in session.feed(frame, faces, t + 1).message
    profile = session.build_profile()
    assert profile.vectors.shape == (6, 944) and profile.embedder == EMB.name
    assert "6 samples" in profile.loo_hint


class _FrameSource:
    def __init__(self, frames: list[Image | None]) -> None:
        self._frames = frames
        self._i = 0

    def read(self) -> FramePacket | None:
        if self._i >= len(self._frames):
            return None
        i, self._i = self._i, self._i + 1
        return FramePacket(self._frames[i], i * 0.6, self._frames[i] is not None, i + 1)

    def close(self) -> None:
        return None


def test_run_enrollment_end_to_end_and_incomplete_cases() -> None:
    cfg = dataclasses.replace(_cfg(), identity=dataclasses.replace(ID_CFG, enroll_samples=5))
    frame, faces = _face(Person(OWNER, 0.5, 0.55, 250))
    messages: list[str] = []
    profile = run_enrollment(cfg, _FrameSource([frame] * 12), ScriptedBackend(lambda _i: faces), EMB, messages.append)
    assert profile.vectors.shape[0] == 5 and messages[-1].startswith("[5/5]")
    with pytest.raises(ProfileError, match="incomplete"):
        run_enrollment(cfg, _FrameSource([frame, None, frame]), ScriptedBackend(lambda _i: faces), EMB, messages.append)
    assert "Camera not available" in messages
    slow = dataclasses.replace(cfg, identity=dataclasses.replace(cfg.identity, enroll_timeout_s=1.0))
    with pytest.raises(ProfileError):
        run_enrollment(slow, _FrameSource([frame] * 5), ScriptedBackend(lambda _i: faces), EMB, messages.append,
                       clock=iter([0.0, 0.5, 5.0, 6.0, 7.0, 8.0]).__next__)


# ------------------------------------------------------------------ CLI helpers
def test_cli_forget_owner_and_verifier_loading(tmp_path: Path) -> None:
    profile_path, key = tmp_path / "owner.enc", tmp_path / "k.key"
    conf = tmp_path / "c.yaml"
    conf.write_text(
        f"identity:\n  enabled: true\n  profile_file: {profile_path}\nprivacy:\n  key_file: {key}\n", encoding="utf-8"
    )
    cfg = load_config(conf)
    with pytest.raises(ProfileError, match="--enroll"):
        main_module._build_verifier(cfg)
    assert main_module._build_verifier(AppConfig()) is None
    save_profile(OwnerProfile.build(_owner_vectors(), EMB.name, ID_CFG), profile_path, key)
    assert main_module._build_verifier(cfg) is not None
    assert main_module.main(["--config", str(conf), "--forget-owner"]) == 0
    assert not profile_path.exists()
    assert main_module.main(["--config", str(conf), "--forget-owner"]) == 0  # idempotent
