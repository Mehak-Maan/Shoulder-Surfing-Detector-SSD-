"""Geometry, finder scheduling, tracker and owner/intruder classification tests."""

from __future__ import annotations

import dataclasses

import pytest
from helpers import FRAME_W, ScriptedBackend, blank, make_face

from config import DetectorConfig, OwnerConfig, TrackerConfig
from detector import (
    Box,
    FaceFinder,
    FrameAnalysis,
    IntruderClassifier,
    IoUTracker,
    RawFace,
    Role,
    estimate_yaw_deg,
    nms,
    plan_tiles,
)

DT = 1 / 30


def test_box_iou_and_containment() -> None:
    a = Box(0.0, 0.0, 0.5, 0.5)
    assert a.iou(a) == pytest.approx(1.0)
    assert a.iou(Box(0.5, 0.5, 1.0, 1.0)) == 0.0
    assert a.iou(Box(0.25, 0.0, 0.75, 0.5)) == pytest.approx(1 / 3)
    assert a.overlap_of_smaller(Box(0.1, 0.1, 0.2, 0.2)) == pytest.approx(1.0)


def test_nms_removes_duplicates_and_contained_boxes() -> None:
    big = RawFace(Box(0.4, 0.4, 0.6, 0.6), 0.9)
    dup = RawFace(Box(0.41, 0.41, 0.61, 0.61), 0.8)
    inner = RawFace(Box(0.45, 0.45, 0.52, 0.52), 0.95)
    other = RawFace(Box(0.8, 0.1, 0.9, 0.3), 0.7)
    kept = nms([big, dup, inner, other], 0.3)
    assert inner in kept and other in kept and big not in kept  # higher-score inner wins over big (contained)
    assert dup not in kept


def test_plan_tiles_covers_roi_with_overlap() -> None:
    roi = (0.0, 0.0, 1.0, 0.7)
    tiles = plan_tiles(roi, 3, 2, 0.25)
    assert len(tiles) == 6
    assert all(0.0 <= t.x1 < t.x2 <= 1.0 and 0.0 <= t.y1 < t.y2 <= 0.7 + 1e-9 for t in tiles)
    assert tiles[0].x2 > tiles[1].x1  # neighbouring tiles overlap
    assert min(t.x1 for t in tiles) == 0.0 and max(t.x2 for t in tiles) == pytest.approx(1.0)
    assert max(t.y2 for t in tiles) == pytest.approx(0.7)


# ------------------------------------------------------------------ finder
def _finder_cfg(**kw: object) -> DetectorConfig:
    return dataclasses.replace(DetectorConfig(), **kw)  # type: ignore[arg-type]


def test_far_field_tiles_are_scheduled_round_robin_with_period() -> None:
    backend = ScriptedBackend(lambda _i: [make_face(0.5, 0.5, 0.1)])
    finder = FaceFinder(_finder_cfg(), backend)
    frame = blank()
    runs = [len(finder.find(frame, t).tiles_run) for t in (0.0, 0.01, 0.02, 0.03)]
    assert runs == [2, 2, 2, 0]  # 6 tiles, at most 2 per frame, then nothing is due
    assert backend.calls == 3 * 3 + 1
    assert len(finder.find(frame, 0.35).tiles_run) == 2  # period elapsed -> oldest tiles again


def test_far_field_can_be_disabled() -> None:
    backend = ScriptedBackend(lambda _i: [])
    finder = FaceFinder(_finder_cfg(far_field_enabled=False), backend)
    result = finder.find(blank(), 0.0)
    assert backend.calls == 1 and result.tiles_run == () and result.all_tiles == ()


def test_tile_detections_are_mapped_and_cut_faces_dropped() -> None:
    cfg = _finder_cfg(far_field_cols=2, far_field_rows=1, far_field_roi=(0.0, 0.0, 1.0, 1.0), far_field_overlap=0.0)
    good = RawFace(Box(0.2, 0.3, 0.4, 0.6), 0.9)
    cut_right = RawFace(Box(0.8, 0.3, 1.0, 0.6), 0.9)
    cut_left = RawFace(Box(0.0, 0.3, 0.3, 0.6), 0.9)
    script = {0: [], 1: [cut_right, good], 2: [cut_left]}
    finder = FaceFinder(cfg, ScriptedBackend(lambda i: script.get(i, [])))
    faces = finder.find(blank(), 0.0).faces
    assert len(faces) == 1
    assert faces[0].box.x1 == pytest.approx(0.1) and faces[0].box.x2 == pytest.approx(0.2)  # tile 0 is x in [0, .5]


def test_tile_face_keypoints_are_mapped_to_frame() -> None:
    tile = Box(0.5, 0.0, 1.0, 0.5)
    face = make_face(0.5, 0.5, 0.2).from_tile(tile)
    assert face.box.cx == pytest.approx(0.75) and face.box.cy == pytest.approx(0.25)
    assert face.keypoints is not None and face.keypoints[2][0] == pytest.approx(0.75)


# ------------------------------------------------------------------ tracker
def test_tracker_keeps_stable_id_for_moving_face() -> None:
    tracker = IoUTracker(TrackerConfig())
    ids = set()
    for i in range(60):
        tracks = tracker.update([make_face(0.3 + 0.004 * i, 0.5, 0.15)], i * DT)
        ids |= {t.id for t in tracks}
    assert ids == {1}


def test_tracker_separates_two_faces() -> None:
    tracker = IoUTracker(TrackerConfig())
    for i in range(30):
        tracks = tracker.update([make_face(0.25, 0.5, 0.15), make_face(0.75, 0.5, 0.1)], i * DT)
    assert sorted(t.id for t in tracks) == [1, 2]
    assert all(t.hits == 30 for t in tracks)


def test_tracker_ttl_bridges_short_gaps_only() -> None:
    tracker = IoUTracker(TrackerConfig())
    tracker.update([make_face(0.5, 0.5, 0.15)], 0.0)
    coasting = tracker.update([], 0.5)
    assert [t.id for t in coasting] == [1] and not coasting[0].visible
    back = tracker.update([make_face(0.5, 0.5, 0.15)], 0.55)
    assert [t.id for t in back] == [1] and back[0].visible
    tracker.update([], 1.0)
    new = tracker.update([make_face(0.5, 0.5, 0.15)], 1.7)  # > ttl since last sighting
    assert [t.id for t in new] == [2]


def test_tracker_centroid_fallback_when_iou_is_low() -> None:
    tracker = IoUTracker(TrackerConfig(smoothing=1.0))
    tracker.update([make_face(0.40, 0.5, 0.10)], 0.0)
    jumped = make_face(0.48, 0.5, 0.10)  # IoU ~0.11 < 0.2, centroid within one box
    assert tracker._affinity(Box(0.35, 0.4, 0.45, 0.6), jumped.box) > 0
    tracks = tracker.update([jumped], DT)
    assert [t.id for t in tracks] == [1]


# ------------------------------------------------------------------ yaw
def test_yaw_estimation() -> None:
    cfg = DetectorConfig()
    frontal = make_face(0.5, 0.5, 0.1, nose_shift=0.0)
    assert estimate_yaw_deg(frontal.keypoints, 128, cfg) == pytest.approx(0.0, abs=1e-6)
    turned = make_face(0.5, 0.5, 0.1, nose_shift=0.3)
    yaw = estimate_yaw_deg(turned.keypoints, 128, cfg)
    assert yaw is not None and 25.0 < yaw < 35.0
    profile = make_face(0.5, 0.5, 0.1, nose_shift=1.5)
    assert estimate_yaw_deg(profile.keypoints, 128, cfg) == pytest.approx(90.0)
    assert estimate_yaw_deg(frontal.keypoints, 20, cfg) is None  # face too small for a reliable estimate
    assert estimate_yaw_deg(None, 128, cfg) is None


# ------------------------------------------------------------------ classification
class Rig:
    """Tracker + classifier fed with synthetic faces at 30 FPS."""

    def __init__(self, owner_cfg: OwnerConfig | None = None) -> None:
        self.tracker = IoUTracker(TrackerConfig())
        self.clf = IntruderClassifier(DetectorConfig(), TrackerConfig(), owner_cfg or OwnerConfig())
        self.t = 0.0
        self.last = FrameAnalysis()

    def feed(self, faces: list[RawFace], seconds: float = 0.2) -> FrameAnalysis:
        for _ in range(max(1, round(seconds / DT))):
            self.t += DT
            tracks = self.tracker.update(faces, self.t)
            self.last = self.clf.classify(tracks, self.t, FRAME_W)
        return self.last

    def roles(self) -> list[tuple[str, str]]:
        return [(f.role.value, f.reason) for f in self.last.faces]


OWNER = make_face(0.45, 0.55, 0.22)


def test_largest_face_is_owner_and_second_face_is_intruder() -> None:
    rig = Rig()
    analysis = rig.feed([OWNER, make_face(0.82, 0.4, 0.09)])
    roles = {f.track.id: f.role for f in analysis.faces}
    assert sorted(roles.values(), key=lambda r: r.value) == [Role.INTRUDER, Role.OWNER]
    assert analysis.intruder_present and analysis.owner_id == 1


def test_small_far_face_is_ignored() -> None:
    rig = Rig()
    rig.feed([OWNER, make_face(0.82, 0.4, 0.03)])
    assert ("ignored", "too_small") in rig.roles() and not rig.last.intruder_present


def test_face_looking_away_is_ignored_but_side_angle_is_not() -> None:
    rig = Rig()
    rig.feed([OWNER, make_face(0.82, 0.4, 0.09, nose_shift=0.55)])
    assert ("ignored", "not_facing") in rig.roles()
    rig = Rig()
    rig.feed([OWNER, make_face(0.82, 0.4, 0.09, nose_shift=0.3)])  # ~30 deg: still counts
    assert rig.last.intruder_present


def test_owner_is_sticky_when_intruder_gets_larger() -> None:
    rig = Rig()
    rig.feed([OWNER, make_face(0.82, 0.4, 0.09)])
    rig.feed([make_face(0.45, 0.55, 0.15), make_face(0.8, 0.4, 0.30)])
    assert rig.last.owner_id == 1 and rig.last.intruder_present


def test_duplicate_detection_of_owner_is_not_an_intruder() -> None:
    rig = Rig()
    rig.feed([OWNER, make_face(0.46, 0.56, 0.2)])
    assert not rig.last.intruder_present
    assert ("ignored", "duplicate_of_owner") in rig.roles()


def test_nothing_is_flagged_before_tracks_are_confirmed() -> None:
    rig = Rig()
    rig.feed([OWNER, make_face(0.82, 0.4, 0.09)], seconds=DT)  # a single frame
    assert not rig.last.intruder_present and rig.last.owner_id is None


def test_owner_track_id_switch_is_recovered_by_continuity() -> None:
    rig = Rig()
    rig.feed([OWNER], seconds=1.0)
    rig.feed([], seconds=0.7)  # detector drops the owner longer than the track ttl
    for _ in range(12):
        analysis = rig.feed([OWNER], seconds=DT)
        assert not analysis.intruder_present
    assert rig.last.owner_id == 2  # new track, but still the owner


def test_stranger_sitting_down_after_owner_left_is_an_intruder() -> None:
    rig = Rig()
    rig.feed([OWNER], seconds=1.0)
    rig.feed([], seconds=3.0)
    analysis = rig.feed([make_face(0.46, 0.55, 0.2)], seconds=0.5)
    assert analysis.intruder_present


def test_without_enrollment_largest_face_is_adopted_after_reacquire_time() -> None:
    rig = Rig(OwnerConfig(reacquire_s=10.0))
    rig.feed([OWNER], seconds=1.0)
    rig.feed([], seconds=2.0)
    assert rig.feed([OWNER], seconds=5.0).intruder_present
    assert not rig.feed([OWNER], seconds=6.0).intruder_present
