"""Tracker and stabilizer: association, expiry, promotion, hysteresis, decay.

Scripted observation sequences with fake clocks — deterministic, no camera.
"""

import numpy as np

from identity_lab.identity.matcher import (
    REASON_BELOW_THRESHOLD,
    REASON_MATCH,
    MatchCandidate,
    MatchResult,
)
from identity_lab.tracking.stabilizer import (
    DEMOTE_N,
    PROMOTE_N,
    SWITCH_N,
    IdentityStabilizer,
)
from identity_lab.tracking.tracker import FaceTracker, iou
from identity_lab.vision.types import DetectedFace

BBOX = (100, 100, 300, 300)


def face(bbox=BBOX):
    return DetectedFace(
        bbox=bbox, det_score=0.9, landmarks=None, kps=None,
        embedding=np.ones(4, dtype=np.float32),
    )


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def known(ident="id-cam", name="Cam", score=0.7, second=None):
    return MatchResult(
        is_known=True, identity_id=ident, display_name=name, similarity=score,
        best=MatchCandidate(ident, name, score), second_best=second,
        reason=REASON_MATCH,
    )


def unknown(score=0.2):
    return MatchResult(
        is_known=False, identity_id=None, display_name=None, similarity=score,
        best=None, second_best=None, reason=REASON_BELOW_THRESHOLD,
    )


# ------------------------------------------------------------------ iou

def test_iou_basics():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == (50 / 150)
    assert iou((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0  # degenerate box


# ------------------------------------------------------------------ tracker

def test_track_id_stable_across_small_movement():
    clock = Clock()
    tracker = FaceTracker(clock=clock)
    [(tid1, _)] = tracker.update([face((100, 100, 300, 300))])
    clock.t += 0.2
    [(tid2, _)] = tracker.update([face((120, 110, 320, 310))])  # moved a bit
    assert tid1 == tid2 == "001"


def test_new_face_gets_new_id():
    clock = Clock()
    tracker = FaceTracker(clock=clock)
    tracker.update([face((100, 100, 300, 300))])
    clock.t += 0.2
    pairs = tracker.update(
        [face((100, 100, 300, 300)), face((600, 100, 800, 300))]
    )
    assert [tid for tid, _ in pairs] == ["001", "002"]
    assert tracker.active_count == 2


def test_two_faces_keep_their_ids_in_any_order():
    clock = Clock()
    tracker = FaceTracker(clock=clock)
    tracker.update([face((100, 100, 300, 300)), face((600, 100, 800, 300))])
    clock.t += 0.2
    pairs = tracker.update(
        [face((610, 105, 810, 305)), face((105, 100, 305, 300))]  # swapped order
    )
    by_pos = {p[1].bbox[0]: p[0] for p in pairs}
    assert by_pos[105] == "001"   # left face keeps id 001
    assert by_pos[610] == "002"   # right face keeps id 002


def test_track_expires_after_absence_and_id_not_reused():
    clock = Clock()
    tracker = FaceTracker(expiry_s=1.0, clock=clock)
    [(tid1, _)] = tracker.update([face()])
    assert tid1 == "001"
    clock.t += 2.0
    pairs = tracker.update([face()])  # same spot, but long gone -> new track
    assert tracker.pop_expired() == ["001"]
    assert [tid for tid, _ in pairs] == ["002"]  # never reuse 001


def test_track_survives_brief_absence():
    clock = Clock()
    tracker = FaceTracker(expiry_s=1.0, clock=clock)
    [(tid1, _)] = tracker.update([face()])
    clock.t += 0.5
    tracker.update([])          # one missed update, still within expiry
    clock.t += 0.3
    [(tid2, _)] = tracker.update([face()])
    assert tid2 == tid1
    assert tracker.pop_expired() == []


def test_reset_expires_everything():
    tracker = FaceTracker(clock=Clock())
    tracker.update([face()])
    tracker.reset()
    assert tracker.active_count == 0
    assert tracker.pop_expired() == ["001"]


# ------------------------------------------------------------------ stabilizer

def test_promotion_requires_consecutive_agreement():
    stab = IdentityStabilizer()
    for i in range(PROMOTE_N - 1):
        obs = stab.observe("t1", known(), BBOX)
        assert obs.is_known is False
        assert f"PENDING CAM {i + 1}/{PROMOTE_N}" == obs.reason
    obs = stab.observe("t1", known(), BBOX)
    assert obs.is_known is True
    assert obs.display_name == "Cam"
    assert obs.reason == "CONFIRMED"


def test_mixed_observations_do_not_promote():
    stab = IdentityStabilizer()
    stab.observe("t1", known(), BBOX)
    stab.observe("t1", unknown(), BBOX)          # breaks the streak
    obs = stab.observe("t1", known(), BBOX)
    assert obs.is_known is False
    assert obs.reason == f"PENDING CAM 1/{PROMOTE_N}"


def test_flicker_does_not_drop_confirmed_name():
    stab = IdentityStabilizer()
    for _ in range(PROMOTE_N):
        stab.observe("t1", known(), BBOX)
    for i in range(DEMOTE_N - 1):
        obs = stab.observe("t1", unknown(), BBOX)
        assert obs.is_known is True              # held through the flicker
        assert obs.display_name == "Cam"
        assert obs.reason == f"HOLD {i + 1}/{DEMOTE_N}"
    obs = stab.observe("t1", known(), BBOX)      # match returns
    assert obs.is_known and obs.reason == "CONFIRMED"


def test_decay_to_unknown_after_sustained_misses():
    stab = IdentityStabilizer()
    for _ in range(PROMOTE_N):
        stab.observe("t1", known(), BBOX)
    for _ in range(DEMOTE_N - 1):
        stab.observe("t1", unknown(), BBOX)
    obs = stab.observe("t1", unknown(), BBOX)
    assert obs.is_known is False
    assert obs.reason == "DECAYED TO UNKNOWN"
    assert obs.display_name is None


def test_hysteresis_against_name_switching():
    stab = IdentityStabilizer()
    for _ in range(PROMOTE_N):
        stab.observe("t1", known("id-cam", "Cam"), BBOX)
    for i in range(SWITCH_N - 1):
        obs = stab.observe("t1", known("id-ale", "Ale"), BBOX)
        assert obs.display_name == "Cam"         # still Cam while Ale builds up
        assert f"CANDIDATE ALE {i + 1}/{SWITCH_N}" in obs.reason
    obs = stab.observe("t1", known("id-ale", "Ale"), BBOX)
    assert obs.display_name == "Ale"
    assert obs.reason == "SWITCHED"


def test_switch_candidate_streak_broken_by_original():
    stab = IdentityStabilizer()
    for _ in range(PROMOTE_N):
        stab.observe("t1", known("id-cam", "Cam"), BBOX)
    stab.observe("t1", known("id-ale", "Ale"), BBOX)
    stab.observe("t1", known("id-cam", "Cam"), BBOX)  # original reasserts
    obs = stab.observe("t1", known("id-ale", "Ale"), BBOX)
    assert obs.display_name == "Cam"
    assert f"CANDIDATE ALE 1/{SWITCH_N}" in obs.reason  # streak restarted


def test_forget_resets_track_state():
    stab = IdentityStabilizer()
    for _ in range(PROMOTE_N):
        stab.observe("t1", known(), BBOX)
    stab.forget("t1")
    obs = stab.observe("t1", known(), BBOX)
    assert obs.is_known is False                 # must re-earn promotion
    assert obs.reason == f"PENDING CAM 1/{PROMOTE_N}"


def test_tracks_are_independent():
    stab = IdentityStabilizer()
    for _ in range(PROMOTE_N):
        stab.observe("t1", known("id-cam", "Cam"), BBOX)
    obs2 = stab.observe("t2", known("id-ale", "Ale"), BBOX)
    assert obs2.is_known is False                # t2 starts from scratch
    obs1 = stab.observe("t1", known("id-cam", "Cam"), BBOX)
    assert obs1.display_name == "Cam"


def test_observation_carries_raw_scores_and_metadata():
    stab = IdentityStabilizer()
    second = MatchCandidate("id-ale", "Ale", 0.31)
    obs = stab.observe("t9", known(score=0.66, second=second), (1, 2, 3, 4))
    assert obs.track_id == "t9"
    assert obs.similarity == 0.66
    assert obs.second_best_similarity == 0.31
    assert obs.bbox == (1, 2, 3, 4)
    assert obs.timestamp is not None
    assert obs.is_known is False                 # pending, not yet promoted


def test_no_match_data_observation():
    stab = IdentityStabilizer()
    obs = stab.observe("t1", None, BBOX)
    assert obs.is_known is False
    assert obs.similarity is None
    assert obs.reason == "NO MATCH DATA"
