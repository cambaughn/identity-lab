"""Enrollment session: gate order, pacing, stage progression, save semantics.

Synthetic DetectedFaces and frames only — no camera, no model.
"""

import numpy as np
import pytest

from identity_lab.identity.enrollment import (
    ALL_STAGES,
    MIN_DET_SCORE,
    MIN_FACE_PX,
    MIN_SAMPLE_GAP_S,
    OUTLIER_CHECK_AFTER,
    REQUIRED_STAGES,
    REQUIRED_TARGET,
    SAMPLES_PER_STAGE,
    EnrollmentSession,
    FeedbackCode,
    PoseStage,
    save_enrollment,
)
from identity_lab.identity.errors import DuplicateIdentityError
from identity_lab.identity.store import IdentityStore
from identity_lab.vision.types import DetectedFace

RNG = np.random.default_rng(7)
SHARP_FRAME = RNG.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
BLURRY_FRAME = np.full((480, 640, 3), 128, dtype=np.uint8)  # zero variance
BBOX = (100, 100, 300, 340)  # 200x240 face — comfortably above MIN_FACE_PX


class Ticker:
    """Fake monotonic clock advancing a fixed step per accepted call."""

    def __init__(self, step=MIN_SAMPLE_GAP_S + 0.01):
        self.t = 0.0
        self.step = step

    def __call__(self):
        self.t += self.step
        return self.t


def emb(seed=0, dim=512):
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def face(embedding=None, bbox=BBOX, score=0.9):
    return DetectedFace(
        bbox=bbox, det_score=score, landmarks=None, kps=None,
        embedding=embedding if embedding is not None else emb(),
    )


def session():
    return EnrollmentSession(clock=Ticker())


def fill_required(s, base_seed=100):
    """Accept exactly REQUIRED_TARGET consistent samples."""
    base = emb(base_seed)
    i = 0
    while not s.required_complete:
        jitter = np.random.default_rng(1000 + i).normal(0, 0.02, 512).astype(np.float32)
        v = base + jitter
        v /= np.linalg.norm(v)
        fb = s.process((face(v),), SHARP_FRAME)
        assert fb.code is FeedbackCode.ACCEPTED, fb.code
        i += 1
    return i


# ------------------------------------------------------------------ gates

def test_no_face_rejected():
    fb = session().process((), SHARP_FRAME)
    assert fb.code is FeedbackCode.NO_FACE


def test_multiple_faces_hard_hold():
    fb = session().process((face(), face(emb(2))), SHARP_FRAME)
    assert fb.code is FeedbackCode.MULTIPLE_FACES


def test_small_face_rejected():
    small = face(bbox=(0, 0, MIN_FACE_PX - 1, MIN_FACE_PX - 1))
    fb = session().process((small,), SHARP_FRAME)
    assert fb.code is FeedbackCode.TOO_SMALL


def test_low_confidence_rejected():
    fb = session().process((face(score=MIN_DET_SCORE - 0.05),), SHARP_FRAME)
    assert fb.code is FeedbackCode.LOW_CONFIDENCE


def test_pacing_blocks_adjacent_frames():
    s = EnrollmentSession(clock=Ticker(step=0.01))  # frames 10ms apart
    first = s.process((face(emb(1)),), SHARP_FRAME)
    assert first.code is FeedbackCode.ACCEPTED
    second = s.process((face(emb(1)),), SHARP_FRAME)
    assert second.code is FeedbackCode.PACING
    assert s.accepted_total == 1


def test_missing_embedding_waits():
    f = DetectedFace(bbox=BBOX, det_score=0.9, landmarks=None, kps=None, embedding=None)
    fb = session().process((f,), SHARP_FRAME)
    assert fb.code is FeedbackCode.WAITING_EMBEDDING


def test_blurry_frame_rejected():
    fb = session().process((face(),), BLURRY_FRAME)
    assert fb.code is FeedbackCode.TOO_BLURRY


def test_outlier_rejected_after_warmup():
    s = session()
    base = emb(1)
    for i in range(OUTLIER_CHECK_AFTER):
        v = base + np.random.default_rng(i).normal(0, 0.02, 512).astype(np.float32)
        v /= np.linalg.norm(v)
        assert s.process((face(v),), SHARP_FRAME).code is FeedbackCode.ACCEPTED
    # An (almost surely) orthogonal embedding = a different person.
    stranger = emb(999)
    fb = s.process((face(stranger),), SHARP_FRAME)
    assert fb.code is FeedbackCode.INCONSISTENT
    assert s.accepted_total == OUTLIER_CHECK_AFTER


def test_gate_order_multiple_faces_beats_size():
    tiny = face(bbox=(0, 0, 10, 10))
    fb = session().process((tiny, tiny), SHARP_FRAME)
    assert fb.code is FeedbackCode.MULTIPLE_FACES  # count checked first


# ------------------------------------------------------------------ stages

def test_stage_progression_in_order():
    s = session()
    seen = []
    for _ in range(len(REQUIRED_STAGES) * SAMPLES_PER_STAGE):
        seen.append(s.current_stage)
        assert s.process((face(emb(1)),), SHARP_FRAME).accepted
    expected = [st for st in REQUIRED_STAGES for _ in range(SAMPLES_PER_STAGE)]
    assert seen == expected
    assert s.required_complete
    assert not s.fully_complete            # optional glasses stage remains
    assert s.current_stage is PoseStage.GLASSES


def test_optional_stage_completes_session():
    s = session()
    fill_required(s)
    base = emb(100)  # stay consistent with fill_required's identity
    for i in range(SAMPLES_PER_STAGE):
        jitter = np.random.default_rng(2000 + i).normal(0, 0.02, 512).astype(np.float32)
        v = base + jitter
        v /= np.linalg.norm(v)
        assert s.process((face(v),), SHARP_FRAME).accepted
    assert s.fully_complete
    assert s.process((face(base),), SHARP_FRAME).code is FeedbackCode.COMPLETE
    assert s.accepted_total == REQUIRED_TARGET + SAMPLES_PER_STAGE


def test_stage_statuses_snapshot():
    s = session()
    assert s.process((face(emb(1)),), SHARP_FRAME).accepted
    statuses = {st.stage: st for st in s.stage_statuses()}
    assert statuses[PoseStage.FORWARD].accepted == 1
    assert statuses[PoseStage.LEFT].accepted == 0
    assert statuses[PoseStage.GLASSES].required is False
    assert len(statuses) == len(ALL_STAGES)


def test_thumbnail_captured_on_first_accept():
    s = session()
    assert s.thumbnail_png is None
    assert s.process((face(emb(1)),), SHARP_FRAME).accepted
    assert s.thumbnail_png is not None
    assert bytes(s.thumbnail_png[:8]) == b"\x89PNG\r\n\x1a\n"
    first = s.thumbnail_png
    assert s.process((face(emb(1)),), SHARP_FRAME).accepted
    assert s.thumbnail_png is first  # not replaced


# ------------------------------------------------------------------ saving

@pytest.fixture
def store(tmp_path):
    s = IdentityStore(tmp_path / "identities.db")
    yield s
    s.close()


def test_save_enrollment_persists_everything(store):
    s = session()
    fill_required(s)
    record = save_enrollment(store, "Cam", s, "buffalo_l")
    assert record.display_name == "Cam"
    assert record.sample_count == REQUIRED_TARGET
    assert record.model_id == "buffalo_l"
    samples = store.get_embedding_samples(record.identity_id)
    assert len(samples) == REQUIRED_TARGET
    for stored, original in zip(samples, s.embeddings()):
        assert np.array_equal(stored.embedding, original)
    assert store.get_identity_thumbnail(record.identity_id) == s.thumbnail_png


def test_save_incomplete_session_refused(store):
    s = session()
    assert s.process((face(emb(1)),), SHARP_FRAME).accepted
    with pytest.raises(ValueError):
        save_enrollment(store, "Cam", s, "buffalo_l")
    assert store.list_identities() == []


def test_save_duplicate_name_leaves_no_partial_identity(store):
    s1 = session()
    fill_required(s1)
    save_enrollment(store, "Cam", s1, "buffalo_l")
    s2 = session()
    fill_required(s2, base_seed=200)
    with pytest.raises(DuplicateIdentityError):
        save_enrollment(store, "CAM", s2, "buffalo_l")
    assert len(store.list_identities()) == 1


def test_cancel_is_simply_not_saving(store):
    s = session()
    fill_required(s)
    # ... user hits CANCEL: no save call. Nothing may exist in the store.
    assert store.list_identities() == []
