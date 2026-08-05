"""Vision-layer logic: size filter, scheduler, and inference worker with a
fake engine — no model download, no webcam."""

import numpy as np

from identity_lab.vision.types import (
    DetectedFace,
    InferenceScheduler,
    filter_small_faces,
)
from identity_lab.vision.worker import InferenceWorker, ModelState


def face(x1, y1, x2, y2, score=0.9):
    return DetectedFace(
        bbox=(x1, y1, x2, y2),
        det_score=score,
        landmarks=None,
        kps=None,
        embedding=None,
    )


# -- min-face-size filtering --

def test_filter_uses_smaller_bbox_side():
    wide_but_short = face(0, 0, 300, 40)  # min side 40
    assert filter_small_faces((wide_but_short,), 50) == ()
    assert filter_small_faces((wide_but_short,), 40) == (wide_but_short,)


def test_filter_keeps_large_and_drops_small():
    big, small = face(0, 0, 200, 200), face(0, 0, 30, 30)
    assert filter_small_faces((big, small), 60) == (big,)
    assert filter_small_faces((big, small), 0) == (big, small)


# -- inference scheduling --

def test_scheduler_every_frame():
    s = InferenceScheduler(every_n=1)
    assert [s.should_submit() for _ in range(4)] == [True, True, True, True]


def test_scheduler_every_third():
    s = InferenceScheduler(every_n=3)
    assert [s.should_submit() for _ in range(7)] == [
        True, False, False, True, False, False, True,
    ]


def test_scheduler_change_and_reset():
    s = InferenceScheduler(every_n=2)
    assert s.should_submit() is True
    assert s.should_submit() is False
    s.reset()
    assert s.should_submit() is True


def test_scheduler_invalid_n_treated_as_one():
    s = InferenceScheduler(every_n=0)
    assert [s.should_submit() for _ in range(3)] == [True, True, True]


# -- inference worker (fake engine, synchronous signals) --

class FakeEngine:
    def __init__(self, init_error=None, analyze_error=None, on_analyze=None):
        self.init_error = init_error
        self.analyze_error = analyze_error
        self.on_analyze = on_analyze
        self.analyzed_frames = []
        self.embedding_flags = []

    def initialize(self):
        if self.init_error:
            raise self.init_error

    def analyze(self, frame, with_embeddings=False):
        self.analyzed_frames.append(frame)
        self.embedding_flags.append(with_embeddings)
        if self.on_analyze:
            self.on_analyze(self)
        if self.analyze_error:
            raise self.analyze_error
        return (face(0, 0, 100, 100),)


def run_worker(worker):
    states, errors, results = [], [], []
    worker.model_state_changed.connect(states.append)
    worker.error.connect(errors.append)
    worker.result_ready.connect(results.append)
    worker.run()
    return states, errors, results


def frame_with_marker(marker):
    f = np.zeros((8, 8, 3), dtype=np.uint8)
    f[0, 0, 0] = marker
    return f


def test_worker_init_failure_propagates_and_finishes():
    engine = FakeEngine(init_error=RuntimeError("no model"))
    worker = InferenceWorker(engine_factory=lambda: engine)
    finished = []
    worker.finished.connect(lambda: finished.append(True))
    states, errors, results = run_worker(worker)
    assert states == [ModelState.LOADING.value, ModelState.ERROR.value]
    assert len(errors) == 1 and "no model" in errors[0]
    assert results == []
    assert finished == [True]


def test_worker_processes_latest_submitted_frame_only():
    engine = FakeEngine(on_analyze=lambda e: worker.request_stop())
    worker = InferenceWorker(engine_factory=lambda: engine)
    worker.submit(frame_with_marker(1))
    worker.submit(frame_with_marker(2))  # replaces frame 1 before run starts
    states, errors, results = run_worker(worker)
    assert states == [ModelState.LOADING.value, ModelState.READY.value]
    assert errors == []
    assert len(engine.analyzed_frames) == 1
    assert engine.analyzed_frames[0][0, 0, 0] == 2  # newest frame won
    assert len(results) == 1
    result = results[0]
    assert result.frame_w == 8 and result.frame_h == 8
    assert len(result.faces) == 1
    assert result.latency_ms >= 0


def test_worker_analyze_errors_do_not_kill_until_threshold():
    calls = []

    def failing_analyze(engine):
        calls.append(1)
        if len(calls) < InferenceWorker.MAX_CONSECUTIVE_FAILURES:
            worker.submit(frame_with_marker(len(calls)))  # keep feeding

    engine = FakeEngine(analyze_error=ValueError("bad frame"), on_analyze=failing_analyze)
    worker = InferenceWorker(engine_factory=lambda: engine)
    worker.submit(frame_with_marker(0))
    states, errors, results = run_worker(worker)
    assert len(calls) == InferenceWorker.MAX_CONSECUTIVE_FAILURES
    assert states[-1] == ModelState.ERROR.value
    assert any("repeatedly" in e for e in errors)
    assert results == []


def test_worker_per_frame_embeddings_flag_reaches_engine():
    engine = FakeEngine(on_analyze=lambda e: worker.request_stop())
    worker = InferenceWorker(engine_factory=lambda: engine)
    worker.submit(frame_with_marker(1), with_embeddings=True)
    _, _, results = run_worker(worker)
    assert engine.embedding_flags == [True]
    assert results[0].has_embeddings is True


def test_worker_embeddings_default_off():
    engine = FakeEngine(on_analyze=lambda e: worker.request_stop())
    worker = InferenceWorker(engine_factory=lambda: engine)
    worker.submit(frame_with_marker(1))
    _, _, results = run_worker(worker)
    assert engine.embedding_flags == [False]
    assert results[0].has_embeddings is False
    assert results[0].frame is not None  # analyzed frame rides along


def test_worker_latest_wins_keeps_newest_flag_and_token():
    engine = FakeEngine(on_analyze=lambda e: worker.request_stop())
    worker = InferenceWorker(engine_factory=lambda: engine)
    worker.submit(frame_with_marker(1), with_embeddings=False, token=7)
    worker.submit(frame_with_marker(2), with_embeddings=True, token=8)
    _, _, results = run_worker(worker)
    assert engine.analyzed_frames[0][0, 0, 0] == 2
    assert engine.embedding_flags == [True]
    assert results[0].token == 8  # token rides with its frame


def test_worker_stop_before_run_exits_cleanly():
    engine = FakeEngine()
    worker = InferenceWorker(engine_factory=lambda: engine)
    worker.request_stop()
    states, errors, results = run_worker(worker)
    # Model still loads (state reaches READY), then the loop exits at once.
    assert states == [ModelState.LOADING.value, ModelState.READY.value]
    assert errors == [] and results == []
    assert engine.analyzed_frames == []
