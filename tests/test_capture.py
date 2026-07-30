"""Camera worker state logic with a fake capture backend — no webcam.

CameraWorker.run() is called directly (blocking) in the test thread; Qt
signals connected in the same thread fire synchronously, so state sequences
and release behavior can be asserted deterministically.
"""

import numpy as np

from identity_lab.camera.capture import CameraState, CameraWorker, LatestFrame


class FakeCapture:
    def __init__(self, frames=None, opened=True, fail_after=None):
        self._frames = list(frames or [])
        self._opened = opened
        self._fail_after = fail_after  # deliver this many, then fail reads
        self._reads = 0
        self.released = False

    def isOpened(self):  # noqa: N802 (cv2 API)
        return self._opened

    def read(self):
        self._reads += 1
        if self._fail_after is not None and self._reads > self._fail_after:
            return False, None
        if self._frames:
            return True, self._frames.pop(0)
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def run_worker(worker: CameraWorker):
    """Run the worker loop, recording emitted states/errors synchronously."""
    states, errors = [], []
    worker.state_changed.connect(states.append)
    worker.error.connect(errors.append)
    worker.run()
    return states, errors


def test_latest_frame_coalesces_notifications():
    box = LatestFrame()
    a = np.zeros((2, 2, 3), dtype=np.uint8)
    b = np.ones((2, 2, 3), dtype=np.uint8)
    assert box.put(a) is True      # first frame -> notify
    assert box.put(b) is False     # unconsumed -> replace, no second notify
    taken = box.take()
    assert np.array_equal(taken, b)  # consumer gets the newest frame
    assert box.take() is None
    assert box.put(a) is True      # after consumption, notify again


def test_normal_stop_releases_and_reaches_offline():
    fake = FakeCapture()
    worker = CameraWorker(0, capture_factory=lambda i: fake)
    frames_seen = []

    def on_frame():
        frames_seen.append(worker.take_frame())
        if len(frames_seen) >= 3:
            worker.request_stop()

    worker.frame_available.connect(on_frame)
    states, errors = run_worker(worker)
    assert fake.released
    assert errors == []
    assert states[0] == CameraState.INITIALIZING.value
    assert CameraState.READY.value in states
    assert states[-1] == CameraState.OFFLINE.value
    assert len(frames_seen) == 3


def test_open_failure_emits_error_and_offline():
    fake = FakeCapture(opened=False)
    worker = CameraWorker(3, capture_factory=lambda i: fake)
    states, errors = run_worker(worker)
    assert fake.released
    assert len(errors) == 1
    assert "index 3" in errors[0]
    assert CameraState.ERROR.value in states
    assert states[-1] == CameraState.OFFLINE.value


def test_capture_factory_exception_is_contained():
    def exploding_factory(i):
        raise RuntimeError("backend exploded")

    worker = CameraWorker(0, capture_factory=exploding_factory)
    states, errors = run_worker(worker)  # must not raise
    assert len(errors) == 1
    assert "backend exploded" in errors[0]
    assert states[-1] == CameraState.OFFLINE.value


def test_persistent_read_failure_errors_and_releases():
    fake = FakeCapture(fail_after=2)
    worker = CameraWorker(0, capture_factory=lambda i: fake)
    worker.MAX_CONSECUTIVE_READ_FAILURES = 3  # keep the test fast
    states, errors = run_worker(worker)
    assert fake.released
    assert len(errors) == 1
    assert "stopped delivering frames" in errors[0]
    assert states[-1] == CameraState.OFFLINE.value


def test_stop_before_first_frame_is_clean():
    fake = FakeCapture()
    worker = CameraWorker(0, capture_factory=lambda i: fake)
    worker.request_stop()  # stop requested before run() begins
    states, errors = run_worker(worker)
    assert fake.released
    assert errors == []
    assert states[-1] == CameraState.OFFLINE.value
