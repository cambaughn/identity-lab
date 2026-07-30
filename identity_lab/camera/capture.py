"""Camera capture worker — owns cv2.VideoCapture on a background thread.

Design (see plan): the UI never touches the camera. A CameraWorker is moved
to a QThread; its run() loop reads frames and publishes them through a
LatestFrame holder that coalesces — the UI is notified at most once per
unconsumed frame, so a slow UI can never build up a frame queue. The camera
is released in a finally block on every exit path (stop, error, exception).

The capture backend is injectable (capture_factory) so all state logic is
testable without a webcam.
"""

import threading
import time
from enum import Enum

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot


class CameraState(str, Enum):
    OFFLINE = "CAMERA OFFLINE"
    INITIALIZING = "CAMERA INITIALIZING"
    READY = "CAMERA READY"
    ERROR = "CAMERA ERROR"


class LatestFrame:
    """Thread-safe single-slot frame holder with notification coalescing.

    put() stores the newest frame, replacing any unconsumed one, and returns
    True only when the consumer has no notification outstanding — the caller
    should emit its "frame available" signal only in that case. take() returns
    the newest frame and re-arms notification.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._pending = False

    def put(self, frame: np.ndarray) -> bool:
        with self._lock:
            self._frame = frame
            if self._pending:
                return False
            self._pending = True
            return True

    def take(self) -> np.ndarray | None:
        with self._lock:
            frame, self._frame = self._frame, None
            self._pending = False
            return frame


def _default_capture_factory(index: int) -> cv2.VideoCapture:
    return cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)


class CameraWorker(QObject):
    """Reads frames from one camera until asked to stop.

    Signals are the only outward channel; errors never raise across the
    thread boundary. request_stop() is safe to call from any thread.
    """

    frame_available = Signal()
    state_changed = Signal(str)   # CameraState value
    error = Signal(str)           # human-readable detail
    finished = Signal()

    MAX_CONSECUTIVE_READ_FAILURES = 30

    def __init__(self, camera_index: int, capture_factory=None) -> None:
        super().__init__()
        self.camera_index = camera_index
        self._capture_factory = capture_factory or _default_capture_factory
        self._stop = threading.Event()
        self._latest = LatestFrame()

    # -- consumer API (called from the UI thread) --

    def take_frame(self) -> np.ndarray | None:
        return self._latest.take()

    def request_stop(self) -> None:
        self._stop.set()

    # -- worker thread --

    @Slot()
    def run(self) -> None:
        cap = None
        try:
            self.state_changed.emit(CameraState.INITIALIZING.value)
            cap = self._capture_factory(self.camera_index)
            if cap is None or not cap.isOpened():
                self.state_changed.emit(CameraState.ERROR.value)
                self.error.emit(
                    f"Camera index {self.camera_index} could not be opened. "
                    "Check the camera permission for the terminal app and "
                    "that no other app is using the camera."
                )
                return

            failures = 0
            first_frame = True
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    failures += 1
                    if failures >= self.MAX_CONSECUTIVE_READ_FAILURES:
                        self.state_changed.emit(CameraState.ERROR.value)
                        self.error.emit(
                            f"Camera index {self.camera_index} stopped "
                            "delivering frames."
                        )
                        return
                    time.sleep(0.05)
                    continue
                failures = 0
                if first_frame:
                    first_frame = False
                    self.state_changed.emit(CameraState.READY.value)
                if self._latest.put(frame):
                    self.frame_available.emit()
        except Exception as exc:  # never let an exception cross the thread
            self.state_changed.emit(CameraState.ERROR.value)
            self.error.emit(f"Unexpected camera failure: {exc}")
        finally:
            if cap is not None:
                cap.release()
            self.state_changed.emit(CameraState.OFFLINE.value)
            self.finished.emit()
