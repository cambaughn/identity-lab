"""Inference worker — runs the vision engine on its own thread.

Frames arrive via submit() (any thread) into a LatestFrame holder: while an
inference pass is running, newer frames replace the waiting one, so
inference can never build a backlog. The engine is injectable for tests.
"""

import threading
import time
from enum import Enum

from PySide6.QtCore import QObject, Signal, Slot

from identity_lab.camera.capture import LatestFrame
from identity_lab.vision.engine import VisionEngine
from identity_lab.vision.types import InferenceResult


class ModelState(str, Enum):
    LOADING = "MODEL LOADING"
    READY = "MODEL READY"
    ERROR = "MODEL ERROR"


class InferenceWorker(QObject):
    model_state_changed = Signal(str)   # ModelState value
    result_ready = Signal(object)       # InferenceResult
    error = Signal(str)
    finished = Signal()

    MAX_CONSECUTIVE_FAILURES = 5

    def __init__(self, engine_factory=VisionEngine) -> None:
        super().__init__()
        self._engine_factory = engine_factory
        self._latest = LatestFrame()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._with_embeddings = False

    # -- producer API (any thread) --

    def submit(self, frame) -> None:
        self._latest.put(frame)
        self._wake.set()

    def set_embeddings_enabled(self, enabled: bool) -> None:
        """Embeddings cost ~50ms/face; only enrollment/recognition need them.
        Plain bool assignment — atomic in CPython, read once per loop pass."""
        self._with_embeddings = enabled

    def request_stop(self) -> None:
        self._stop.set()
        self._wake.set()

    # -- worker thread --

    @Slot()
    def run(self) -> None:
        try:
            self.model_state_changed.emit(ModelState.LOADING.value)
            try:
                engine = self._engine_factory()
                t0 = time.perf_counter()
                engine.initialize()
                init_s = time.perf_counter() - t0
            except Exception as exc:
                self.model_state_changed.emit(ModelState.ERROR.value)
                self.error.emit(f"Model initialization failed: {exc}")
                return
            self.model_state_changed.emit(ModelState.READY.value)
            self.error_free_init_s = init_s

            failures = 0
            while not self._stop.is_set():
                if not self._wake.wait(timeout=0.2):
                    continue
                self._wake.clear()
                if self._stop.is_set():
                    break
                frame = self._latest.take()
                if frame is None:
                    continue
                try:
                    t0 = time.perf_counter()
                    faces = engine.analyze(
                        frame, with_embeddings=self._with_embeddings
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000
                except Exception as exc:
                    failures += 1
                    self.error.emit(f"Inference failed: {exc}")
                    if failures >= self.MAX_CONSECUTIVE_FAILURES:
                        self.model_state_changed.emit(ModelState.ERROR.value)
                        self.error.emit(
                            "Inference failing repeatedly; stopping analysis."
                        )
                        return
                    continue
                failures = 0
                h, w = frame.shape[:2]
                self.result_ready.emit(
                    InferenceResult(
                        faces=faces,
                        latency_ms=latency_ms,
                        frame_w=w,
                        frame_h=h,
                        frame=frame,
                    )
                )
        finally:
            self.finished.emit()
