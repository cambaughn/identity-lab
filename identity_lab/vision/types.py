"""Vision-layer result types."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DetectedFace:
    """One detected face in original frame coordinates."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    det_score: float
    landmarks: np.ndarray | None     # (106, 2) fine landmarks, if available
    kps: np.ndarray | None           # (5, 2) coarse keypoints
    embedding: np.ndarray | None     # 512-d normed embedding (never logged)

    @property
    def size_px(self) -> int:
        """Smaller side of the bbox — used for min-face-size filtering."""
        x1, y1, x2, y2 = self.bbox
        return int(min(x2 - x1, y2 - y1))


@dataclass(frozen=True)
class InferenceResult:
    """Output of one inference pass on one frame.

    frame is the analyzed BGR frame (a reference, not a copy) — consumers
    like enrollment need it for blur checks and thumbnails. Never persisted.
    """

    faces: tuple[DetectedFace, ...]
    latency_ms: float
    frame_w: int
    frame_h: int
    frame: np.ndarray | None = None
    has_embeddings: bool = True  # False for detection-only passes
    token: int = 0               # caller's submit token, echoed back


def filter_small_faces(
    faces: tuple[DetectedFace, ...] | list[DetectedFace], min_size_px: int
) -> tuple[DetectedFace, ...]:
    return tuple(f for f in faces if f.size_px >= min_size_px)


@dataclass
class InferenceScheduler:
    """Decides which frames are submitted for inference (every Nth)."""

    every_n: int = 2
    _count: int = field(default=0, repr=False)

    def should_submit(self) -> bool:
        n = max(1, self.every_n)
        submit = self._count % n == 0
        self._count += 1
        return submit

    def reset(self) -> None:
        self._count = 0
