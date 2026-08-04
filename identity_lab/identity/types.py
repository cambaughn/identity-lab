"""Identity-layer record types."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

ENROLLMENT_VERSION = 1  # bump when the enrollment procedure changes materially


@dataclass(frozen=True)
class IdentityRecord:
    identity_id: str
    display_name: str
    created_at: datetime
    enrollment_version: int
    sample_count: int
    model_id: str | None   # None until the first sample is stored
    dim: int | None        # None until the first sample is stored


@dataclass(frozen=True)
class IdentityObservation:
    """One stabilized per-track recognition observation (spec structure).

    identity/display_name reflect the *stabilized* decision; similarity and
    second_best_similarity are the raw scores from the current frame's match
    (None when the frame had no usable match). reason explains the current
    stabilizer state in console language.
    """

    track_id: str
    identity_id: str | None
    display_name: str | None
    similarity: float | None
    second_best_similarity: float | None
    is_known: bool
    reason: str
    bbox: tuple[int, int, int, int]
    timestamp: datetime


@dataclass(frozen=True, eq=False)  # eq=False: ndarray members don't compare
class EmbeddingSample:
    sample_id: str
    identity_id: str
    embedding: np.ndarray  # float32, 1-D, dim elements
    dim: int
    dtype: str
    model_id: str
    created_at: datetime
