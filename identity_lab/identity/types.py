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


@dataclass(frozen=True, eq=False)  # eq=False: ndarray members don't compare
class EmbeddingSample:
    sample_id: str
    identity_id: str
    embedding: np.ndarray  # float32, 1-D, dim elements
    dim: int
    dtype: str
    model_id: str
    created_at: datetime
