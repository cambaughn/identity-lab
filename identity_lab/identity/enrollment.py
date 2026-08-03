"""Enrollment session — guided multi-pose capture with simple, deterministic
quality gates.

Philosophy (CP7): a working, understandable flow over a sophisticated one.
Gates are checked in a fixed, documented order and each rejection has one
plain reason code, so the UI can always tell the user exactly what the
system wants. Pose prompts guide sample variety but are NOT enforced by
pose estimation — deliberately simple for the first pass.

Gate order per observation:
  1. exactly one face visible (0 -> NO_FACE, >1 -> MULTIPLE_FACES hard hold)
  2. face large enough        (TOO_SMALL)
  3. detector confident       (LOW_CONFIDENCE)
  4. paced over time          (PACING — not an error, just "keep holding")
  5. embedding present        (WAITING_EMBEDDING)
  6. not badly blurred        (TOO_BLURRY; Laplacian variance)
  7. consistent with session  (INCONSISTENT; cosine vs mean of accepted —
     guards against a person swap mid-enrollment)

Nothing touches the database until save_enrollment() — cancel is simply
"never call save", which guarantees no partial identities.
"""

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from identity_lab.identity.store import IdentityStore
from identity_lab.identity.types import ENROLLMENT_VERSION, IdentityRecord
from identity_lab.vision.quality import face_sharpness, face_thumbnail_png
from identity_lab.vision.types import DetectedFace

# -- tuning constants (first pass: conservative and simple) --
SAMPLES_PER_STAGE = 4
MIN_SAMPLE_GAP_S = 0.4       # never adjacent frames; spreads capture over time
MIN_FACE_PX = 120            # enrollment wants a close, clear face
MIN_DET_SCORE = 0.60
MIN_SHARPNESS = 30.0         # Laplacian variance on 112px gray crop
OUTLIER_MIN_COSINE = 0.40    # vs mean of accepted samples
OUTLIER_CHECK_AFTER = 3      # need a few samples before consistency checks


class PoseStage(str, Enum):
    FORWARD = "FACE FORWARD"
    LEFT = "TURN SLIGHTLY LEFT"
    RIGHT = "TURN SLIGHTLY RIGHT"
    UP = "LOOK SLIGHTLY UP"
    DOWN = "LOOK SLIGHTLY DOWN"
    GLASSES = "OPTIONAL: GLASSES ON / OFF"


REQUIRED_STAGES = [
    PoseStage.FORWARD,
    PoseStage.LEFT,
    PoseStage.RIGHT,
    PoseStage.UP,
    PoseStage.DOWN,
]
OPTIONAL_STAGES = [PoseStage.GLASSES]
ALL_STAGES = REQUIRED_STAGES + OPTIONAL_STAGES
REQUIRED_TARGET = len(REQUIRED_STAGES) * SAMPLES_PER_STAGE  # 20


class FeedbackCode(str, Enum):
    ACCEPTED = "SAMPLE ACCEPTED"
    NO_FACE = "NO FACE VISIBLE"
    MULTIPLE_FACES = "MULTIPLE FACES — ENROLLMENT ON HOLD"
    TOO_SMALL = "MOVE CLOSER TO THE CAMERA"
    LOW_CONFIDENCE = "LOW DETECTION CONFIDENCE — FACE THE CAMERA"
    PACING = "SAMPLING — HOLD POSE"
    WAITING_EMBEDDING = "PROCESSING — HOLD POSE"
    TOO_BLURRY = "HOLD STILL — IMAGE BLURRED"
    INCONSISTENT = "SAMPLE INCONSISTENT — REJECTED"
    COMPLETE = "ALL SAMPLES CAPTURED"


@dataclass(frozen=True)
class EnrollmentFeedback:
    code: FeedbackCode
    stage: PoseStage | None
    accepted_total: int
    required_target: int

    @property
    def accepted(self) -> bool:
        return self.code is FeedbackCode.ACCEPTED


@dataclass(frozen=True)
class StageStatus:
    stage: PoseStage
    accepted: int
    target: int
    required: bool

    @property
    def done(self) -> bool:
        return self.accepted >= self.target


class EnrollmentSession:
    """Collects quality-gated embeddings across guided pose stages.

    Pure logic: feed it detection results via process(); it never touches
    the camera, the UI, or the database.
    """

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._counts: dict[PoseStage, int] = {s: 0 for s in ALL_STAGES}
        self._embeddings: list[np.ndarray] = []
        self._last_accept_t: float | None = None
        self.thumbnail_png: bytes | None = None

    # -- progress introspection --

    @property
    def current_stage(self) -> PoseStage | None:
        for stage in ALL_STAGES:
            if self._counts[stage] < SAMPLES_PER_STAGE:
                return stage
        return None  # everything, including optional, is done

    @property
    def accepted_total(self) -> int:
        return len(self._embeddings)

    @property
    def required_complete(self) -> bool:
        return all(
            self._counts[s] >= SAMPLES_PER_STAGE for s in REQUIRED_STAGES
        )

    @property
    def fully_complete(self) -> bool:
        return self.current_stage is None

    def stage_statuses(self) -> list[StageStatus]:
        return [
            StageStatus(
                stage=s,
                accepted=self._counts[s],
                target=SAMPLES_PER_STAGE,
                required=s in REQUIRED_STAGES,
            )
            for s in ALL_STAGES
        ]

    def embeddings(self) -> list[np.ndarray]:
        return list(self._embeddings)

    # -- the gate pipeline --

    def process(
        self, faces: tuple[DetectedFace, ...], frame_bgr: np.ndarray | None
    ) -> EnrollmentFeedback:
        stage = self.current_stage
        if stage is None:
            return self._feedback(FeedbackCode.COMPLETE, None)

        if len(faces) == 0:
            return self._feedback(FeedbackCode.NO_FACE, stage)
        if len(faces) > 1:
            return self._feedback(FeedbackCode.MULTIPLE_FACES, stage)
        face = faces[0]

        if face.size_px < MIN_FACE_PX:
            return self._feedback(FeedbackCode.TOO_SMALL, stage)
        if face.det_score < MIN_DET_SCORE:
            return self._feedback(FeedbackCode.LOW_CONFIDENCE, stage)

        now = self._clock()
        if (
            self._last_accept_t is not None
            and now - self._last_accept_t < MIN_SAMPLE_GAP_S
        ):
            return self._feedback(FeedbackCode.PACING, stage)

        if face.embedding is None:
            return self._feedback(FeedbackCode.WAITING_EMBEDDING, stage)

        if frame_bgr is not None:
            if face_sharpness(frame_bgr, face.bbox) < MIN_SHARPNESS:
                return self._feedback(FeedbackCode.TOO_BLURRY, stage)

        embedding = np.asarray(face.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm <= 0:
            return self._feedback(FeedbackCode.WAITING_EMBEDDING, stage)
        embedding = embedding / norm

        if len(self._embeddings) >= OUTLIER_CHECK_AFTER:
            mean = np.mean(self._embeddings, axis=0)
            mean /= np.linalg.norm(mean)
            if float(np.dot(embedding, mean)) < OUTLIER_MIN_COSINE:
                return self._feedback(FeedbackCode.INCONSISTENT, stage)

        # Accepted.
        self._embeddings.append(embedding)
        self._counts[stage] += 1
        self._last_accept_t = now
        if self.thumbnail_png is None and frame_bgr is not None:
            self.thumbnail_png = face_thumbnail_png(frame_bgr, face.bbox)
        return self._feedback(FeedbackCode.ACCEPTED, stage)

    def _feedback(
        self, code: FeedbackCode, stage: PoseStage | None
    ) -> EnrollmentFeedback:
        return EnrollmentFeedback(
            code=code,
            stage=stage,
            accepted_total=self.accepted_total,
            required_target=REQUIRED_TARGET,
        )


def save_enrollment(
    store: IdentityStore,
    display_name: str,
    session: EnrollmentSession,
    model_id: str,
) -> IdentityRecord:
    """Persist a completed session as a new identity (all-or-nothing).

    Requires the session's required stages to be complete. If any sample
    write fails, the partially created identity is removed before the
    error propagates — the store never retains a half-enrolled person.
    """
    if not session.required_complete:
        raise ValueError(
            f"enrollment incomplete: {session.accepted_total} accepted, "
            f"{REQUIRED_TARGET} required"
        )
    record = store.create_identity(
        display_name, enrollment_version=ENROLLMENT_VERSION
    )
    try:
        for embedding in session.embeddings():
            store.add_embedding_sample(record.identity_id, embedding, model_id)
        if session.thumbnail_png is not None:
            store.set_identity_thumbnail(record.identity_id, session.thumbnail_png)
    except Exception:
        store.delete_identity(record.identity_id)
        raise
    return store.get_identity(record.identity_id)
