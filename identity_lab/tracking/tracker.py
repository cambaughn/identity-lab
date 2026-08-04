"""Short-lived face tracks — greedy IOU association frame to frame.

Pure logic (injectable clock, no Qt). Track ids are monotonically numbered
and never reused within a session, so a debug overlay reading TRACK 007 can
only ever mean one physical appearance of a face.
"""

import time
from dataclasses import dataclass, field

from identity_lab.vision.types import DetectedFace

IOU_THRESHOLD = 0.30   # minimum overlap to associate a face with a track
TRACK_EXPIRY_S = 1.5   # unseen for this long -> track dies (survives brief
                       # occlusion at ~5 inference updates per second)


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: str
    bbox: tuple[int, int, int, int]
    last_seen: float
    hits: int = field(default=1)


class FaceTracker:
    """Associates each inference result's faces with stable track ids."""

    def __init__(
        self,
        iou_threshold: float = IOU_THRESHOLD,
        expiry_s: float = TRACK_EXPIRY_S,
        clock=time.monotonic,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._expiry_s = expiry_s
        self._clock = clock
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._expired: list[str] = []

    def update(
        self, faces: tuple[DetectedFace, ...] | list[DetectedFace]
    ) -> list[tuple[str, DetectedFace]]:
        """Associate faces with tracks. Returns [(track_id, face)] in the
        input face order. Unmatched faces start new tracks; tracks unseen
        longer than expiry_s are retired (collect via pop_expired())."""
        now = self._clock()

        # Retire stale tracks BEFORE association — a track that has been
        # gone longer than expiry_s must not be resurrected by a new face
        # appearing in the same spot (that is a new appearance, new id).
        alive: list[_Track] = []
        for track in self._tracks:
            if now - track.last_seen > self._expiry_s:
                self._expired.append(track.track_id)
            else:
                alive.append(track)
        self._tracks = alive

        # Greedy best-overlap-first assignment.
        candidates = []
        for t_idx, track in enumerate(self._tracks):
            for f_idx, face in enumerate(faces):
                overlap = iou(track.bbox, face.bbox)
                if overlap >= self._iou_threshold:
                    candidates.append((overlap, t_idx, f_idx))
        candidates.sort(reverse=True)
        assigned_tracks: set[int] = set()
        assigned_faces: set[int] = set()
        face_to_track: dict[int, _Track] = {}
        for overlap, t_idx, f_idx in candidates:
            if t_idx in assigned_tracks or f_idx in assigned_faces:
                continue
            assigned_tracks.add(t_idx)
            assigned_faces.add(f_idx)
            track = self._tracks[t_idx]
            track.bbox = faces[f_idx].bbox
            track.last_seen = now
            track.hits += 1
            face_to_track[f_idx] = track

        result: list[tuple[str, DetectedFace]] = []
        for f_idx, face in enumerate(faces):
            track = face_to_track.get(f_idx)
            if track is None:
                track = _Track(
                    track_id=f"{self._next_id:03d}",
                    bbox=face.bbox,
                    last_seen=now,
                )
                self._next_id += 1  # ids are never reused
                self._tracks.append(track)
            result.append((track.track_id, face))
        return result

    def pop_expired(self) -> list[str]:
        """Track ids retired since the last call (for stabilizer cleanup)."""
        expired, self._expired = self._expired, []
        return expired

    def reset(self) -> None:
        """Forget all tracks (camera stopped). Ids still never reused."""
        self._expired.extend(t.track_id for t in self._tracks)
        self._tracks = []

    @property
    def active_count(self) -> int:
        return len(self._tracks)
