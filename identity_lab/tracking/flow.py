"""Per-track sparse optical-flow box propagation (Lucas–Kanade).

Between full detector passes (~5 Hz), each track's displayed box is moved
by *measured* pixel motion: a small grid of points inside the box is
tracked frame-to-frame with pyramidal LK optical flow on a downscaled
grayscale image; the median point translation moves the box and the change
in point spread rescales it. This is visual tracking of actual motion —
not interpolation between stale detections.

Failure behavior is explicit: when too few flow points remain valid
(occlusion, blur, textureless input) the track COASTS — its box freezes in
place — until the next detection re-anchors it or the track expires. No
blind extrapolation, ever.

Core OpenCV only (calcOpticalFlowPyrLK); no contrib dependency. Pure
logic + numpy/cv2, no Qt — fully testable with synthetic frames.
"""

import cv2
import numpy as np

DOWNSCALE = 0.5            # flow runs on a half-resolution grayscale image
GRID = 7                   # GRID x GRID seed points per face box
INNER_FRACTION = 0.8       # seed points within the central 80% of the box
MIN_VALID_FRACTION = 0.4   # fewer valid flow points -> coasting
MAX_DELTA_MAD = 6.0        # downscaled px; incoherent point motion -> coasting
                           # (faces are non-rigid: blinks/rotation scatter
                           # points — this is deliberately loose)
MAX_STEP_PX = 80.0         # full-res px; larger single-frame jumps -> coasting
SCALE_CLAMP = (0.95, 1.05) # max per-frame raw box scale change
SCALE_DAMPING = 0.5        # apply only half of each frame's scale change
SCALE_DEADBAND = 0.005     # ignore sub-0.5% scale noise entirely
ANCHOR_BLEND = 0.3         # healthy-track drift correction strength
ANCHOR_DEADBAND_PX = 4.0   # ignore detection disagreement smaller than this
ANCHOR_SNAP_IOU = 0.4      # below this overlap the detection wins outright

_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)


def _to_flow_gray(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, None, fx=DOWNSCALE, fy=DOWNSCALE)


def compensate_bbox(
    detected_bbox: tuple[int, int, int, int],
    center_at_submit: tuple[float, float] | None,
    center_now: tuple[float, float] | None,
) -> tuple[float, float, float, float]:
    """Shift a stale detection by the flow motion accumulated since the
    analyzed frame was submitted.

    A detection describes where the face was ~100-200ms ago. Anchoring to it
    directly drags a moving face's box backward. Given the track's flow-box
    center when the frame was submitted and its center now, shifting the
    detection by the difference yields where the detection says the face is
    NOW. Returns the bbox unchanged when either center is unknown."""
    if center_at_submit is None or center_now is None:
        return detected_bbox
    dx = center_now[0] - center_at_submit[0]
    dy = center_now[1] - center_at_submit[1]
    x1, y1, x2, y2 = detected_bbox
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def bbox_center(bbox) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _region_has_texture(
    gray: np.ndarray, bbox: tuple[float, float, float, float]
) -> bool:
    x1, y1, x2, y2 = (int(v * DOWNSCALE) for v in bbox)
    h, w = gray.shape[:2]
    crop = gray[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
    return crop.size > 0 and float(crop.std()) > 5.0


def _seed_points(bbox: tuple[float, float, float, float]) -> np.ndarray:
    """GRID x GRID points (downscaled coords) inside the central box area."""
    x1, y1, x2, y2 = (v * DOWNSCALE for v in bbox)
    w, h = x2 - x1, y2 - y1
    pad_x = w * (1 - INNER_FRACTION) / 2
    pad_y = h * (1 - INNER_FRACTION) / 2
    xs = np.linspace(x1 + pad_x, x2 - pad_x, GRID, dtype=np.float32)
    ys = np.linspace(y1 + pad_y, y2 - pad_y, GRID, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    return grid.astype(np.float32)


class _FlowTrack:
    def __init__(self, bbox: tuple[float, float, float, float]) -> None:
        self.bbox = tuple(float(v) for v in bbox)
        self.points: np.ndarray | None = None  # (N, 2) downscaled coords
        self.coasting = False


class FlowTracker:
    def __init__(self) -> None:
        self._prev_gray: np.ndarray | None = None
        self._tracks: dict[str, _FlowTrack] = {}

    # -- per preview frame --

    def step(self, frame_bgr: np.ndarray) -> dict[str, tuple[int, int, int, int]]:
        """Advance all boxes by measured flow. Returns {track_id: bbox}."""
        gray = _to_flow_gray(frame_bgr)
        prev = self._prev_gray
        self._prev_gray = gray
        if prev is None or prev.shape != gray.shape:
            return self.boxes()

        for track in self._tracks.values():
            if track.points is None or len(track.points) < 4:
                # Lost points: re-acquire from the current frame if the box
                # region has texture again (flow resumes next step);
                # otherwise keep coasting.
                if _region_has_texture(gray, track.bbox):
                    track.points = _seed_points(track.bbox)
                track.coasting = True
                continue
            new_pts, status, _err = cv2.calcOpticalFlowPyrLK(
                prev, gray, track.points.reshape(-1, 1, 2), None, **_LK_PARAMS
            )
            status = status.ravel().astype(bool)
            if new_pts is None or status.sum() < len(track.points) * MIN_VALID_FRACTION:
                track.coasting = True
                track.points = None  # force reseed at next anchor
                continue
            old = track.points[status]
            new = new_pts.reshape(-1, 2)[status]

            # Translation: median point motion (robust to a few bad points).
            deltas = new - old
            delta = np.median(deltas, axis=0)
            # Coherence gate: genuine motion moves points together; LK on
            # vanished/blurred texture scatters them. Median absolute
            # deviation catches that, and a hard step cap catches the rest.
            mad = float(np.median(np.abs(deltas - delta)))
            if (
                mad > MAX_DELTA_MAD
                or float(np.linalg.norm(delta)) > MAX_STEP_PX * DOWNSCALE
            ):
                track.coasting = True
                track.points = None  # reseed at next anchor
                continue
            track.coasting = False
            # Scale: change in median spread around the point centroid.
            old_spread = np.median(np.linalg.norm(old - old.mean(axis=0), axis=1))
            new_spread = np.median(np.linalg.norm(new - new.mean(axis=0), axis=1))
            scale = 1.0
            if old_spread > 1e-3:
                raw = float(np.clip(new_spread / old_spread, *SCALE_CLAMP))
                # Damped + deadbanded: stops per-frame "size breathing" on
                # non-rigid faces while still following real scale changes.
                scale = 1.0 + (raw - 1.0) * SCALE_DAMPING
                if abs(scale - 1.0) < SCALE_DEADBAND:
                    scale = 1.0

            x1, y1, x2, y2 = track.bbox
            cx = (x1 + x2) / 2 + float(delta[0]) / DOWNSCALE
            cy = (y1 + y2) / 2 + float(delta[1]) / DOWNSCALE
            half_w = (x2 - x1) / 2 * scale
            half_h = (y2 - y1) / 2 * scale
            track.bbox = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
            track.points = new
        return self.boxes()

    # -- detector corrections --

    def anchor(self, track_id: str, detected_bbox: tuple[int, int, int, int]) -> None:
        """Detection arrived for this track: correct drift and reseed flow.

        The caller is responsible for compensating detection staleness
        (see compensate_bbox) — this method treats detected_bbox as
        describing the face's position NOW.

        Correction policy: coasting/new tracks snap; gross disagreement
        (IOU below ANCHOR_SNAP_IOU) snaps; disagreement below the deadband
        is ignored (flow is fresher than a stale detection); otherwise a
        gentle blend corrects drift without visible nudging.
        """
        from identity_lab.tracking.tracker import iou

        track = self._tracks.get(track_id)
        if track is None:
            track = _FlowTrack(detected_bbox)
            self._tracks[track_id] = track
        elif track.coasting or track.points is None:
            track.bbox = tuple(float(v) for v in detected_bbox)
        elif iou(
            tuple(int(v) for v in track.bbox),
            tuple(int(v) for v in detected_bbox),
        ) < ANCHOR_SNAP_IOU:
            track.bbox = tuple(float(v) for v in detected_bbox)
        else:
            diffs = [abs(f - d) for f, d in zip(track.bbox, detected_bbox)]
            if max(diffs) >= ANCHOR_DEADBAND_PX:
                b = ANCHOR_BLEND
                track.bbox = tuple(
                    (1 - b) * f + b * d
                    for f, d in zip(track.bbox, detected_bbox)
                )
            # else: within deadband — keep the flow box untouched.
        track.coasting = False
        # Reseed on every anchor: stale drifted points must not linger.
        track.points = _seed_points(track.bbox)

    # -- bookkeeping --

    def boxes(self) -> dict[str, tuple[int, int, int, int]]:
        return {
            tid: tuple(int(round(v)) for v in t.bbox)
            for tid, t in self._tracks.items()
        }

    def is_coasting(self, track_id: str) -> bool:
        track = self._tracks.get(track_id)
        return bool(track and track.coasting)

    def forget(self, track_id: str) -> None:
        self._tracks.pop(track_id, None)

    def reset(self) -> None:
        self._tracks.clear()
        self._prev_gray = None
