"""Simple, well-understood image-quality helpers for enrollment.

Deliberately unsophisticated (per CP7 philosophy): one blur measure and one
thumbnail helper, both operating on a face crop. No quality-scoring system.
"""

import cv2
import numpy as np

_CROP_ANALYSIS_SIZE = 112  # normalize crop size so the blur measure is
                           # roughly independent of how close the face is


def _crop_face(
    frame_bgr: np.ndarray, bbox: tuple[int, int, int, int], margin: float
) -> np.ndarray | None:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    mx = int((x2 - x1) * margin)
    my = int((y2 - y1) * margin)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return frame_bgr[y1:y2, x1:x2]


def face_sharpness(
    frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]
) -> float:
    """Variance of the Laplacian over the (size-normalized, grayscale) face
    crop. Higher = sharper. Returns 0.0 for degenerate crops."""
    crop = _crop_face(frame_bgr, bbox, margin=0.0)
    if crop is None:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (_CROP_ANALYSIS_SIZE, _CROP_ANALYSIS_SIZE))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_thumbnail_png(
    frame_bgr: np.ndarray, bbox: tuple[int, int, int, int], size: int = 128
) -> bytes | None:
    """Small square PNG of the face (with margin), for UI presentation only.
    Returns None if the crop or encoding fails — callers treat the thumbnail
    as strictly optional."""
    crop = _crop_face(frame_bgr, bbox, margin=0.3)
    if crop is None:
        return None
    thumb = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", thumb)
    return buf.tobytes() if ok else None
