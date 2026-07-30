"""Frame conversion helpers — pure functions, testable without a webcam."""

import numpy as np
from PySide6.QtGui import QImage


def mirror_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Horizontally mirror a frame (selfie view)."""
    return np.ascontiguousarray(frame_bgr[:, ::-1])


def to_display_qimage(frame_bgr: np.ndarray, mirrored: bool = True) -> QImage:
    """Convert an OpenCV BGR frame to a QImage for display.

    Returns a QImage that owns its pixel data (safe after the source
    array goes away).
    """
    if mirrored:
        frame_bgr = mirror_frame(frame_bgr)
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
