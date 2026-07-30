"""Frame conversion helpers — no webcam required."""

import numpy as np
from PySide6.QtGui import QImage

from identity_lab.camera.frames import mirror_frame, to_display_qimage


def _gradient_frame(h=4, w=6):
    """BGR frame whose blue channel encodes the column index."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = np.arange(w, dtype=np.uint8)[None, :]
    return frame


def test_mirror_flips_columns():
    frame = _gradient_frame()
    mirrored = mirror_frame(frame)
    assert np.array_equal(mirrored[:, :, 0], frame[:, ::-1, 0])
    assert mirrored.flags["C_CONTIGUOUS"]


def test_mirror_twice_is_identity():
    frame = _gradient_frame()
    assert np.array_equal(mirror_frame(mirror_frame(frame)), frame)


def test_qimage_dimensions_and_format():
    img = to_display_qimage(_gradient_frame(h=4, w=6), mirrored=False)
    assert img.width() == 6
    assert img.height() == 4
    assert img.format() == QImage.Format.Format_RGB888


def test_qimage_converts_bgr_to_rgb():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # pure blue in BGR
    img = to_display_qimage(frame, mirrored=False)
    pixel = img.pixelColor(0, 0)
    assert (pixel.red(), pixel.green(), pixel.blue()) == (0, 0, 255)


def test_qimage_mirroring_moves_marker_pixel():
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    frame[0, 0, 2] = 255  # red marker at left edge (BGR)
    img = to_display_qimage(frame, mirrored=True)
    assert img.pixelColor(3, 0).red() == 255
    assert img.pixelColor(0, 0).red() == 0


def test_qimage_owns_its_data():
    frame = _gradient_frame()
    img = to_display_qimage(frame, mirrored=False)
    frame[:] = 0  # mutate source after conversion
    assert img.pixelColor(5, 0).blue() == 5
