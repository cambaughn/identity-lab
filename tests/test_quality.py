"""Quality helpers: sharpness measure and thumbnail generation."""

import cv2
import numpy as np

from identity_lab.vision.quality import face_sharpness, face_thumbnail_png

RNG = np.random.default_rng(3)
NOISY = RNG.integers(0, 255, size=(400, 400, 3), dtype=np.uint8)
BBOX = (50, 50, 350, 350)


def test_sharp_beats_blurred():
    sharp = face_sharpness(NOISY, BBOX)
    blurred_frame = cv2.GaussianBlur(NOISY, (31, 31), 12)
    blurred = face_sharpness(blurred_frame, BBOX)
    assert sharp > blurred * 5  # decisively distinguishable
    assert blurred >= 0


def test_uniform_image_has_zero_sharpness():
    flat = np.full((200, 200, 3), 77, dtype=np.uint8)
    assert face_sharpness(flat, (10, 10, 190, 190)) == 0.0


def test_degenerate_bbox_returns_zero():
    assert face_sharpness(NOISY, (100, 100, 100, 100)) == 0.0
    assert face_sharpness(NOISY, (390, 390, 800, 800)) < np.inf  # clamped, no crash


def test_thumbnail_is_png_of_requested_size():
    png = face_thumbnail_png(NOISY, BBOX, size=128)
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (128, 128, 3)


def test_thumbnail_degenerate_bbox_returns_none():
    assert face_thumbnail_png(NOISY, (100, 100, 100, 100)) is None


def test_thumbnail_bbox_clamped_to_frame():
    png = face_thumbnail_png(NOISY, (-50, -50, 500, 500), size=64)
    assert png is not None
