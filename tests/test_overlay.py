"""Mirrored overlay coordinate transforms — pure math, no Qt."""

import pytest

from identity_lab.ui.overlay import display_transform, map_bbox, map_point


def test_transform_identity_when_sizes_match():
    scale, off_x, off_y = display_transform(640, 480, 640, 480)
    assert (scale, off_x, off_y) == (1.0, 0.0, 0.0)


def test_transform_letterboxes_wide_widget():
    # 640x480 frame in a 1280x480 widget: fit height, center horizontally.
    scale, off_x, off_y = display_transform(640, 480, 1280, 480)
    assert scale == 1.0
    assert off_x == 320.0
    assert off_y == 0.0


def test_transform_scales_down():
    scale, off_x, off_y = display_transform(1920, 1080, 960, 540)
    assert scale == 0.5
    assert (off_x, off_y) == (0.0, 0.0)


def test_transform_degenerate_sizes_safe():
    assert display_transform(0, 480, 640, 480) == (1.0, 0.0, 0.0)
    assert display_transform(640, 480, 0, 0) == (1.0, 0.0, 0.0)


def test_map_bbox_unmirrored_identity_transform():
    x, y, w, h = map_bbox((10, 20, 110, 220), 640, 1.0, 0.0, 0.0, mirrored=False)
    assert (x, y, w, h) == (10, 20, 100, 200)


def test_map_bbox_mirrored_flips_horizontally():
    # Face near the frame's left edge must render near the display's right edge.
    x, y, w, h = map_bbox((0, 0, 100, 50), 640, 1.0, 0.0, 0.0, mirrored=True)
    assert (x, y, w, h) == (540, 0, 100, 50)


def test_map_bbox_mirror_preserves_size_and_vertical():
    bbox = (200, 100, 350, 300)
    mx, my, mw, mh = map_bbox(bbox, 640, 1.0, 0.0, 0.0, mirrored=True)
    assert (mw, mh) == (150, 200)
    assert my == 100


def test_map_bbox_scale_and_offset():
    x, y, w, h = map_bbox((100, 100, 200, 200), 640, 0.5, 10.0, 20.0, mirrored=False)
    assert (x, y, w, h) == (60.0, 70.0, 50.0, 50.0)


def test_map_point_mirrored_center_is_fixed():
    x, y = map_point((320, 240), 640, 1.0, 0.0, 0.0, mirrored=True)
    assert (x, y) == (320, 240)


def test_map_point_consistent_with_bbox_mirror():
    frame_w, scale, off_x, off_y = 640, 0.75, 5.0, 7.0
    bbox = (100, 50, 300, 250)
    bx, by, bw, bh = map_bbox(bbox, frame_w, scale, off_x, off_y, mirrored=True)
    # The bbox's left edge in display space corresponds to frame x2.
    px, py = map_point((bbox[2], bbox[1]), frame_w, scale, off_x, off_y, mirrored=True)
    assert px == pytest.approx(bx)
    assert py == pytest.approx(by)
