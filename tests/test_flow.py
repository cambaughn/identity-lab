"""FlowTracker: synthetic textured-patch sequences — no camera.

A noise-textured square moves across a gray background; the flow tracker
must follow it, detect scale changes, coast on textureless frames, and
re-anchor cleanly.
"""

import cv2
import numpy as np

from identity_lab.tracking.flow import FlowTracker

RNG = np.random.default_rng(11)
PATCH = RNG.integers(0, 255, size=(120, 120, 3), dtype=np.uint8)
FRAME_H, FRAME_W = 480, 640


def frame_with_patch(x: int, y: int, patch=PATCH) -> np.ndarray:
    frame = np.full((FRAME_H, FRAME_W, 3), 90, dtype=np.uint8)
    h, w = patch.shape[:2]
    frame[y : y + h, x : x + w] = patch
    return frame


def center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def test_flow_follows_translation():
    tracker = FlowTracker()
    x, y = 100, 100
    tracker.step(frame_with_patch(x, y))
    tracker.anchor("t1", (x, y, x + 120, y + 120))
    for _ in range(10):
        x += 8  # move right 8px per frame
        boxes = tracker.step(frame_with_patch(x, y))
    cx, cy = center(boxes["t1"])
    assert abs(cx - (x + 60)) < 10  # within 10px of the true center
    assert abs(cy - (y + 60)) < 10
    assert not tracker.is_coasting("t1")


def test_flow_follows_diagonal_and_vertical():
    tracker = FlowTracker()
    x, y = 200, 80
    tracker.step(frame_with_patch(x, y))
    tracker.anchor("t1", (x, y, x + 120, y + 120))
    for _ in range(8):
        x += 5
        y += 6
        boxes = tracker.step(frame_with_patch(x, y))
    cx, cy = center(boxes["t1"])
    assert abs(cx - (x + 60)) < 10
    assert abs(cy - (y + 60)) < 10


def test_flow_tracks_scale_growth():
    tracker = FlowTracker()
    size = 100
    x, y = 200, 150
    tracker.step(frame_with_patch(x, y, cv2.resize(PATCH, (size, size))))
    tracker.anchor("t1", (x, y, x + size, y + size))
    for _ in range(10):
        size = int(size * 1.05)
        tracker.step(frame_with_patch(x, y, cv2.resize(PATCH, (size, size))))
    boxes = tracker.boxes()
    x1, y1, x2, y2 = boxes["t1"]
    assert (x2 - x1) > 115  # grew from 100 toward ~160 (clamped per frame)


def test_textureless_frames_cause_coasting_not_explosion():
    tracker = FlowTracker()
    tracker.step(frame_with_patch(100, 100))
    tracker.anchor("t1", (100, 100, 220, 220))
    before = tracker.boxes()["t1"]
    for _ in range(5):
        boxes = tracker.step(np.full((FRAME_H, FRAME_W, 3), 90, dtype=np.uint8))
    after = boxes["t1"]
    assert tracker.is_coasting("t1")
    # Box must be frozen (or near-frozen), never flung off-screen.
    assert abs(after[0] - before[0]) < 20 and abs(after[1] - before[1]) < 20


def test_anchor_recovers_from_coasting():
    tracker = FlowTracker()
    tracker.step(frame_with_patch(100, 100))
    tracker.anchor("t1", (100, 100, 220, 220))
    for _ in range(3):
        tracker.step(np.full((FRAME_H, FRAME_W, 3), 90, dtype=np.uint8))
    assert tracker.is_coasting("t1")
    # Detection re-anchors at a new position: snap (no blend from a coast).
    tracker.anchor("t1", (300, 200, 420, 320))
    assert tracker.boxes()["t1"] == (300, 200, 420, 320)
    assert not tracker.is_coasting("t1")
    # And flow resumes from the new spot.
    x, y = 300, 200
    tracker.step(frame_with_patch(x, y))
    for _ in range(5):
        x += 6
        boxes = tracker.step(frame_with_patch(x, y))
    assert abs(center(boxes["t1"])[0] - (x + 60)) < 10


def test_anchor_blends_when_tracking_well():
    tracker = FlowTracker()
    tracker.step(frame_with_patch(100, 100))
    tracker.anchor("t1", (100, 100, 220, 220))
    tracker.step(frame_with_patch(100, 100))  # healthy flow, no motion
    tracker.anchor("t1", (110, 100, 230, 220))  # detection slightly right
    x1 = tracker.boxes()["t1"][0]
    assert 100 < x1 < 110  # blended, not snapped


def test_anchor_deadband_ignores_tiny_disagreement():
    tracker = FlowTracker()
    tracker.step(frame_with_patch(100, 100))
    tracker.anchor("t1", (100, 100, 220, 220))
    tracker.step(frame_with_patch(100, 100))
    tracker.anchor("t1", (102, 101, 222, 221))  # < deadband everywhere
    assert tracker.boxes()["t1"] == (100, 100, 220, 220)  # flow box kept


def test_anchor_snaps_on_gross_disagreement():
    tracker = FlowTracker()
    tracker.step(frame_with_patch(100, 100))
    tracker.anchor("t1", (100, 100, 220, 220))
    tracker.step(frame_with_patch(100, 100))
    tracker.anchor("t1", (400, 300, 520, 420))  # IOU ~0 -> detection wins
    assert tracker.boxes()["t1"] == (400, 300, 520, 420)


def test_compensate_bbox_shifts_by_flow_motion():
    from identity_lab.tracking.flow import bbox_center, compensate_bbox

    stale_det = (100, 100, 220, 220)
    at_submit = bbox_center((100, 100, 220, 220))   # box center at submit
    now = bbox_center((140, 110, 260, 230))         # flow moved +40, +10
    corrected = compensate_bbox(stale_det, at_submit, now)
    assert corrected == (140, 110, 260, 230)
    # Unknown centers -> unchanged
    assert compensate_bbox(stale_det, None, now) == stale_det
    assert compensate_bbox(stale_det, at_submit, None) == stale_det


def test_two_independent_tracks():
    tracker = FlowTracker()
    patch_b = RNG.integers(0, 255, size=(120, 120, 3), dtype=np.uint8)

    def two_patch_frame(xa, xb):
        frame = np.full((FRAME_H, FRAME_W, 3), 90, dtype=np.uint8)
        frame[100:220, xa : xa + 120] = PATCH
        frame[300:420, xb : xb + 120] = patch_b
        return frame

    xa, xb = 50, 400
    tracker.step(two_patch_frame(xa, xb))
    tracker.anchor("a", (xa, 100, xa + 120, 220))
    tracker.anchor("b", (xb, 300, xb + 120, 420))
    for _ in range(8):
        xa += 7
        xb -= 7  # moving toward each other vertically separated
        boxes = tracker.step(two_patch_frame(xa, xb))
    assert abs(center(boxes["a"])[0] - (xa + 60)) < 12
    assert abs(center(boxes["b"])[0] - (xb + 60)) < 12
    assert center(boxes["a"])[1] < center(boxes["b"])[1]  # never swapped


def test_forget_and_reset():
    tracker = FlowTracker()
    tracker.step(frame_with_patch(100, 100))
    tracker.anchor("t1", (100, 100, 220, 220))
    tracker.forget("t1")
    assert tracker.boxes() == {}
    tracker.anchor("t2", (10, 10, 60, 60))
    tracker.reset()
    assert tracker.boxes() == {}
    assert not tracker.is_coasting("t2")
