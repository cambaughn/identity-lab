"""Overlay coordinate math — pure functions, no Qt types.

The preview is drawn mirrored, scaled to fit, and centered. Overlays are
produced in original frame coordinates by the vision layer, so they must go
through the same mirror + scale + offset transform to land on the face.
"""


def display_transform(
    frame_w: int, frame_h: int, widget_w: int, widget_h: int
) -> tuple[float, float, float]:
    """Scale factor and top-left offsets for an aspect-fit centered image."""
    if frame_w <= 0 or frame_h <= 0 or widget_w <= 0 or widget_h <= 0:
        return 1.0, 0.0, 0.0
    scale = min(widget_w / frame_w, widget_h / frame_h)
    off_x = (widget_w - frame_w * scale) / 2
    off_y = (widget_h - frame_h * scale) / 2
    return scale, off_x, off_y


def map_bbox(
    bbox: tuple[float, float, float, float],
    frame_w: int,
    scale: float,
    off_x: float,
    off_y: float,
    mirrored: bool = True,
) -> tuple[float, float, float, float]:
    """Map a frame-space bbox (x1, y1, x2, y2) to widget-space (x, y, w, h)."""
    x1, y1, x2, y2 = bbox
    if mirrored:
        x1, x2 = frame_w - x2, frame_w - x1
    return (
        off_x + x1 * scale,
        off_y + y1 * scale,
        (x2 - x1) * scale,
        (y2 - y1) * scale,
    )


def map_point(
    point: tuple[float, float],
    frame_w: int,
    scale: float,
    off_x: float,
    off_y: float,
    mirrored: bool = True,
) -> tuple[float, float]:
    """Map a frame-space point to widget-space."""
    x, y = point
    if mirrored:
        x = frame_w - x
    return off_x + x * scale, off_y + y * scale
