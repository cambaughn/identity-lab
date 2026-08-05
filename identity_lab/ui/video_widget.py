"""Video preview area — live frame, console state text, and face overlays."""

from dataclasses import dataclass, field

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from identity_lab.camera.capture import CameraState
from identity_lab.ui import theme
from identity_lab.ui.overlay import display_transform, map_bbox, map_point
from identity_lab.vision.types import DetectedFace

OVERLAY_FONT_PX = 11
LABEL_FONT_PX = 13


@dataclass(frozen=True)
class OverlayFace:
    """One face plus its display geometry/strings (composed upstream).

    bbox is the *displayed* box — flow-tracked between detector passes, so
    it may differ from face.bbox (the last detection). landmarks are already
    shifted to match bbox."""

    face: DetectedFace
    bbox: tuple[int, int, int, int]
    label: str                          # "CAM 0.62" or "UNKNOWN"
    known: bool
    debug_lines: tuple[str, ...] = field(default_factory=tuple)
    landmarks: object = None            # np.ndarray (106, 2) or None


class VideoWidget(QWidget):
    """Black instrument-panel viewport: live video with face overlays when
    running, otherwise a large amber state label."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self._image: QImage | None = None
        self._state: CameraState = CameraState.OFFLINE
        self._error_detail: str = ""
        self._entries: tuple[OverlayFace, ...] = ()
        self._frame_size: tuple[int, int] | None = None  # (w, h)
        self._show_landmarks = False
        self._debug = False

    def set_frame(self, image: QImage) -> None:
        self._image = image
        self.update()

    def set_overlay(
        self,
        entries: tuple[OverlayFace, ...],
        frame_w: int,
        frame_h: int,
        show_landmarks: bool,
        debug: bool,
    ) -> None:
        self._entries = entries
        self._frame_size = (frame_w, frame_h)
        self._show_landmarks = show_landmarks
        self._debug = debug
        self.update()

    def clear_overlay(self) -> None:
        self._entries = ()
        self._frame_size = None
        self.update()

    def set_state(self, state: CameraState, error_detail: str = "") -> None:
        self._state = state
        self._error_detail = error_detail
        if state is not CameraState.READY:
            self._image = None
            self.clear_overlay()
        self.update()

    # -- painting --

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.BG))
        pen = QPen(QColor(theme.AMBER_DIM))
        pen.setWidth(theme.BORDER_W)
        painter.setPen(pen)
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

        if self._state is CameraState.READY and self._image is not None:
            scaled = self._image.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawImage(x, y, scaled)
            self._draw_overlays(painter)
        else:
            self._draw_state_text(painter)
        painter.end()

    def _draw_overlays(self, painter: QPainter) -> None:
        if not self._entries or self._frame_size is None:
            return
        frame_w, frame_h = self._frame_size
        scale, off_x, off_y = display_transform(
            frame_w, frame_h, self.width(), self.height()
        )
        label_font = QFont()
        label_font.setFamilies(["Menlo", "SF Mono", "Monaco", "Courier New"])
        label_font.setPixelSize(LABEL_FONT_PX)
        debug_font = QFont(label_font)
        debug_font.setPixelSize(OVERLAY_FONT_PX)

        landmark_pen = QPen(QColor(theme.AMBER_DIM))
        landmark_pen.setWidth(2)

        for entry in self._entries:
            face = entry.face
            color = QColor(theme.AMBER if entry.known else theme.AMBER_DIM)
            box_pen = QPen(color)
            box_pen.setWidth(theme.BORDER_W)
            # NOTE: the overlay mirror must match the preview mirror.
            x, y, w, h = map_bbox(
                entry.bbox, frame_w, scale, off_x, off_y, mirrored=True
            )
            painter.setPen(box_pen)
            painter.drawRect(QRectF(x, y, w, h))

            # Name label above the box.
            painter.setFont(label_font)
            label_rect = QRectF(
                x, y - LABEL_FONT_PX - 8, max(w, 160), LABEL_FONT_PX + 6
            )
            painter.fillRect(label_rect, QColor(0, 0, 0, 190))
            painter.setPen(QPen(color))
            painter.drawText(
                label_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                f" {entry.label}",
            )

            # Debug lines below the box.
            if self._debug and entry.debug_lines:
                painter.setFont(debug_font)
                painter.setPen(QPen(QColor(theme.AMBER_DIM)))
                line_h = OVERLAY_FONT_PX + 3
                for i, line in enumerate(entry.debug_lines):
                    dbg_rect = QRectF(
                        x, y + h + 2 + i * line_h, max(w, 230), line_h
                    )
                    painter.fillRect(dbg_rect, QColor(0, 0, 0, 170))
                    painter.drawText(
                        dbg_rect,
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                        f" {line}",
                    )

            if self._show_landmarks and entry.landmarks is not None:
                painter.setPen(landmark_pen)
                points = [
                    QPointF(*map_point((px, py), frame_w, scale, off_x, off_y))
                    for px, py in entry.landmarks
                ]
                painter.drawPoints(points)

    def _draw_state_text(self, painter: QPainter) -> None:
        is_error = self._state is CameraState.ERROR
        color = QColor(theme.ERROR if is_error else theme.AMBER)
        font = QFont()
        font.setFamilies(["Menlo", "SF Mono", "Monaco", "Courier New"])
        font.setPixelSize(theme.FONT_SIZE_STATE)
        painter.setFont(font)
        painter.setPen(color)

        rect = self.rect()
        text = f"** {self._state.value} **"
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

        if is_error and self._error_detail:
            font.setPixelSize(theme.FONT_SIZE)
            painter.setFont(font)
            painter.setPen(QColor(theme.AMBER_DIM))
            detail_rect = rect.adjusted(20, rect.height() // 2 + 24, -20, -20)
            painter.drawText(
                detail_rect,
                Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap,
                self._error_detail,
            )
