"""Video preview area — live frame, console state text, and face overlays."""

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from identity_lab.camera.capture import CameraState
from identity_lab.ui import theme
from identity_lab.ui.overlay import display_transform, map_bbox, map_point
from identity_lab.vision.types import DetectedFace

OVERLAY_FONT_PX = 11


class VideoWidget(QWidget):
    """Black instrument-panel viewport: live video with face overlays when
    running, otherwise a large amber state label."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self._image: QImage | None = None
        self._state: CameraState = CameraState.OFFLINE
        self._error_detail: str = ""
        self._faces: tuple[DetectedFace, ...] = ()
        self._frame_size: tuple[int, int] | None = None  # (w, h)
        self._show_landmarks = False
        self._debug = False

    def set_frame(self, image: QImage) -> None:
        self._image = image
        self.update()

    def set_overlay(
        self,
        faces: tuple[DetectedFace, ...],
        frame_w: int,
        frame_h: int,
        show_landmarks: bool,
        debug: bool,
    ) -> None:
        self._faces = faces
        self._frame_size = (frame_w, frame_h)
        self._show_landmarks = show_landmarks
        self._debug = debug
        self.update()

    def clear_overlay(self) -> None:
        self._faces = ()
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
        if not self._faces or self._frame_size is None:
            return
        frame_w, frame_h = self._frame_size
        scale, off_x, off_y = display_transform(
            frame_w, frame_h, self.width(), self.height()
        )
        font = QFont()
        font.setFamilies(["Menlo", "SF Mono", "Monaco", "Courier New"])
        font.setPixelSize(OVERLAY_FONT_PX)
        painter.setFont(font)

        box_pen = QPen(QColor(theme.AMBER))
        box_pen.setWidth(theme.BORDER_W)
        landmark_pen = QPen(QColor(theme.AMBER_DIM))
        landmark_pen.setWidth(2)

        for face in self._faces:
            # NOTE: the overlay mirror must match the preview mirror.
            x, y, w, h = map_bbox(
                face.bbox, frame_w, scale, off_x, off_y, mirrored=True
            )
            painter.setPen(box_pen)
            painter.drawRect(QRectF(x, y, w, h))

            if self._debug:
                label = f"DET {face.det_score:.2f}  {face.size_px}PX"
                text_rect = QRectF(x, y - OVERLAY_FONT_PX - 6, max(w, 130), OVERLAY_FONT_PX + 4)
                painter.fillRect(text_rect, QColor(0, 0, 0, 180))
                painter.drawText(
                    text_rect,
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    label,
                )

            if self._show_landmarks and face.landmarks is not None:
                painter.setPen(landmark_pen)
                points = [
                    QPointF(*map_point((px, py), frame_w, scale, off_x, off_y))
                    for px, py in face.landmarks
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
