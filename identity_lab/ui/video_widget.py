"""Video preview area — draws the live frame or a console state message."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from identity_lab.camera.capture import CameraState
from identity_lab.ui import theme


class VideoWidget(QWidget):
    """Black instrument-panel viewport: live video when running, otherwise a
    large amber state label (CAMERA OFFLINE / INITIALIZING / ERROR)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self._image: QImage | None = None
        self._state: CameraState = CameraState.OFFLINE
        self._error_detail: str = ""

    def set_frame(self, image: QImage) -> None:
        self._image = image
        self.update()

    def set_state(self, state: CameraState, error_detail: str = "") -> None:
        self._state = state
        self._error_detail = error_detail
        if state is not CameraState.READY:
            self._image = None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.BG))

        # Thin border around the viewport, like an instrument bezel.
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
        else:
            self._draw_state_text(painter)
        painter.end()

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
