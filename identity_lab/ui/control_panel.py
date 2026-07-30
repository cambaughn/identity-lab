"""Right-hand control panel — camera controls and status readouts."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from identity_lab.camera.capture import CameraState
from identity_lab.ui import theme

PROBE_CAMERA_INDICES = 4  # macOS exposes no device names via OpenCV; offer 0..3


class ControlPanel(QWidget):
    start_requested = Signal()
    stop_requested = Signal()
    camera_index_changed = Signal(int)

    def __init__(self, initial_camera_index: int = 0, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(230)
        root = QVBoxLayout(self)
        root.setSpacing(theme.SPACING)
        root.setContentsMargins(0, 0, 0, 0)

        cam_box = QGroupBox("CAMERA")
        cam_layout = QVBoxLayout(cam_box)
        cam_layout.setSpacing(theme.SPACING)

        self.start_button = QPushButton("START CAMERA")
        self.stop_button = QPushButton("STOP CAMERA")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_requested)
        self.stop_button.clicked.connect(self.stop_requested)

        input_label = QLabel("INPUT")
        input_label.setObjectName("secondary")
        self.camera_selector = QComboBox()
        for i in range(PROBE_CAMERA_INDICES):
            self.camera_selector.addItem(f"CAM {i}", i)
        if 0 <= initial_camera_index < PROBE_CAMERA_INDICES:
            self.camera_selector.setCurrentIndex(initial_camera_index)
        self.camera_selector.currentIndexChanged.connect(
            lambda _: self.camera_index_changed.emit(self.selected_camera_index())
        )

        cam_layout.addWidget(self.start_button)
        cam_layout.addWidget(self.stop_button)
        cam_layout.addWidget(input_label)
        cam_layout.addWidget(self.camera_selector)

        status_box = QGroupBox("STATUS")
        status_layout = QVBoxLayout(status_box)
        status_layout.setSpacing(theme.SPACING // 2)
        self.state_label = QLabel(CameraState.OFFLINE.value)
        self.resolution_label = QLabel("RES  ----x----")
        self.resolution_label.setObjectName("secondary")
        self.message_label = QLabel("")
        self.message_label.setObjectName("secondary")
        self.message_label.setWordWrap(True)
        status_layout.addWidget(self.state_label)
        status_layout.addWidget(self.resolution_label)
        status_layout.addWidget(self.message_label)

        root.addWidget(cam_box)
        root.addWidget(status_box)
        root.addStretch(1)

    def selected_camera_index(self) -> int:
        return int(self.camera_selector.currentData())

    def set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def show_state(self, state: CameraState) -> None:
        self.state_label.setText(state.value)
        is_error = state is CameraState.ERROR
        self.state_label.setObjectName("error" if is_error else "")
        # Re-polish so the objectName-based style applies immediately.
        self.state_label.style().unpolish(self.state_label)
        self.state_label.style().polish(self.state_label)
        if not is_error:
            self.message_label.setText("")

    def show_message(self, text: str) -> None:
        self.message_label.setText(text)

    def show_resolution(self, width: int | None, height: int | None) -> None:
        if width and height:
            self.resolution_label.setText(f"RES  {width}x{height}")
        else:
            self.resolution_label.setText("RES  ----x----")
