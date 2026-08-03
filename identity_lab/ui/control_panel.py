"""Right-hand control panel — camera/detection controls, readouts, event log."""

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QListView,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from identity_lab.camera.capture import CameraState
from identity_lab.camera.devices import CameraDeviceInfo
from identity_lab.ui import theme

INFER_INTERVAL_CHOICES = [1, 2, 3, 5, 10]
NO_CAMERA_TEXT = "NO CAMERAS DETECTED"


class ConsoleComboBox(QComboBox):
    """QComboBox whose dropdown is pinned flush below the control.

    On macOS, Qt positions the popup in native overlap mode with its own
    margins regardless of the `combobox-popup: 0` style hint (measured:
    7px left shift, wrong width, overlapping the control). Setting the
    popup container's geometry explicitly after showPopup() is the only
    reliable way to make control + dropdown read as one component.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setView(QListView())

    def showPopup(self) -> None:  # noqa: N802 (Qt override)
        super().showPopup()
        container = self.view().window()
        rows = min(self.count(), self.maxVisibleItems())
        row_h = sum(self.view().sizeHintForRow(i) for i in range(rows))
        # Container frame + view frame borders (frameWidth() reports 0 under
        # stylesheets, so measure the actual chrome around the view instead).
        chrome = container.height() - self.view().height()
        if chrome < 0 or chrome > 20:
            chrome = 2
        below = self.mapToGlobal(QPoint(0, self.height()))
        container.setGeometry(below.x(), below.y(), self.width(), row_h + chrome + 2)


class ControlPanel(QWidget):
    start_requested = Signal()
    stop_requested = Signal()
    enroll_requested = Signal()
    device_selected = Signal(object)      # CameraDeviceInfo
    landmarks_toggled = Signal(bool)
    debug_toggled = Signal(bool)
    infer_every_n_changed = Signal(int)
    min_face_px_changed = Signal(int)
    threshold_changed = Signal(float)
    margin_changed = Signal(float)
    top_k_changed = Signal(int)

    def __init__(
        self,
        show_landmarks: bool = False,
        debug_mode: bool = False,
        infer_every_n: int = 2,
        min_face_px: int = 60,
        recognition_threshold: float = 0.40,
        match_margin: float = 0.08,
        top_k: int = 3,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setFixedWidth(250)
        self._devices: list[CameraDeviceInfo] = []

        root = QVBoxLayout(self)
        root.setSpacing(theme.SPACING)
        root.setContentsMargins(0, 0, 0, 0)

        # -- CAMERA --
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
        self.camera_selector = ConsoleComboBox()
        self.camera_selector.currentIndexChanged.connect(self._on_selector_changed)
        cam_layout.addWidget(self.start_button)
        cam_layout.addWidget(self.stop_button)
        cam_layout.addWidget(input_label)
        cam_layout.addWidget(self.camera_selector)

        # -- IDENTITY --
        id_box = QGroupBox("IDENTITY")
        id_layout = QVBoxLayout(id_box)
        id_layout.setSpacing(theme.SPACING)
        self.enroll_button = QPushButton("ENROLL PERSON")
        self.enroll_button.setEnabled(False)
        self.enroll_button.setToolTip(
            "Requires a running camera and a loaded model"
        )
        self.enroll_button.clicked.connect(self.enroll_requested)
        id_layout.addWidget(self.enroll_button)

        # -- DETECTION --
        det_box = QGroupBox("DETECTION")
        det_layout = QVBoxLayout(det_box)
        det_layout.setSpacing(theme.SPACING // 2)
        self.landmarks_check = QCheckBox("LANDMARKS")
        self.landmarks_check.setChecked(show_landmarks)
        self.landmarks_check.toggled.connect(self.landmarks_toggled)
        self.debug_check = QCheckBox("DEBUG INFO")
        self.debug_check.setChecked(debug_mode)
        self.debug_check.toggled.connect(self.debug_toggled)

        interval_label = QLabel("INFER INTERVAL")
        interval_label.setObjectName("secondary")
        self.interval_selector = ConsoleComboBox()
        for n in INFER_INTERVAL_CHOICES:
            label = "EVERY FRAME" if n == 1 else f"EVERY {n} FRAMES"
            self.interval_selector.addItem(label, n)
        idx = (
            INFER_INTERVAL_CHOICES.index(infer_every_n)
            if infer_every_n in INFER_INTERVAL_CHOICES
            else 1
        )
        self.interval_selector.setCurrentIndex(idx)
        self.interval_selector.currentIndexChanged.connect(
            lambda _: self.infer_every_n_changed.emit(
                int(self.interval_selector.currentData())
            )
        )

        self.min_face_label = QLabel("")
        self.min_face_label.setObjectName("secondary")
        self.min_face_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_face_slider.setRange(0, 300)
        self.min_face_slider.setSingleStep(10)
        self.min_face_slider.setValue(min_face_px)
        self._update_min_face_label(min_face_px)
        self.min_face_slider.valueChanged.connect(self._on_min_face_changed)

        det_layout.addWidget(self.landmarks_check)
        det_layout.addWidget(self.debug_check)
        det_layout.addWidget(interval_label)
        det_layout.addWidget(self.interval_selector)
        det_layout.addWidget(self.min_face_label)
        det_layout.addWidget(self.min_face_slider)

        # -- RECOGNITION --
        rec_box = QGroupBox("RECOGNITION")
        rec_layout = QVBoxLayout(rec_box)
        rec_layout.setSpacing(theme.SPACING // 2)

        self.threshold_label = QLabel("")
        self.threshold_label.setObjectName("secondary")
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(5, 95)  # 0.05 .. 0.95
        self.threshold_slider.setValue(round(recognition_threshold * 100))
        self._update_threshold_label(self.threshold_slider.value())
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)

        self.margin_label = QLabel("")
        self.margin_label.setObjectName("secondary")
        self.margin_slider = QSlider(Qt.Orientation.Horizontal)
        self.margin_slider.setRange(0, 50)  # 0.00 .. 0.50
        self.margin_slider.setValue(round(match_margin * 100))
        self._update_margin_label(self.margin_slider.value())
        self.margin_slider.valueChanged.connect(self._on_margin_changed)

        topk_label = QLabel("TOP-K SAMPLES")
        topk_label.setObjectName("secondary")
        self.topk_selector = ConsoleComboBox()
        for k in (1, 3, 5):
            self.topk_selector.addItem(f"TOP {k}", k)
        self.topk_selector.setCurrentIndex(
            {1: 0, 3: 1, 5: 2}.get(top_k, 1)
        )
        self.topk_selector.currentIndexChanged.connect(
            lambda _: self.top_k_changed.emit(int(self.topk_selector.currentData()))
        )

        rec_layout.addWidget(self.threshold_label)
        rec_layout.addWidget(self.threshold_slider)
        rec_layout.addWidget(self.margin_label)
        rec_layout.addWidget(self.margin_slider)
        rec_layout.addWidget(topk_label)
        rec_layout.addWidget(self.topk_selector)

        # -- READOUTS --
        readout_box = QGroupBox("READOUTS")
        readout_layout = QVBoxLayout(readout_box)
        readout_layout.setSpacing(theme.SPACING // 2)
        self.state_label = QLabel(CameraState.OFFLINE.value)
        self.model_label = QLabel("MODEL --")
        self.fps_label = QLabel("FPS --.-")
        self.fps_label.setObjectName("secondary")
        self.latency_label = QLabel("LATENCY --- MS")
        self.latency_label.setObjectName("secondary")
        self.faces_label = QLabel("FACES --")
        self.id_label = QLabel("ID --")
        self.message_label = QLabel("")
        self.message_label.setObjectName("secondary")
        self.message_label.setWordWrap(True)
        for w in (
            self.state_label,
            self.model_label,
            self.fps_label,
            self.latency_label,
            self.faces_label,
            self.id_label,
            self.message_label,
        ):
            readout_layout.addWidget(w)

        # -- EVENT LOG --
        log_box = QGroupBox("EVENT LOG")
        log_layout = QVBoxLayout(log_box)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(200)
        self.log_view.setFixedHeight(120)
        log_layout.addWidget(self.log_view)

        root.addWidget(cam_box)
        root.addWidget(id_box)
        root.addWidget(det_box)
        root.addWidget(rec_box)
        root.addWidget(readout_box)
        root.addWidget(log_box)
        root.addStretch(1)

    # -- camera device list --

    def set_devices(
        self, devices: list[CameraDeviceInfo], selected_id: str | None
    ) -> None:
        """Rebuild the selector; keep selection on selected_id when present."""
        self._devices = devices
        self.camera_selector.blockSignals(True)
        self.camera_selector.clear()
        if not devices:
            self.camera_selector.addItem(NO_CAMERA_TEXT, None)
            self.camera_selector.setEnabled(False)
            self.start_button.setEnabled(False)
        else:
            self.camera_selector.setEnabled(True)
            if not self.stop_button.isEnabled():
                self.start_button.setEnabled(True)
            for dev in devices:
                self.camera_selector.addItem(dev.name.upper(), dev.device_id)
            index = next(
                (i for i, d in enumerate(devices) if d.device_id == selected_id), 0
            )
            self.camera_selector.setCurrentIndex(index)
        self.camera_selector.blockSignals(False)

    def selected_device(self) -> CameraDeviceInfo | None:
        idx = self.camera_selector.currentIndex()
        if 0 <= idx < len(self._devices):
            return self._devices[idx]
        return None

    def _on_selector_changed(self, _index: int) -> None:
        dev = self.selected_device()
        if dev is not None:
            self.device_selected.emit(dev)

    # -- detection controls --

    def _on_min_face_changed(self, value: int) -> None:
        self._update_min_face_label(value)
        self.min_face_px_changed.emit(value)

    def _update_min_face_label(self, value: int) -> None:
        self.min_face_label.setText(f"MIN FACE {value:03d} PX")

    # -- recognition controls --

    def _on_threshold_changed(self, value: int) -> None:
        self._update_threshold_label(value)
        self.threshold_changed.emit(value / 100.0)

    def _update_threshold_label(self, value: int) -> None:
        self.threshold_label.setText(f"THRESHOLD {value / 100.0:.2f}")

    def _on_margin_changed(self, value: int) -> None:
        self._update_margin_label(value)
        self.margin_changed.emit(value / 100.0)

    def _update_margin_label(self, value: int) -> None:
        self.margin_label.setText(f"MARGIN {value / 100.0:.2f}")

    # -- run state & readouts --

    def set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running and bool(self._devices))
        self.stop_button.setEnabled(running)

    def show_state(self, state: CameraState) -> None:
        self.state_label.setText(state.value)
        is_error = state is CameraState.ERROR
        self.state_label.setObjectName("error" if is_error else "")
        self.state_label.style().unpolish(self.state_label)
        self.state_label.style().polish(self.state_label)
        if not is_error:
            self.message_label.setText("")

    def show_model_state(self, text: str, is_error: bool = False) -> None:
        self.model_label.setText(text)
        self.model_label.setObjectName("error" if is_error else "")
        self.model_label.style().unpolish(self.model_label)
        self.model_label.style().polish(self.model_label)

    def show_message(self, text: str) -> None:
        self.message_label.setText(text)

    def show_fps(self, fps: float) -> None:
        self.fps_label.setText(f"FPS {fps:4.1f}")

    def show_latency(self, ms: float | None) -> None:
        if ms is None:
            self.latency_label.setText("LATENCY --- MS")
        else:
            self.latency_label.setText(f"LATENCY {ms:3.0f} MS")

    def show_face_count(self, count: int | None) -> None:
        self.faces_label.setText("FACES --" if count is None else f"FACES {count:02d}")

    def show_recognition_result(self, text: str | None) -> None:
        self.id_label.setText("ID --" if text is None else f"ID {text}")

    def set_enroll_enabled(self, enabled: bool) -> None:
        self.enroll_button.setEnabled(enabled)

    def append_log(self, line: str) -> None:
        self.log_view.appendPlainText(line)
