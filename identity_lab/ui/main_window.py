"""Main window — wires the video widget, control panel, and camera worker.

Camera lifecycle: each Start creates a fresh CameraWorker on a fresh QThread;
Stop (or window close, or a camera error) asks the worker to stop, the worker
releases the camera in its finally block and signals finished, and the thread
is then shut down and discarded. Repeated start/stop cycles therefore never
reuse stale camera state.
"""

from pathlib import Path

from PySide6.QtCore import QByteArray, QThread
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget

from identity_lab.camera.capture import CameraState, CameraWorker
from identity_lab.camera.frames import to_display_qimage
from identity_lab.config.settings import AppSettings, load_settings, save_settings
from identity_lab.ui import theme
from identity_lab.ui.control_panel import ControlPanel
from identity_lab.ui.video_widget import VideoWidget

CONSENT_NOTICE = (
    "EXPERIMENTAL LOCAL FACE-RECOGNITION RESEARCH TOOL — USE ONLY WITH THE "
    "INFORMED CONSENT OF EVERYONE ON CAMERA — ALL DATA REMAINS ON THIS MACHINE"
)


class MainWindow(QMainWindow):
    def __init__(self, settings_file: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("IDENTITY LAB")
        self._settings_file = settings_file
        self._settings: AppSettings = load_settings(settings_file)

        self._thread: QThread | None = None
        self._worker: CameraWorker | None = None
        self._pending_restart = False
        self._camera_state = CameraState.OFFLINE

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(theme.SPACING, theme.SPACING, theme.SPACING, theme.SPACING)
        outer.setSpacing(theme.SPACING)

        content = QHBoxLayout()
        content.setSpacing(theme.SPACING)
        self.video = VideoWidget()
        self.panel = ControlPanel(initial_camera_index=self._settings.camera_index)
        content.addWidget(self.video, stretch=1)
        content.addWidget(self.panel)

        consent = QLabel(CONSENT_NOTICE)
        consent.setObjectName("consent")
        consent.setWordWrap(True)

        outer.addLayout(content, stretch=1)
        outer.addWidget(consent)
        self.setCentralWidget(central)

        self.panel.start_requested.connect(self.start_camera)
        self.panel.stop_requested.connect(self.stop_camera)
        self.panel.camera_index_changed.connect(self._on_camera_index_changed)

        if self._settings.window_geometry:
            self.restoreGeometry(
                QByteArray.fromBase64(self._settings.window_geometry.encode())
            )
        else:
            self.resize(960, 600)

    # -- camera lifecycle --

    def start_camera(self) -> None:
        if self._thread is not None:
            return
        self.panel.set_running(True)
        worker = CameraWorker(self.panel.selected_camera_index())
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.frame_available.connect(self._on_frame_available)
        worker.state_changed.connect(self._on_camera_state)
        worker.error.connect(self._on_camera_error)
        worker.finished.connect(thread.quit)
        thread.finished.connect(self._on_thread_finished)
        self._worker = worker
        self._thread = thread
        thread.start()

    def stop_camera(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()

    def _on_thread_finished(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self.panel.set_running(False)
        self.panel.show_resolution(None, None)
        if self._pending_restart:
            self._pending_restart = False
            self.start_camera()

    # -- worker signal handlers (UI thread) --

    def _on_frame_available(self) -> None:
        if self._worker is None:
            return
        frame = self._worker.take_frame()
        if frame is None:
            return
        h, w = frame.shape[:2]
        self.panel.show_resolution(w, h)
        self.video.set_frame(to_display_qimage(frame, mirrored=True))

    def _on_camera_state(self, state_value: str) -> None:
        state = CameraState(state_value)
        # Keep an ERROR display visible; don't let the worker's final
        # OFFLINE transition overwrite it.
        if state is CameraState.OFFLINE and self._camera_state is CameraState.ERROR:
            return
        self._camera_state = state
        self.video.set_state(state)
        self.panel.show_state(state)

    def _on_camera_error(self, detail: str) -> None:
        self._camera_state = CameraState.ERROR
        self.video.set_state(CameraState.ERROR, detail)
        self.panel.show_state(CameraState.ERROR)
        self.panel.show_message(detail)

    def _on_camera_index_changed(self, index: int) -> None:
        self._settings.camera_index = index
        self._save_settings()
        if self._thread is not None:
            # Restart on the new input once the current worker fully stops.
            self._pending_restart = True
            self.stop_camera()

    # -- shutdown & persistence --

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._pending_restart = False
        if self._worker is not None:
            self._worker.request_stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        self._save_settings(include_geometry=True)
        super().closeEvent(event)

    def _save_settings(self, include_geometry: bool = False) -> None:
        if include_geometry:
            self._settings.window_geometry = bytes(
                self.saveGeometry().toBase64()
            ).decode()
        save_settings(self._settings, self._settings_file)
