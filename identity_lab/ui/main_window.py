"""Main window — wires camera, inference, panel, and diagnostics together.

Threads: one CameraWorker thread per camera session (fresh per Start), and
one long-lived InferenceWorker thread that loads the model at launch and
analyzes submitted frames until the app closes. Frame flow per camera frame:
display immediately (mirrored), and — when the model is ready and the
scheduler admits the frame — submit it for inference. Both hand-offs use
latest-frame-wins holders, so neither display nor inference can queue up.
"""

from pathlib import Path

from PySide6.QtCore import QByteArray, QThread
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget

from identity_lab.camera.capture import CameraState, CameraWorker
from identity_lab.camera.devices import resolve_device, snapshot_devices
from identity_lab.camera.frames import to_display_qimage
from identity_lab.config.settings import AppSettings, load_settings, save_settings
from identity_lab.diagnostics.metrics import FpsCounter, StatusLog
from identity_lab.ui import theme
from identity_lab.ui.control_panel import ControlPanel
from identity_lab.ui.video_widget import VideoWidget
from identity_lab.vision.types import InferenceResult, InferenceScheduler, filter_small_faces
from identity_lab.vision.worker import InferenceWorker, ModelState

CONSENT_NOTICE = (
    "EXPERIMENTAL LOCAL FACE-RECOGNITION RESEARCH TOOL — USE ONLY WITH THE "
    "INFORMED CONSENT OF EVERYONE ON CAMERA — ALL DATA REMAINS ON THIS MACHINE"
)


class MainWindow(QMainWindow):
    def __init__(
        self, settings_file: Path | None = None, start_inference: bool = True
    ) -> None:
        super().__init__()
        self.setWindowTitle("IDENTITY LAB")
        self._settings_file = settings_file
        self._settings: AppSettings = load_settings(settings_file)

        self._cam_thread: QThread | None = None
        self._cam_worker: CameraWorker | None = None
        self._pending_restart = False
        self._camera_state = CameraState.OFFLINE

        self._inf_thread: QThread | None = None
        self._inf_worker: InferenceWorker | None = None
        self._model_ready = False
        self._scheduler = InferenceScheduler(every_n=self._settings.infer_every_n)
        self._last_result: InferenceResult | None = None

        self._fps = FpsCounter()
        self._status_log = StatusLog()

        # -- layout --
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(
            theme.SPACING, theme.SPACING, theme.SPACING, theme.SPACING
        )
        outer.setSpacing(theme.SPACING)
        content = QHBoxLayout()
        content.setSpacing(theme.SPACING)
        self.video = VideoWidget()
        self.panel = ControlPanel(
            show_landmarks=self._settings.show_landmarks,
            debug_mode=self._settings.debug_mode,
            infer_every_n=self._settings.infer_every_n,
            min_face_px=self._settings.min_face_px,
        )
        content.addWidget(self.video, stretch=1)
        content.addWidget(self.panel)
        consent = QLabel(CONSENT_NOTICE)
        consent.setObjectName("consent")
        consent.setWordWrap(True)
        outer.addLayout(content, stretch=1)
        outer.addWidget(consent)
        self.setCentralWidget(central)

        # -- panel signals --
        self.panel.start_requested.connect(self.start_camera)
        self.panel.stop_requested.connect(self.stop_camera)
        self.panel.device_selected.connect(self._on_device_selected)
        self.panel.landmarks_toggled.connect(self._on_landmarks_toggled)
        self.panel.debug_toggled.connect(self._on_debug_toggled)
        self.panel.infer_every_n_changed.connect(self._on_interval_changed)
        self.panel.min_face_px_changed.connect(self._on_min_face_changed)

        # -- camera device list (Qt enumeration + hot-plug updates) --
        self._media_devices = QMediaDevices(self)
        self._media_devices.videoInputsChanged.connect(self._refresh_devices)
        self._refresh_devices(initial=True)

        if self._settings.window_geometry:
            self.restoreGeometry(
                QByteArray.fromBase64(self._settings.window_geometry.encode())
            )
        else:
            self.resize(1040, 640)

        self.log_event("SYSTEM START")
        if start_inference:
            self._start_inference_worker()

    # -- status log --

    def log_event(self, message: str) -> None:
        event = self._status_log.add(message)
        self.panel.append_log(StatusLog.format_event(event))

    # -- camera device list --

    def _refresh_devices(self, initial: bool = False) -> None:
        devices = snapshot_devices()
        selected, reason = resolve_device(self._settings.camera_device_id, devices)
        self.panel.set_devices(
            devices, selected.device_id if selected else None
        )
        if initial:
            names = ", ".join(d.name for d in devices) or "none"
            self.log_event(f"CAMERAS DETECTED: {names}")
        else:
            self.log_event(f"CAMERA LIST CHANGED ({len(devices)} DEVICE(S))")
        if selected is None:
            self.log_event("NO CAMERAS AVAILABLE")
            if self._cam_thread is not None:
                self.stop_camera()
            return
        if reason != "saved" and self._settings.camera_device_id is not None:
            self.log_event(f"SAVED CAMERA NOT FOUND — USING {selected.name.upper()}")
        if self._settings.camera_device_id != selected.device_id:
            self._settings.camera_device_id = selected.device_id
            self._save_settings()
        # If the camera we are currently running disappeared, stop cleanly.
        if self._cam_thread is not None and self._cam_worker is not None:
            current_positions = [d.position for d in devices]
            if self._cam_worker.camera_index not in current_positions:
                self.log_event("ACTIVE CAMERA REMOVED — STOPPING")
                self.stop_camera()

    def _on_device_selected(self, device) -> None:
        if self._settings.camera_device_id == device.device_id:
            return
        self._settings.camera_device_id = device.device_id
        self._save_settings()
        self.log_event(f"INPUT SELECTED: {device.name.upper()}")
        if self._cam_thread is not None:
            self._pending_restart = True
            self.stop_camera()

    # -- camera lifecycle --

    def start_camera(self) -> None:
        if self._cam_thread is not None:
            return
        device = self.panel.selected_device()
        if device is None:
            self.log_event("START IGNORED — NO CAMERA")
            return
        self.panel.set_running(True)
        self.log_event(f"CAMERA START: {device.name.upper()}")
        worker = CameraWorker(device.position)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.frame_available.connect(self._on_frame_available)
        worker.state_changed.connect(self._on_camera_state)
        worker.error.connect(self._on_camera_error)
        worker.finished.connect(thread.quit)
        thread.finished.connect(self._on_cam_thread_finished)
        self._cam_worker = worker
        self._cam_thread = thread
        self._fps.reset()
        self._scheduler.reset()
        thread.start()

    def stop_camera(self) -> None:
        if self._cam_worker is not None:
            self._cam_worker.request_stop()

    def _on_cam_thread_finished(self) -> None:
        if self._cam_worker is not None:
            self._cam_worker.deleteLater()
        if self._cam_thread is not None:
            self._cam_thread.deleteLater()
        self._cam_worker = None
        self._cam_thread = None
        self.panel.set_running(False)
        self.panel.show_fps(0.0)
        self.panel.show_latency(None)
        self.panel.show_face_count(None)
        self._last_result = None
        self.video.clear_overlay()
        self.log_event("CAMERA STOPPED")
        if self._pending_restart:
            self._pending_restart = False
            self.start_camera()

    # -- inference worker lifecycle --

    def _start_inference_worker(self) -> None:
        worker = InferenceWorker()
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.model_state_changed.connect(self._on_model_state)
        worker.result_ready.connect(self._on_inference_result)
        worker.error.connect(self._on_inference_error)
        worker.finished.connect(thread.quit)
        self._inf_worker = worker
        self._inf_thread = thread
        thread.start()

    def _on_model_state(self, state_value: str) -> None:
        self._model_ready = state_value == ModelState.READY.value
        is_error = state_value == ModelState.ERROR.value
        self.panel.show_model_state(state_value, is_error=is_error)
        self.log_event(state_value)

    def _on_inference_error(self, detail: str) -> None:
        self.log_event(f"INFERENCE ERROR: {detail}")

    # -- frame & result flow (UI thread) --

    def _on_frame_available(self) -> None:
        if self._cam_worker is None:
            return
        frame = self._cam_worker.take_frame()
        if frame is None:
            return
        self._fps.tick()
        self.panel.show_fps(self._fps.fps)
        self.video.set_frame(to_display_qimage(frame, mirrored=True))
        if (
            self._model_ready
            and self._inf_worker is not None
            and self._scheduler.should_submit()
        ):
            self._inf_worker.submit(frame)

    def _on_inference_result(self, result: InferenceResult) -> None:
        if self._cam_worker is None:
            return  # camera stopped while inference was in flight
        self._last_result = result
        self.panel.show_latency(result.latency_ms)
        self._apply_overlay()

    def _apply_overlay(self) -> None:
        result = self._last_result
        if result is None:
            return
        faces = filter_small_faces(result.faces, self._settings.min_face_px)
        self.panel.show_face_count(len(faces))
        self.video.set_overlay(
            faces,
            result.frame_w,
            result.frame_h,
            show_landmarks=self._settings.show_landmarks,
            debug=self._settings.debug_mode,
        )

    # -- camera state handlers --

    def _on_camera_state(self, state_value: str) -> None:
        state = CameraState(state_value)
        if state is CameraState.OFFLINE and self._camera_state is CameraState.ERROR:
            return
        self._camera_state = state
        self.video.set_state(state)
        self.panel.show_state(state)
        if state is CameraState.READY:
            self.log_event("CAMERA READY")

    def _on_camera_error(self, detail: str) -> None:
        self._camera_state = CameraState.ERROR
        self.video.set_state(CameraState.ERROR, detail)
        self.panel.show_state(CameraState.ERROR)
        self.panel.show_message(detail)
        self.log_event(f"CAMERA ERROR: {detail}")

    # -- detection settings handlers --

    def _on_landmarks_toggled(self, on: bool) -> None:
        self._settings.show_landmarks = on
        self._save_settings()
        self._apply_overlay()

    def _on_debug_toggled(self, on: bool) -> None:
        self._settings.debug_mode = on
        self._save_settings()
        self._apply_overlay()

    def _on_interval_changed(self, n: int) -> None:
        self._settings.infer_every_n = n
        self._scheduler.every_n = n
        self._save_settings()
        self.log_event(f"INFER INTERVAL: EVERY {n} FRAME(S)")

    def _on_min_face_changed(self, px: int) -> None:
        self._settings.min_face_px = px
        self._save_settings()
        self._apply_overlay()

    # -- shutdown & persistence --

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._pending_restart = False
        if self._cam_worker is not None:
            self._cam_worker.request_stop()
        if self._inf_worker is not None:
            self._inf_worker.request_stop()
        for thread in (self._cam_thread, self._inf_thread):
            if thread is not None:
                thread.quit()
                thread.wait(5000)
        self._save_settings(include_geometry=True)
        super().closeEvent(event)

    def _save_settings(self, include_geometry: bool = False) -> None:
        if include_geometry:
            self._settings.window_geometry = bytes(
                self.saveGeometry().toBase64()
            ).decode()
        save_settings(self._settings, self._settings_file)
