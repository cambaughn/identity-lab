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
from identity_lab.config.settings import (
    APP_DATA_DIR,
    AppSettings,
    load_settings,
    save_settings,
)
from identity_lab.diagnostics.metrics import FpsCounter, StatusLog
from identity_lab.identity.errors import IdentityStoreError
from identity_lab.identity.matcher import Matcher, MatchResult, build_gallery
from identity_lab.identity.store import DEFAULT_DB_FILENAME, IdentityStore
from identity_lab.ui import theme
from identity_lab.ui.control_panel import ControlPanel
from identity_lab.ui.enroll_dialog import EnrollDialog
from identity_lab.ui.video_widget import OverlayFace, VideoWidget
from identity_lab.vision.engine import MODEL_PACK
from identity_lab.vision.types import InferenceResult, InferenceScheduler, filter_small_faces
from identity_lab.vision.worker import InferenceWorker, ModelState

CONSENT_NOTICE = (
    "EXPERIMENTAL LOCAL FACE-RECOGNITION RESEARCH TOOL — USE ONLY WITH THE "
    "INFORMED CONSENT OF EVERYONE ON CAMERA — ALL DATA REMAINS ON THIS MACHINE"
)


class MainWindow(QMainWindow):
    def __init__(
        self,
        settings_file: Path | None = None,
        start_inference: bool = True,
        db_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("IDENTITY LAB")
        self._settings_file = settings_file
        self._settings: AppSettings = load_settings(settings_file)

        self._store: IdentityStore | None = None
        self._store_error: str | None = None
        try:
            self._store = IdentityStore(
                db_path or (APP_DATA_DIR / DEFAULT_DB_FILENAME)
            )
        except IdentityStoreError as exc:
            self._store_error = str(exc)

        self._cam_thread: QThread | None = None
        self._cam_worker: CameraWorker | None = None
        self._pending_restart = False
        self._camera_state = CameraState.OFFLINE

        self._inf_thread: QThread | None = None
        self._inf_worker: InferenceWorker | None = None
        self._model_ready = False
        self._scheduler = InferenceScheduler(every_n=self._settings.infer_every_n)
        self._last_result: InferenceResult | None = None
        self._last_matches: list[MatchResult | None] = []
        self._matcher = Matcher([])

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
            recognition_threshold=self._settings.recognition_threshold,
            match_margin=self._settings.match_margin,
            top_k=self._settings.top_k,
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
        self.panel.enroll_requested.connect(self.open_enrollment)
        self.panel.device_selected.connect(self._on_device_selected)
        self.panel.landmarks_toggled.connect(self._on_landmarks_toggled)
        self.panel.debug_toggled.connect(self._on_debug_toggled)
        self.panel.infer_every_n_changed.connect(self._on_interval_changed)
        self.panel.min_face_px_changed.connect(self._on_min_face_changed)
        self.panel.threshold_changed.connect(self._on_threshold_changed)
        self.panel.margin_changed.connect(self._on_margin_changed)
        self.panel.top_k_changed.connect(self._on_top_k_changed)

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
        if self._store_error:
            self.log_event(f"IDENTITY DB ERROR: {self._store_error}")
            self.panel.show_message(
                "Identity database unavailable — enrollment disabled. "
                "See event log."
            )
        self._reload_gallery(initial=True)
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
        self.panel.show_recognition_result(None)
        self._last_result = None
        self._last_matches = []
        self.video.clear_overlay()
        self.log_event("CAMERA STOPPED")
        if self._pending_restart:
            self._pending_restart = False
            self.start_camera()

    # -- inference worker lifecycle --

    def _start_inference_worker(self) -> None:
        worker = InferenceWorker()
        # Recognition needs embeddings on every pass now (CP8+). The
        # detection-only fast path remains available via this flag.
        worker.set_embeddings_enabled(True)
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
        self._update_enroll_enabled()
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
        # Match once per result (cheap: microseconds against a small gallery);
        # overlay refreshes reuse these matches until the next result.
        self._last_matches = [
            self._matcher.match(
                face.embedding,
                threshold=self._settings.recognition_threshold,
                margin=self._settings.match_margin,
                top_k=self._settings.top_k,
            )
            if face.embedding is not None
            else None
            for face in result.faces
        ]
        self.panel.show_latency(result.latency_ms)
        self._apply_overlay()

    def _compose_entry(
        self, face, match: MatchResult | None
    ) -> OverlayFace:
        if match is None:
            label, known = "UNKNOWN", False
        elif match.is_known:
            label, known = f"{match.display_name.upper()}  {match.similarity:.2f}", True
        elif match.similarity is not None:
            label, known = f"UNKNOWN  {match.similarity:.2f}", False
        else:
            label, known = "UNKNOWN", False

        debug_lines: list[str] = [f"DET {face.det_score:.2f}  {face.size_px}PX"]
        if match is not None:
            if match.best is not None:
                debug_lines.append(
                    f"BEST {match.best.display_name.upper()} {match.best.score:.2f}"
                )
            if match.second_best is not None:
                debug_lines.append(
                    f"2ND  {match.second_best.display_name.upper()} "
                    f"{match.second_best.score:.2f}"
                )
            debug_lines.append(match.reason)
        return OverlayFace(
            face=face, label=label, known=known, debug_lines=tuple(debug_lines)
        )

    def _apply_overlay(self) -> None:
        result = self._last_result
        if result is None:
            return
        paired = list(zip(result.faces, self._last_matches))
        visible = [
            (face, match)
            for face, match in paired
            if face.size_px >= self._settings.min_face_px
        ]
        self.panel.show_face_count(len(visible))
        entries = tuple(
            self._compose_entry(face, match) for face, match in visible
        )
        self.video.set_overlay(
            entries,
            result.frame_w,
            result.frame_h,
            show_landmarks=self._settings.show_landmarks,
            debug=self._settings.debug_mode,
        )
        known_names = [
            m.display_name.upper() for _, m in visible if m is not None and m.is_known
        ]
        if known_names:
            self.panel.show_recognition_result(", ".join(known_names))
        elif visible:
            self.panel.show_recognition_result("UNKNOWN")
        else:
            self.panel.show_recognition_result(None)

    # -- recognition gallery --

    def _reload_gallery(self, initial: bool = False) -> None:
        if self._store is None:
            self._matcher = Matcher([])
            return
        try:
            gallery, warnings = build_gallery(self._store, MODEL_PACK)
        except IdentityStoreError as exc:
            self._matcher = Matcher([])
            self.log_event(f"GALLERY LOAD FAILED: {exc}")
            return
        self._matcher = Matcher(gallery)
        for warning in warnings:
            self.log_event(f"GALLERY WARNING: {warning}")
        self.log_event(
            f"GALLERY {'LOADED' if initial else 'RELOADED'}: "
            f"{self._matcher.identity_count} IDENTITIES, "
            f"{self._matcher.sample_count} SAMPLES"
        )

    def _on_threshold_changed(self, value: float) -> None:
        self._settings.recognition_threshold = value
        self._save_settings()

    def _on_margin_changed(self, value: float) -> None:
        self._settings.match_margin = value
        self._save_settings()

    def _on_top_k_changed(self, value: int) -> None:
        self._settings.top_k = value
        self._save_settings()
        self.log_event(f"TOP-K SET TO {value}")

    # -- enrollment --

    def _update_enroll_enabled(self) -> None:
        self.panel.set_enroll_enabled(
            self._store is not None
            and self._model_ready
            and self._camera_state is CameraState.READY
        )

    def open_enrollment(self) -> None:
        if (
            self._store is None
            or self._inf_worker is None
            or not self._model_ready
            or self._camera_state is not CameraState.READY
        ):
            self.log_event("ENROLL IGNORED — CAMERA/MODEL/DB NOT READY")
            return
        self.log_event("ENROLLMENT STARTED")
        dialog = EnrollDialog(self._store, MODEL_PACK, parent=self)
        self._inf_worker.result_ready.connect(dialog.handle_result)
        try:
            dialog.exec()
        finally:
            self._inf_worker.result_ready.disconnect(dialog.handle_result)
        if dialog.created_identity is not None:
            record = dialog.created_identity
            self.log_event(
                f"IDENTITY ENROLLED: {record.display_name.upper()} "
                f"({record.sample_count} SAMPLES)"
            )
            self._reload_gallery()
        else:
            self.log_event("ENROLLMENT CANCELED — NOTHING STORED")

    # -- camera state handlers --

    def _on_camera_state(self, state_value: str) -> None:
        state = CameraState(state_value)
        if state is CameraState.OFFLINE and self._camera_state is CameraState.ERROR:
            return
        self._camera_state = state
        self.video.set_state(state)
        self.panel.show_state(state)
        self._update_enroll_enabled()
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
        if self._store is not None:
            self._store.close()
        self._save_settings(include_geometry=True)
        super().closeEvent(event)

    def _save_settings(self, include_geometry: bool = False) -> None:
        if include_geometry:
            self._settings.window_geometry = bytes(
                self.saveGeometry().toBase64()
            ).decode()
        save_settings(self._settings, self._settings_file)
