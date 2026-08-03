"""Guided enrollment dialog — name entry, checklist capture, success state.

The dialog owns an EnrollmentSession and receives InferenceResults forwarded
by the main window while it is open. Nothing is written to the store until
the user presses FINISH; cancel/close at any point persists nothing.
"""

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from identity_lab.identity.enrollment import (
    ALL_STAGES,
    REQUIRED_TARGET,
    EnrollmentSession,
    FeedbackCode,
    PoseStage,
    save_enrollment,
)
from identity_lab.identity.errors import DuplicateIdentityError, IdentityStoreError
from identity_lab.identity.store import IdentityStore
from identity_lab.identity.types import IdentityRecord
from identity_lab.ui import theme
from identity_lab.vision.types import InferenceResult

STAGE_SHORT = {
    PoseStage.FORWARD: "FORWARD",
    PoseStage.LEFT: "LEFT",
    PoseStage.RIGHT: "RIGHT",
    PoseStage.UP: "UP",
    PoseStage.DOWN: "DOWN",
    PoseStage.GLASSES: "GLASSES",
}

# Rejections worth showing in the warning color (vs. neutral guidance).
_ERROR_CODES = {
    FeedbackCode.MULTIPLE_FACES,
    FeedbackCode.TOO_BLURRY,
    FeedbackCode.INCONSISTENT,
}


class EnrollDialog(QDialog):
    def __init__(
        self, store: IdentityStore, model_id: str, parent=None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("ENROLL PERSON")
        self.setModal(True)
        self.setMinimumWidth(420)

        self._store = store
        self._model_id = model_id
        self._session = EnrollmentSession()
        self._capturing = False
        self.created_identity: IdentityRecord | None = None

        self._pages = QStackedWidget()
        self._pages.addWidget(self._build_name_page())     # 0
        self._pages.addWidget(self._build_capture_page())  # 1
        self._pages.addWidget(self._build_success_page())  # 2
        root = QVBoxLayout(self)
        root.setContentsMargins(theme.SPACING, theme.SPACING, theme.SPACING, theme.SPACING)
        root.addWidget(self._pages)

    # ---------------------------------------------------------- page builders

    def _build_name_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(theme.SPACING)
        title = QLabel("ENROLL PERSON")
        consent = QLabel(
            "ENROLL ONLY WITH THIS PERSON'S INFORMED CONSENT. "
            "EMBEDDINGS AND ONE SMALL THUMBNAIL ARE STORED LOCALLY."
        )
        consent.setObjectName("secondary")
        consent.setWordWrap(True)
        name_label = QLabel("DISPLAY NAME")
        name_label.setObjectName("secondary")
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. CAM")
        self._name_error = QLabel("")
        self._name_error.setObjectName("error")
        self._name_error.setWordWrap(True)

        buttons = QHBoxLayout()
        start = QPushButton("START CAPTURE")
        cancel = QPushButton("CANCEL")
        start.clicked.connect(self._on_start_clicked)
        cancel.clicked.connect(self.reject)
        self._name_edit.returnPressed.connect(self._on_start_clicked)
        buttons.addWidget(start)
        buttons.addWidget(cancel)

        layout.addWidget(title)
        layout.addWidget(consent)
        layout.addSpacing(theme.SPACING)
        layout.addWidget(name_label)
        layout.addWidget(self._name_edit)
        layout.addWidget(self._name_error)
        layout.addStretch(1)
        layout.addLayout(buttons)
        return page

    def _build_capture_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(theme.SPACING)

        self._prompt_label = QLabel("")
        self._prompt_label.setStyleSheet(
            f"font-size: {theme.FONT_SIZE_STATE}px;"
        )
        self._prompt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._stage_labels: dict[PoseStage, QLabel] = {}
        checklist = QVBoxLayout()
        checklist.setSpacing(2)
        for stage in ALL_STAGES:
            label = QLabel("")
            self._stage_labels[stage] = label
            checklist.addWidget(label)

        self._progress_label = QLabel("")
        self._feedback_label = QLabel("STAND BY")
        self._feedback_label.setWordWrap(True)

        buttons = QHBoxLayout()
        self._finish_button = QPushButton("FINISH")
        self._finish_button.setEnabled(False)
        self._finish_button.clicked.connect(self._on_finish_clicked)
        cancel = QPushButton("CANCEL")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(self._finish_button)
        buttons.addWidget(cancel)

        layout.addWidget(self._prompt_label)
        layout.addSpacing(theme.SPACING)
        layout.addLayout(checklist)
        layout.addSpacing(theme.SPACING)
        layout.addWidget(self._progress_label)
        layout.addWidget(self._feedback_label)
        layout.addStretch(1)
        layout.addLayout(buttons)
        return page

    def _build_success_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(theme.SPACING)
        done = QLabel("** IDENTITY CREATED **")
        done.setStyleSheet(f"font-size: {theme.FONT_SIZE_STATE}px;")
        done.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._success_name = QLabel("")
        self._success_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._success_detail = QLabel("")
        self._success_detail.setObjectName("secondary")
        self._success_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        close = QPushButton("CLOSE")
        close.clicked.connect(self.accept)
        layout.addStretch(1)
        layout.addWidget(done)
        layout.addWidget(self._success_name)
        layout.addWidget(self._success_detail)
        layout.addStretch(1)
        layout.addWidget(close)
        return page

    # ---------------------------------------------------------- flow

    def _on_start_clicked(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            self._name_error.setText("NAME REQUIRED")
            return
        taken = any(
            r.display_name.lower() == name.lower()
            for r in self._store.list_identities()
        )
        if taken:
            self._name_error.setText(f"'{name.upper()}' ALREADY EXISTS")
            return
        self._display_name = name
        self._capturing = True
        self._pages.setCurrentIndex(1)
        self._refresh_capture_ui(None)

    @Slot(object)
    def handle_result(self, result: InferenceResult) -> None:
        """Called (UI thread) for every inference result while open."""
        if not self._capturing:
            return
        feedback = self._session.process(result.faces, result.frame)
        self._refresh_capture_ui(feedback)

    def _refresh_capture_ui(self, feedback) -> None:
        session = self._session
        stage = session.current_stage
        self._prompt_label.setText(stage.value if stage else "DONE")

        for status in session.stage_statuses():
            mark = "✓" if status.done else ("…" if status.stage is stage else " ")
            suffix = "" if status.required else "  (OPTIONAL)"
            label = self._stage_labels[status.stage]
            label.setText(
                f"{mark}  {STAGE_SHORT[status.stage]:<8} "
                f"{status.accepted}/{status.target}{suffix}"
            )
            label.setObjectName("" if status.done else "secondary")
            label.style().unpolish(label)
            label.style().polish(label)

        self._progress_label.setText(
            f"SAMPLES {session.accepted_total:02d}/{REQUIRED_TARGET}"
            + (" +OPTIONAL" if session.required_complete else "")
        )

        if feedback is not None:
            is_err = feedback.code in _ERROR_CODES
            self._feedback_label.setText(feedback.code.value)
            self._feedback_label.setObjectName("error" if is_err else "secondary")
            self._feedback_label.style().unpolish(self._feedback_label)
            self._feedback_label.style().polish(self._feedback_label)

        self._finish_button.setEnabled(session.required_complete)
        if session.fully_complete:
            self._on_finish_clicked()

    def _on_finish_clicked(self) -> None:
        if not self._session.required_complete or self.created_identity:
            return
        self._capturing = False
        try:
            record = save_enrollment(
                self._store, self._display_name, self._session, self._model_id
            )
        except (IdentityStoreError, DuplicateIdentityError) as exc:
            self._capturing = True
            self._feedback_label.setText(f"SAVE FAILED: {exc}")
            self._feedback_label.setObjectName("error")
            self._feedback_label.style().unpolish(self._feedback_label)
            self._feedback_label.style().polish(self._feedback_label)
            return
        self.created_identity = record
        self._success_name.setText(record.display_name.upper())
        self._success_detail.setText(
            f"{record.sample_count} SAMPLES ACCEPTED\nREADY FOR RECOGNITION"
        )
        self._pages.setCurrentIndex(2)
