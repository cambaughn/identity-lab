"""Identity management dialog — list, rename, delete, reset-all.

Every mutation emits identities_changed so the main window reloads the
recognition gallery immediately; nothing requires a restart. Destructive
actions use explicit console-styled confirmations in the warning color.
"""

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from identity_lab.identity.errors import DuplicateIdentityError, IdentityStoreError
from identity_lab.identity.store import IdentityStore
from identity_lab.identity.types import IdentityRecord
from identity_lab.ui import theme

THUMB_SIZE = 48


def format_identity_row(record: IdentityRecord) -> str:
    """Two-line console row text for one identity."""
    created = record.created_at.strftime("%Y-%m-%d")
    model = (record.model_id or "—").upper()
    return (
        f"{record.display_name.upper()}\n"
        f"{record.sample_count} SAMPLES · {created} · {model}"
    )


class _ConfirmDialog(QDialog):
    """Unmistakable destructive confirmation: warning color, explicit verb."""

    def __init__(self, title: str, message: str, confirm_label: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(380)
        layout = QVBoxLayout(self)
        layout.setSpacing(theme.SPACING)
        head = QLabel(f"** {title} **")
        head.setObjectName("error")
        head.setStyleSheet(f"font-size: {theme.FONT_SIZE_STATE}px;")
        head.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body = QLabel(message)
        body.setWordWrap(True)
        buttons = QHBoxLayout()
        confirm = QPushButton(confirm_label)
        confirm.setObjectName("danger")
        cancel = QPushButton("CANCEL")
        confirm.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        cancel.setDefault(True)  # safety: Enter cancels, never confirms
        buttons.addWidget(cancel)
        buttons.addWidget(confirm)
        layout.addWidget(head)
        layout.addWidget(body)
        layout.addLayout(buttons)


class _RenameDialog(QDialog):
    def __init__(self, current_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RENAME IDENTITY")
        self.setModal(True)
        self.setMinimumWidth(340)
        layout = QVBoxLayout(self)
        layout.setSpacing(theme.SPACING)
        label = QLabel("NEW DISPLAY NAME")
        label.setObjectName("secondary")
        self.name_edit = QLineEdit(current_name)
        self.name_edit.selectAll()
        self.error_label = QLabel("")
        self.error_label.setObjectName("error")
        self.error_label.setWordWrap(True)
        buttons = QHBoxLayout()
        ok = QPushButton("RENAME")
        cancel = QPushButton("CANCEL")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        self.name_edit.returnPressed.connect(self.accept)
        buttons.addWidget(ok)
        buttons.addWidget(cancel)
        layout.addWidget(label)
        layout.addWidget(self.name_edit)
        layout.addWidget(self.error_label)
        layout.addLayout(buttons)


class ManageDialog(QDialog):
    identities_changed = Signal()

    def __init__(self, store: IdentityStore, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MANAGE IDENTITIES")
        self.setModal(True)
        self.setMinimumSize(460, 420)
        self._store = store

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            theme.SPACING, theme.SPACING, theme.SPACING, theme.SPACING
        )
        layout.setSpacing(theme.SPACING)

        title = QLabel("ENROLLED IDENTITIES")
        title.setObjectName("secondary")
        self.list = QListWidget()
        self.list.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self.list.itemSelectionChanged.connect(self._update_buttons)

        self.status_label = QLabel("")
        self.status_label.setObjectName("secondary")
        self.status_label.setWordWrap(True)

        row = QHBoxLayout()
        self.rename_button = QPushButton("RENAME")
        self.delete_button = QPushButton("DELETE")
        self.delete_button.setObjectName("danger")
        self.reset_button = QPushButton("RESET ALL")
        self.reset_button.setObjectName("danger")
        close = QPushButton("CLOSE")
        self.rename_button.clicked.connect(self._rename_selected)
        self.delete_button.clicked.connect(self._delete_selected)
        self.reset_button.clicked.connect(self._reset_all)
        close.clicked.connect(self.accept)
        row.addWidget(self.rename_button)
        row.addWidget(self.delete_button)
        row.addWidget(self.reset_button)
        row.addStretch(1)
        row.addWidget(close)

        layout.addWidget(title)
        layout.addWidget(self.list, stretch=1)
        layout.addWidget(self.status_label)
        layout.addLayout(row)
        self.refresh()

    # -- list population --

    def refresh(self) -> None:
        self.list.clear()
        try:
            records = self._store.list_identities()
        except IdentityStoreError as exc:
            self.status_label.setText(f"STORE ERROR: {exc}")
            records = []
        for record in records:
            item = QListWidgetItem(format_identity_row(record))
            item.setData(Qt.ItemDataRole.UserRole, record)
            icon = self._thumbnail_icon(record.identity_id)
            if icon is not None:
                item.setIcon(icon)
            self.list.addItem(item)
        if not records:
            placeholder = QListWidgetItem("NO IDENTITIES ENROLLED")
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            self.list.addItem(placeholder)
        self._update_buttons()

    def _thumbnail_icon(self, identity_id: str) -> QIcon | None:
        try:
            png = self._store.get_identity_thumbnail(identity_id)
        except IdentityStoreError:
            return None
        if not png:
            return None
        pixmap = QPixmap()
        if not pixmap.loadFromData(png, "PNG"):
            return None
        return QIcon(
            pixmap.scaled(
                THUMB_SIZE,
                THUMB_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _selected_record(self) -> IdentityRecord | None:
        items = self.list.selectedItems()
        if not items:
            return None
        return items[0].data(Qt.ItemDataRole.UserRole)

    def _update_buttons(self) -> None:
        has_selection = self._selected_record() is not None
        self.rename_button.setEnabled(has_selection)
        self.delete_button.setEnabled(has_selection)
        try:
            any_rows = bool(self._store.list_identities())
        except IdentityStoreError:
            any_rows = False
        self.reset_button.setEnabled(any_rows)

    # -- actions --

    def _rename_selected(self) -> None:
        record = self._selected_record()
        if record is None:
            return
        dialog = _RenameDialog(record.display_name, self)
        while True:
            if not dialog.exec():
                return
            new_name = dialog.name_edit.text().strip()
            if not new_name:
                dialog.error_label.setText("NAME REQUIRED")
                continue
            try:
                self._store.rename_identity(record.identity_id, new_name)
            except DuplicateIdentityError:
                dialog.error_label.setText(f"'{new_name.upper()}' ALREADY EXISTS")
                continue
            except IdentityStoreError as exc:
                self.status_label.setText(f"RENAME FAILED: {exc}")
                return
            break
        self.status_label.setText(
            f"RENAMED {record.display_name.upper()} → {new_name.upper()}"
        )
        self.refresh()
        self.identities_changed.emit()

    def _delete_selected(self) -> None:
        record = self._selected_record()
        if record is None:
            return
        confirm = _ConfirmDialog(
            "DELETE IDENTITY",
            f"Permanently delete {record.display_name.upper()} — "
            f"{record.sample_count} stored embeddings and the thumbnail?\n\n"
            "This cannot be undone. The person will immediately read as "
            "UNKNOWN in live recognition.",
            f"DELETE {record.display_name.upper()}",
            self,
        )
        if not confirm.exec():
            return
        try:
            self._store.delete_identity(record.identity_id)
        except IdentityStoreError as exc:
            self.status_label.setText(f"DELETE FAILED: {exc}")
            return
        self.status_label.setText(f"DELETED {record.display_name.upper()}")
        self.refresh()
        self.identities_changed.emit()

    def _reset_all(self) -> None:
        try:
            records = self._store.list_identities()
        except IdentityStoreError as exc:
            self.status_label.setText(f"STORE ERROR: {exc}")
            return
        total_samples = sum(r.sample_count for r in records)
        confirm = _ConfirmDialog(
            "RESET ALL IDENTITY DATA",
            f"Permanently delete ALL {len(records)} identities and "
            f"{total_samples} stored embeddings?\n\n"
            "This wipes every enrolled person from this machine and cannot "
            "be undone.",
            "RESET ALL DATA",
            self,
        )
        if not confirm.exec():
            return
        try:
            self._store.reset_database()
        except IdentityStoreError as exc:
            self.status_label.setText(f"RESET FAILED: {exc}")
            return
        self.status_label.setText("ALL IDENTITY DATA DELETED")
        self.refresh()
        self.identities_changed.emit()
