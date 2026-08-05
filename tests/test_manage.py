"""Identity-manager presentation helpers (store behavior tested in
test_identity_store.py; the dialog itself is exercised offscreen)."""

from datetime import datetime

from identity_lab.identity.types import IdentityRecord
from identity_lab.ui.manage_dialog import format_identity_row


def record(**overrides):
    base = dict(
        identity_id="abc",
        display_name="Cam",
        created_at=datetime(2026, 8, 3, 10, 0, 0),
        enrollment_version=1,
        sample_count=20,
        model_id="buffalo_l",
        dim=512,
    )
    base.update(overrides)
    return IdentityRecord(**base)


def test_row_format_contains_all_metadata():
    row = format_identity_row(record())
    assert "CAM" in row
    assert "20 SAMPLES" in row
    assert "2026-08-03" in row
    assert "BUFFALO_L" in row


def test_row_format_handles_missing_model():
    row = format_identity_row(record(model_id=None, sample_count=0))
    assert "0 SAMPLES" in row
    assert "—" in row
