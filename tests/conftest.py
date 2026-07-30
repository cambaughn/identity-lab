"""Shared fixtures. A QCoreApplication (no display needed) backs Qt signal
machinery in tests; none of these tests require a webcam or a window."""

import pytest
from PySide6.QtCore import QCoreApplication


@pytest.fixture(scope="session", autouse=True)
def qt_core_app():
    app = QCoreApplication.instance() or QCoreApplication([])
    yield app
