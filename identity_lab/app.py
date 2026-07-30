"""Application bootstrap."""

import sys

from PySide6.QtWidgets import QApplication

from identity_lab.ui import theme
from identity_lab.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Identity Lab")
    app.setStyleSheet(theme.stylesheet())
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
