"""Application entry point — launches the Meridian terminal."""

import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtGui import QFont

from portopt.constants import APP_NAME, Fonts
from portopt.gui.theme import apply_theme
from portopt.gui.main_window import MainWindow


def _make_labels_selectable(root):
    """Walk all QLabel children and make them text-selectable by mouse."""
    for label in root.findChildren(QLabel):
        label.setTextInteractionFlags(
            label.textInteractionFlags() | Qt.TextInteractionFlag.TextSelectableByMouse
        )


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)

    # Set default font
    font = QFont(Fonts.SANS, Fonts.SIZE_NORMAL)
    app.setFont(font)

    # Apply Meridian deep-space theme
    apply_theme(app)

    # Launch main window
    window = MainWindow()
    window.show()

    # Make all QLabel text selectable/copyable after UI is fully constructed.
    # Initial sweep + periodic re-scan for dynamically created labels.
    _make_labels_selectable(window)
    _label_timer = QTimer(window)
    _label_timer.timeout.connect(lambda: _make_labels_selectable(window))
    _label_timer.start(3000)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
