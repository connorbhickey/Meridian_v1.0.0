"""Application entry point — launches the Meridian terminal."""

import sys

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtGui import QFont

from portopt.constants import APP_NAME, Fonts
from portopt.gui.theme import apply_theme
from portopt.gui.main_window import MainWindow


def _make_labels_selectable(root):
    """Walk all QLabel children of root and make them text-selectable."""
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
    # Run once at startup and periodically to catch dynamically created labels.
    _make_labels_selectable(window)
    _label_timer = QTimer(window)
    _label_timer.timeout.connect(lambda: _make_labels_selectable(window))
    _label_timer.start(2000)  # Re-scan every 2 seconds for new labels

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
