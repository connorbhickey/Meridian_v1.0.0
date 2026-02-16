"""Application entry point — launches the Meridian terminal."""

import sys

from PySide6.QtCore import QObject, QEvent, Qt
from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtGui import QFont

from portopt.constants import APP_NAME, Fonts
from portopt.gui.theme import apply_theme
from portopt.gui.main_window import MainWindow


class _LabelSelectableFilter(QObject):
    """Event filter that makes all QLabel widgets text-selectable by mouse.

    Installed on QApplication so it sees every widget's Polish event (fired
    once when the widget is first styled). This avoids manually setting the
    flag on every label across the entire codebase.
    """

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Polish and isinstance(obj, QLabel):
            obj.setTextInteractionFlags(
                obj.textInteractionFlags() | Qt.TextInteractionFlag.TextSelectableByMouse
            )
        return False


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)

    # Set default font
    font = QFont(Fonts.SANS, Fonts.SIZE_NORMAL)
    app.setFont(font)

    # Make all QLabel text selectable/copyable
    label_filter = _LabelSelectableFilter(app)
    app.installEventFilter(label_filter)

    # Apply Meridian deep-space theme
    apply_theme(app)

    # Launch main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
