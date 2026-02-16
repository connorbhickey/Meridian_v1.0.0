"""Application entry point — launches the Meridian terminal."""

import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from portopt.constants import APP_NAME, Fonts
from portopt.gui.theme import apply_theme
from portopt.gui.main_window import MainWindow


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

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
