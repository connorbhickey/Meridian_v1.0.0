"""Dockable CONSOLE panel — log/status messages display."""

import logging
from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QTextEdit, QWidget,
)

from portopt.constants import Colors, Fonts
from portopt.gui.panels.base_panel import BasePanel


class ConsolePanel(BasePanel):
    panel_id = "console"
    panel_title = "CONSOLE"

    def __init__(self, parent=None):
        super().__init__(parent)

        # Header bar
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("LOG")
        title.setProperty("header", True)
        header_layout.addWidget(title)
        header_layout.addStretch()

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedHeight(20)
        clear_btn.clicked.connect(self._clear)
        header_layout.addWidget(clear_btn)

        self._layout.addWidget(header)

        # Log text area
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setFont(QFont(Fonts.MONO, Fonts.SIZE_SMALL))
        self._text.setStyleSheet(
            f"background-color: {Colors.BG_PRIMARY}; "
            f"border: 1px solid {Colors.BORDER}; "
            f"color: {Colors.TEXT_SECONDARY};"
        )
        self._layout.addWidget(self._text)

    def log(self, message: str, level: str = "INFO"):
        """Append a log message to the console."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        color_map = {
            "INFO": Colors.TEXT_SECONDARY,
            "SUCCESS": Colors.PROFIT,
            "WARNING": Colors.WARNING,
            "ERROR": Colors.LOSS,
            "DATA": Colors.ACCENT,
        }
        color = color_map.get(level, Colors.TEXT_SECONDARY)
        html = (
            f'<span style="color:{Colors.TEXT_MUTED}">[{timestamp}]</span> '
            f'<span style="color:{color}">[{level}]</span> '
            f'<span style="color:{Colors.TEXT_PRIMARY}">{message}</span>'
        )
        self._text.append(html)
        cursor = self._text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._text.setTextCursor(cursor)

    def log_info(self, msg: str):
        self.log(msg, "INFO")

    def log_success(self, msg: str):
        self.log(msg, "SUCCESS")

    def log_warning(self, msg: str):
        self.log(msg, "WARNING")

    def log_error(self, msg: str):
        self.log(msg, "ERROR")

    def log_data(self, msg: str):
        self.log(msg, "DATA")

    def _clear(self):
        self._text.clear()


class ConsoleLogHandler(logging.Handler):
    """Python logging handler that forwards to the ConsolePanel."""

    def __init__(self, console_panel: ConsolePanel):
        super().__init__()
        self._console = console_panel

    def emit(self, record):
        level_map = {
            logging.DEBUG: "INFO",
            logging.INFO: "INFO",
            logging.WARNING: "WARNING",
            logging.ERROR: "ERROR",
            logging.CRITICAL: "ERROR",
        }
        level = level_map.get(record.levelno, "INFO")
        msg = self.format(record)
        self._console.log(msg, level)
