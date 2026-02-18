"""Keyboard shortcuts reference dialog."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QWidget,
)

from portopt.constants import Colors, Fonts

SHORTCUTS = [
    ("General", [
        ("Ctrl+I", "Import portfolio (CSV/OFX)"),
        ("Ctrl+Shift+E", "Export data"),
        ("Ctrl+,", "Preferences"),
        ("Ctrl+Q", "Exit"),
    ]),
    ("View", [
        ("Ctrl+L", "Strategy Lab"),
        ("Ctrl+Shift+S", "Save current view"),
        ("Ctrl+Shift+L", "Manage saved views"),
    ]),
    ("Data", [
        ("Ctrl+F", "Fidelity connection"),
        ("F5", "Refresh positions"),
    ]),
    ("Optimization", [
        ("Ctrl+O", "Run optimization"),
        ("Ctrl+Shift+B", "Black-Litterman views"),
        ("Ctrl+Shift+C", "Constraints editor"),
    ]),
    ("Backtest", [
        ("Ctrl+B", "Run backtest"),
    ]),
    ("AI", [
        ("Ctrl+Shift+A", "Open Copilot"),
        ("Ctrl+R", "Generate report"),
    ]),
]


class ShortcutsDialog(QDialog):
    """Keyboard shortcuts cheat sheet."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setFixedSize(400, 480)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        header = QLabel("Keyboard Shortcuts")
        header.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-size: 13px; font-weight: bold;"
        )
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background: {Colors.BG_SECONDARY};
            }}
        """)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(10)

        for section_name, bindings in SHORTCUTS:
            section_label = QLabel(section_name)
            section_label.setStyleSheet(
                f"color: {Colors.ACCENT}; font-size: 11px; font-weight: bold; "
                f"margin-top: 4px;"
            )
            content_layout.addWidget(section_label)

            for key, desc in bindings:
                row = QHBoxLayout()
                key_label = QLabel(key)
                key_label.setFixedWidth(130)
                key_label.setStyleSheet(
                    f"color: {Colors.TEXT_PRIMARY}; font-family: {Fonts.MONO}; "
                    f"font-size: 10px; background: {Colors.BG_TERTIARY}; "
                    f"border: 1px solid {Colors.BORDER}; border-radius: 3px; "
                    f"padding: 2px 6px;"
                )
                desc_label = QLabel(desc)
                desc_label.setStyleSheet(
                    f"color: {Colors.TEXT_SECONDARY}; font-size: 10px;"
                )
                row.addWidget(key_label)
                row.addWidget(desc_label, 1)
                content_layout.addLayout(row)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        # Close button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(80)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: {Colors.BG_PRIMARY};
                font-weight: bold;
                border: none;
                border-radius: 3px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{ background: {Colors.ACCENT_HOVER}; }}
        """)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)
