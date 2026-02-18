"""About dialog showing version, dependencies, and links."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
)

from portopt.constants import APP_NAME, APP_VERSION, Colors, Fonts


def _get_dep_versions() -> list[tuple[str, str]]:
    """Collect installed versions of key dependencies."""
    deps = []
    for pkg in [
        "PySide6", "numpy", "pandas", "scipy", "scikit-learn", "cvxpy",
        "pyqtgraph", "matplotlib", "yfinance", "networkx", "hmmlearn",
        "jinja2", "anthropic", "openpyxl", "keyring",
    ]:
        try:
            from importlib.metadata import version
            deps.append((pkg, version(pkg)))
        except Exception:
            deps.append((pkg, "not installed"))
    return deps


class AboutDialog(QDialog):
    """Application about dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About {APP_NAME}")
        self.setFixedSize(420, 440)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Title
        title = QLabel(f"{APP_NAME}")
        title.setStyleSheet(
            f"color: {Colors.ACCENT}; font-size: 22px; font-weight: bold; "
            f"font-family: {Fonts.SANS}; letter-spacing: 4px;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        version_label = QLabel(f"v{APP_VERSION}")
        version_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 11px;")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_label)

        subtitle = QLabel("Quantitative Portfolio Terminal")
        subtitle.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        python_label = QLabel(f"Python {sys.version.split()[0]}")
        python_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px;")
        python_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(python_label)

        # Dependencies
        dep_label = QLabel("Dependencies")
        dep_label.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-size: 11px; font-weight: bold;"
        )
        layout.addWidget(dep_label)

        dep_text = QTextEdit()
        dep_text.setReadOnly(True)
        dep_text.setStyleSheet(f"""
            QTextEdit {{
                background: {Colors.BG_INPUT};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                color: {Colors.TEXT_SECONDARY};
                font-family: {Fonts.MONO};
                font-size: 9px;
                padding: 6px;
            }}
        """)
        lines = []
        for pkg, ver in _get_dep_versions():
            lines.append(f"{pkg:20s} {ver}")
        dep_text.setPlainText("\n".join(lines))
        layout.addWidget(dep_text)

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
