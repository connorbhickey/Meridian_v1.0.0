"""First-run welcome wizard — API key setup + sample portfolio loader."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QFrame,
)

from portopt.constants import APP_NAME, APP_VERSION, Colors, Fonts
from portopt.samples import SAMPLE_PORTFOLIOS
from portopt.utils.credentials import (
    ANTHROPIC_API_KEY, FRED_API_KEY, TIINGO_API_KEY, ALPHA_VANTAGE_API_KEY,
    get_credential, store_credential,
)


class WelcomeDialog(QDialog):
    """First-run welcome wizard."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Welcome to {APP_NAME}")
        self.setFixedSize(500, 520)
        self._selected_sample: str | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        # Header
        title = QLabel(APP_NAME)
        title.setStyleSheet(
            f"color: {Colors.ACCENT}; font-size: 24px; font-weight: bold; "
            f"font-family: {Fonts.SANS}; letter-spacing: 4px;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel(f"v{APP_VERSION} — Quantitative Portfolio Terminal")
        subtitle.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"color: {Colors.BORDER};")
        layout.addWidget(div)

        # Step 1: API Keys
        step1 = QLabel("1. API Keys (optional — enter any you have)")
        step1.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-size: 11px; font-weight: bold;"
        )
        layout.addWidget(step1)

        input_style = f"""
            QLineEdit {{
                background: {Colors.BG_INPUT};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                color: {Colors.TEXT_PRIMARY};
                padding: 5px;
                font-family: {Fonts.MONO};
                font-size: 9px;
            }}
            QLineEdit:focus {{ border-color: {Colors.ACCENT}; }}
        """

        self._key_inputs: dict[str, QLineEdit] = {}
        keys = [
            (ANTHROPIC_API_KEY, "Anthropic (for AI Copilot)", "sk-ant-..."),
            (FRED_API_KEY, "FRED (macro data)", "abc123..."),
            (TIINGO_API_KEY, "Tiingo (backup prices)", "abc123..."),
            (ALPHA_VANTAGE_API_KEY, "Alpha Vantage (backup prices)", "ABC123..."),
        ]
        for cred_key, label, placeholder in keys:
            row = QHBoxLayout()
            lbl = QLabel(f"{label}:")
            lbl.setFixedWidth(180)
            lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 10px;")
            inp = QLineEdit()
            inp.setEchoMode(QLineEdit.EchoMode.Password)
            inp.setPlaceholderText(placeholder)
            inp.setStyleSheet(input_style)
            existing = get_credential(cred_key)
            if existing:
                inp.setText(existing)
            self._key_inputs[cred_key] = inp
            row.addWidget(lbl)
            row.addWidget(inp, 1)
            layout.addLayout(row)

        # Divider
        div2 = QFrame()
        div2.setFrameShape(QFrame.Shape.HLine)
        div2.setStyleSheet(f"color: {Colors.BORDER};")
        layout.addWidget(div2)

        # Step 2: Sample Portfolio
        step2 = QLabel("2. Load a sample portfolio to explore")
        step2.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-size: 11px; font-weight: bold;"
        )
        layout.addWidget(step2)

        self._sample_combo = QComboBox()
        self._sample_combo.addItem("None — I'll import my own")
        for name in SAMPLE_PORTFOLIOS:
            self._sample_combo.addItem(name)
        self._sample_combo.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.BG_INPUT};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                color: {Colors.TEXT_PRIMARY};
                padding: 5px;
                font-size: 10px;
            }}
        """)
        layout.addWidget(self._sample_combo)

        layout.addStretch()

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        skip_btn = QPushButton("Skip")
        skip_btn.setFixedWidth(80)
        skip_btn.clicked.connect(self.reject)
        btn_row.addWidget(skip_btn)

        start_btn = QPushButton("Get Started")
        start_btn.setFixedWidth(100)
        start_btn.setStyleSheet(f"""
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
        start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(start_btn)

        layout.addLayout(btn_row)

    def _on_start(self):
        # Save any entered API keys
        for cred_key, inp in self._key_inputs.items():
            value = inp.text().strip()
            if value:
                store_credential(cred_key, value)

        # Record sample selection
        idx = self._sample_combo.currentIndex()
        if idx > 0:
            name = self._sample_combo.currentText()
            self._selected_sample = str(SAMPLE_PORTFOLIOS.get(name, ""))

        self.accept()

    @property
    def selected_sample_path(self) -> str | None:
        return self._selected_sample
