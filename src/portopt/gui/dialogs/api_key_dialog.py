"""Dialog for entering and storing API keys (Anthropic, FRED, Tiingo, Alpha Vantage)."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFormLayout,
)

from portopt.constants import Colors, Fonts
from portopt.utils.credentials import (
    ANTHROPIC_API_KEY, FRED_API_KEY, TIINGO_API_KEY, ALPHA_VANTAGE_API_KEY,
    get_credential, store_credential, delete_credential,
)

# Each entry: (credential key, display label, placeholder, description)
_KEY_DEFS = [
    (ANTHROPIC_API_KEY, "Anthropic", "sk-ant-...", "AI Copilot (console.anthropic.com)"),
    (FRED_API_KEY, "FRED", "abcdef1234...", "Macro data (fred.stlouisfed.org)"),
    (TIINGO_API_KEY, "Tiingo", "abcdef1234...", "Backup price data (tiingo.com)"),
    (ALPHA_VANTAGE_API_KEY, "Alpha Vantage", "ABCDEF1234...", "Backup price data (alphavantage.co)"),
]


class ApiKeyDialog(QDialog):
    """Dialog for managing all API keys used by Meridian."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("API Keys")
        self.setMinimumWidth(500)
        self._inputs: dict[str, QLineEdit] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        header = QLabel("Configure API Keys")
        header.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-size: 13px; font-weight: bold;"
        )
        layout.addWidget(header)

        desc = QLabel(
            "Keys are stored securely in Windows Credential Manager.\n"
            "Only Anthropic is required (for AI Copilot). Others are optional."
        )
        desc.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        input_style = f"""
            QLineEdit {{
                background: {Colors.BG_INPUT};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                color: {Colors.TEXT_PRIMARY};
                padding: 6px;
                font-family: {Fonts.MONO};
                font-size: 10px;
            }}
            QLineEdit:focus {{ border-color: {Colors.ACCENT}; }}
        """

        form = QFormLayout()
        form.setSpacing(8)

        for cred_key, label, placeholder, description in _KEY_DEFS:
            row = QVBoxLayout()
            row.setSpacing(2)

            hint = QLabel(description)
            hint.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 9px;")
            row.addWidget(hint)

            field_row = QHBoxLayout()
            inp = QLineEdit()
            inp.setEchoMode(QLineEdit.EchoMode.Password)
            inp.setPlaceholderText(placeholder)
            inp.setStyleSheet(input_style)

            existing = get_credential(cred_key)
            if existing:
                inp.setText(existing)

            self._inputs[cred_key] = inp
            field_row.addWidget(inp)

            clear_btn = QPushButton("Clear")
            clear_btn.setFixedWidth(50)
            clear_btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    color: {Colors.LOSS};
                    border: 1px solid {Colors.LOSS};
                    border-radius: 3px;
                    padding: 4px;
                    font-size: 9px;
                }}
                QPushButton:hover {{ background: {Colors.LOSS_DIM}; }}
            """)
            clear_btn.clicked.connect(lambda checked=False, k=cred_key: self._on_clear(k))
            field_row.addWidget(clear_btn)

            row.addLayout(field_row)

            lbl = QLabel(f"{label}:")
            lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; font-weight: bold;")
            lbl.setFixedWidth(100)
            form.addRow(lbl, row)

        layout.addLayout(form)
        layout.addStretch()

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(80)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton("Save All")
        save_btn.setFixedWidth(80)
        save_btn.setStyleSheet(f"""
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
        save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(save_btn)

        layout.addLayout(btn_row)

    def _on_clear(self, cred_key: str):
        delete_credential(cred_key)
        self._inputs[cred_key].clear()

    def _on_save(self):
        for cred_key, inp in self._inputs.items():
            value = inp.text().strip()
            if value:
                store_credential(cred_key, value)
            else:
                delete_credential(cred_key)
        self.accept()

    def get_api_key(self) -> str | None:
        """Return the Anthropic key (backward compat with copilot controller)."""
        inp = self._inputs.get(ANTHROPIC_API_KEY)
        return inp.text().strip() if inp and inp.text().strip() else None
