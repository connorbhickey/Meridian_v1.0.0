"""Dialog for entering and storing the Anthropic API key."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QCheckBox, QPushButton,
)

from portopt.constants import Colors, Fonts
from portopt.utils.credentials import (
    ANTHROPIC_API_KEY, get_credential, store_credential, delete_credential,
)


class ApiKeyDialog(QDialog):
    """Simple dialog for Anthropic API key entry."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Anthropic API Key")
        self.setMinimumWidth(420)
        self._api_key: str | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header
        header = QLabel("Configure Anthropic API Key")
        header.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-size: 13px; font-weight: bold;"
        )
        layout.addWidget(header)

        desc = QLabel(
            "Enter your Anthropic API key to enable the AI Copilot.\n"
            "Get a key at console.anthropic.com"
        )
        desc.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Key input
        self._key_input = QLineEdit()
        self._key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._key_input.setPlaceholderText("sk-ant-...")
        self._key_input.setStyleSheet(f"""
            QLineEdit {{
                background: {Colors.BG_INPUT};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                color: {Colors.TEXT_PRIMARY};
                padding: 8px;
                font-family: {Fonts.MONO};
                font-size: 10px;
            }}
            QLineEdit:focus {{ border-color: {Colors.ACCENT}; }}
        """)
        layout.addWidget(self._key_input)

        # Pre-fill from keyring if available
        existing = get_credential(ANTHROPIC_API_KEY)
        if existing:
            self._key_input.setText(existing)

        # Save to keyring checkbox
        self._save_check = QCheckBox("Save securely in OS keyring")
        self._save_check.setChecked(True)
        self._save_check.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 10px;")
        layout.addWidget(self._save_check)

        layout.addStretch()

        # Buttons
        btn_row = QHBoxLayout()

        # Delete button (only show if key exists)
        if existing:
            delete_btn = QPushButton("Remove Key")
            delete_btn.setFixedWidth(90)
            delete_btn.setStyleSheet(f"""
                QPushButton {{
                    background: {Colors.LOSS_DIM};
                    color: {Colors.LOSS};
                    border: 1px solid {Colors.LOSS};
                    border-radius: 3px;
                    padding: 6px;
                    font-size: 10px;
                }}
                QPushButton:hover {{ background: {Colors.LOSS}; color: {Colors.BG_PRIMARY}; }}
            """)
            delete_btn.clicked.connect(self._on_delete)
            btn_row.addWidget(delete_btn)

        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(80)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton("Save")
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

    def _on_save(self):
        key = self._key_input.text().strip()
        if not key:
            return
        self._api_key = key
        if self._save_check.isChecked():
            store_credential(ANTHROPIC_API_KEY, key)
        self.accept()

    def _on_delete(self):
        delete_credential(ANTHROPIC_API_KEY)
        self._key_input.clear()
        self._api_key = None
        self.accept()

    def get_api_key(self) -> str | None:
        """Return the entered key (after dialog accepted)."""
        return self._api_key
