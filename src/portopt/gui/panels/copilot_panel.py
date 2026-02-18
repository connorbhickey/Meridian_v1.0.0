"""Dockable COPILOT panel — AI chat interface with streaming responses."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QWidget,
)

from portopt.constants import Colors, Fonts
from portopt.gui.panels.base_panel import BasePanel


class CopilotPanel(BasePanel):
    panel_id = "copilot"
    panel_title = "AI COPILOT"

    message_submitted = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_streaming = False
        self._build_ui()

    def _build_ui(self):
        # ── Header bar ──────────────────────────────────────────────
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)

        title = QLabel("COPILOT")
        title.setStyleSheet(
            f"color: {Colors.ACCENT}; font-size: 10px; font-weight: bold; "
            f"font-family: {Fonts.SANS};"
        )
        header_layout.addWidget(title)

        model_label = QLabel("claude-sonnet")
        model_label.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-size: 9px; "
            f"font-family: {Fonts.MONO};"
        )
        header_layout.addWidget(model_label)

        header_layout.addStretch()

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedHeight(20)
        clear_btn.setFixedWidth(50)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {Colors.TEXT_MUTED};
                border: 1px solid {Colors.BORDER};
                border-radius: 3px;
                font-size: 9px;
            }}
            QPushButton:hover {{
                color: {Colors.TEXT_PRIMARY};
                border-color: {Colors.BORDER_LIGHT};
            }}
        """)
        clear_btn.clicked.connect(self._clear_chat)
        header_layout.addWidget(clear_btn)

        self._layout.addWidget(header)

        # ── Chat display ────────────────────────────────────────────
        self._chat = QTextEdit()
        self._chat.setReadOnly(True)
        self._chat.setFont(QFont(Fonts.MONO, Fonts.SIZE_SMALL))
        self._chat.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.BG_PRIMARY};
                border: 1px solid {Colors.BORDER};
                color: {Colors.TEXT_SECONDARY};
                padding: 6px;
            }}
        """)
        self._layout.addWidget(self._chat)

        # ── Welcome message ─────────────────────────────────────────
        self._chat.setHtml(
            f'<div style="color:{Colors.TEXT_MUTED}; font-size:10px; padding:12px;">'
            f'<b style="color:{Colors.ACCENT}">Meridian Copilot</b><br><br>'
            f'Ask me about your portfolio — metrics, risk analysis, optimization, '
            f'stress tests, Monte Carlo projections, and more.<br><br>'
            f'<i>Run an optimization first to provide portfolio context.</i>'
            f'</div>'
        )

        # ── Input bar ──────────────────────────────────────────────
        input_widget = QWidget()
        input_layout = QHBoxLayout(input_widget)
        input_layout.setContentsMargins(0, 4, 0, 0)
        input_layout.setSpacing(4)

        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask about your portfolio...")
        self._input.setFont(QFont(Fonts.MONO, Fonts.SIZE_SMALL))
        self._input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {Colors.BG_INPUT};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                color: {Colors.TEXT_PRIMARY};
                padding: 6px 8px;
            }}
            QLineEdit:focus {{
                border-color: {Colors.ACCENT};
            }}
        """)
        self._input.returnPressed.connect(self._on_submit)
        input_layout.addWidget(self._input)

        self._send_btn = QPushButton("SEND")
        self._send_btn.setFixedHeight(30)
        self._send_btn.setFixedWidth(60)
        self._send_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT_DIM};
                color: {Colors.ACCENT};
                border: 1px solid {Colors.ACCENT};
                border-radius: 4px;
                font-family: {Fonts.SANS};
                font-size: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT};
                color: {Colors.BG_PRIMARY};
            }}
            QPushButton:disabled {{
                background: {Colors.BG_INPUT};
                color: {Colors.TEXT_MUTED};
                border-color: {Colors.BORDER};
            }}
        """)
        self._send_btn.clicked.connect(self._on_submit)
        input_layout.addWidget(self._send_btn)

        self._layout.addWidget(input_widget)

    # ── Public API ──────────────────────────────────────────────────

    def append_user_message(self, text: str):
        """Display a user message bubble."""
        html = (
            f'<div style="text-align:right; margin:6px 0;">'
            f'<span style="background:{Colors.ACCENT_DIM}; '
            f'color:{Colors.TEXT_PRIMARY}; padding:6px 10px; '
            f'border-radius:8px; display:inline-block; max-width:80%; '
            f'font-size:10px; text-align:left;">'
            f'{_escape_html(text)}'
            f'</span></div>'
        )
        self._chat.append(html)
        self._scroll_to_bottom()

    def start_assistant_message(self):
        """Begin a new assistant response bubble."""
        self._is_streaming = True
        html = (
            f'<div style="margin:6px 0;" id="assistant-msg">'
            f'<span style="color:{Colors.ACCENT}; font-size:9px; '
            f'font-weight:bold;">COPILOT</span><br>'
            f'<span style="color:{Colors.TEXT_SECONDARY}; font-size:10px;">'
        )
        self._chat.append(html)
        self._scroll_to_bottom()

    def append_assistant_chunk(self, text: str):
        """Append a text chunk to the current assistant bubble."""
        if not self._is_streaming:
            return
        # Insert text at the end of the document (before closing tags)
        cursor = self._chat.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self._scroll_to_bottom()

    def finish_assistant_message(self):
        """Close the current assistant bubble."""
        if self._is_streaming:
            cursor = self._chat.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertHtml('</span></div><hr style="border:none; '
                              f'border-top:1px solid {Colors.BORDER}; margin:4px 0;">')
            self._is_streaming = False
            self._scroll_to_bottom()

    def show_tool_use(self, tool_name: str):
        """Show a tool-use indicator in the chat."""
        display_name = tool_name.replace("_", " ").title()
        html = (
            f'<div style="margin:2px 0;">'
            f'<span style="color:{Colors.WARNING}; font-size:9px; '
            f'font-style:italic;">⚡ Using {display_name}...</span></div>'
        )
        self._chat.append(html)
        self._scroll_to_bottom()

    def show_error(self, msg: str):
        """Display an error message in the chat."""
        html = (
            f'<div style="margin:6px 0;">'
            f'<span style="background:{Colors.LOSS_DIM}; '
            f'color:{Colors.LOSS}; padding:6px 10px; border-radius:4px; '
            f'display:inline-block; font-size:10px;">'
            f'⚠ {_escape_html(msg)}'
            f'</span></div>'
        )
        self._chat.append(html)
        self._is_streaming = False
        self.set_waiting(False)
        self._scroll_to_bottom()

    def set_waiting(self, waiting: bool):
        """Toggle input enabled/disabled state."""
        self._input.setEnabled(not waiting)
        self._send_btn.setEnabled(not waiting)
        if waiting:
            self._input.setPlaceholderText("Thinking...")
        else:
            self._input.setPlaceholderText("Ask about your portfolio...")
            self._input.setFocus()

    # ── Internal ────────────────────────────────────────────────────

    def _on_submit(self):
        """Handle send button or Enter key."""
        text = self._input.text().strip()
        if not text:
            return
        self._input.clear()
        self.message_submitted.emit(text)

    def _clear_chat(self):
        """Clear the chat display."""
        self._chat.clear()
        self._is_streaming = False

    def _scroll_to_bottom(self):
        """Scroll chat to the bottom."""
        cursor = self._chat.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._chat.setTextCursor(cursor)
        self._chat.ensureCursorVisible()


def _escape_html(text: str) -> str:
    """Escape HTML special characters but preserve newlines."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
