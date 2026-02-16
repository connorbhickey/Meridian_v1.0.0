"""Dialog for saving and loading dock layout presets."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QListWidget, QListWidgetItem,
)

from portopt.constants import Colors, Fonts


class LayoutDialog(QDialog):
    """Dialog for managing dock layout presets."""

    layout_save_requested = Signal(str)    # layout name
    layout_load_requested = Signal(str)    # layout name
    layout_delete_requested = Signal(str)  # layout name

    def __init__(self, saved_layouts: list[str], parent=None):
        super().__init__(parent)
        self._saved = list(saved_layouts)
        self.setWindowTitle("Layout Manager")
        self.setMinimumSize(350, 350)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        header = QLabel("Dock Layout Presets")
        header.setFont(QFont(Fonts.SANS, 12, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY};")
        layout.addWidget(header)

        # Save new
        save_row = QHBoxLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("New layout name...")
        save_row.addWidget(self._name_edit)

        save_btn = QPushButton("Save Current")
        save_btn.setFixedWidth(100)
        save_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: {Colors.BG_PRIMARY};
                font-weight: bold;
                border: none;
                border-radius: 3px;
                padding: 4px 8px;
            }}
        """)
        save_btn.clicked.connect(self._on_save)
        save_row.addWidget(save_btn)
        layout.addLayout(save_row)

        # Saved layouts list
        layout.addWidget(QLabel("Saved Layouts:"))
        self._list = QListWidget()
        for name in self._saved:
            self._list.addItem(QListWidgetItem(name))
        layout.addWidget(self._list, 1)

        # Action buttons
        btn_row = QHBoxLayout()

        load_btn = QPushButton("Load Selected")
        load_btn.clicked.connect(self._on_load)
        btn_row.addWidget(load_btn)

        delete_btn = QPushButton("Delete")
        delete_btn.setStyleSheet(f"color: {Colors.LOSS};")
        delete_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(delete_btn)

        btn_row.addStretch()

        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(80)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)

        layout.addLayout(btn_row)

    def _on_save(self):
        name = self._name_edit.text().strip()
        if not name:
            return
        self.layout_save_requested.emit(name)
        if name not in self._saved:
            self._saved.append(name)
            self._list.addItem(QListWidgetItem(name))
        self._name_edit.clear()

    def _on_load(self):
        item = self._list.currentItem()
        if item:
            self.layout_load_requested.emit(item.text())
            self.accept()

    def _on_delete(self):
        item = self._list.currentItem()
        if item:
            name = item.text()
            self.layout_delete_requested.emit(name)
            row = self._list.row(item)
            self._list.takeItem(row)
            if name in self._saved:
                self._saved.remove(name)
