"""Right-click context menu helper for QTableWidget instances.

B2 feature: provides Copy Cell, Copy Row, Copy All, and Export as CSV
actions for any data table in the application.
"""

from __future__ import annotations

import csv
import io
import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QMenu, QTableWidget,
)

logger = logging.getLogger(__name__)


def setup_table_context_menu(
    table: QTableWidget,
    extra_actions: list[tuple[str, callable]] | None = None,
):
    """Attach a right-click context menu to a QTableWidget.

    Args:
        table: The table widget to enhance.
        extra_actions: Optional list of (label, callback) tuples for
            panel-specific menu items.
    """
    table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    table.customContextMenuRequested.connect(
        lambda pos: _show_menu(table, pos, extra_actions or [])
    )


def _show_menu(
    table: QTableWidget,
    pos,
    extra_actions: list[tuple[str, callable]],
):
    menu = QMenu(table)
    menu.setStyleSheet("""
        QMenu {
            background: #161b22;
            color: #e6edf3;
            border: 1px solid #30363d;
            padding: 4px;
        }
        QMenu::item:selected {
            background: #0a3d5c;
        }
    """)

    # Copy Cell
    copy_cell = menu.addAction("Copy Cell")
    copy_cell.triggered.connect(lambda: _copy_cell(table))

    # Copy Row
    copy_row = menu.addAction("Copy Row")
    copy_row.triggered.connect(lambda: _copy_row(table))

    menu.addSeparator()

    # Copy All
    copy_all = menu.addAction("Copy All")
    copy_all.triggered.connect(lambda: _copy_all(table))

    # Export CSV
    export_csv = menu.addAction("Export as CSV…")
    export_csv.triggered.connect(lambda: _export_csv(table))

    # Extra panel-specific actions
    if extra_actions:
        menu.addSeparator()
        for label, callback in extra_actions:
            action = menu.addAction(label)
            action.triggered.connect(callback)

    menu.exec(table.viewport().mapToGlobal(pos))


def _copy_cell(table: QTableWidget):
    """Copy the current cell text to clipboard."""
    item = table.currentItem()
    if item:
        QApplication.clipboard().setText(item.text())


def _copy_row(table: QTableWidget):
    """Copy the current row as tab-separated values."""
    row = table.currentRow()
    if row < 0:
        return
    cols = table.columnCount()
    values = []
    for c in range(cols):
        item = table.item(row, c)
        values.append(item.text() if item else "")
    QApplication.clipboard().setText("\t".join(values))


def _copy_all(table: QTableWidget):
    """Copy entire table contents (with headers) as tab-separated values."""
    lines = []

    # Headers
    headers = []
    for c in range(table.columnCount()):
        h = table.horizontalHeaderItem(c)
        headers.append(h.text() if h else f"Col{c}")
    lines.append("\t".join(headers))

    # Rows
    for r in range(table.rowCount()):
        row_vals = []
        for c in range(table.columnCount()):
            item = table.item(r, c)
            row_vals.append(item.text() if item else "")
        lines.append("\t".join(row_vals))

    QApplication.clipboard().setText("\n".join(lines))


def _export_csv(table: QTableWidget):
    """Export entire table to a CSV file via file dialog."""
    path, _ = QFileDialog.getSaveFileName(
        table, "Export Table as CSV", "", "CSV Files (*.csv);;All Files (*)"
    )
    if not path:
        return

    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Headers
            headers = []
            for c in range(table.columnCount()):
                h = table.horizontalHeaderItem(c)
                headers.append(h.text() if h else f"Col{c}")
            writer.writerow(headers)

            # Rows
            for r in range(table.rowCount()):
                row_vals = []
                for c in range(table.columnCount()):
                    item = table.item(r, c)
                    row_vals.append(item.text() if item else "")
                writer.writerow(row_vals)

        logger.info("Table exported to %s", path)
    except OSError as e:
        logger.error("Failed to export CSV: %s", e)
