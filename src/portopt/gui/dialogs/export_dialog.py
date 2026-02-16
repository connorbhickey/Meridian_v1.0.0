"""Dialog for exporting data (CSV weights, PDF reports)."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QFileDialog,
    QGroupBox, QLineEdit,
)

from portopt.constants import Colors, Fonts

logger = logging.getLogger(__name__)


class ExportDialog(QDialog):
    """Dialog for exporting optimization/backtest results."""

    export_requested = Signal(dict)  # export config

    def __init__(self, parent=None, has_weights=False, has_backtest=False):
        super().__init__(parent)
        self._has_weights = has_weights
        self._has_backtest = has_backtest
        self.setWindowTitle("Export Results")
        self.setMinimumSize(400, 300)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        header = QLabel("Export Results")
        header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 14px; font-weight: bold;")
        layout.addWidget(header)

        # Format
        format_group = QGroupBox("Export Format")
        form = QFormLayout()

        self._format_combo = QComboBox()
        self._format_combo.addItems(["CSV — Weights", "CSV — Trades", "CSV — Metrics"])
        self._format_combo.currentTextChanged.connect(self._on_format_changed)
        form.addRow("Format:", self._format_combo)

        self._include_metadata = QCheckBox("Include metadata header")
        self._include_metadata.setChecked(True)
        form.addRow(self._include_metadata)

        format_group.setLayout(form)
        layout.addWidget(format_group)

        # Output path
        path_group = QGroupBox("Output")
        path_layout = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Select output file...")
        path_layout.addWidget(self._path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse)
        path_layout.addWidget(browse_btn)

        path_group.setLayout(path_layout)
        layout.addWidget(path_group)

        layout.addStretch()

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(80)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        export_btn = QPushButton("Export")
        export_btn.setFixedWidth(80)
        export_btn.setStyleSheet(f"""
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
        export_btn.clicked.connect(self._submit)
        btn_row.addWidget(export_btn)

        layout.addLayout(btn_row)

    def _on_format_changed(self, text: str):
        pass

    def _browse(self):
        fmt = self._format_combo.currentText()
        if "CSV" in fmt:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export CSV", "", "CSV Files (*.csv);;All Files (*)"
            )
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export PDF", "", "PDF Files (*.pdf);;All Files (*)"
            )
        if path:
            self._path_edit.setText(path)

    def _submit(self):
        path = self._path_edit.text().strip()
        if not path:
            return

        config = {
            "format": self._format_combo.currentText(),
            "path": path,
            "include_metadata": self._include_metadata.isChecked(),
        }
        self.export_requested.emit(config)
        self.accept()


def export_weights_csv(weights: dict[str, float], path: str,
                       metadata: dict | None = None):
    """Write weights to CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if metadata:
            for k, v in metadata.items():
                writer.writerow([f"# {k}", v])
        writer.writerow(["Symbol", "Weight"])
        for sym, w in sorted(weights.items(), key=lambda x: -x[1]):
            writer.writerow([sym, f"{w:.6f}"])
    logger.info("Exported weights to %s", path)


def export_trades_csv(trades: list[dict], path: str):
    """Write trade blotter to CSV file."""
    if not trades:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trades[0].keys())
        writer.writeheader()
        writer.writerows(trades)
    logger.info("Exported %d trades to %s", len(trades), path)


def export_metrics_csv(metrics: dict[str, float], path: str):
    """Write metrics to CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in metrics.items():
            writer.writerow([k, f"{v:.6f}" if isinstance(v, float) else v])
    logger.info("Exported metrics to %s", path)
