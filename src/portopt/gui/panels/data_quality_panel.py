"""Dockable DATA QUALITY panel — coverage, staleness, anomalies."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QProgressBar,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,
    QWidget, QLabel, QGroupBox, QFormLayout,
)
import pyqtgraph as pg
import numpy as np

from portopt.gui.panels.base_panel import BasePanel
from portopt.constants import Colors, Fonts


def _pg_color(hex_color: str) -> QColor:
    return QColor(hex_color)


class DataQualityPanel(BasePanel):
    panel_id = "data_quality"
    panel_title = "DATA QUALITY"

    refresh_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header
        header = QHBoxLayout()
        header.setSpacing(8)

        title = QLabel("Data Coverage & Quality")
        title.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-family: {Fonts.SANS}; "
            f"font-size: 10px; font-weight: bold;"
        )
        header.addWidget(title)
        header.addStretch()

        self._refresh_btn = QPushButton("REFRESH")
        self._refresh_btn.setFixedHeight(28)
        self._refresh_btn.setFixedWidth(100)
        self._refresh_btn.setStyleSheet(f"""
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
        """)
        self._refresh_btn.clicked.connect(self.refresh_requested.emit)
        header.addWidget(self._refresh_btn)

        layout.addLayout(header)

        self._progress = QProgressBar()
        self._progress.setFixedHeight(4)
        self._progress.setTextVisible(False)
        self._progress.setRange(0, 0)
        self._progress.hide()
        self._progress.setStyleSheet(f"""
            QProgressBar {{ background: {Colors.BG_INPUT}; border: none; }}
            QProgressBar::chunk {{ background: {Colors.ACCENT}; }}
        """)
        layout.addWidget(self._progress)

        # Summary cards
        summary_layout = QHBoxLayout()
        summary_layout.setSpacing(8)

        self._symbols_card = self._make_card("Symbols", "0")
        self._obs_card = self._make_card("Observations", "0")
        self._coverage_card = self._make_card("Avg Coverage", "0%")
        self._anomalies_card = self._make_card("Anomalies", "0")
        self._cache_card = self._make_card("Cache Size", "0 MB")

        for card in [self._symbols_card, self._obs_card, self._coverage_card,
                     self._anomalies_card, self._cache_card]:
            summary_layout.addWidget(card)

        layout.addLayout(summary_layout)

        splitter = QSplitter(Qt.Vertical)

        # ── Coverage table ──────────────────────────────────────────
        self._coverage_table = QTableWidget(0, 7)
        self._coverage_table.setHorizontalHeaderLabels([
            "Symbol", "First Date", "Last Date", "Trading Days",
            "Missing", "Coverage %", "Staleness",
        ])
        self._coverage_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        for i in range(1, 7):
            self._coverage_table.horizontalHeader().setSectionResizeMode(
                i, QHeaderView.Stretch
            )
        self._coverage_table.verticalHeader().hide()
        self._coverage_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._coverage_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._coverage_table.setStyleSheet(f"""
            QTableWidget {{
                background: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                gridline-color: {Colors.BORDER};
                font-family: {Fonts.MONO};
                font-size: 10px;
                border: 1px solid {Colors.BORDER};
            }}
            QHeaderView::section {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_MUTED};
                border: 1px solid {Colors.BORDER};
                padding: 4px;
                font-family: {Fonts.SANS};
                font-size: 9px;
                font-weight: bold;
            }}
            QTableWidget::item:selected {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        splitter.addWidget(self._coverage_table)

        # ── Anomalies table ─────────────────────────────────────────
        self._anomaly_table = QTableWidget(0, 5)
        self._anomaly_table.setHorizontalHeaderLabels([
            "Symbol", "Date", "Type", "Description", "Severity",
        ])
        self._anomaly_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        for i in range(1, 5):
            self._anomaly_table.horizontalHeader().setSectionResizeMode(
                i, QHeaderView.Stretch
            )
        self._anomaly_table.verticalHeader().hide()
        self._anomaly_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._anomaly_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._anomaly_table.setStyleSheet(self._coverage_table.styleSheet())
        splitter.addWidget(self._anomaly_table)

        splitter.setSizes([300, 200])
        layout.addWidget(splitter)
        self.content_layout.addLayout(layout)

    def _make_card(self, label: str, value: str) -> QGroupBox:
        """Create a small summary card widget."""
        card = QGroupBox()
        card.setFixedHeight(60)
        card.setStyleSheet(f"""
            QGroupBox {{
                background: {Colors.BG_TERTIARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
            }}
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 4, 8, 4)
        card_layout.setSpacing(2)

        lbl = QLabel(label.upper())
        lbl.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-family: {Fonts.SANS}; "
            f"font-size: 8px; font-weight: bold;"
        )
        lbl.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(lbl)

        val = QLabel(value)
        val.setObjectName(f"card_value_{label.lower().replace(' ', '_')}")
        val.setStyleSheet(
            f"color: {Colors.ACCENT}; font-family: {Fonts.MONO}; "
            f"font-size: 14px; font-weight: bold;"
        )
        val.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(val)

        return card

    def _get_card_value(self, card: QGroupBox) -> QLabel:
        """Get the value label from a card."""
        for child in card.findChildren(QLabel):
            if child.objectName().startswith("card_value_"):
                return child
        return QLabel()

    # ── Public API ────────────────────────────────────────────────────

    def set_running(self, running: bool):
        self._refresh_btn.setEnabled(not running)
        self._progress.setVisible(running)

    def set_report(self, report):
        """Populate from a QualityReport.

        Args:
            report: portopt.data.quality.QualityReport
        """
        # Summary cards
        self._get_card_value(self._symbols_card).setText(str(report.total_symbols))
        self._get_card_value(self._obs_card).setText(f"{report.total_observations:,}")
        self._get_card_value(self._coverage_card).setText(f"{report.avg_coverage_pct:.0f}%")
        self._get_card_value(self._anomalies_card).setText(str(len(report.anomalies)))
        self._get_card_value(self._cache_card).setText(f"{report.cache_size_mb:.1f} MB")

        # Coverage table
        self._coverage_table.setRowCount(len(report.coverage))
        for i, c in enumerate(sorted(report.coverage, key=lambda x: x.symbol)):
            self._coverage_table.setItem(i, 0, QTableWidgetItem(c.symbol))

            first = QTableWidgetItem(str(c.first_date) if c.first_date else "—")
            first.setTextAlignment(Qt.AlignCenter)
            self._coverage_table.setItem(i, 1, first)

            last = QTableWidgetItem(str(c.last_date) if c.last_date else "—")
            last.setTextAlignment(Qt.AlignCenter)
            self._coverage_table.setItem(i, 2, last)

            days = QTableWidgetItem(f"{c.trading_days:,}")
            days.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._coverage_table.setItem(i, 3, days)

            missing = QTableWidgetItem(str(c.missing_days))
            missing.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if c.missing_days > 10:
                missing.setForeground(_pg_color(Colors.WARNING))
            self._coverage_table.setItem(i, 4, missing)

            cov = QTableWidgetItem(f"{c.coverage_pct:.0f}%")
            cov.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if c.coverage_pct < 90:
                cov.setForeground(_pg_color(Colors.WARNING))
            elif c.coverage_pct >= 95:
                cov.setForeground(_pg_color(Colors.PROFIT))
            self._coverage_table.setItem(i, 5, cov)

            stale = QTableWidgetItem(f"{c.staleness_days}d")
            stale.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if c.staleness_days > 5:
                stale.setForeground(_pg_color(Colors.WARNING))
            elif c.staleness_days > 14:
                stale.setForeground(_pg_color(Colors.LOSS))
            self._coverage_table.setItem(i, 6, stale)

        # Anomalies table
        anomalies = sorted(report.anomalies, key=lambda a: (a.severity == "info", a.symbol))
        self._anomaly_table.setRowCount(len(anomalies))
        for i, a in enumerate(anomalies):
            self._anomaly_table.setItem(i, 0, QTableWidgetItem(a.symbol))

            dt = QTableWidgetItem(str(a.date))
            dt.setTextAlignment(Qt.AlignCenter)
            self._anomaly_table.setItem(i, 1, dt)

            atype = QTableWidgetItem(a.anomaly_type.replace("_", " ").title())
            self._anomaly_table.setItem(i, 2, atype)

            desc = QTableWidgetItem(a.description)
            self._anomaly_table.setItem(i, 3, desc)

            sev = QTableWidgetItem(a.severity.upper())
            sev.setTextAlignment(Qt.AlignCenter)
            if a.severity == "error":
                sev.setForeground(_pg_color(Colors.LOSS))
            elif a.severity == "warning":
                sev.setForeground(_pg_color(Colors.WARNING))
            else:
                sev.setForeground(_pg_color(Colors.TEXT_MUTED))
            self._anomaly_table.setItem(i, 4, sev)
