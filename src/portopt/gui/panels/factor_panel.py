"""Dockable FACTOR ANALYSIS panel — factor exposure bar chart + regression table."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QProgressBar,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView, QWidget, QLabel,
)
import pyqtgraph as pg
import numpy as np

from portopt.gui.panels.base_panel import BasePanel
from portopt.constants import Colors, Fonts


def _pg_color(hex_color: str) -> QColor:
    return QColor(hex_color)


class FactorAnalysisPanel(BasePanel):
    panel_id = "factor_analysis"
    panel_title = "FACTOR ANALYSIS"

    run_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header row
        header = QHBoxLayout()
        header.setSpacing(8)

        title = QLabel("Fama-French 3-Factor Exposures")
        title.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-family: {Fonts.SANS}; "
            f"font-size: 10px; font-weight: bold;"
        )
        header.addWidget(title)
        header.addStretch()

        self._run_btn = QPushButton("RUN ANALYSIS")
        self._run_btn.setFixedHeight(28)
        self._run_btn.setFixedWidth(140)
        self._run_btn.setStyleSheet(f"""
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
        self._run_btn.clicked.connect(self.run_requested.emit)
        header.addWidget(self._run_btn)

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

        splitter = QSplitter(Qt.Vertical)

        # ── Bar chart (portfolio-level factor betas) ─────────────────
        self._beta_plot = pg.PlotWidget(title="Portfolio Factor Betas")
        self._beta_plot.setBackground(Colors.BG_SECONDARY)
        self._beta_plot.showGrid(x=False, y=True, alpha=0.15)
        self._beta_plot.setLabel("left", "Beta", color=Colors.TEXT_SECONDARY)
        splitter.addWidget(self._beta_plot)

        # ── Per-asset exposure table ─────────────────────────────────
        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels([
            "Symbol", "MKT-RF β", "SMB β", "HML β", "R²", "Alpha t-stat",
        ])
        for i in range(6):
            mode = QHeaderView.Stretch if i == 0 else QHeaderView.ResizeToContents
            self._table.horizontalHeader().setSectionResizeMode(i, mode)
        self._table.verticalHeader().hide()
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setStyleSheet(f"""
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
        splitter.addWidget(self._table)
        splitter.setSizes([200, 200])

        layout.addWidget(splitter)
        self.content_layout.addLayout(layout)

    # ── Public API ────────────────────────────────────────────────────

    def set_running(self, running: bool):
        self._run_btn.setEnabled(not running)
        self._progress.setVisible(running)

    def set_result(self, result):
        """Populate chart and table from a FactorModelResult.

        Args:
            result: portopt.engine.factors.FactorModelResult
        """
        self._update_beta_chart(result.portfolio_exposures)
        self._update_table(result.asset_exposures, result.r_squared)

    def _update_beta_chart(self, portfolio_exposures):
        self._beta_plot.clear()
        if not portfolio_exposures:
            return

        names = [e.factor_name for e in portfolio_exposures]
        betas = [e.beta for e in portfolio_exposures]
        x_pos = np.arange(len(names))

        colors = [
            _pg_color(Colors.ACCENT) if b >= 0 else _pg_color(Colors.LOSS)
            for b in betas
        ]
        brushes = [pg.mkBrush(c) for c in colors]

        bar = pg.BarGraphItem(
            x=x_pos, height=betas, width=0.5,
            brushes=brushes, pens=[pg.mkPen(None)] * len(betas),
        )
        self._beta_plot.addItem(bar)

        # X-axis labels
        bottom_axis = self._beta_plot.getAxis("bottom")
        bottom_axis.setTicks([list(zip(x_pos, names))])

        # Zero line
        zero_line = pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen(Colors.TEXT_MUTED, width=1, style=Qt.DashLine),
        )
        self._beta_plot.addItem(zero_line)

    def _update_table(self, asset_exposures, r_squared):
        self._table.setRowCount(len(asset_exposures))
        for row, (symbol, exps) in enumerate(sorted(asset_exposures.items())):
            self._table.setItem(row, 0, QTableWidgetItem(symbol))

            exp_map = {e.factor_name: e for e in exps}

            for col, factor in enumerate(["MKT-RF", "SMB", "HML"], start=1):
                exp = exp_map.get(factor)
                if exp:
                    item = QTableWidgetItem(f"{exp.beta:+.3f}")
                    color = Colors.PROFIT if exp.beta >= 0 else Colors.LOSS
                    item.setForeground(_pg_color(color))
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                else:
                    item = QTableWidgetItem("—")
                self._table.setItem(row, col, item)

            # R²
            r2 = r_squared.get(symbol, 0.0)
            r2_item = QTableWidgetItem(f"{r2:.3f}")
            r2_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(row, 4, r2_item)

            # Alpha t-stat (intercept significance — use MKT-RF t-stat as proxy)
            mkt = exp_map.get("MKT-RF")
            t_item = QTableWidgetItem(f"{mkt.t_stat:.2f}" if mkt else "—")
            t_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(row, 5, t_item)
