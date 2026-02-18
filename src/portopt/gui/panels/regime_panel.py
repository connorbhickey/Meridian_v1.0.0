"""Dockable REGIME DETECTION panel — HMM regime chart, transition matrix, stats."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QFormLayout, QSpinBox, QPushButton,
    QProgressBar, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QWidget, QLabel, QGroupBox,
)
import pyqtgraph as pg
import numpy as np

from portopt.gui.panels.base_panel import BasePanel
from portopt.constants import Colors, Fonts


def _pg_color(hex_color: str) -> QColor:
    return QColor(hex_color)


class RegimePanel(BasePanel):
    panel_id = "regime"
    panel_title = "REGIME DETECTION"

    run_requested = Signal(int)  # n_regimes

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header controls
        header = QHBoxLayout()
        header.setSpacing(8)

        lbl = QLabel("Regimes:")
        lbl.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-family: {Fonts.SANS}; font-size: 10px;"
        )
        header.addWidget(lbl)

        self._n_regimes_spin = QSpinBox()
        self._n_regimes_spin.setRange(2, 4)
        self._n_regimes_spin.setValue(3)
        self._n_regimes_spin.setFixedWidth(50)
        header.addWidget(self._n_regimes_spin)

        header.addStretch()

        self._run_btn = QPushButton("DETECT REGIMES")
        self._run_btn.setFixedHeight(28)
        self._run_btn.setFixedWidth(150)
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
        self._run_btn.clicked.connect(self._on_run)
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

        # ── Regime timeline chart ────────────────────────────────────
        self._regime_plot = pg.PlotWidget(title="Market Regime Timeline")
        self._regime_plot.setBackground(Colors.BG_SECONDARY)
        self._regime_plot.showGrid(x=True, y=True, alpha=0.15)
        self._regime_plot.setLabel("left", "Return", color=Colors.TEXT_SECONDARY)
        splitter.addWidget(self._regime_plot)

        # ── Stats section ────────────────────────────────────────────
        stats_widget = QWidget()
        stats_layout = QHBoxLayout(stats_widget)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(8)

        # Regime stats table
        self._stats_table = QTableWidget(0, 4)
        self._stats_table.setHorizontalHeaderLabels([
            "Regime", "Ann. Return", "Ann. Vol", "Probability",
        ])
        for i in range(4):
            self._stats_table.horizontalHeader().setSectionResizeMode(
                i, QHeaderView.Stretch
            )
        self._stats_table.verticalHeader().hide()
        self._stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._stats_table.setStyleSheet(f"""
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
        """)
        stats_layout.addWidget(self._stats_table, stretch=1)

        # Transition matrix table
        self._trans_table = QTableWidget(0, 0)
        self._trans_table.verticalHeader().hide()
        self._trans_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._trans_table.setStyleSheet(f"""
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
        """)
        stats_layout.addWidget(self._trans_table, stretch=1)

        splitter.addWidget(stats_widget)
        splitter.setSizes([300, 150])

        layout.addWidget(splitter)

        # Current regime label
        self._current_label = QLabel("")
        self._current_label.setAlignment(Qt.AlignCenter)
        self._current_label.setStyleSheet(
            f"color: {Colors.ACCENT}; font-family: {Fonts.MONO}; "
            f"font-size: 12px; font-weight: bold; padding: 4px;"
        )
        layout.addWidget(self._current_label)

        self.content_layout.addLayout(layout)

    # ── Public API ────────────────────────────────────────────────────

    def set_running(self, running: bool):
        self._run_btn.setEnabled(not running)
        self._progress.setVisible(running)

    def set_result(self, result):
        """Populate chart and tables from a RegimeResult.

        Args:
            result: portopt.engine.regime.RegimeResult
        """
        self._update_regime_chart(result)
        self._update_stats_table(result.regimes)
        self._update_transition_matrix(result.transition_matrix, result.regimes)
        self._current_label.setText(
            f"CURRENT REGIME: {result.current_regime_name.upper()} "
            f"(BIC: {result.bic:.0f})"
        )
        self._current_label.setStyleSheet(
            f"color: {result.regimes[result.current_regime].color}; "
            f"font-family: {Fonts.MONO}; font-size: 12px; "
            f"font-weight: bold; padding: 4px;"
        )

    def _update_regime_chart(self, result):
        self._regime_plot.clear()
        if not result.dates or len(result.regime_sequence) == 0:
            return

        # Convert dates to epoch seconds for DateAxisItem
        try:
            dates_epoch = np.array([d.timestamp() for d in result.dates], dtype=float)
            self._regime_plot.setAxisItems({"bottom": pg.DateAxisItem()})
        except (AttributeError, TypeError):
            dates_epoch = np.arange(len(result.dates), dtype=float)

        # Color-coded background regions for each regime
        regime_seq = result.regime_sequence
        for regime_info in result.regimes:
            regime_idx = result.regimes.index(regime_info)
            mask = regime_seq == regime_idx
            if not mask.any():
                continue

            x_vals = dates_epoch[mask]
            # Draw regime probability as filled area
            probs = result.regime_probabilities[mask, regime_idx]
            color = QColor(regime_info.color)
            color.setAlpha(50)
            brush = pg.mkBrush(color)
            fill = pg.FillBetweenItem(
                pg.PlotDataItem(x_vals, probs),
                pg.PlotDataItem(x_vals, np.zeros_like(probs)),
                brush=brush,
            )
            self._regime_plot.addItem(fill)

        # Legend via small scatter points
        for regime_info in result.regimes:
            self._regime_plot.plot(
                [], [], pen=None,
                symbol="s", symbolSize=8,
                symbolBrush=pg.mkBrush(QColor(regime_info.color)),
                name=regime_info.name,
            )

        self._regime_plot.addLegend(offset=(-10, 10))

    def _update_stats_table(self, regimes):
        self._stats_table.setRowCount(len(regimes))
        for i, r in enumerate(regimes):
            name_item = QTableWidgetItem(r.name)
            name_item.setForeground(_pg_color(r.color))
            self._stats_table.setItem(i, 0, name_item)

            ret_item = QTableWidgetItem(f"{r.mean_return:+.1%}")
            color = Colors.PROFIT if r.mean_return >= 0 else Colors.LOSS
            ret_item.setForeground(_pg_color(color))
            ret_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._stats_table.setItem(i, 1, ret_item)

            vol_item = QTableWidgetItem(f"{r.volatility:.1%}")
            vol_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._stats_table.setItem(i, 2, vol_item)

            prob_item = QTableWidgetItem(f"{r.stationary_prob:.1%}")
            prob_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._stats_table.setItem(i, 3, prob_item)

    def _update_transition_matrix(self, trans_matrix, regimes):
        n = len(regimes)
        names = [r.name for r in regimes]

        self._trans_table.setColumnCount(n + 1)
        self._trans_table.setRowCount(n)
        self._trans_table.setHorizontalHeaderLabels(["From \\ To"] + names)
        for i in range(n + 1):
            self._trans_table.horizontalHeader().setSectionResizeMode(
                i, QHeaderView.Stretch
            )

        for i in range(n):
            from_item = QTableWidgetItem(names[i])
            from_item.setForeground(_pg_color(regimes[i].color))
            self._trans_table.setItem(i, 0, from_item)

            for j in range(n):
                val = trans_matrix[i, j]
                item = QTableWidgetItem(f"{val:.2%}")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if i == j:
                    item.setForeground(_pg_color(Colors.ACCENT))
                self._trans_table.setItem(i, j + 1, item)

    # ── Internal ──────────────────────────────────────────────────────

    def _on_run(self):
        self.run_requested.emit(self._n_regimes_spin.value())
