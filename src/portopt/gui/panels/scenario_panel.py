"""Dockable SCENARIO panel — interactive what-if analysis with return shocks.

B5 feature: after each optimization, users can adjust per-asset return
shocks via sliders and instantly see the re-optimized weights and
metrics compared to the baseline.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QPushButton, QScrollArea,
    QSlider, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from portopt.constants import Colors, Fonts, OptMethod
from portopt.data.models import OptimizationResult
from portopt.gui.panels.base_panel import BasePanel
from portopt.gui.widgets.table_context_menu import setup_table_context_menu

logger = logging.getLogger(__name__)


class ScenarioPanel(BasePanel):
    panel_id = "scenario"
    panel_title = "WHAT-IF"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._base_mu: pd.Series | None = None
        self._base_cov: pd.DataFrame | None = None
        self._base_weights: dict[str, float] = {}
        self._sliders: dict[str, QSlider] = {}
        self._slider_labels: dict[str, QLabel] = {}
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self._compute_scenario)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        self._status_label = QLabel("Run optimization first")
        self._status_label.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-family: {Fonts.SANS}; font-size: 9px;"
        )
        toolbar.addWidget(self._status_label)
        toolbar.addStretch()

        reset_btn = QPushButton("RESET")
        reset_btn.setFixedHeight(22)
        reset_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_INPUT};
                color: {Colors.ACCENT};
                border: 1px solid {Colors.BORDER};
                border-radius: 3px;
                font-family: {Fonts.SANS};
                font-size: 9px;
                font-weight: bold;
                padding: 0 10px;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
                border-color: {Colors.ACCENT};
            }}
        """)
        reset_btn.clicked.connect(self._reset_sliders)
        toolbar.addWidget(reset_btn)
        layout.addLayout(toolbar)

        # Splitter: sliders left, results right
        splitter = QSplitter(Qt.Horizontal)

        # ── Left: Sliders ─────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background: {Colors.BG_SECONDARY};
                border: 1px solid {Colors.BORDER};
            }}
        """)
        self._slider_container = QWidget()
        self._slider_layout = QVBoxLayout(self._slider_container)
        self._slider_layout.setContentsMargins(6, 6, 6, 6)
        self._slider_layout.setSpacing(4)
        self._slider_layout.addStretch()
        scroll.setWidget(self._slider_container)
        splitter.addWidget(scroll)

        # ── Right: Results table ──────────────────────────────────
        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.NoSelection)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(f"""
            QTableWidget {{
                background: {Colors.BG_SECONDARY};
                gridline-color: {Colors.BORDER};
                color: {Colors.TEXT_PRIMARY};
                font-family: {Fonts.MONO};
                font-size: 10px;
                border: 1px solid {Colors.BORDER};
            }}
            QTableWidget::item {{
                padding: 2px 6px;
            }}
            QHeaderView::section {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_SECONDARY};
                border: 1px solid {Colors.BORDER};
                padding: 3px;
                font-size: 9px;
                font-weight: bold;
            }}
        """)
        setup_table_context_menu(self._table)
        splitter.addWidget(self._table)
        splitter.setSizes([200, 300])

        layout.addWidget(splitter)
        self.content_layout.addLayout(layout)

    # ── Public API ───────────────────────────────────────────────

    def set_base_data(
        self,
        mu: pd.Series,
        cov: pd.DataFrame,
        weights: dict[str, float],
    ):
        """Set base optimization data for scenario analysis.

        Called after each optimization completes.
        """
        self._base_mu = mu.copy()
        self._base_cov = cov.copy()
        self._base_weights = dict(weights)
        self._status_label.setText(f"{len(mu)} assets loaded — adjust sliders to see impact")
        self._build_sliders()
        self._compute_scenario()

    # ── Sliders ──────────────────────────────────────────────────

    def _build_sliders(self):
        """Create one slider per asset."""
        # Clear existing sliders
        while self._slider_layout.count() > 1:  # keep the stretch
            item = self._slider_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._sliders.clear()
        self._slider_labels.clear()

        if self._base_mu is None:
            return

        for sym in self._base_mu.index:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            sym_label = QLabel(sym)
            sym_label.setFixedWidth(50)
            sym_label.setStyleSheet(
                f"color: {Colors.TEXT_PRIMARY}; font-family: {Fonts.MONO}; "
                f"font-size: 9px; font-weight: bold;"
            )
            row_layout.addWidget(sym_label)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(-50, 50)  # -50% to +50% shock
            slider.setValue(0)
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            slider.setTickInterval(10)
            slider.setStyleSheet(f"""
                QSlider::groove:horizontal {{
                    background: {Colors.BG_INPUT};
                    height: 4px;
                    border-radius: 2px;
                }}
                QSlider::handle:horizontal {{
                    background: {Colors.ACCENT};
                    width: 10px;
                    margin: -4px 0;
                    border-radius: 5px;
                }}
                QSlider::sub-page:horizontal {{
                    background: {Colors.ACCENT_DIM};
                }}
            """)
            slider.valueChanged.connect(self._on_slider_changed)
            row_layout.addWidget(slider)

            pct_label = QLabel("0%")
            pct_label.setFixedWidth(40)
            pct_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            pct_label.setStyleSheet(
                f"color: {Colors.TEXT_SECONDARY}; font-family: {Fonts.MONO}; font-size: 9px;"
            )
            row_layout.addWidget(pct_label)

            self._sliders[sym] = slider
            self._slider_labels[sym] = pct_label

            # Insert before the stretch
            self._slider_layout.insertWidget(self._slider_layout.count() - 1, row)

    def _on_slider_changed(self, _value):
        """Update label and start debounce timer."""
        for sym, slider in self._sliders.items():
            val = slider.value()
            label = self._slider_labels[sym]
            label.setText(f"{val:+d}%")
            if val > 0:
                label.setStyleSheet(
                    f"color: {Colors.PROFIT}; font-family: {Fonts.MONO}; font-size: 9px;"
                )
            elif val < 0:
                label.setStyleSheet(
                    f"color: {Colors.LOSS}; font-family: {Fonts.MONO}; font-size: 9px;"
                )
            else:
                label.setStyleSheet(
                    f"color: {Colors.TEXT_SECONDARY}; font-family: {Fonts.MONO}; font-size: 9px;"
                )

        self._debounce.start()

    def _reset_sliders(self):
        """Reset all sliders to 0."""
        for slider in self._sliders.values():
            slider.setValue(0)

    # ── Computation ──────────────────────────────────────────────

    def _compute_scenario(self):
        """Re-optimize with shocked expected returns."""
        if self._base_mu is None or self._base_cov is None:
            return

        try:
            from portopt.engine.optimization.mean_variance import MeanVarianceOptimizer

            # Build shocked mu
            shocked_mu = self._base_mu.copy()
            for sym, slider in self._sliders.items():
                shock = slider.value() / 100.0  # Convert percentage to decimal
                shocked_mu[sym] += shock

            # Re-optimize
            opt = MeanVarianceOptimizer(
                expected_returns=shocked_mu,
                covariance=self._base_cov,
                method=OptMethod.MAX_SHARPE,
            )
            result = opt.optimize()

            self._update_table(result.weights, result)

        except Exception as e:
            logger.warning("Scenario computation failed: %s", e)
            self._status_label.setText(f"Scenario error: {e}")

    def _update_table(self, shocked_weights: dict, result: OptimizationResult):
        """Display base vs shocked weights and metrics."""
        symbols = list(self._base_mu.index)
        n_assets = len(symbols)

        # Asset weight rows + 3 summary rows
        self._table.setRowCount(n_assets + 4)
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Asset", "Base", "Shocked", "Delta"])

        for i, sym in enumerate(symbols):
            base_w = self._base_weights.get(sym, 0.0)
            shock_w = shocked_weights.get(sym, 0.0)
            delta = shock_w - base_w

            sym_item = QTableWidgetItem(sym)
            sym_item.setFont(QFont(Fonts.MONO, 9))
            self._table.setItem(i, 0, sym_item)

            base_item = QTableWidgetItem(f"{base_w:.2%}")
            base_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(i, 1, base_item)

            shock_item = QTableWidgetItem(f"{shock_w:.2%}")
            shock_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(i, 2, shock_item)

            delta_item = QTableWidgetItem(f"{delta:+.2%}")
            delta_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if delta > 0.001:
                delta_item.setForeground(pg_color(Colors.PROFIT))
            elif delta < -0.001:
                delta_item.setForeground(pg_color(Colors.LOSS))
            self._table.setItem(i, 3, delta_item)

        # Separator row
        sep_row = n_assets
        for c in range(4):
            item = QTableWidgetItem("─" * 8 if c == 0 else "")
            item.setForeground(pg_color(Colors.BORDER))
            self._table.setItem(sep_row, c, item)

        # Summary: Return, Volatility, Sharpe
        from portopt.engine.optimization.mean_variance import MeanVarianceOptimizer
        base_opt = MeanVarianceOptimizer(
            expected_returns=self._base_mu,
            covariance=self._base_cov,
            method=OptMethod.MAX_SHARPE,
        )
        base_result = base_opt.optimize()

        summaries = [
            ("Return", base_result.expected_return, result.expected_return),
            ("Volatility", base_result.volatility, result.volatility),
            ("Sharpe", base_result.sharpe_ratio, result.sharpe_ratio),
        ]

        for j, (metric_name, base_val, shock_val) in enumerate(summaries):
            row = sep_row + 1 + j
            delta = shock_val - base_val

            name_item = QTableWidgetItem(metric_name)
            name_item.setFont(QFont(Fonts.SANS, 9))
            name_item.setForeground(pg_color(Colors.ACCENT))
            self._table.setItem(row, 0, name_item)

            if metric_name == "Sharpe":
                fmt = "{:.3f}"
                dfmt = "{:+.3f}"
            else:
                fmt = "{:.2%}"
                dfmt = "{:+.2%}"

            base_item = QTableWidgetItem(fmt.format(base_val))
            base_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(row, 1, base_item)

            shock_item = QTableWidgetItem(fmt.format(shock_val))
            shock_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(row, 2, shock_item)

            delta_item = QTableWidgetItem(dfmt.format(delta))
            delta_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if delta > 0.001:
                delta_item.setForeground(pg_color(Colors.PROFIT))
            elif delta < -0.001:
                delta_item.setForeground(pg_color(Colors.LOSS))
            self._table.setItem(row, 3, delta_item)


def pg_color(hex_color: str):
    """Create a QColor from a hex string for table item foreground."""
    from PySide6.QtGui import QColor
    return QColor(hex_color)
