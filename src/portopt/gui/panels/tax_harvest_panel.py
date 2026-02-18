"""Dockable TAX HARVEST panel — candidate table, replacement suggestions, summary."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QFormLayout, QDoubleSpinBox,
    QPushButton, QProgressBar, QSplitter, QTableWidget,
    QTableWidgetItem, QHeaderView, QWidget, QLabel, QGroupBox,
)
import pyqtgraph as pg
import numpy as np

from portopt.gui.panels.base_panel import BasePanel
from portopt.constants import Colors, Fonts


def _pg_color(hex_color: str) -> QColor:
    return QColor(hex_color)


class TaxHarvestPanel(BasePanel):
    panel_id = "tax_harvest"
    panel_title = "TAX HARVEST"

    run_requested = Signal(float)  # tax_rate

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

        title = QLabel("Tax-Loss Harvesting Analysis")
        title.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-family: {Fonts.SANS}; "
            f"font-size: 10px; font-weight: bold;"
        )
        header.addWidget(title)
        header.addStretch()

        rate_lbl = QLabel("Tax Rate:")
        rate_lbl.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-family: {Fonts.SANS}; font-size: 10px;"
        )
        header.addWidget(rate_lbl)

        self._tax_rate_spin = QDoubleSpinBox()
        self._tax_rate_spin.setRange(0, 60)
        self._tax_rate_spin.setValue(35.0)
        self._tax_rate_spin.setSuffix("%")
        self._tax_rate_spin.setDecimals(1)
        self._tax_rate_spin.setFixedWidth(80)
        header.addWidget(self._tax_rate_spin)

        self._run_btn = QPushButton("ANALYZE")
        self._run_btn.setFixedHeight(28)
        self._run_btn.setFixedWidth(100)
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

        # ── Candidates table ─────────────────────────────────────────
        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels([
            "Symbol", "Quantity", "Cost Basis", "Market Value",
            "Loss", "Loss %", "Tax Savings",
        ])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for i in range(1, 7):
            self._table.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)
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

        # ── Bottom: summary + bar chart ──────────────────────────────
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)

        # Summary group
        summary_group = QGroupBox("Summary")
        summary_group.setStyleSheet(f"""
            QGroupBox {{
                color: {Colors.TEXT_SECONDARY};
                font-family: {Fonts.SANS};
                font-size: 10px;
                font-weight: bold;
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }}
        """)
        summary_form = QFormLayout()
        summary_form.setSpacing(4)

        self._total_loss_label = QLabel("$0.00")
        self._total_loss_label.setStyleSheet(
            f"color: {Colors.LOSS}; font-family: {Fonts.MONO}; font-size: 12px; font-weight: bold;"
        )
        summary_form.addRow("Total Harvestable Loss:", self._total_loss_label)

        self._total_savings_label = QLabel("$0.00")
        self._total_savings_label.setStyleSheet(
            f"color: {Colors.PROFIT}; font-family: {Fonts.MONO}; font-size: 12px; font-weight: bold;"
        )
        summary_form.addRow("Estimated Tax Savings:", self._total_savings_label)

        self._candidates_label = QLabel("0")
        self._candidates_label.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-family: {Fonts.MONO}; font-size: 12px;"
        )
        summary_form.addRow("Candidates:", self._candidates_label)

        self._replacements_label = QLabel("")
        self._replacements_label.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-family: {Fonts.MONO}; font-size: 10px;"
        )
        self._replacements_label.setWordWrap(True)
        summary_form.addRow("Replacements:", self._replacements_label)

        summary_group.setLayout(summary_form)
        bottom_layout.addWidget(summary_group, stretch=1)

        # Bar chart of tax savings per symbol
        self._savings_plot = pg.PlotWidget(title="Tax Savings by Position")
        self._savings_plot.setBackground(Colors.BG_SECONDARY)
        self._savings_plot.showGrid(x=False, y=True, alpha=0.15)
        self._savings_plot.setLabel("left", "Savings ($)", color=Colors.TEXT_SECONDARY)
        bottom_layout.addWidget(self._savings_plot, stretch=2)

        splitter.addWidget(bottom)
        splitter.setSizes([250, 200])

        layout.addWidget(splitter)
        self.content_layout.addLayout(layout)

    # ── Public API ────────────────────────────────────────────────────

    def set_running(self, running: bool):
        self._run_btn.setEnabled(not running)
        self._progress.setVisible(running)

    def set_result(self, recommendation):
        """Populate the panel from a HarvestRecommendation.

        Args:
            recommendation: portopt.engine.tax_harvest.HarvestRecommendation
        """
        candidates = recommendation.candidates

        # Table
        self._table.setRowCount(len(candidates))
        for i, c in enumerate(candidates):
            self._table.setItem(i, 0, QTableWidgetItem(c.symbol))

            qty = QTableWidgetItem(f"{c.quantity:,.0f}")
            qty.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(i, 1, qty)

            cb = QTableWidgetItem(f"${c.cost_basis:,.0f}")
            cb.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(i, 2, cb)

            mv = QTableWidgetItem(f"${c.market_value:,.0f}")
            mv.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(i, 3, mv)

            loss = QTableWidgetItem(f"${c.unrealized_loss:,.0f}")
            loss.setForeground(_pg_color(Colors.LOSS))
            loss.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(i, 4, loss)

            pct = QTableWidgetItem(f"{c.loss_pct:.1f}%")
            pct.setForeground(_pg_color(Colors.LOSS))
            pct.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(i, 5, pct)

            sav = QTableWidgetItem(f"${c.tax_savings:,.0f}")
            sav.setForeground(_pg_color(Colors.PROFIT))
            sav.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(i, 6, sav)

        # Summary
        self._total_loss_label.setText(f"${recommendation.total_harvestable_loss:,.0f}")
        self._total_savings_label.setText(f"${recommendation.total_tax_savings:,.0f}")
        self._candidates_label.setText(str(len(candidates)))

        # Replacements text
        repl_text = []
        for sym, alts in recommendation.replacement_suggestions.items():
            repl_text.append(f"{sym} -> {', '.join(alts)}")
        self._replacements_label.setText("\n".join(repl_text) if repl_text else "None found")

        # Bar chart
        self._update_savings_chart(candidates)

    def _update_savings_chart(self, candidates):
        self._savings_plot.clear()
        if not candidates:
            return

        symbols = [c.symbol for c in candidates]
        savings = [c.tax_savings for c in candidates]
        x_pos = np.arange(len(symbols))

        bar = pg.BarGraphItem(
            x=x_pos, height=savings, width=0.5,
            brush=pg.mkBrush(QColor(Colors.PROFIT)),
            pen=pg.mkPen(None),
        )
        self._savings_plot.addItem(bar)

        bottom = self._savings_plot.getAxis("bottom")
        bottom.setTicks([list(zip(x_pos, symbols))])

    # ── Internal ──────────────────────────────────────────────────────

    def _on_run(self):
        self.run_requested.emit(self._tax_rate_spin.value() / 100)
