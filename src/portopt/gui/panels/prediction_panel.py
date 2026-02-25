"""Dockable PREDICTION panel — 25-method ensemble stock predictor.

Five tabs: Summary, Methods, Monte Carlo, Signals, Statistics.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portopt.constants import Colors, Fonts
from portopt.engine.prediction.ensemble import PredictionResult
from portopt.gui.panels.base_panel import BasePanel


class PredictionPanel(BasePanel):
    panel_id = "prediction"
    panel_title = "STOCK PREDICTOR"

    run_requested = Signal(str, int)  # (symbol, horizon_days)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result: PredictionResult | None = None
        self._build_ui()

    # ──────────────────────────────────────────────────────────────
    # UI Construction
    # ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout()
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # ── Top bar: symbol input + horizon + run button ──
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)

        sym_label = QLabel("Symbol:")
        sym_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-family: {Fonts.SANS};")
        top_bar.addWidget(sym_label)

        self._sym_input = QLineEdit()
        self._sym_input.setPlaceholderText("AAPL")
        self._sym_input.setMaximumWidth(120)
        self._sym_input.setStyleSheet(f"""
            QLineEdit {{
                background: {Colors.BG_INPUT};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 3px;
                padding: 4px 8px;
                font-family: {Fonts.MONO};
                font-size: 12px;
            }}
        """)
        self._sym_input.returnPressed.connect(self._on_run)
        top_bar.addWidget(self._sym_input)

        hz_label = QLabel("Horizon:")
        hz_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-family: {Fonts.SANS};")
        top_bar.addWidget(hz_label)

        self._horizon_spin = QSpinBox()
        self._horizon_spin.setRange(21, 504)
        self._horizon_spin.setValue(252)
        self._horizon_spin.setSingleStep(21)
        self._horizon_spin.setSuffix(" days")
        self._horizon_spin.setMaximumWidth(110)
        top_bar.addWidget(self._horizon_spin)

        self._run_btn = QPushButton("▶ PREDICT")
        self._run_btn.setFixedHeight(32)
        self._run_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT_DIM};
                color: {Colors.ACCENT};
                border: 1px solid {Colors.ACCENT};
                border-radius: 4px;
                font-family: {Fonts.SANS};
                font-size: 11px;
                font-weight: bold;
                padding: 0 16px;
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
        top_bar.addWidget(self._run_btn)

        self._progress = QProgressBar()
        self._progress.setFixedHeight(4)
        self._progress.setTextVisible(False)
        self._progress.setRange(0, 0)
        self._progress.hide()
        self._progress.setStyleSheet(f"""
            QProgressBar {{ background: {Colors.BG_INPUT}; border: none; }}
            QProgressBar::chunk {{ background: {Colors.ACCENT}; }}
        """)
        top_bar.addWidget(self._progress)
        top_bar.addStretch()

        outer.addLayout(top_bar)

        # ── Status bar ──
        self._status_label = QLabel("Enter a symbol and click PREDICT")
        self._status_label.setStyleSheet(f"""
            color: {Colors.TEXT_MUTED};
            font-family: {Fonts.MONO};
            font-size: 10px;
            padding: 2px 4px;
        """)
        outer.addWidget(self._status_label)

        # ── Tab widget ──
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {Colors.BORDER};
                background: {Colors.BG_SECONDARY};
            }}
            QTabBar::tab {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_MUTED};
                border: 1px solid {Colors.BORDER};
                padding: 6px 14px;
                font-family: {Fonts.SANS};
                font-size: 10px;
                font-weight: bold;
            }}
            QTabBar::tab:selected {{
                background: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                border-bottom: 2px solid {Colors.ACCENT};
            }}
        """)

        self._build_summary_tab()
        self._build_methods_tab()
        self._build_mc_tab()
        self._build_signals_tab()
        self._build_stats_tab()

        outer.addWidget(self._tabs)
        self.content_layout.addLayout(outer)

    # ── Tab 1: Summary ────────────────────────────────────────────

    def _build_summary_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Headline card
        self._headline = QLabel("No prediction yet")
        self._headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._headline.setStyleSheet(f"""
            font-family: {Fonts.SANS};
            font-size: 18px;
            font-weight: bold;
            color: {Colors.TEXT_PRIMARY};
            background: {Colors.BG_TERTIARY};
            border: 1px solid {Colors.BORDER};
            border-radius: 6px;
            padding: 16px;
        """)
        layout.addWidget(self._headline)

        # Key metrics row
        metrics_row = QHBoxLayout()
        self._metric_labels = {}
        for key in ["Confidence", "Vol Scale", "J-S Coeff", "Kelly %", "Model Fidelity"]:
            card = self._make_metric_card(key)
            metrics_row.addWidget(card)
        layout.addLayout(metrics_row)

        # Prediction interval summary
        self._pi_label = QLabel("")
        self._pi_label.setStyleSheet(f"""
            color: {Colors.TEXT_SECONDARY};
            font-family: {Fonts.MONO};
            font-size: 10px;
            padding: 8px;
            background: {Colors.BG_TERTIARY};
            border: 1px solid {Colors.BORDER};
            border-radius: 4px;
        """)
        self._pi_label.setWordWrap(True)
        layout.addWidget(self._pi_label)

        # Probability table
        self._prob_table = QTableWidget()
        self._prob_table.setColumnCount(2)
        self._prob_table.setHorizontalHeaderLabels(["Scenario", "Probability"])
        self._prob_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._prob_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._prob_table.verticalHeader().setVisible(False)
        self._prob_table.setMaximumHeight(160)
        self._style_table(self._prob_table)
        layout.addWidget(self._prob_table)

        layout.addStretch()
        self._tabs.addTab(w, "SUMMARY")

    # ── Tab 2: Methods ────────────────────────────────────────────

    def _build_methods_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)

        self._methods_table = QTableWidget()
        self._methods_table.setColumnCount(5)
        self._methods_table.setHorizontalHeaderLabels(
            ["Method", "Estimate", "Weight %", "Source", "Vol Scaled"]
        )
        self._methods_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._methods_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._methods_table.verticalHeader().setVisible(False)
        self._methods_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._style_table(self._methods_table)
        layout.addWidget(self._methods_table)

        self._tabs.addTab(w, "METHODS")

    # ── Tab 3: Monte Carlo Distribution ───────────────────────────

    def _build_mc_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)

        self._hist_plot = pg.PlotWidget(title="Terminal Price Distribution (MJD)")
        self._hist_plot.setBackground(Colors.BG_SECONDARY)
        self._hist_plot.showGrid(x=True, y=True, alpha=0.15)
        self._hist_plot.setLabel("left", "Density (%)", color=Colors.TEXT_SECONDARY)
        self._hist_plot.setLabel("bottom", "Price ($)", color=Colors.TEXT_SECONDARY)
        layout.addWidget(self._hist_plot, stretch=3)

        # MC percentile row
        self._mc_label = QLabel("")
        self._mc_label.setStyleSheet(f"""
            color: {Colors.TEXT_SECONDARY};
            font-family: {Fonts.MONO};
            font-size: 10px;
            padding: 6px;
            background: {Colors.BG_TERTIARY};
            border: 1px solid {Colors.BORDER};
            border-radius: 4px;
        """)
        self._mc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._mc_label)

        self._tabs.addTab(w, "MONTE CARLO")

    # ── Tab 4: Signal Details ─────────────────────────────────────

    def _build_signals_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)

        self._signals_table = QTableWidget()
        self._signals_table.setColumnCount(4)
        self._signals_table.setHorizontalHeaderLabels(
            ["Signal", "Estimate", "Detail", "Label"]
        )
        self._signals_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._signals_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._signals_table.verticalHeader().setVisible(False)
        self._signals_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._style_table(self._signals_table)
        layout.addWidget(self._signals_table)

        self._tabs.addTab(w, "SIGNALS")

    # ── Tab 5: Statistics ─────────────────────────────────────────

    def _build_stats_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)

        # Weight bar chart
        self._weight_plot = pg.PlotWidget(title="Method Weights")
        self._weight_plot.setBackground(Colors.BG_SECONDARY)
        self._weight_plot.showGrid(x=False, y=True, alpha=0.15)
        self._weight_plot.setLabel("left", "Weight (%)", color=Colors.TEXT_SECONDARY)
        layout.addWidget(self._weight_plot, stretch=2)

        # Bootstrap / PI info
        self._boot_label = QLabel("")
        self._boot_label.setStyleSheet(f"""
            color: {Colors.TEXT_SECONDARY};
            font-family: {Fonts.MONO};
            font-size: 10px;
            padding: 8px;
            background: {Colors.BG_TERTIARY};
            border: 1px solid {Colors.BORDER};
            border-radius: 4px;
        """)
        self._boot_label.setWordWrap(True)
        layout.addWidget(self._boot_label)

        self._tabs.addTab(w, "STATISTICS")

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def set_result(self, result: PredictionResult):
        """Update all tabs with prediction results."""
        self._result = result
        self._update_summary(result)
        self._update_methods(result)
        self._update_mc(result)
        self._update_signals(result)
        self._update_stats(result)

    def set_running(self, running: bool):
        self._run_btn.setEnabled(not running)
        self._progress.setVisible(running)

    def set_status(self, text: str):
        self._status_label.setText(text)

    # ──────────────────────────────────────────────────────────────
    # Tab Update Methods
    # ──────────────────────────────────────────────────────────────

    def _update_summary(self, r: PredictionResult):
        # Headline
        s_price = r.methods[0].est if r.methods else 0  # we need current price from data
        # Use first method's raw data or compute from ensemble
        direction = "▲" if r.ensemble_return_pct > 0 else "▼"
        color = Colors.PROFIT if r.ensemble_return_pct > 0 else Colors.LOSS

        kelly = r.signals.get("kelly", {})
        confidence = kelly.get("confidence", 0)

        self._headline.setText(
            f"{r.symbol}  •  ${r.ensemble_point:.2f}  "
            f"({direction}{abs(r.ensemble_return_pct):.1f}%)  •  "
            f"{kelly.get('label', 'N/A')} CONFIDENCE"
        )
        self._headline.setStyleSheet(f"""
            font-family: {Fonts.SANS};
            font-size: 18px;
            font-weight: bold;
            color: {color};
            background: {Colors.BG_TERTIARY};
            border: 1px solid {Colors.BORDER};
            border-radius: 6px;
            padding: 16px;
        """)

        # Metric cards
        self._set_metric("Confidence", f"{confidence:.0f}%")
        self._set_metric("Vol Scale", f"{r.vol_scale:.2f}x")
        self._set_metric("J-S Coeff", f"{r.js_coeff:.4f}")
        self._set_metric("Kelly %", f"{kelly.get('kellyHalf', 0):.1f}%")
        self._set_metric("Model Fidelity", f"{r.prediction_interval.fidelity:.1%}")

        # Prediction interval
        pi = r.prediction_interval
        self._pi_label.setText(
            f"90% Prediction Interval: ${pi.total_90[0]:.2f} — ${pi.total_90[1]:.2f}  |  "
            f"Model uncertainty: {pi.model_pct:.0f}%  |  "
            f"Market uncertainty: {pi.market_pct:.0f}%  |  "
            f"Total σ: ${pi.total_std:.2f}"
        )

        # Probability table
        probs = r.probabilities
        self._prob_table.setRowCount(len(probs))
        for i, p in enumerate(probs):
            self._prob_table.setItem(i, 0, QTableWidgetItem(p["label"]))
            val_item = QTableWidgetItem(f"{p['value']:.1f}%")
            val_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._prob_table.setItem(i, 1, val_item)

    def _update_methods(self, r: PredictionResult):
        methods = r.methods
        self._methods_table.setRowCount(len(methods))

        for i, m in enumerate(methods):
            name_item = QTableWidgetItem(m.name)
            name_item.setForeground(pg.mkColor(m.color))
            self._methods_table.setItem(i, 0, name_item)

            est_item = QTableWidgetItem(f"${m.est:.2f}")
            est_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._methods_table.setItem(i, 1, est_item)

            wt_item = QTableWidgetItem(f"{m.weight * 100:.1f}%")
            wt_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._methods_table.setItem(i, 2, wt_item)

            self._methods_table.setItem(i, 3, QTableWidgetItem(m.source))

            vs_text = "Yes" if m.vol_scaled else "No"
            self._methods_table.setItem(i, 4, QTableWidgetItem(vs_text))

    def _update_mc(self, r: PredictionResult):
        self._hist_plot.clear()
        hist = r.histogram
        if not hist:
            return

        centers = [h["c"] for h in hist]
        densities = [h["d"] for h in hist]
        width = (centers[1] - centers[0]) * 0.85 if len(centers) > 1 else 1.0

        bg = pg.BarGraphItem(
            x=centers, height=densities, width=width,
            brush=pg.mkBrush(59, 130, 246, 120),
            pen=pg.mkPen(59, 130, 246, 200),
        )
        self._hist_plot.addItem(bg)

        # Ensemble point line
        ens_pen = pg.mkPen(Colors.ACCENT, width=2, style=Qt.PenStyle.DashLine)
        self._hist_plot.addItem(
            pg.InfiniteLine(
                pos=r.ensemble_point, angle=90, pen=ens_pen,
                label=f"Ensemble ${r.ensemble_point:.2f}",
                labelOpts={"color": Colors.ACCENT, "position": 0.9},
            )
        )

        mc = r.mc
        self._mc_label.setText(
            f"P5: ${mc.get('p5', 0):.2f}  |  "
            f"P25: ${mc.get('p25', 0):.2f}  |  "
            f"Median: ${mc.get('p50', 0):.2f}  |  "
            f"P75: ${mc.get('p75', 0):.2f}  |  "
            f"P95: ${mc.get('p95', 0):.2f}  |  "
            f"Mean: ${mc.get('mean', 0):.2f}"
        )

    def _update_signals(self, r: PredictionResult):
        sig = r.signals
        # Map signal keys to display info
        signal_rows = [
            ("Regime", "reg", "regime", "regime"),
            ("Momentum", "mom", "rsi", "label"),
            ("Sector RS", "srs", "spread", "label"),
            ("Vol Regime", "volR", "vix", "regime"),
            ("Inst Sentiment", "inst", "comp", "label"),
            ("EPS Revision", "epsRev", "revision", "label"),
            ("Size Factor", "size", "capB", "label"),
            ("Value Factor", "val", "score", "label"),
            ("Quality", "qual", "score", "label"),
            ("Investment", "inv", "capex", "label"),
            ("Low Vol", "lowV", "vol", "label"),
            ("PEAD", "pead", "surprise", "label"),
            ("Seasonality", "season", "month", None),
            ("Options Skew", "opts", "ivRank", "label"),
            ("Insider", "insider", "net", "label"),
            ("Rev Accel", "revAcc", "accel", "label"),
            ("FCF Yield", "fcf", "yield", "label"),
            ("Leverage", "lev", "de", "label"),
            ("Buyback", "buyback", "change", "label"),
            ("Macro", "macro", "comp", "label"),
            ("Kelly", "kelly", "confidence", "label"),
        ]

        self._signals_table.setRowCount(len(signal_rows))
        for i, (name, key, detail_key, label_key) in enumerate(signal_rows):
            s = sig.get(key, {})
            self._signals_table.setItem(i, 0, QTableWidgetItem(name))

            est = s.get("est", 0)
            est_item = QTableWidgetItem(f"${est:.2f}" if isinstance(est, (int, float)) else str(est))
            est_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._signals_table.setItem(i, 1, est_item)

            detail = s.get(detail_key, "")
            self._signals_table.setItem(i, 2, QTableWidgetItem(str(detail)))

            label = s.get(label_key, "") if label_key else ""
            self._signals_table.setItem(i, 3, QTableWidgetItem(str(label)))

    def _update_stats(self, r: PredictionResult):
        # Weight bar chart
        self._weight_plot.clear()
        methods = r.methods
        if not methods:
            return

        names = [m.name[:12] for m in methods]
        weights = [m.weight * 100 for m in methods]
        colors = [pg.mkBrush(m.color) for m in methods]

        x = np.arange(len(methods))
        for i in range(len(methods)):
            bar = pg.BarGraphItem(
                x=[i], height=[weights[i]], width=0.7,
                brush=colors[i],
            )
            self._weight_plot.addItem(bar)

        # X-axis labels
        xax = self._weight_plot.getAxis("bottom")
        xax.setTicks([list(zip(x, names))])

        # Bootstrap / PI info
        b = r.bootstrap
        pi = r.prediction_interval
        self._boot_label.setText(
            f"Bootstrap (5K resamples):\n"
            f"  Mean: ${b.mean:.2f}  |  Std: ${b.std:.2f}\n"
            f"  68% CI: ${b.ci68[0]:.2f} — ${b.ci68[1]:.2f}\n"
            f"  90% CI: ${b.ci90[0]:.2f} — ${b.ci90[1]:.2f}\n"
            f"  95% CI: ${b.ci95[0]:.2f} — ${b.ci95[1]:.2f}\n\n"
            f"Prediction Interval:\n"
            f"  Model σ: ${pi.model_std:.2f} ({pi.model_pct:.0f}% of variance)\n"
            f"  Market σ: ${pi.market_std:.2f} ({pi.market_pct:.0f}% of variance)\n"
            f"  Total σ: ${pi.total_std:.2f}\n"
            f"  90% Total: ${pi.total_90[0]:.2f} — ${pi.total_90[1]:.2f}\n"
            f"  Signal Fidelity: {pi.fidelity:.1%}"
        )

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _on_run(self):
        symbol = self._sym_input.text().strip().upper()
        if symbol:
            horizon = self._horizon_spin.value()
            self.run_requested.emit(symbol, horizon)

    def _make_metric_card(self, title: str) -> QWidget:
        card = QWidget()
        card.setStyleSheet(f"""
            background: {Colors.BG_TERTIARY};
            border: 1px solid {Colors.BORDER};
            border-radius: 4px;
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            color: {Colors.TEXT_MUTED};
            font-family: {Fonts.SANS};
            font-size: 9px;
            border: none;
            background: transparent;
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        value_label = QLabel("—")
        value_label.setStyleSheet(f"""
            color: {Colors.TEXT_PRIMARY};
            font-family: {Fonts.MONO};
            font-size: 13px;
            font-weight: bold;
            border: none;
            background: transparent;
        """)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(value_label)

        self._metric_labels[title] = value_label
        return card

    def _set_metric(self, title: str, value: str):
        label = self._metric_labels.get(title)
        if label:
            label.setText(value)

    def _style_table(self, table: QTableWidget):
        table.setStyleSheet(f"""
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
        table.setAlternatingRowColors(True)
