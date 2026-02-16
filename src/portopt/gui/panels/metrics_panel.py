"""Dockable METRICS panel — dense Bloomberg-style performance metrics grid."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QVBoxLayout, QGridLayout, QLabel, QFrame, QScrollArea, QWidget,
)

from portopt.gui.panels.base_panel import BasePanel
from portopt.constants import Colors, Fonts

# Metrics display config: (key, label, format_fn)
METRIC_DEFS = [
    ("total_return", "Total Return", lambda v: f"{v:+.2%}"),
    ("annual_return", "Annual Return", lambda v: f"{v:+.2%}"),
    ("annual_volatility", "Annual Vol", lambda v: f"{v:.2%}"),
    ("sharpe_ratio", "Sharpe Ratio", lambda v: f"{v:.3f}"),
    ("sortino_ratio", "Sortino Ratio", lambda v: f"{v:.3f}"),
    ("calmar_ratio", "Calmar Ratio", lambda v: f"{v:.3f}"),
    ("max_drawdown", "Max Drawdown", lambda v: f"{v:.2%}"),
    ("avg_drawdown", "Avg Drawdown", lambda v: f"{v:.2%}"),
    ("max_drawdown_duration", "Max DD Duration", lambda v: f"{v:.0f}d"),
    ("var_95", "VaR (95%)", lambda v: f"{v:.2%}"),
    ("cvar_95", "CVaR (95%)", lambda v: f"{v:.2%}"),
    ("skewness", "Skewness", lambda v: f"{v:.3f}"),
    ("kurtosis", "Kurtosis", lambda v: f"{v:.3f}"),
    ("omega_ratio", "Omega Ratio", lambda v: f"{v:.3f}"),
    ("gain_to_pain", "Gain/Pain", lambda v: f"{v:.3f}"),
    ("tail_ratio", "Tail Ratio", lambda v: f"{v:.3f}"),
    ("win_rate", "Win Rate", lambda v: f"{v:.1%}"),
    ("profit_factor", "Profit Factor", lambda v: f"{v:.3f}"),
    ("best_day", "Best Day", lambda v: f"{v:+.2%}"),
    ("worst_day", "Worst Day", lambda v: f"{v:+.2%}"),
    ("avg_daily_return", "Avg Daily", lambda v: f"{v:+.4%}"),
    ("daily_vol", "Daily Vol", lambda v: f"{v:.4%}"),
    ("information_ratio", "Info Ratio", lambda v: f"{v:.3f}"),
    ("tracking_error", "Track Error", lambda v: f"{v:.2%}"),
]


class MetricsPanel(BasePanel):
    panel_id = "metrics"
    panel_title = "METRICS"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cells = {}  # key -> (label_widget, value_widget)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background: {Colors.BG_SECONDARY};
                border: 1px solid {Colors.BORDER};
            }}
        """)

        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(1)

        n_cols = 3  # 3 metric cells per row
        for i, (key, label, _fmt) in enumerate(METRIC_DEFS):
            row = i // n_cols
            col = i % n_cols

            cell = self._make_cell(key, label)
            grid.addWidget(cell, row, col)

        grid.setRowStretch(len(METRIC_DEFS) // n_cols + 1, 1)
        scroll.setWidget(container)
        layout.addWidget(scroll)
        self.content_layout.addLayout(layout)

    def _make_cell(self, key: str, label: str) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background: {Colors.BG_TERTIARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 2px;
            }}
        """)
        frame.setFixedHeight(48)

        vlayout = QVBoxLayout(frame)
        vlayout.setContentsMargins(6, 3, 6, 3)
        vlayout.setSpacing(0)

        lbl = QLabel(label)
        lbl.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-family: {Fonts.SANS}; "
            f"font-size: 8px; font-weight: bold; text-transform: uppercase; border: none;"
        )
        lbl.setAlignment(Qt.AlignLeft)
        vlayout.addWidget(lbl)

        val = QLabel("—")
        val.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-family: {Fonts.MONO}; "
            f"font-size: 13px; font-weight: bold; border: none;"
        )
        val.setAlignment(Qt.AlignLeft)
        vlayout.addWidget(val)

        self._cells[key] = (lbl, val)
        return frame

    # ── Public API ───────────────────────────────────────────────────

    def set_metrics(self, metrics: dict):
        """Update displayed metrics from a dict of key -> float values."""
        for key, label, fmt in METRIC_DEFS:
            if key not in self._cells:
                continue
            _, val_widget = self._cells[key]

            value = metrics.get(key)
            if value is None:
                val_widget.setText("—")
                val_widget.setStyleSheet(
                    f"color: {Colors.TEXT_MUTED}; font-family: {Fonts.MONO}; "
                    f"font-size: 13px; font-weight: bold; border: none;"
                )
                continue

            try:
                text = fmt(value)
            except (TypeError, ValueError):
                text = str(value)

            # Color coding: green for positive, red for negative for return-like metrics
            color = Colors.TEXT_PRIMARY
            colored_keys = {
                "total_return", "annual_return", "sharpe_ratio", "sortino_ratio",
                "calmar_ratio", "omega_ratio", "gain_to_pain", "best_day", "worst_day",
                "avg_daily_return", "information_ratio",
            }
            if key in colored_keys:
                if value > 0:
                    color = Colors.PROFIT
                elif value < 0:
                    color = Colors.LOSS

            # Drawdown metrics are always negative — color red
            if key in ("max_drawdown", "avg_drawdown", "var_95", "cvar_95"):
                color = Colors.LOSS if value < 0 else Colors.TEXT_PRIMARY

            val_widget.setText(text)
            val_widget.setStyleSheet(
                f"color: {color}; font-family: {Fonts.MONO}; "
                f"font-size: 13px; font-weight: bold; border: none;"
            )

    def clear_metrics(self):
        for key, (_, val_widget) in self._cells.items():
            val_widget.setText("—")
            val_widget.setStyleSheet(
                f"color: {Colors.TEXT_MUTED}; font-family: {Fonts.MONO}; "
                f"font-size: 13px; font-weight: bold; border: none;"
            )
