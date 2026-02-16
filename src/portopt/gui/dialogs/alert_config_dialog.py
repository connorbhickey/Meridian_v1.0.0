"""Dialog for configuring risk alert thresholds.

B4 feature: lets users set warning thresholds for key risk metrics.
When a metric breaches its threshold, visual indicators appear on the
risk panel gauges and an alert signal is emitted.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QLabel, QVBoxLayout,
)

from portopt.constants import Colors, Fonts

# Default thresholds (None = disabled)
DEFAULT_ALERTS = {
    "var_95": {"enabled": False, "threshold": -0.03, "label": "VaR 95%"},
    "cvar_95": {"enabled": False, "threshold": -0.05, "label": "CVaR 95%"},
    "max_drawdown": {"enabled": True, "threshold": -0.20, "label": "Max Drawdown"},
    "annual_volatility": {"enabled": True, "threshold": 0.30, "label": "Annual Volatility"},
}


class AlertConfigDialog(QDialog):
    """Dialog for setting risk metric alert thresholds."""

    def __init__(self, current_alerts: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Risk Alert Configuration")
        self.setFixedWidth(380)
        self.setStyleSheet(f"""
            QDialog {{
                background: {Colors.BG_PRIMARY};
                color: {Colors.TEXT_PRIMARY};
            }}
            QLabel {{
                color: {Colors.TEXT_SECONDARY};
                font-family: {Fonts.SANS};
                font-size: 10px;
            }}
        """)

        self._alerts = current_alerts or dict(DEFAULT_ALERTS)
        self._widgets: dict[str, tuple[QCheckBox, QDoubleSpinBox]] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        title = QLabel("Set thresholds for risk metric alerts")
        title.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-size: 12px; font-weight: bold;"
        )
        layout.addWidget(title)

        form = QFormLayout()
        form.setSpacing(6)

        for key, config in self._alerts.items():
            row_layout = QFormLayout()

            check = QCheckBox()
            check.setChecked(config.get("enabled", False))

            spin = QDoubleSpinBox()
            spin.setRange(-1.0, 1.0)
            spin.setSingleStep(0.01)
            spin.setDecimals(3)
            spin.setValue(config.get("threshold", 0.0))
            spin.setEnabled(config.get("enabled", False))
            spin.setStyleSheet(f"""
                QDoubleSpinBox {{
                    background: {Colors.BG_INPUT};
                    color: {Colors.TEXT_PRIMARY};
                    border: 1px solid {Colors.BORDER};
                    border-radius: 3px;
                    padding: 2px 4px;
                    font-family: {Fonts.MONO};
                }}
            """)

            check.toggled.connect(spin.setEnabled)

            self._widgets[key] = (check, spin)
            label = config.get("label", key)
            form.addRow(f"{label}:", check)
            form.addRow("  Threshold:", spin)

        layout.addLayout(form)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_INPUT};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 3px;
                padding: 4px 16px;
                font-family: {Fonts.SANS};
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
                border-color: {Colors.ACCENT};
            }}
        """)
        layout.addWidget(buttons)

    def get_alerts(self) -> dict:
        """Return the configured alert settings."""
        result = {}
        for key, (check, spin) in self._widgets.items():
            config = dict(self._alerts[key])
            config["enabled"] = check.isChecked()
            config["threshold"] = spin.value()
            result[key] = config
        return result
