"""Dockable OPTIMIZATION panel — method selection, parameters, and run control."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox,
    QDoubleSpinBox, QCheckBox, QPushButton, QProgressBar,
    QLabel, QGroupBox,
)

from portopt.gui.panels.base_panel import BasePanel
from portopt.constants import (
    Colors, Fonts, OptMethod, CovEstimator, ReturnEstimator,
    LinkageMethod, RiskMeasure,
)

# Human-readable names for optimization methods
METHOD_LABELS = {
    OptMethod.INVERSE_VARIANCE: "Inverse Variance",
    OptMethod.MIN_VOLATILITY: "Minimum Volatility",
    OptMethod.MAX_SHARPE: "Maximum Sharpe Ratio",
    OptMethod.EFFICIENT_RISK: "Efficient Risk",
    OptMethod.EFFICIENT_RETURN: "Efficient Return",
    OptMethod.MAX_QUADRATIC_UTILITY: "Max Quadratic Utility",
    OptMethod.MAX_DIVERSIFICATION: "Max Diversification",
    OptMethod.MAX_DECORRELATION: "Max Decorrelation",
    OptMethod.BLACK_LITTERMAN: "Black-Litterman",
    OptMethod.HRP: "Hierarchical Risk Parity",
    OptMethod.HERC: "Hierarchical Equal Risk Contribution",
    OptMethod.TIC: "Theory-Implied Correlation",
}


class OptimizationPanel(BasePanel):
    panel_id = "optimization"
    panel_title = "OPTIMIZATION"

    run_requested = Signal(dict)  # emits config dict
    save_requested = Signal()     # B3: request to save result for comparison

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # ── Method ───────────────────────────────────────────────────
        method_group = QGroupBox("Method")
        method_group.setStyleSheet(self._group_style())
        method_layout = QFormLayout()
        method_layout.setSpacing(4)

        self._method_combo = QComboBox()
        for method, label in METHOD_LABELS.items():
            self._method_combo.addItem(label, method)
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_layout.addRow("Optimizer:", self._method_combo)

        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # ── Estimators ───────────────────────────────────────────────
        est_group = QGroupBox("Estimators")
        est_group.setStyleSheet(self._group_style())
        est_layout = QFormLayout()
        est_layout.setSpacing(4)

        self._cov_combo = QComboBox()
        for e in CovEstimator:
            self._cov_combo.addItem(e.name.replace("_", " ").title(), e)
        est_layout.addRow("Covariance:", self._cov_combo)

        self._ret_combo = QComboBox()
        for e in ReturnEstimator:
            self._ret_combo.addItem(e.name.replace("_", " ").title(), e)
        est_layout.addRow("Returns:", self._ret_combo)

        self._linkage_combo = QComboBox()
        for lm in LinkageMethod:
            self._linkage_combo.addItem(lm.value.title(), lm)
        self._linkage_combo.setEnabled(False)
        est_layout.addRow("Linkage:", self._linkage_combo)

        self._risk_combo = QComboBox()
        for rm in RiskMeasure:
            self._risk_combo.addItem(rm.name.replace("_", " ").title(), rm)
        self._risk_combo.setEnabled(False)
        est_layout.addRow("Risk Measure:", self._risk_combo)

        est_group.setLayout(est_layout)
        layout.addWidget(est_group)

        # ── Parameters ───────────────────────────────────────────────
        params_group = QGroupBox("Parameters")
        params_group.setStyleSheet(self._group_style())
        params_layout = QFormLayout()
        params_layout.setSpacing(4)

        self._rf_spin = QDoubleSpinBox()
        self._rf_spin.setRange(0, 0.20)
        self._rf_spin.setSingleStep(0.005)
        self._rf_spin.setValue(0.02)
        self._rf_spin.setDecimals(3)
        self._rf_spin.setSuffix(" ")
        params_layout.addRow("Risk-Free Rate:", self._rf_spin)

        self._ra_spin = QDoubleSpinBox()
        self._ra_spin.setRange(0.1, 20.0)
        self._ra_spin.setSingleStep(0.5)
        self._ra_spin.setValue(1.0)
        self._ra_spin.setDecimals(1)
        params_layout.addRow("Risk Aversion:", self._ra_spin)

        self._target_ret_spin = QDoubleSpinBox()
        self._target_ret_spin.setRange(0, 1.0)
        self._target_ret_spin.setSingleStep(0.01)
        self._target_ret_spin.setValue(0.10)
        self._target_ret_spin.setDecimals(3)
        self._target_ret_spin.setEnabled(False)
        params_layout.addRow("Target Return:", self._target_ret_spin)

        self._target_risk_spin = QDoubleSpinBox()
        self._target_risk_spin.setRange(0, 1.0)
        self._target_risk_spin.setSingleStep(0.01)
        self._target_risk_spin.setValue(0.15)
        self._target_risk_spin.setDecimals(3)
        self._target_risk_spin.setEnabled(False)
        params_layout.addRow("Target Risk:", self._target_risk_spin)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # ── Constraints ──────────────────────────────────────────────
        constr_group = QGroupBox("Constraints")
        constr_group.setStyleSheet(self._group_style())
        constr_layout = QFormLayout()
        constr_layout.setSpacing(4)

        self._long_only_check = QCheckBox()
        self._long_only_check.setChecked(True)
        constr_layout.addRow("Long Only:", self._long_only_check)

        self._min_weight_spin = QDoubleSpinBox()
        self._min_weight_spin.setRange(-1.0, 1.0)
        self._min_weight_spin.setSingleStep(0.01)
        self._min_weight_spin.setValue(0.0)
        self._min_weight_spin.setDecimals(3)
        constr_layout.addRow("Min Weight:", self._min_weight_spin)

        self._max_weight_spin = QDoubleSpinBox()
        self._max_weight_spin.setRange(0.0, 1.0)
        self._max_weight_spin.setSingleStep(0.05)
        self._max_weight_spin.setValue(1.0)
        self._max_weight_spin.setDecimals(3)
        constr_layout.addRow("Max Weight:", self._max_weight_spin)

        constr_group.setLayout(constr_layout)
        layout.addWidget(constr_group)

        # ── Run Button ───────────────────────────────────────────────
        self._run_btn = QPushButton("RUN OPTIMIZATION")
        self._run_btn.setFixedHeight(36)
        self._run_btn.setCursor(pg_cursor_hand())
        self._run_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT_DIM};
                color: {Colors.ACCENT};
                border: 1px solid {Colors.ACCENT};
                border-radius: 4px;
                font-family: {Fonts.SANS};
                font-size: 12px;
                font-weight: bold;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT};
                color: {Colors.BG_PRIMARY};
            }}
            QPushButton:pressed {{
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
        layout.addWidget(self._run_btn)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setFixedHeight(4)
        self._progress.setTextVisible(False)
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.hide()
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                background: {Colors.BG_INPUT};
                border: none;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background: {Colors.ACCENT};
                border-radius: 2px;
            }}
        """)
        layout.addWidget(self._progress)

        # B3: Save result for comparison
        self._save_btn = QPushButton("SAVE TO COMPARE")
        self._save_btn.setFixedHeight(28)
        self._save_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_INPUT};
                color: {Colors.PROFIT};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                font-family: {Fonts.SANS};
                font-size: 10px;
                font-weight: bold;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background: {Colors.PROFIT_DIM};
                border-color: {Colors.PROFIT};
            }}
            QPushButton:disabled {{
                color: {Colors.TEXT_MUTED};
                border-color: {Colors.BORDER};
            }}
        """)
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(lambda: self.save_requested.emit())
        layout.addWidget(self._save_btn)

        layout.addStretch()
        self.content_layout.addLayout(layout)

    # ── Public API ───────────────────────────────────────────────────

    def get_config(self) -> dict:
        """Return current optimization configuration."""
        method = self._method_combo.currentData()
        return {
            "method": method,
            "cov_estimator": self._cov_combo.currentData(),
            "return_estimator": self._ret_combo.currentData(),
            "linkage": self._linkage_combo.currentData(),
            "risk_measure": self._risk_combo.currentData(),
            "risk_free_rate": self._rf_spin.value(),
            "risk_aversion": self._ra_spin.value(),
            "target_return": self._target_ret_spin.value(),
            "target_risk": self._target_risk_spin.value(),
            "long_only": self._long_only_check.isChecked(),
            "min_weight": self._min_weight_spin.value(),
            "max_weight": self._max_weight_spin.value(),
        }

    def set_running(self, running: bool):
        """Toggle run/progress state."""
        self._run_btn.setEnabled(not running)
        self._progress.setVisible(running)

    def set_has_result(self, has_result: bool):
        """Enable/disable the save button based on whether a result exists."""
        self._save_btn.setEnabled(has_result)

    # ── Internal ─────────────────────────────────────────────────────

    def _on_method_changed(self, _idx):
        method = self._method_combo.currentData()
        is_hierarchical = method in (OptMethod.HRP, OptMethod.HERC, OptMethod.TIC)
        self._linkage_combo.setEnabled(is_hierarchical)
        self._risk_combo.setEnabled(method == OptMethod.HERC)

        self._target_ret_spin.setEnabled(method == OptMethod.EFFICIENT_RETURN)
        self._target_risk_spin.setEnabled(method == OptMethod.EFFICIENT_RISK)

    def _on_run(self):
        self.run_requested.emit(self.get_config())

    def _group_style(self) -> str:
        return f"""
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
        """


def pg_cursor_hand():
    """Return a pointing hand cursor."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QCursor
    return QCursor(Qt.PointingHandCursor)
