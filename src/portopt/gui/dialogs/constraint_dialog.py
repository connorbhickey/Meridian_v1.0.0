"""Dialog for editing portfolio optimization constraints."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QPushButton, QDoubleSpinBox, QCheckBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
)

from portopt.constants import Colors, Fonts
from portopt.engine.constraints import PortfolioConstraints


class ConstraintDialog(QDialog):
    """Dialog for editing optimization constraints."""

    constraints_updated = Signal(object)  # PortfolioConstraints

    def __init__(self, symbols: list[str],
                 current: PortfolioConstraints | None = None,
                 parent=None):
        super().__init__(parent)
        self.symbols = sorted(symbols)
        self._current = current or PortfolioConstraints()
        self.setWindowTitle("Optimization Constraints")
        self.setMinimumSize(500, 450)
        self.resize(550, 500)
        self._build_ui()
        self._load_current()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Global constraints
        global_group = QGroupBox("Global Constraints")
        global_form = QFormLayout()
        global_form.setSpacing(4)

        self._long_only = QCheckBox("Long Only (no shorting)")
        global_form.addRow(self._long_only)

        self._market_neutral = QCheckBox("Market Neutral (weights sum to 0)")
        global_form.addRow(self._market_neutral)

        self._min_weight = QDoubleSpinBox()
        self._min_weight.setRange(-1.0, 1.0)
        self._min_weight.setSingleStep(0.01)
        self._min_weight.setDecimals(3)
        global_form.addRow("Min Weight:", self._min_weight)

        self._max_weight = QDoubleSpinBox()
        self._max_weight.setRange(0.01, 1.0)
        self._max_weight.setSingleStep(0.05)
        self._max_weight.setDecimals(3)
        self._max_weight.setValue(1.0)
        global_form.addRow("Max Weight:", self._max_weight)

        self._leverage = QDoubleSpinBox()
        self._leverage.setRange(0.1, 5.0)
        self._leverage.setSingleStep(0.1)
        self._leverage.setDecimals(2)
        self._leverage.setValue(1.0)
        global_form.addRow("Leverage:", self._leverage)

        self._max_turnover = QDoubleSpinBox()
        self._max_turnover.setRange(0.0, 10.0)
        self._max_turnover.setSingleStep(0.1)
        self._max_turnover.setDecimals(2)
        self._max_turnover.setSpecialValueText("Unconstrained")
        global_form.addRow("Max Turnover:", self._max_turnover)

        global_group.setLayout(global_form)
        layout.addWidget(global_group)

        # Per-asset bounds table
        bounds_group = QGroupBox("Per-Asset Weight Bounds (optional)")
        bounds_layout = QVBoxLayout()

        self._bounds_table = QTableWidget(len(self.symbols), 3)
        self._bounds_table.setHorizontalHeaderLabels(["Symbol", "Min", "Max"])
        self._bounds_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._bounds_table.setMaximumHeight(200)

        for i, sym in enumerate(self.symbols):
            item = QTableWidgetItem(sym)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._bounds_table.setItem(i, 0, item)

            min_spin = QDoubleSpinBox()
            min_spin.setRange(-1.0, 1.0)
            min_spin.setDecimals(3)
            min_spin.setSpecialValueText("—")
            self._bounds_table.setCellWidget(i, 1, min_spin)

            max_spin = QDoubleSpinBox()
            max_spin.setRange(0.0, 1.0)
            max_spin.setDecimals(3)
            max_spin.setValue(0.0)
            max_spin.setSpecialValueText("—")
            self._bounds_table.setCellWidget(i, 2, max_spin)

        bounds_layout.addWidget(self._bounds_table)
        bounds_group.setLayout(bounds_layout)
        layout.addWidget(bounds_group)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset)
        btn_row.addWidget(reset_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(80)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        apply_btn = QPushButton("Apply")
        apply_btn.setFixedWidth(80)
        apply_btn.setStyleSheet(f"""
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
        apply_btn.clicked.connect(self._submit)
        btn_row.addWidget(apply_btn)

        layout.addLayout(btn_row)

    def _load_current(self):
        c = self._current
        self._long_only.setChecked(c.long_only)
        self._market_neutral.setChecked(c.market_neutral)
        self._min_weight.setValue(c.min_weight)
        self._max_weight.setValue(c.max_weight)
        self._leverage.setValue(c.leverage)
        self._max_turnover.setValue(c.max_turnover or 0.0)

    def _reset(self):
        self._current = PortfolioConstraints()
        self._load_current()
        for i in range(self._bounds_table.rowCount()):
            self._bounds_table.cellWidget(i, 1).setValue(0.0)
            self._bounds_table.cellWidget(i, 2).setValue(0.0)

    def _submit(self):
        weight_bounds = {}
        for i, sym in enumerate(self.symbols):
            lo = self._bounds_table.cellWidget(i, 1).value()
            hi = self._bounds_table.cellWidget(i, 2).value()
            if lo != 0.0 or hi != 0.0:
                weight_bounds[sym] = (lo, hi if hi > 0 else 1.0)

        constraints = PortfolioConstraints(
            long_only=self._long_only.isChecked(),
            market_neutral=self._market_neutral.isChecked(),
            min_weight=self._min_weight.value(),
            max_weight=self._max_weight.value(),
            leverage=self._leverage.value(),
            max_turnover=self._max_turnover.value() or None,
            weight_bounds=weight_bounds,
        )

        self.constraints_updated.emit(constraints)
        self.accept()
