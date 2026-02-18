"""Preferences dialog — general settings, data, appearance."""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QSpinBox,
    QTabWidget, QWidget, QGroupBox, QDoubleSpinBox,
)

from portopt.config import get_settings
from portopt.constants import Colors, Fonts

logger = logging.getLogger(__name__)

# Settings keys
KEY_THEME = "appearance/theme"
KEY_FONT_SIZE = "appearance/font_size"
KEY_SHOW_TICKER = "appearance/show_ticker_bar"
KEY_CACHE_DAYS = "data/cache_days"
KEY_DEFAULT_LOOKBACK = "data/default_lookback_years"
KEY_AUTO_FETCH = "data/auto_fetch_on_import"
KEY_RISK_FREE_RATE = "optimization/risk_free_rate"
KEY_LONG_ONLY_DEFAULT = "optimization/long_only_default"
KEY_DEFAULT_METHOD = "optimization/default_method"


class PreferencesDialog(QDialog):
    """Application preferences dialog with tabs for each category."""

    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumSize(480, 400)
        self._settings = get_settings()
        self._build_ui()
        self._load_settings()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        header = QLabel("Preferences")
        header.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-size: 14px; font-weight: bold;"
        )
        layout.addWidget(header)

        # Tab widget
        tabs = QTabWidget()
        tabs.addTab(self._build_general_tab(), "General")
        tabs.addTab(self._build_data_tab(), "Data")
        tabs.addTab(self._build_optimization_tab(), "Optimization")
        layout.addWidget(tabs)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        defaults_btn = QPushButton("Restore Defaults")
        defaults_btn.setFixedWidth(120)
        defaults_btn.clicked.connect(self._restore_defaults)
        btn_row.addWidget(defaults_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(80)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton("Save")
        save_btn.setFixedWidth(80)
        save_btn.setStyleSheet(f"""
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
        save_btn.clicked.connect(self._save_and_close)
        btn_row.addWidget(save_btn)

        layout.addLayout(btn_row)

    def _build_general_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Appearance
        appearance = QGroupBox("Appearance")
        form = QFormLayout()

        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["Dark (Deep Space)", "Dark (Classic)"])
        form.addRow("Theme:", self._theme_combo)

        self._font_size_spin = QSpinBox()
        self._font_size_spin.setRange(8, 16)
        self._font_size_spin.setValue(10)
        self._font_size_spin.setSuffix(" px")
        form.addRow("Font Size:", self._font_size_spin)

        self._show_ticker = QCheckBox("Show ticker bar")
        self._show_ticker.setChecked(True)
        form.addRow(self._show_ticker)

        appearance.setLayout(form)
        layout.addWidget(appearance)
        layout.addStretch()
        return widget

    def _build_data_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        cache_group = QGroupBox("Price Cache")
        form = QFormLayout()

        self._cache_days_spin = QSpinBox()
        self._cache_days_spin.setRange(1, 365)
        self._cache_days_spin.setValue(7)
        self._cache_days_spin.setSuffix(" days")
        form.addRow("Cache Expiry:", self._cache_days_spin)

        self._auto_fetch = QCheckBox("Auto-fetch prices on portfolio import")
        self._auto_fetch.setChecked(True)
        form.addRow(self._auto_fetch)

        cache_group.setLayout(form)
        layout.addWidget(cache_group)

        lookback_group = QGroupBox("Historical Data")
        form2 = QFormLayout()

        self._lookback_spin = QSpinBox()
        self._lookback_spin.setRange(1, 20)
        self._lookback_spin.setValue(3)
        self._lookback_spin.setSuffix(" years")
        form2.addRow("Default Lookback:", self._lookback_spin)

        lookback_group.setLayout(form2)
        layout.addWidget(lookback_group)
        layout.addStretch()
        return widget

    def _build_optimization_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        defaults_group = QGroupBox("Default Parameters")
        form = QFormLayout()

        self._risk_free_spin = QDoubleSpinBox()
        self._risk_free_spin.setRange(0, 20)
        self._risk_free_spin.setValue(4.0)
        self._risk_free_spin.setSuffix("%")
        self._risk_free_spin.setDecimals(2)
        form.addRow("Risk-Free Rate:", self._risk_free_spin)

        self._long_only = QCheckBox("Long-only by default")
        self._long_only.setChecked(True)
        form.addRow(self._long_only)

        self._method_combo = QComboBox()
        self._method_combo.addItems([
            "Max Sharpe", "Min Volatility", "Risk Parity (HRP)",
            "Equal Risk Contribution", "Black-Litterman",
        ])
        form.addRow("Default Method:", self._method_combo)

        defaults_group.setLayout(form)
        layout.addWidget(defaults_group)
        layout.addStretch()
        return widget

    def _load_settings(self):
        """Load current settings into widgets."""
        s = self._settings

        self._theme_combo.setCurrentIndex(
            int(s.value(KEY_THEME, 0))
        )
        self._font_size_spin.setValue(
            int(s.value(KEY_FONT_SIZE, 10))
        )
        self._show_ticker.setChecked(
            s.value(KEY_SHOW_TICKER, True) in (True, "true", "True", 1)
        )
        self._cache_days_spin.setValue(
            int(s.value(KEY_CACHE_DAYS, 7))
        )
        self._lookback_spin.setValue(
            int(s.value(KEY_DEFAULT_LOOKBACK, 3))
        )
        self._auto_fetch.setChecked(
            s.value(KEY_AUTO_FETCH, True) in (True, "true", "True", 1)
        )
        self._risk_free_spin.setValue(
            float(s.value(KEY_RISK_FREE_RATE, 4.0))
        )
        self._long_only.setChecked(
            s.value(KEY_LONG_ONLY_DEFAULT, True) in (True, "true", "True", 1)
        )
        self._method_combo.setCurrentIndex(
            int(s.value(KEY_DEFAULT_METHOD, 0))
        )

    def _save_and_close(self):
        """Write settings and close dialog."""
        s = self._settings

        s.setValue(KEY_THEME, self._theme_combo.currentIndex())
        s.setValue(KEY_FONT_SIZE, self._font_size_spin.value())
        s.setValue(KEY_SHOW_TICKER, self._show_ticker.isChecked())
        s.setValue(KEY_CACHE_DAYS, self._cache_days_spin.value())
        s.setValue(KEY_DEFAULT_LOOKBACK, self._lookback_spin.value())
        s.setValue(KEY_AUTO_FETCH, self._auto_fetch.isChecked())
        s.setValue(KEY_RISK_FREE_RATE, self._risk_free_spin.value())
        s.setValue(KEY_LONG_ONLY_DEFAULT, self._long_only.isChecked())
        s.setValue(KEY_DEFAULT_METHOD, self._method_combo.currentIndex())

        s.sync()
        self.settings_changed.emit()
        logger.info("Preferences saved")
        self.accept()

    def _restore_defaults(self):
        """Reset all widgets to defaults."""
        self._theme_combo.setCurrentIndex(0)
        self._font_size_spin.setValue(10)
        self._show_ticker.setChecked(True)
        self._cache_days_spin.setValue(7)
        self._lookback_spin.setValue(3)
        self._auto_fetch.setChecked(True)
        self._risk_free_spin.setValue(4.0)
        self._long_only.setChecked(True)
        self._method_combo.setCurrentIndex(0)
