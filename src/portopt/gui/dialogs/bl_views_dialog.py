"""Dialog for entering Black-Litterman views."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QDoubleSpinBox, QLineEdit,
    QComboBox, QFrame, QScrollArea, QWidget,
)

from portopt.constants import Colors, Fonts
from portopt.engine.optimization.black_litterman import BLView


class BLViewsDialog(QDialog):
    """Dialog for inputting Black-Litterman views."""

    views_submitted = Signal(list)  # list[BLView]

    def __init__(self, symbols: list[str], parent=None):
        super().__init__(parent)
        self.symbols = sorted(symbols)
        self._view_rows: list[dict] = []
        self.setWindowTitle("Black-Litterman Views")
        self.setMinimumSize(600, 400)
        self.resize(700, 500)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Header
        header = QLabel("Define your views on asset returns:")
        header.setFont(QFont(Fonts.SANS, 11, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY};")
        layout.addWidget(header)

        hint = QLabel(
            "Absolute: 'AAPL returns 10%' | "
            "Relative: 'AAPL outperforms MSFT by 2%'"
        )
        hint.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 9px;")
        layout.addWidget(hint)

        # Scrollable view area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._container = QWidget()
        self._views_layout = QVBoxLayout(self._container)
        self._views_layout.setSpacing(4)
        self._views_layout.addStretch()
        scroll.setWidget(self._container)
        layout.addWidget(scroll, 1)

        # Add view button
        add_btn = QPushButton("+ Add View")
        add_btn.setFixedWidth(120)
        add_btn.clicked.connect(self._add_view_row)
        layout.addWidget(add_btn)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(80)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        apply_btn = QPushButton("Apply Views")
        apply_btn.setFixedWidth(100)
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

        # Start with one view row
        self._add_view_row()

    def _add_view_row(self):
        """Add a new view input row."""
        row_frame = QFrame()
        row_frame.setStyleSheet(f"""
            QFrame {{
                background: {Colors.BG_TERTIARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 3px;
            }}
        """)
        grid = QGridLayout(row_frame)
        grid.setContentsMargins(8, 4, 8, 4)
        grid.setSpacing(4)

        # View type
        type_combo = QComboBox()
        type_combo.addItems(["Absolute", "Relative"])
        type_combo.setFixedWidth(80)
        grid.addWidget(QLabel("Type:"), 0, 0)
        grid.addWidget(type_combo, 0, 1)

        # Asset 1
        asset1 = QComboBox()
        asset1.addItems(self.symbols)
        asset1.setFixedWidth(80)
        grid.addWidget(QLabel("Asset:"), 0, 2)
        grid.addWidget(asset1, 0, 3)

        # Asset 2 (for relative)
        vs_label = QLabel("vs:")
        asset2 = QComboBox()
        asset2.addItems(self.symbols)
        asset2.setFixedWidth(80)
        grid.addWidget(vs_label, 0, 4)
        grid.addWidget(asset2, 0, 5)

        # Expected return
        ret_spin = QDoubleSpinBox()
        ret_spin.setRange(-100.0, 100.0)
        ret_spin.setSuffix("%")
        ret_spin.setDecimals(1)
        ret_spin.setValue(5.0)
        ret_spin.setFixedWidth(80)
        grid.addWidget(QLabel("Return:"), 0, 6)
        grid.addWidget(ret_spin, 0, 7)

        # Confidence
        conf_spin = QDoubleSpinBox()
        conf_spin.setRange(0.0, 1.0)
        conf_spin.setSingleStep(0.1)
        conf_spin.setValue(0.5)
        conf_spin.setDecimals(2)
        conf_spin.setFixedWidth(60)
        grid.addWidget(QLabel("Conf:"), 0, 8)
        grid.addWidget(conf_spin, 0, 9)

        # Remove button
        remove_btn = QPushButton("x")
        remove_btn.setFixedSize(20, 20)
        remove_btn.setStyleSheet(f"color: {Colors.LOSS}; border: none; font-weight: bold;")
        grid.addWidget(remove_btn, 0, 10)

        # Toggle relative fields
        def on_type_changed(text):
            is_rel = text == "Relative"
            vs_label.setVisible(is_rel)
            asset2.setVisible(is_rel)

        type_combo.currentTextChanged.connect(on_type_changed)
        on_type_changed(type_combo.currentText())

        row_data = {
            "frame": row_frame,
            "type": type_combo,
            "asset1": asset1,
            "asset2": asset2,
            "return": ret_spin,
            "confidence": conf_spin,
        }
        self._view_rows.append(row_data)

        remove_btn.clicked.connect(lambda: self._remove_view_row(row_data))

        # Insert before stretch
        idx = self._views_layout.count() - 1
        self._views_layout.insertWidget(idx, row_frame)

    def _remove_view_row(self, row_data: dict):
        if len(self._view_rows) <= 1:
            return
        row_data["frame"].deleteLater()
        self._view_rows.remove(row_data)

    def _submit(self):
        views = []
        for row in self._view_rows:
            view_type = row["type"].currentText()
            asset1 = row["asset1"].currentText()
            ret_pct = row["return"].value() / 100.0
            confidence = row["confidence"].value()

            if view_type == "Absolute":
                views.append(BLView(
                    assets=[asset1],
                    weights=[1.0],
                    view_return=ret_pct,
                    confidence=confidence,
                ))
            else:
                asset2 = row["asset2"].currentText()
                if asset1 == asset2:
                    continue
                views.append(BLView(
                    assets=[asset1, asset2],
                    weights=[1.0, -1.0],
                    view_return=ret_pct,
                    confidence=confidence,
                ))

        self.views_submitted.emit(views)
        self.accept()
