"""Dark trading terminal theme — Bloomberg/Citadel aesthetic.

Layered on top of qdarkstyle as a base, with custom QSS overrides
for dense information display, monospace data fonts, and P&L coloring.
"""

from portopt.constants import Colors, Fonts

TERMINAL_QSS = f"""
/* ── Global ──────────────────────────────────────────────────────── */
QMainWindow {{
    background-color: {Colors.BG_PRIMARY};
}}

QWidget {{
    color: {Colors.TEXT_PRIMARY};
    font-family: "{Fonts.SANS}";
    font-size: {Fonts.SIZE_NORMAL}pt;
}}

/* ── Dock Widgets (Panels) ───────────────────────────────────────── */
QDockWidget {{
    color: {Colors.TEXT_PRIMARY};
    font-family: "{Fonts.SANS}";
    font-size: {Fonts.SIZE_NORMAL}pt;
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}

QDockWidget::title {{
    background-color: {Colors.BG_SECONDARY};
    border: 1px solid {Colors.BORDER};
    border-bottom: 2px solid {Colors.ACCENT};
    padding: 4px 8px;
    font-weight: bold;
    font-size: {Fonts.SIZE_SMALL}pt;
    text-transform: uppercase;
}}

QDockWidget::close-button, QDockWidget::float-button {{
    border: none;
    background: transparent;
    padding: 2px;
}}

QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
    background-color: {Colors.BG_TERTIARY};
}}

/* ── Tab Bar (stacked dock widgets) ──────────────────────────────── */
QTabBar {{
    background-color: {Colors.BG_PRIMARY};
    border: none;
}}

QTabBar::tab {{
    background-color: {Colors.BG_SECONDARY};
    color: {Colors.TEXT_SECONDARY};
    border: 1px solid {Colors.BORDER};
    border-bottom: none;
    padding: 4px 12px;
    margin-right: 1px;
    font-size: {Fonts.SIZE_SMALL}pt;
}}

QTabBar::tab:selected {{
    background-color: {Colors.BG_TERTIARY};
    color: {Colors.ACCENT};
    border-bottom: 2px solid {Colors.ACCENT};
}}

QTabBar::tab:hover:!selected {{
    background-color: {Colors.BG_TERTIARY};
    color: {Colors.TEXT_PRIMARY};
}}

/* ── Tables ──────────────────────────────────────────────────────── */
QTableWidget, QTableView, QTreeView {{
    background-color: {Colors.BG_PRIMARY};
    alternate-background-color: {Colors.BG_SECONDARY};
    gridline-color: {Colors.BORDER};
    border: 1px solid {Colors.BORDER};
    font-family: "{Fonts.MONO}";
    font-size: {Fonts.SIZE_SMALL}pt;
    selection-background-color: {Colors.ACCENT_DIM};
    selection-color: {Colors.TEXT_PRIMARY};
}}

QHeaderView::section {{
    background-color: {Colors.BG_SECONDARY};
    color: {Colors.TEXT_SECONDARY};
    border: 1px solid {Colors.BORDER};
    padding: 3px 6px;
    font-family: "{Fonts.SANS}";
    font-size: {Fonts.SIZE_SMALL}pt;
    font-weight: bold;
}}

/* ── Inputs ──────────────────────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QDateEdit {{
    background-color: {Colors.BG_INPUT};
    border: 1px solid {Colors.BORDER};
    border-radius: 2px;
    padding: 3px 6px;
    color: {Colors.TEXT_PRIMARY};
    font-family: "{Fonts.MONO}";
    font-size: {Fonts.SIZE_NORMAL}pt;
    selection-background-color: {Colors.ACCENT_DIM};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {Colors.ACCENT};
}}

QComboBox {{
    background-color: {Colors.BG_INPUT};
    border: 1px solid {Colors.BORDER};
    border-radius: 2px;
    padding: 3px 6px;
    color: {Colors.TEXT_PRIMARY};
    font-size: {Fonts.SIZE_NORMAL}pt;
    min-width: 80px;
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox QAbstractItemView {{
    background-color: {Colors.BG_SECONDARY};
    border: 1px solid {Colors.BORDER};
    selection-background-color: {Colors.ACCENT_DIM};
}}

/* ── Buttons ─────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {Colors.BG_TERTIARY};
    border: 1px solid {Colors.BORDER};
    border-radius: 2px;
    padding: 4px 14px;
    color: {Colors.TEXT_PRIMARY};
    font-size: {Fonts.SIZE_NORMAL}pt;
    font-weight: bold;
}}

QPushButton:hover {{
    background-color: {Colors.ACCENT_DIM};
    border-color: {Colors.ACCENT};
    color: {Colors.ACCENT};
}}

QPushButton:pressed {{
    background-color: {Colors.BG_INPUT};
}}

QPushButton:disabled {{
    color: {Colors.TEXT_MUTED};
    border-color: {Colors.BORDER};
}}

/* Primary action buttons */
QPushButton[primary="true"] {{
    background-color: {Colors.ACCENT_DIM};
    border-color: {Colors.ACCENT};
    color: {Colors.ACCENT};
}}

QPushButton[primary="true"]:hover {{
    background-color: {Colors.ACCENT};
    color: {Colors.BG_PRIMARY};
}}

/* ── Scrollbars ──────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background-color: {Colors.BG_PRIMARY};
    width: 8px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background-color: {Colors.BORDER_LIGHT};
    min-height: 30px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {Colors.TEXT_MUTED};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar:horizontal {{
    background-color: {Colors.BG_PRIMARY};
    height: 8px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background-color: {Colors.BORDER_LIGHT};
    min-width: 30px;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {Colors.TEXT_MUTED};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* ── Splitters ───────────────────────────────────────────────────── */
QSplitter::handle {{
    background-color: {Colors.BORDER};
}}

QSplitter::handle:horizontal {{
    width: 2px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}

/* ── Status Bar ──────────────────────────────────────────────────── */
QStatusBar {{
    background-color: {Colors.BG_SECONDARY};
    border-top: 1px solid {Colors.BORDER};
    color: {Colors.TEXT_SECONDARY};
    font-family: "{Fonts.MONO}";
    font-size: {Fonts.SIZE_SMALL}pt;
}}

/* ── Menu Bar ────────────────────────────────────────────────────── */
QMenuBar {{
    background-color: {Colors.BG_SECONDARY};
    border-bottom: 1px solid {Colors.BORDER};
    color: {Colors.TEXT_PRIMARY};
    font-size: {Fonts.SIZE_NORMAL}pt;
}}

QMenuBar::item:selected {{
    background-color: {Colors.BG_TERTIARY};
}}

QMenu {{
    background-color: {Colors.BG_SECONDARY};
    border: 1px solid {Colors.BORDER};
    color: {Colors.TEXT_PRIMARY};
}}

QMenu::item:selected {{
    background-color: {Colors.ACCENT_DIM};
    color: {Colors.ACCENT};
}}

QMenu::separator {{
    height: 1px;
    background-color: {Colors.BORDER};
    margin: 4px 8px;
}}

/* ── Tooltips ────────────────────────────────────────────────────── */
QToolTip {{
    background-color: {Colors.BG_TERTIARY};
    border: 1px solid {Colors.ACCENT};
    color: {Colors.TEXT_PRIMARY};
    padding: 4px 6px;
    font-family: "{Fonts.SANS}";
    font-size: {Fonts.SIZE_SMALL}pt;
}}

/* ── Group Boxes ─────────────────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {Colors.BORDER};
    border-radius: 3px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: bold;
    color: {Colors.TEXT_SECONDARY};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
    color: {Colors.ACCENT};
}}

/* ── Progress Bar ────────────────────────────────────────────────── */
QProgressBar {{
    background-color: {Colors.BG_INPUT};
    border: 1px solid {Colors.BORDER};
    border-radius: 2px;
    text-align: center;
    color: {Colors.TEXT_PRIMARY};
    font-size: {Fonts.SIZE_SMALL}pt;
    height: 16px;
}}

QProgressBar::chunk {{
    background-color: {Colors.ACCENT};
    border-radius: 1px;
}}

/* ── Labels (data-value styling) ─────────────────────────────────── */
QLabel[dataValue="true"] {{
    font-family: "{Fonts.MONO}";
    font-size: {Fonts.SIZE_NORMAL}pt;
}}

QLabel[profit="true"] {{
    color: {Colors.PROFIT};
}}

QLabel[loss="true"] {{
    color: {Colors.LOSS};
}}

QLabel[header="true"] {{
    color: {Colors.ACCENT};
    font-weight: bold;
    font-size: {Fonts.SIZE_SMALL}pt;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
"""


def apply_theme(app):
    """Apply the trading terminal dark theme to a QApplication."""
    import qdarkstyle
    # Base dark style
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyside6"))
    # Layer our terminal overrides on top
    app.setStyleSheet(app.styleSheet() + "\n" + TERMINAL_QSS)
