"""Meridian terminal theme — supports Dark (deep-space) and Light palettes.

The QSS is built dynamically from the current Colors values so that
apply_palette('light') in constants.py is respected.
"""

from portopt.constants import Colors, Fonts, apply_palette
from portopt.config import get_settings

# Theme index constants (match preferences_dialog combo order)
THEME_DARK_DEEP_SPACE = 0
THEME_DARK_CLASSIC = 1
THEME_LIGHT = 2


def _build_terminal_qss() -> str:
    """Build the terminal QSS string from current Colors/Fonts values."""
    C = Colors
    F = Fonts
    return f"""
/* ── Global ──────────────────────────────────────────────────────── */
QMainWindow {{
    background-color: {C.BG_PRIMARY};
}}

QWidget {{
    color: {C.TEXT_PRIMARY};
    font-family: "{F.SANS}", "{F.SANS_FALLBACK}", sans-serif;
    font-size: {F.SIZE_NORMAL}pt;
}}

/* ── Dock Widgets (Panels) ───────────────────────────────────────── */
QDockWidget {{
    color: {C.TEXT_PRIMARY};
    font-family: "{F.SANS}", "{F.SANS_FALLBACK}", sans-serif;
    font-size: {F.SIZE_NORMAL}pt;
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}

QDockWidget::title {{
    background-color: {C.BG_SECONDARY};
    border: 1px solid {C.BORDER};
    border-bottom: 2px solid {C.ACCENT};
    padding: 4px 8px;
    font-weight: bold;
    font-size: {F.SIZE_SMALL}pt;
    text-transform: uppercase;
}}

QDockWidget::close-button, QDockWidget::float-button {{
    border: none;
    background: transparent;
    padding: 2px;
}}

QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
    background-color: {C.BG_TERTIARY};
}}

/* ── Tab Bar (stacked dock widgets) ──────────────────────────────── */
QTabBar {{
    background-color: {C.BG_PRIMARY};
    border: none;
}}

QTabBar::tab {{
    background-color: {C.BG_SECONDARY};
    color: {C.TEXT_SECONDARY};
    border: 1px solid {C.BORDER};
    border-bottom: none;
    padding: 4px 12px;
    margin-right: 1px;
    font-size: {F.SIZE_SMALL}pt;
}}

QTabBar::tab:selected {{
    background-color: {C.BG_TERTIARY};
    color: {C.ACCENT};
    border-bottom: 2px solid {C.ACCENT};
}}

QTabBar::tab:hover:!selected {{
    background-color: {C.BG_TERTIARY};
    color: {C.TEXT_PRIMARY};
}}

/* ── Tables ──────────────────────────────────────────────────────── */
QTableWidget, QTableView, QTreeView {{
    background-color: {C.BG_PRIMARY};
    alternate-background-color: {C.BG_SECONDARY};
    gridline-color: {C.BORDER};
    border: 1px solid {C.BORDER};
    font-family: "{F.MONO}", "{F.MONO_FALLBACK}", monospace;
    font-size: {F.SIZE_SMALL}pt;
    selection-background-color: {C.ACCENT_DIM};
    selection-color: {C.TEXT_PRIMARY};
}}

QHeaderView::section {{
    background-color: {C.BG_SECONDARY};
    color: {C.TEXT_SECONDARY};
    border: 1px solid {C.BORDER};
    padding: 3px 6px;
    font-family: "{F.SANS}", "{F.SANS_FALLBACK}", sans-serif;
    font-size: {F.SIZE_SMALL}pt;
    font-weight: bold;
}}

/* ── Inputs ──────────────────────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QDateEdit {{
    background-color: {C.BG_INPUT};
    border: 1px solid {C.BORDER};
    border-radius: 2px;
    padding: 3px 6px;
    color: {C.TEXT_PRIMARY};
    font-family: "{F.MONO}", "{F.MONO_FALLBACK}", monospace;
    font-size: {F.SIZE_NORMAL}pt;
    selection-background-color: {C.ACCENT_DIM};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {C.ACCENT};
}}

QComboBox {{
    background-color: {C.BG_INPUT};
    border: 1px solid {C.BORDER};
    border-radius: 2px;
    padding: 3px 6px;
    color: {C.TEXT_PRIMARY};
    font-size: {F.SIZE_NORMAL}pt;
    min-width: 80px;
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox QAbstractItemView {{
    background-color: {C.BG_SECONDARY};
    border: 1px solid {C.BORDER};
    selection-background-color: {C.ACCENT_DIM};
}}

/* ── Buttons ─────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {C.BG_TERTIARY};
    border: 1px solid {C.BORDER};
    border-radius: 2px;
    padding: 4px 14px;
    color: {C.TEXT_PRIMARY};
    font-size: {F.SIZE_NORMAL}pt;
    font-weight: bold;
}}

QPushButton:hover {{
    background-color: {C.BG_ELEVATED};
    border-color: {C.ACCENT_HOVER};
    color: {C.ACCENT_HOVER};
}}

QPushButton:pressed {{
    background-color: {C.BG_INPUT};
}}

QPushButton:disabled {{
    color: {C.TEXT_MUTED};
    border-color: {C.BORDER};
}}

/* Primary action buttons */
QPushButton[primary="true"] {{
    background-color: {C.ACCENT_DIM};
    border-color: {C.ACCENT};
    color: {C.ACCENT};
}}

QPushButton[primary="true"]:hover {{
    background-color: {C.ACCENT_HOVER};
    color: {C.BG_PRIMARY};
}}

/* ── Scrollbars ──────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background-color: {C.BG_PRIMARY};
    width: 8px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background-color: {C.BORDER_LIGHT};
    min-height: 30px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {C.TEXT_MUTED};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar:horizontal {{
    background-color: {C.BG_PRIMARY};
    height: 8px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background-color: {C.BORDER_LIGHT};
    min-width: 30px;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {C.TEXT_MUTED};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* ── Splitters ───────────────────────────────────────────────────── */
QSplitter::handle {{
    background-color: {C.BORDER};
}}

QSplitter::handle:horizontal {{
    width: 2px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}

/* ── Status Bar ──────────────────────────────────────────────────── */
QStatusBar {{
    background-color: {C.BG_SECONDARY};
    border-top: 1px solid {C.BORDER};
    color: {C.TEXT_SECONDARY};
    font-family: "{F.MONO}", "{F.MONO_FALLBACK}", monospace;
    font-size: {F.SIZE_SMALL}pt;
}}

/* ── Menu Bar ────────────────────────────────────────────────────── */
QMenuBar {{
    background-color: {C.BG_SECONDARY};
    border-bottom: 1px solid {C.BORDER};
    color: {C.TEXT_PRIMARY};
    font-size: {F.SIZE_NORMAL}pt;
}}

QMenuBar::item:selected {{
    background-color: {C.BG_TERTIARY};
}}

QMenu {{
    background-color: {C.BG_SECONDARY};
    border: 1px solid {C.BORDER};
    color: {C.TEXT_PRIMARY};
}}

QMenu::item:selected {{
    background-color: {C.ACCENT_DIM};
    color: {C.ACCENT};
}}

QMenu::separator {{
    height: 1px;
    background-color: {C.BORDER};
    margin: 4px 8px;
}}

/* ── Tooltips ────────────────────────────────────────────────────── */
QToolTip {{
    background-color: {C.BG_TERTIARY};
    border: 1px solid {C.ACCENT};
    color: {C.TEXT_PRIMARY};
    padding: 4px 6px;
    font-family: "{F.SANS}", "{F.SANS_FALLBACK}", sans-serif;
    font-size: {F.SIZE_SMALL}pt;
}}

/* ── Group Boxes ─────────────────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {C.BORDER};
    border-radius: 3px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: bold;
    color: {C.TEXT_SECONDARY};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
    color: {C.ACCENT};
}}

/* ── Progress Bar ────────────────────────────────────────────────── */
QProgressBar {{
    background-color: {C.BG_INPUT};
    border: 1px solid {C.BORDER};
    border-radius: 2px;
    text-align: center;
    color: {C.TEXT_PRIMARY};
    font-size: {F.SIZE_SMALL}pt;
    height: 16px;
}}

QProgressBar::chunk {{
    background-color: {C.ACCENT};
    border-radius: 1px;
}}

/* ── Labels (data-value styling) ─────────────────────────────────── */
QLabel[dataValue="true"] {{
    font-family: "{F.MONO}", "{F.MONO_FALLBACK}", monospace;
    font-size: {F.SIZE_NORMAL}pt;
}}

QLabel[profit="true"] {{
    color: {C.PROFIT};
}}

QLabel[loss="true"] {{
    color: {C.LOSS};
}}

QLabel[header="true"] {{
    color: {C.ACCENT};
    font-weight: bold;
    font-size: {F.SIZE_SMALL}pt;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
"""


def get_theme_index() -> int:
    """Read the current theme index from QSettings."""
    s = get_settings()
    return int(s.value("appearance/theme", THEME_DARK_DEEP_SPACE))


def apply_theme(app) -> None:
    """Apply the selected theme to a QApplication.

    Must be called before creating any widgets so that Colors
    attributes are set correctly for panel stylesheet construction.
    """
    theme_idx = get_theme_index()

    if theme_idx == THEME_LIGHT:
        # Switch Colors to light palette before building QSS
        apply_palette("light")
        # Use terminal QSS only — no qdarkstyle base
        app.setStyleSheet(_build_terminal_qss())
    else:
        # Dark themes: qdarkstyle base + terminal overrides
        apply_palette("dark")
        import qdarkstyle
        base = qdarkstyle.load_stylesheet(qt_api="pyside6")
        app.setStyleSheet(base + "\n" + _build_terminal_qss())
