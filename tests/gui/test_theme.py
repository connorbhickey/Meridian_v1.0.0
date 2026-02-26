"""Tests for the theme system — palette switching, QSS generation, theme index."""

import importlib

import pytest


class TestApplyPalette:
    """Verify apply_palette correctly mutates Colors attributes."""

    def _reload_constants(self):
        """Re-import constants to reset Colors to dark defaults."""
        import portopt.constants
        importlib.reload(portopt.constants)
        return portopt.constants

    def test_dark_palette_is_default(self):
        mod = self._reload_constants()
        assert mod.Colors.BG_PRIMARY == "#0a0e14"
        assert mod.Colors.TEXT_PRIMARY == "#f0f6fc"

    def test_apply_light_palette_changes_backgrounds(self):
        mod = self._reload_constants()
        mod.apply_palette("light")
        assert mod.Colors.BG_PRIMARY == "#f6f8fa"
        assert mod.Colors.BG_SECONDARY == "#ffffff"

    def test_apply_light_palette_changes_text(self):
        mod = self._reload_constants()
        mod.apply_palette("light")
        assert mod.Colors.TEXT_PRIMARY == "#1f2328"
        assert mod.Colors.TEXT_SECONDARY == "#32383f"

    def test_apply_light_palette_changes_accent(self):
        mod = self._reload_constants()
        mod.apply_palette("light")
        assert mod.Colors.ACCENT == "#0096b7"

    def test_apply_light_palette_changes_semantic(self):
        mod = self._reload_constants()
        mod.apply_palette("light")
        assert mod.Colors.PROFIT == "#16a34a"
        assert mod.Colors.LOSS == "#dc2626"

    def test_apply_light_palette_changes_chart_palette(self):
        mod = self._reload_constants()
        mod.apply_palette("light")
        assert mod.Colors.CHART_PALETTE[0] == "#0096b7"
        assert len(mod.Colors.CHART_PALETTE) == 10

    def test_apply_dark_is_noop(self):
        mod = self._reload_constants()
        original_bg = mod.Colors.BG_PRIMARY
        mod.apply_palette("dark")
        assert mod.Colors.BG_PRIMARY == original_bg

    def test_apply_unknown_palette_is_noop(self):
        mod = self._reload_constants()
        original_bg = mod.Colors.BG_PRIMARY
        mod.apply_palette("neon")
        assert mod.Colors.BG_PRIMARY == original_bg

    def test_light_palette_covers_all_color_attrs(self):
        """Every key in _LIGHT_PALETTE maps to an existing Colors attribute."""
        mod = self._reload_constants()
        for key in mod._LIGHT_PALETTE:
            assert hasattr(mod.Colors, key), f"Colors missing attribute: {key}"

    def test_sector_colors_unchanged_by_light(self):
        """Sector colors are not in the light palette and stay the same."""
        mod = self._reload_constants()
        original = dict(mod.Colors.SECTOR_COLORS)
        mod.apply_palette("light")
        assert mod.Colors.SECTOR_COLORS == original


class TestBuildTerminalQSS:
    """Verify QSS builder reads current Colors values."""

    def _reload_all(self):
        import portopt.constants
        importlib.reload(portopt.constants)
        import portopt.gui.theme
        importlib.reload(portopt.gui.theme)
        return portopt.constants, portopt.gui.theme

    def test_dark_qss_contains_dark_bg(self):
        consts, theme = self._reload_all()
        qss = theme._build_terminal_qss()
        assert "#0a0e14" in qss  # BG_PRIMARY dark

    def test_light_qss_contains_light_bg(self):
        consts, theme = self._reload_all()
        consts.apply_palette("light")
        qss = theme._build_terminal_qss()
        assert "#f6f8fa" in qss  # BG_PRIMARY light
        assert "#0a0e14" not in qss

    def test_qss_contains_font_families(self):
        consts, theme = self._reload_all()
        qss = theme._build_terminal_qss()
        assert "JetBrains Mono" in qss
        assert "Inter" in qss

    def test_qss_contains_widget_selectors(self):
        consts, theme = self._reload_all()
        qss = theme._build_terminal_qss()
        for selector in ["QMainWindow", "QDockWidget", "QTableWidget",
                         "QPushButton", "QMenuBar", "QStatusBar"]:
            assert selector in qss


class TestThemeConstants:
    """Verify theme index constants."""

    def test_theme_indices(self):
        from portopt.gui.theme import (
            THEME_DARK_DEEP_SPACE, THEME_DARK_CLASSIC, THEME_LIGHT,
        )
        assert THEME_DARK_DEEP_SPACE == 0
        assert THEME_DARK_CLASSIC == 1
        assert THEME_LIGHT == 2


class TestApplyTheme:
    """Verify apply_theme applies correct stylesheet."""

    pytest.importorskip("PySide6")

    def _reload_all(self):
        import portopt.constants
        importlib.reload(portopt.constants)
        import portopt.gui.theme
        importlib.reload(portopt.gui.theme)
        return portopt.constants, portopt.gui.theme

    def test_apply_dark_theme(self, qtbot, monkeypatch):
        from PySide6.QtWidgets import QApplication
        consts, theme = self._reload_all()
        app = QApplication.instance()
        monkeypatch.setattr(theme, "get_theme_index", lambda: 0)
        theme.apply_theme(app)
        ss = app.styleSheet()
        assert "#0a0e14" in ss  # dark BG present in QSS

    def test_apply_light_theme(self, qtbot, monkeypatch):
        from PySide6.QtWidgets import QApplication
        consts, theme = self._reload_all()
        app = QApplication.instance()
        monkeypatch.setattr(theme, "get_theme_index", lambda: 2)
        theme.apply_theme(app)
        ss = app.styleSheet()
        assert "#f6f8fa" in ss  # light BG present
        assert consts.Colors.BG_PRIMARY == "#f6f8fa"
