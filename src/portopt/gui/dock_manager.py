"""Dock layout manager — save/restore named panel arrangements."""

from PySide6.QtCore import QByteArray
from PySide6.QtWidgets import QMainWindow

from portopt.config import get_settings


LAYOUT_KEY_PREFIX = "layouts/"
LAST_LAYOUT_KEY = "last_layout"


class DockManager:
    """Manage dock widget layout presets for the main window."""

    def __init__(self, main_window: QMainWindow):
        self._window = main_window
        self._settings = get_settings()

    def save_layout(self, name: str):
        """Save current dock arrangement under a named preset."""
        state = self._window.saveState()
        geometry = self._window.saveGeometry()
        self._settings.setValue(f"{LAYOUT_KEY_PREFIX}{name}/state", state)
        self._settings.setValue(f"{LAYOUT_KEY_PREFIX}{name}/geometry", geometry)
        self._settings.setValue(LAST_LAYOUT_KEY, name)

    def restore_layout(self, name: str) -> bool:
        """Restore a named layout preset. Returns True if successful."""
        state = self._settings.value(f"{LAYOUT_KEY_PREFIX}{name}/state")
        geometry = self._settings.value(f"{LAYOUT_KEY_PREFIX}{name}/geometry")
        if state is not None:
            if geometry is not None:
                self._window.restoreGeometry(geometry)
            self._window.restoreState(state)
            self._settings.setValue(LAST_LAYOUT_KEY, name)
            return True
        return False

    def restore_last(self) -> bool:
        """Restore the last used layout."""
        last = self._settings.value(LAST_LAYOUT_KEY)
        if last:
            return self.restore_layout(last)
        return False

    def delete_layout(self, name: str):
        """Delete a saved layout preset."""
        self._settings.remove(f"{LAYOUT_KEY_PREFIX}{name}")

    def list_layouts(self) -> list[str]:
        """Return names of all saved layout presets."""
        self._settings.beginGroup("layouts")
        names = self._settings.childGroups()
        self._settings.endGroup()
        return names

    def save_session(self):
        """Save current state as the session state (auto-restore on next launch)."""
        self.save_layout("__session__")

    def restore_session(self) -> bool:
        """Restore the session state from last close."""
        return self.restore_layout("__session__")
