"""Base class for all dockable terminal panels."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QLabel

from portopt.constants import Colors, Fonts


class BasePanel(QDockWidget):
    """Base dockable panel with standard terminal styling."""

    panel_id: str = ""
    panel_title: str = "Panel"

    def __init__(self, parent=None):
        super().__init__(self.panel_title, parent)
        self.setObjectName(self.panel_id)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )

        # Content container
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(2)
        self.content_layout = self._layout
        self.setWidget(self._container)

    def add_placeholder(self, text: str = ""):
        """Add a centered placeholder label (for stubs)."""
        label = QLabel(text or f"{self.panel_title}\n(Phase build pending)")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFont(QFont(Fonts.MONO, Fonts.SIZE_SMALL))
        label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; padding: 20px;")
        self._layout.addWidget(label)
