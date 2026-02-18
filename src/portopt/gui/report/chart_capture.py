"""Utilities for capturing panel charts as PNG images for report embedding."""

from __future__ import annotations

import base64
import io
import logging

from PySide6.QtCore import QBuffer, QByteArray, QIODevice
from PySide6.QtWidgets import QWidget

logger = logging.getLogger(__name__)


def capture_panel_chart(panel: QWidget, width: int = 800, height: int = 400) -> bytes:
    """Capture a panel's chart area as PNG bytes.

    Looks for a pyqtgraph PlotWidget child first; falls back to grabbing
    the entire panel widget.

    Returns empty bytes if the panel is not visible or has no renderable content.
    """
    try:
        # Try to find a pyqtgraph PlotWidget child
        target = None
        try:
            import pyqtgraph as pg
            for child in panel.findChildren(pg.PlotWidget):
                target = child
                break
        except ImportError:
            pass

        if target is None:
            target = panel

        if not target.isVisible():
            return b""

        pixmap = target.grab()
        if pixmap.isNull():
            return b""

        # Scale to requested dimensions
        scaled = pixmap.scaled(
            width, height,
        )

        buf = QByteArray()
        buffer = QBuffer(buf)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        scaled.save(buffer, "PNG")
        buffer.close()

        return bytes(buf.data())

    except Exception as e:
        logger.warning("Failed to capture panel chart: %s", e)
        return b""


def capture_matplotlib_figure(fig) -> bytes:
    """Capture a matplotlib Figure as PNG bytes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure

    Returns
    -------
    bytes : PNG image data, or empty bytes on failure.
    """
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor="#0d1117", edgecolor="none")
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.warning("Failed to capture matplotlib figure: %s", e)
        return b""


def png_to_data_uri(png_bytes: bytes) -> str:
    """Convert PNG bytes to a data URI for HTML embedding.

    Returns empty string if input is empty.
    """
    if not png_bytes:
        return ""
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"
