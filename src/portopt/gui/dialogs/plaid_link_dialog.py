"""Plaid Link dialog — embeds the Plaid Link web UI for bank authentication.

Uses QWebEngineView to load a minimal HTML page that initializes Plaid Link JS.
Communicates back to Python via QWebChannel when the user completes or exits.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Qt, QUrl, Signal, Slot
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout

from portopt.constants import Colors, Fonts

logger = logging.getLogger(__name__)

# Minimal HTML page that loads Plaid Link
_LINK_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    margin: 0; padding: 20px;
    background: {bg};
    color: {text};
    font-family: {font}, sans-serif;
    display: flex; align-items: center; justify-content: center;
    height: calc(100vh - 40px);
  }}
  .loading {{ text-align: center; }}
  .loading h2 {{ color: {accent}; margin-bottom: 10px; }}
  .loading p {{ color: {muted}; font-size: 14px; }}
</style>
<script src="https://cdn.plaid.com/link/v2/stable/link-initialize.js"></script>
</head>
<body>
<div class="loading" id="status">
  <h2 id="status-heading">Connecting to your bank...</h2>
  <p id="status-detail">Plaid Link is loading. A secure window will appear shortly.</p>
</div>
<script>
var handler = Plaid.create({{
  token: '{link_token}',
  onSuccess: function(public_token, metadata) {{
    var institution = metadata.institution ? metadata.institution.name : '';
    if (window.bridge) {{
      window.bridge.onSuccess(public_token, institution);
    }}
  }},
  onExit: function(err, metadata) {{
    if (window.bridge) {{
      window.bridge.onExit(err ? JSON.stringify(err) : '');
    }}
  }},
  onLoad: function() {{
    document.getElementById('status-heading').textContent = 'Bank login ready';
    document.getElementById('status-detail').textContent = 'Complete the login in the popup window.';
  }}
}});
handler.open();
</script>
</body>
</html>"""


class _PlaidBridge(QObject):
    """JavaScript-to-Python bridge for Plaid Link callbacks."""

    success = Signal(str, str)  # (public_token, institution_name)
    exited = Signal(str)        # error message or empty

    @Slot(str, str)
    def onSuccess(self, public_token: str, institution_name: str):
        logger.info("Plaid Link success: institution=%s", institution_name)
        self.success.emit(public_token, institution_name)

    @Slot(str)
    def onExit(self, error: str):
        if error:
            logger.warning("Plaid Link exited with error: %s", error)
        else:
            logger.info("Plaid Link exited by user")
        self.exited.emit(error)


class PlaidLinkDialog(QDialog):
    """Dialog that embeds Plaid Link for connecting bank accounts.

    Signals:
        public_token_received(str, str): (public_token, institution_name) on success
    """

    public_token_received = Signal(str, str)

    def __init__(self, link_token: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Link Financial Account")
        self.setMinimumSize(500, 700)
        self.setStyleSheet(f"""
            QDialog {{
                background: {Colors.BG_PRIMARY};
            }}
        """)

        self._link_token = link_token
        self._bridge = _PlaidBridge()
        self._bridge.success.connect(self._on_success)
        self._bridge.exited.connect(self._on_exit)

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        try:
            from PySide6.QtWebChannel import QWebChannel
            from PySide6.QtWebEngineWidgets import QWebEngineView

            self._web_view = QWebEngineView()

            # Set up web channel for JS-to-Python communication
            channel = QWebChannel(self._web_view.page())
            channel.registerObject("bridge", self._bridge)
            self._web_view.page().setWebChannel(channel)

            # Build HTML with link token injected
            html = _LINK_HTML_TEMPLATE.format(
                link_token=self._link_token,
                bg=Colors.BG_PRIMARY,
                text=Colors.TEXT_PRIMARY,
                accent=Colors.ACCENT,
                muted=Colors.TEXT_MUTED,
                font=Fonts.SANS,
            )

            # Inject QWebChannel JS before the Plaid script
            qwebchannel_js = '<script src="qrc:///qtwebchannel/qwebchannel.js"></script>'
            bridge_init = """
            <script>
            new QWebChannel(qt.webChannelTransport, function(channel) {
                window.bridge = channel.objects.bridge;
            });
            </script>
            """
            html = html.replace(
                '<script src="https://cdn.plaid.com/link/v2/stable/link-initialize.js"></script>',
                qwebchannel_js + bridge_init
                + '<script src="https://cdn.plaid.com/link/v2/stable/link-initialize.js"></script>',
            )

            self._web_view.setHtml(html, QUrl("https://cdn.plaid.com"))
            layout.addWidget(self._web_view)

        except ImportError:
            # QWebEngineView not available — show fallback message
            logger.warning("QWebEngineView not available, cannot show Plaid Link")
            fallback = QLabel(
                "Plaid Link requires PySide6-WebEngine.\n\n"
                "Install it with:\n  pip install PySide6-WebEngine\n\n"
                "Then restart the application."
            )
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback.setStyleSheet(
                f"color: {Colors.TEXT_MUTED}; font-family: {Fonts.MONO}; font-size: 11px; padding: 40px;"
            )
            layout.addWidget(fallback)

    def _on_success(self, public_token: str, institution_name: str):
        self.public_token_received.emit(public_token, institution_name)
        self.accept()

    def _on_exit(self, error: str):
        self.reject()
