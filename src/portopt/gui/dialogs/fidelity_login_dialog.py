"""Fidelity connection dialog — CSV import (primary) + browser automation (secondary)."""

import webbrowser

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QCheckBox, QStackedWidget, QWidget, QProgressBar,
    QGroupBox, QFormLayout, QFileDialog,
)

from portopt.constants import Colors, Fonts
from portopt.utils.threading import run_in_thread

_FIDELITY_POSITIONS_URL = "https://digital.fidelity.com/ftgw/digital/portfolio/positions"


# Named page constants for stack navigation
class _Page:
    PLAYWRIGHT_SETUP = 0
    CREDENTIALS = 1
    TWOFA = 2
    PROGRESS = 3
    RESULT = 4


class FidelityLoginDialog(QDialog):
    """Fidelity connection dialog with CSV import as primary method."""

    login_requested = Signal(str, str, str)  # username, password, totp_secret
    interactive_login_requested = Signal()   # open browser for manual login
    csv_imported = Signal(str)               # path to CSV file
    twofa_submitted = Signal(str)            # code
    skip_requested = Signal()

    def __init__(self, parent=None, show_playwright_setup: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Connect to Fidelity")
        self.setFixedSize(480, 520)
        self.setModal(True)
        self._worker = None  # prevent GC of install worker

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        # Header
        header = QLabel("FIDELITY CONNECTION")
        header.setFont(QFont(Fonts.SANS, Fonts.SIZE_HEADER, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(f"color: {Colors.ACCENT}; letter-spacing: 2px;")
        header.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(header)

        subtitle = QLabel("Import your Fidelity holdings into Meridian")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SMALL}pt;")
        subtitle.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(subtitle)

        # Stacked pages
        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        self._setup_playwright_page()
        self._setup_login_page()
        self._setup_2fa_page()
        self._setup_progress_page()
        self._setup_result_page()

        # Bottom buttons
        btn_layout = QHBoxLayout()
        self._skip_btn = QPushButton("Cancel")
        self._skip_btn.clicked.connect(self._on_skip)
        self._skip_btn.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        btn_layout.addWidget(self._skip_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Start on playwright setup or credentials page
        if show_playwright_setup:
            self._stack.setCurrentIndex(_Page.PLAYWRIGHT_SETUP)
        else:
            self._stack.setCurrentIndex(_Page.CREDENTIALS)

    def _setup_playwright_page(self):
        """Page 0: Playwright Firefox setup (shown when browser is missing)."""
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 16, 0, 0)

        info = QLabel(
            "Fidelity connection requires a browser engine.\n"
            "Playwright Firefox needs to be installed (one-time setup)."
        )
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        page_layout.addWidget(info)

        page_layout.addSpacing(16)

        self._pw_install_btn = QPushButton("Install Firefox Browser")
        self._pw_install_btn.setProperty("primary", True)
        self._pw_install_btn.clicked.connect(self._on_install_playwright)
        page_layout.addWidget(self._pw_install_btn)

        self._pw_progress = QProgressBar()
        self._pw_progress.setRange(0, 0)  # Indeterminate
        self._pw_progress.hide()
        page_layout.addWidget(self._pw_progress)

        self._pw_status = QLabel("")
        self._pw_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pw_status.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: {Fonts.SIZE_SMALL}pt;")
        page_layout.addWidget(self._pw_status)

        page_layout.addStretch()
        self._stack.addWidget(page)

    def _on_install_playwright(self):
        """Run Playwright Firefox install in a background thread."""
        from portopt.utils.playwright_check import install_playwright_firefox

        self._pw_install_btn.setEnabled(False)
        self._pw_progress.show()
        self._pw_status.setText("Installing Firefox browser engine...")

        self._worker = run_in_thread(
            install_playwright_firefox,
            on_result=self._on_playwright_installed,
            on_error=lambda e: self._on_playwright_install_error(str(e)),
        )

    def _on_playwright_installed(self, result):
        success, output = result
        self._pw_progress.hide()
        if success:
            self._pw_status.setText("Firefox installed successfully!")
            self._pw_status.setStyleSheet(f"color: {Colors.PROFIT}; font-size: {Fonts.SIZE_SMALL}pt;")
            from PySide6.QtCore import QTimer
            QTimer.singleShot(1000, lambda: self._stack.setCurrentIndex(_Page.CREDENTIALS))
        else:
            self._pw_install_btn.setEnabled(True)
            self._pw_status.setText(f"Install failed: {output[:100]}")
            self._pw_status.setStyleSheet(f"color: {Colors.LOSS}; font-size: {Fonts.SIZE_SMALL}pt;")

    def _on_playwright_install_error(self, error: str):
        self._pw_progress.hide()
        self._pw_install_btn.setEnabled(True)
        self._pw_status.setText(f"Error: {error[:100]}")
        self._pw_status.setStyleSheet(f"color: {Colors.LOSS}; font-size: {Fonts.SIZE_SMALL}pt;")

    def _setup_login_page(self):
        """Page 1: CSV import (primary) + browser automation (secondary)."""
        page = QWidget()
        form_layout = QVBoxLayout(page)
        form_layout.setContentsMargins(0, 4, 0, 0)

        # ── Primary: CSV Import ──
        csv_info = QLabel(
            "1. Log in to Fidelity in your browser\n"
            "2. Go to Positions and click \"Download\"\n"
            "3. Import the CSV file below"
        )
        csv_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        csv_info.setWordWrap(True)
        csv_info.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SMALL}pt;")
        csv_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form_layout.addWidget(csv_info)

        form_layout.addSpacing(4)

        open_fidelity_btn = QPushButton("Open Fidelity Positions Page")
        open_fidelity_btn.clicked.connect(
            lambda: webbrowser.open(_FIDELITY_POSITIONS_URL)
        )
        form_layout.addWidget(open_fidelity_btn)

        form_layout.addSpacing(4)

        self._import_csv_btn = QPushButton("Import Positions CSV")
        self._import_csv_btn.setProperty("primary", True)
        self._import_csv_btn.setFixedHeight(38)
        self._import_csv_btn.clicked.connect(self._on_import_csv)
        form_layout.addWidget(self._import_csv_btn)

        form_layout.addSpacing(8)

        # ── Separator ──
        sep_layout = QHBoxLayout()
        sep_line_l = QLabel()
        sep_line_l.setFixedHeight(1)
        sep_line_l.setStyleSheet(f"background: {Colors.BORDER};")
        sep_label = QLabel("OR")
        sep_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: {Fonts.SIZE_SMALL}pt;")
        sep_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sep_line_r = QLabel()
        sep_line_r.setFixedHeight(1)
        sep_line_r.setStyleSheet(f"background: {Colors.BORDER};")
        sep_layout.addWidget(sep_line_l, 1)
        sep_layout.addWidget(sep_label)
        sep_layout.addWidget(sep_line_r, 1)
        form_layout.addLayout(sep_layout)

        # ── Secondary: Auto-Connect ──
        group = QGroupBox("Auto-Connect (Experimental)")
        form = QFormLayout(group)
        form.setSpacing(6)

        self._username_input = QLineEdit()
        self._username_input.setPlaceholderText("Fidelity username")
        form.addRow("Username:", self._username_input)

        self._password_input = QLineEdit()
        self._password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._password_input.setPlaceholderText("Fidelity password")
        form.addRow("Password:", self._password_input)

        self._totp_input = QLineEdit()
        self._totp_input.setPlaceholderText("(Optional) TOTP secret key")
        form.addRow("TOTP Secret:", self._totp_input)

        form_layout.addWidget(group)

        self._remember_check = QCheckBox("Save credentials (Windows Credential Manager)")
        self._remember_check.setChecked(True)
        self._remember_check.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SMALL}pt;")
        form_layout.addWidget(self._remember_check)

        self._login_btn = QPushButton("Auto-Connect")
        self._login_btn.clicked.connect(self._on_login)
        form_layout.addWidget(self._login_btn)

        self._login_error = QLabel("")
        self._login_error.setStyleSheet(f"color: {Colors.LOSS}; font-size: {Fonts.SIZE_SMALL}pt;")
        self._login_error.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._login_error.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._login_error.hide()
        form_layout.addWidget(self._login_error)

        self._stack.addWidget(page)

    def _setup_2fa_page(self):
        """Page 2: 2FA code entry."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 16, 0, 0)

        info = QLabel("A verification code was sent to your phone.\nEnter it below to complete login.")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(info)

        self._twofa_input = QLineEdit()
        self._twofa_input.setPlaceholderText("Enter 6-digit code")
        self._twofa_input.setMaxLength(6)
        self._twofa_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._twofa_input.setFont(QFont(Fonts.MONO, 18))
        self._twofa_input.setFixedHeight(48)
        layout.addWidget(self._twofa_input)

        self._twofa_btn = QPushButton("Verify")
        self._twofa_btn.setProperty("primary", True)
        self._twofa_btn.clicked.connect(self._on_2fa_submit)
        layout.addWidget(self._twofa_btn)

        self._twofa_error = QLabel("")
        self._twofa_error.setStyleSheet(f"color: {Colors.LOSS}; font-size: {Fonts.SIZE_SMALL}pt;")
        self._twofa_error.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._twofa_error.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._twofa_error.hide()
        layout.addWidget(self._twofa_error)

        layout.addStretch()
        self._stack.addWidget(page)

    def _setup_progress_page(self):
        """Page 3: Connection progress."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 40, 0, 0)

        self._progress_label = QLabel("Connecting to Fidelity...")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.setFont(QFont(Fonts.SANS, Fonts.SIZE_NORMAL))
        self._progress_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # Indeterminate
        layout.addWidget(self._progress_bar)

        layout.addStretch()
        self._stack.addWidget(page)

    def _setup_result_page(self):
        """Page 4: Success/failure result."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 30, 0, 0)

        self._result_icon = QLabel()
        self._result_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._result_icon.setFont(QFont(Fonts.SANS, 36))
        layout.addWidget(self._result_icon)

        self._result_label = QLabel()
        self._result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._result_label.setFont(QFont(Fonts.SANS, Fonts.SIZE_LARGE))
        self._result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._result_label)

        self._result_detail = QLabel()
        self._result_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._result_detail.setWordWrap(True)
        self._result_detail.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        self._result_detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._result_detail)

        self._done_btn = QPushButton("Done")
        self._done_btn.setProperty("primary", True)
        self._done_btn.clicked.connect(self.accept)
        self._done_btn.hide()
        layout.addWidget(self._done_btn)

        layout.addStretch()
        self._stack.addWidget(page)

    # ── Actions ──────────────────────────────────────────────────────
    def _on_import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Fidelity Positions CSV",
            "", "CSV Files (*.csv);;All Files (*)",
        )
        if path:
            self.csv_imported.emit(path)

    def _on_login(self):
        username = self._username_input.text().strip()
        password = self._password_input.text().strip()
        totp = self._totp_input.text().strip()
        if not username or not password:
            self._login_error.setText("Username and password are required")
            self._login_error.show()
            return
        self._login_error.hide()
        self.show_progress("Logging in to Fidelity...")
        self.login_requested.emit(username, password, totp)

    def _on_browser_login(self):
        self.show_progress("Opening browser — log in to Fidelity there...")
        self.interactive_login_requested.emit()

    def _on_2fa_submit(self):
        code = self._twofa_input.text().strip()
        if not code or len(code) < 6:
            self._twofa_error.setText("Enter the full 6-digit code")
            self._twofa_error.show()
            return
        self._twofa_error.hide()
        self.show_progress("Verifying code...")
        self.twofa_submitted.emit(code)

    def _on_skip(self):
        self.skip_requested.emit()
        self.reject()

    # ── State Changes (called by controller) ─────────────────────────
    def prefill_credentials(self, username: str, password: str, totp: str):
        """Populate input fields with saved credentials."""
        if username:
            self._username_input.setText(username)
        if password:
            self._password_input.setText(password)
        if totp:
            self._totp_input.setText(totp)

    def show_playwright_setup(self):
        """Switch to the Playwright setup page."""
        self._stack.setCurrentIndex(_Page.PLAYWRIGHT_SETUP)

    def show_2fa(self):
        self._stack.setCurrentIndex(_Page.TWOFA)
        self._twofa_input.setFocus()

    def show_progress(self, text: str = "Connecting..."):
        self._progress_label.setText(text)
        self._stack.setCurrentIndex(_Page.PROGRESS)

    def show_success(self, detail: str = ""):
        self._result_icon.setText("OK")
        self._result_icon.setStyleSheet(f"color: {Colors.PROFIT};")
        self._result_label.setText("Connected to Fidelity")
        self._result_label.setStyleSheet(f"color: {Colors.PROFIT};")
        self._result_detail.setText(detail)
        self._done_btn.show()
        self._skip_btn.hide()
        self._stack.setCurrentIndex(_Page.RESULT)

    def show_error(self, message: str):
        self._result_icon.setText("X")
        self._result_icon.setStyleSheet(f"color: {Colors.LOSS};")
        self._result_label.setText("Connection Failed")
        self._result_label.setStyleSheet(f"color: {Colors.LOSS};")
        self._result_detail.setText(message)
        self._done_btn.setText("Retry")
        self._done_btn.show()
        self._done_btn.clicked.disconnect()
        self._done_btn.clicked.connect(lambda: self._stack.setCurrentIndex(_Page.CREDENTIALS))
        self._stack.setCurrentIndex(_Page.RESULT)

    @property
    def remember_credentials(self) -> bool:
        return self._remember_check.isChecked()
