"""First-run Fidelity login dialog with 2FA support."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QCheckBox, QStackedWidget, QWidget, QProgressBar,
    QGroupBox, QFormLayout,
)

from portopt.constants import Colors, Fonts


class FidelityLoginDialog(QDialog):
    """Multi-step Fidelity login dialog: credentials -> 2FA -> done."""

    login_requested = Signal(str, str, str)  # username, password, totp_secret
    twofa_submitted = Signal(str)            # code
    skip_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connect to Fidelity")
        self.setFixedSize(440, 380)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        # Header
        header = QLabel("FIDELITY CONNECTION")
        header.setFont(QFont(Fonts.SANS, Fonts.SIZE_HEADER, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(f"color: {Colors.ACCENT}; letter-spacing: 2px;")
        layout.addWidget(header)

        subtitle = QLabel("Link your Fidelity account to view holdings")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SMALL}pt;")
        layout.addWidget(subtitle)

        # Stacked pages
        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        self._setup_login_page()
        self._setup_2fa_page()
        self._setup_progress_page()
        self._setup_result_page()

        # Bottom buttons
        btn_layout = QHBoxLayout()
        self._skip_btn = QPushButton("Skip")
        self._skip_btn.clicked.connect(self._on_skip)
        self._skip_btn.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        btn_layout.addWidget(self._skip_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _setup_login_page(self):
        """Page 0: Username/password entry."""
        page = QWidget()
        form_layout = QVBoxLayout(page)
        form_layout.setContentsMargins(0, 8, 0, 0)

        group = QGroupBox("Credentials")
        form = QFormLayout(group)
        form.setSpacing(8)

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

        self._remember_check = QCheckBox("Save credentials securely (Windows Credential Manager)")
        self._remember_check.setChecked(True)
        self._remember_check.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SMALL}pt;")
        form_layout.addWidget(self._remember_check)

        self._login_btn = QPushButton("Connect")
        self._login_btn.setProperty("primary", True)
        self._login_btn.clicked.connect(self._on_login)
        form_layout.addWidget(self._login_btn)

        self._login_error = QLabel("")
        self._login_error.setStyleSheet(f"color: {Colors.LOSS}; font-size: {Fonts.SIZE_SMALL}pt;")
        self._login_error.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._login_error.hide()
        form_layout.addWidget(self._login_error)

        self._stack.addWidget(page)

    def _setup_2fa_page(self):
        """Page 1: 2FA code entry."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 16, 0, 0)

        info = QLabel("A verification code was sent to your phone.\nEnter it below to complete login.")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
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
        self._twofa_error.hide()
        layout.addWidget(self._twofa_error)

        layout.addStretch()
        self._stack.addWidget(page)

    def _setup_progress_page(self):
        """Page 2: Connection progress."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 40, 0, 0)

        self._progress_label = QLabel("Connecting to Fidelity...")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.setFont(QFont(Fonts.SANS, Fonts.SIZE_NORMAL))
        layout.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # Indeterminate
        layout.addWidget(self._progress_bar)

        layout.addStretch()
        self._stack.addWidget(page)

    def _setup_result_page(self):
        """Page 3: Success/failure result."""
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
        layout.addWidget(self._result_label)

        self._result_detail = QLabel()
        self._result_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._result_detail.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        layout.addWidget(self._result_detail)

        self._done_btn = QPushButton("Done")
        self._done_btn.setProperty("primary", True)
        self._done_btn.clicked.connect(self.accept)
        self._done_btn.hide()
        layout.addWidget(self._done_btn)

        layout.addStretch()
        self._stack.addWidget(page)

    # ── Actions ──────────────────────────────────────────────────────
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
    def show_2fa(self):
        self._stack.setCurrentIndex(1)
        self._twofa_input.setFocus()

    def show_progress(self, text: str = "Connecting..."):
        self._progress_label.setText(text)
        self._stack.setCurrentIndex(2)

    def show_success(self, detail: str = ""):
        self._result_icon.setText("OK")
        self._result_icon.setStyleSheet(f"color: {Colors.PROFIT};")
        self._result_label.setText("Connected to Fidelity")
        self._result_label.setStyleSheet(f"color: {Colors.PROFIT};")
        self._result_detail.setText(detail)
        self._done_btn.show()
        self._skip_btn.hide()
        self._stack.setCurrentIndex(3)

    def show_error(self, message: str):
        self._result_icon.setText("X")
        self._result_icon.setStyleSheet(f"color: {Colors.LOSS};")
        self._result_label.setText("Connection Failed")
        self._result_label.setStyleSheet(f"color: {Colors.LOSS};")
        self._result_detail.setText(message)
        self._done_btn.setText("Retry")
        self._done_btn.show()
        self._done_btn.clicked.disconnect()
        self._done_btn.clicked.connect(lambda: self._stack.setCurrentIndex(0))
        self._stack.setCurrentIndex(3)

    @property
    def remember_credentials(self) -> bool:
        return self._remember_check.isChecked()
