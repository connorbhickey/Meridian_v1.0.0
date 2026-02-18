"""Application entry point — launches the Meridian terminal."""

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from portopt.constants import APP_NAME, Fonts


def _setup_file_logging():
    """Configure rotating file log at ~/.meridian/meridian.log."""
    log_dir = Path.home() / ".meridian"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "meridian.log"

    handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.getLogger("portopt").info("Meridian starting — log file: %s", log_file)


def _install_exception_hook():
    """Install a global exception hook that shows a dialog for unhandled exceptions."""
    _original_hook = sys.excepthook

    def _hook(exc_type, exc_value, exc_tb):
        import traceback
        logger = logging.getLogger("portopt")
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.critical("Unhandled exception:\n%s", tb_text)

        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
            app = QApplication.instance()
            if app:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Critical)
                msg.setWindowTitle("Meridian — Unexpected Error")
                msg.setText("An unexpected error occurred.")
                msg.setDetailedText(tb_text)
                msg.setStandardButtons(QMessageBox.StandardButton.Ok)
                msg.exec()
        except Exception:
            pass

        _original_hook(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook


def _make_labels_selectable(root):
    """Walk all QLabel children and make them text-selectable by mouse."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel

    for label in root.findChildren(QLabel):
        label.setTextInteractionFlags(
            label.textInteractionFlags() | Qt.TextInteractionFlag.TextSelectableByMouse
        )


def _run_check():
    """Import all modules and exit — used to validate installs."""
    errors = []
    modules = [
        "portopt.constants",
        "portopt.config",
        "portopt.data.models",
        "portopt.data.cache",
        "portopt.data.manager",
        "portopt.data.providers.yfinance_provider",
        "portopt.data.providers.tiingo_provider",
        "portopt.data.providers.alphavantage_provider",
        "portopt.data.providers.fred_provider",
        "portopt.data.importers.fidelity_csv",
        "portopt.data.importers.schwab_csv",
        "portopt.data.importers.robinhood_csv",
        "portopt.data.importers.generic_csv",
        "portopt.engine.optimization.mean_variance",
        "portopt.engine.optimization.hrp",
        "portopt.engine.optimization.herc",
        "portopt.engine.optimization.black_litterman",
        "portopt.engine.risk",
        "portopt.engine.returns",
        "portopt.engine.metrics",
        "portopt.engine.factors",
        "portopt.engine.regime",
        "portopt.engine.risk_budgeting",
        "portopt.engine.tax_harvest",
        "portopt.engine.strategy_compare",
        "portopt.engine.execution",
        "portopt.engine.network.mst",
        "portopt.backtest.engine",
        "portopt.gui.main_window",
    ]
    import importlib
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as e:
            errors.append(f"  FAIL: {mod} — {e}")

    if errors:
        print(f"Meridian import check FAILED ({len(errors)} errors):")
        for err in errors:
            print(err)
        sys.exit(1)
    else:
        print(f"Meridian import check OK — {len(modules)} modules loaded successfully")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Meridian — Quantitative Portfolio Terminal")
    parser.add_argument("--check", action="store_true", help="Import all modules and exit (validate install)")
    args = parser.parse_args()

    if args.check:
        _run_check()

    _setup_file_logging()
    _install_exception_hook()

    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont

    from portopt.gui.theme import apply_theme
    from portopt.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)

    # Set default font
    font = QFont(Fonts.SANS, Fonts.SIZE_NORMAL)
    app.setFont(font)

    # Apply Meridian deep-space theme
    apply_theme(app)

    # Launch main window
    window = MainWindow()
    window.show()

    # Make all QLabel text selectable/copyable after UI is fully constructed.
    # Initial sweep + periodic re-scan for dynamically created labels.
    _make_labels_selectable(window)
    _label_timer = QTimer(window)
    _label_timer.timeout.connect(lambda: _make_labels_selectable(window))
    _label_timer.start(3000)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
