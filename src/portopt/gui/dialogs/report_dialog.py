"""Dialog for configuring and generating portfolio PDF reports."""

from __future__ import annotations

import logging
import os
import subprocess
import sys

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
    QCheckBox, QLabel, QPushButton, QLineEdit, QFileDialog,
    QMessageBox, QProgressBar,
)

from portopt.constants import Colors, Fonts
from portopt.gui.report.chart_capture import (
    capture_panel_chart, png_to_data_uri,
)
from portopt.gui.report.generator import ReportData, render_html, generate_pdf

logger = logging.getLogger(__name__)


class ReportDialog(QDialog):
    """Dialog for configuring and generating a portfolio report."""

    def __init__(
        self,
        parent=None,
        weights: dict[str, float] | None = None,
        metrics: dict[str, float] | None = None,
        panels: dict | None = None,
        copilot_controller=None,
    ):
        super().__init__(parent)
        self._weights = weights
        self._metrics = metrics
        self._panels = panels or {}
        self._copilot = copilot_controller
        self.setWindowTitle("Generate Report")
        self.setMinimumSize(450, 380)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header
        header = QLabel("Generate Portfolio Report")
        header.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-size: 14px; font-weight: bold;"
        )
        layout.addWidget(header)

        # Sections
        sections_group = QGroupBox("Report Sections")
        sections_layout = QVBoxLayout()

        self._cb_summary = QCheckBox("Executive Summary (AI-generated)")
        self._cb_summary.setChecked(True)
        self._cb_summary.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 10px;")
        sections_layout.addWidget(self._cb_summary)

        self._cb_holdings = QCheckBox("Portfolio Holdings & Weights")
        self._cb_holdings.setChecked(True)
        self._cb_holdings.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 10px;")
        sections_layout.addWidget(self._cb_holdings)

        self._cb_metrics = QCheckBox("Performance Metrics")
        self._cb_metrics.setChecked(True)
        self._cb_metrics.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 10px;")
        sections_layout.addWidget(self._cb_metrics)

        self._cb_charts = QCheckBox("Charts (Frontier, Weights, Correlation)")
        self._cb_charts.setChecked(True)
        self._cb_charts.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 10px;")
        sections_layout.addWidget(self._cb_charts)

        sections_group.setLayout(sections_layout)
        layout.addWidget(sections_group)

        # AI note
        ai_note = QLabel(
            "The Executive Summary requires an Anthropic API key configured via AI → API Key."
        )
        ai_note.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 9px;")
        ai_note.setWordWrap(True)
        layout.addWidget(ai_note)

        # Output path
        path_group = QGroupBox("Output")
        path_layout = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Select output PDF file...")
        self._path_edit.setStyleSheet(f"""
            QLineEdit {{
                background: {Colors.BG_INPUT};
                border: 1px solid {Colors.BORDER};
                border-radius: 3px;
                color: {Colors.TEXT_PRIMARY};
                padding: 5px;
                font-size: 10px;
            }}
        """)
        path_layout.addWidget(self._path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse)
        path_layout.addWidget(browse_btn)

        path_group.setLayout(path_layout)
        layout.addWidget(path_group)

        # Progress
        self._progress = QProgressBar()
        self._progress.setFixedHeight(4)
        self._progress.setRange(0, 0)
        self._progress.hide()
        self._progress.setStyleSheet(f"""
            QProgressBar {{ background: {Colors.BG_INPUT}; border: none; }}
            QProgressBar::chunk {{ background: {Colors.ACCENT}; }}
        """)
        layout.addWidget(self._progress)

        layout.addStretch()

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(80)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self._generate_btn = QPushButton("Generate")
        self._generate_btn.setFixedWidth(90)
        self._generate_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: {Colors.BG_PRIMARY};
                font-weight: bold;
                border: none;
                border-radius: 3px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{ background: {Colors.ACCENT_HOVER}; }}
            QPushButton:disabled {{ background: {Colors.BG_INPUT}; color: {Colors.TEXT_MUTED}; }}
        """)
        self._generate_btn.clicked.connect(self._on_generate)
        btn_row.addWidget(self._generate_btn)

        layout.addLayout(btn_row)

    def _browse(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Report PDF", "portfolio_report.pdf",
            "PDF Files (*.pdf);;All Files (*)",
        )
        if path:
            self._path_edit.setText(path)

    def _on_generate(self):
        output_path = self._path_edit.text().strip()
        if not output_path:
            QMessageBox.warning(self, "No Output Path", "Please select an output file.")
            return

        self._generate_btn.setEnabled(False)
        self._progress.show()

        try:
            self._do_generate(output_path)
        except Exception as e:
            logger.error("Report generation failed: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", f"Report generation failed:\n{e}")
        finally:
            self._generate_btn.setEnabled(True)
            self._progress.hide()

    def _do_generate(self, output_path: str):
        """Build report data, render HTML, and generate PDF."""
        sections = []
        if self._cb_summary.isChecked():
            sections.append("executive_summary")
        if self._cb_holdings.isChecked():
            sections.append("holdings")
        if self._cb_metrics.isChecked():
            sections.append("metrics")
        if self._cb_charts.isChecked():
            sections.append("charts")

        # Capture charts
        chart_images: dict[str, str] = {}
        if "charts" in sections:
            chart_panels = {
                "efficient_frontier": self._panels.get("frontier"),
                "weight_allocation": self._panels.get("weights"),
                "correlation_matrix": self._panels.get("correlation"),
            }
            for name, panel in chart_panels.items():
                if panel is not None:
                    png = capture_panel_chart(panel)
                    if png:
                        chart_images[name] = png_to_data_uri(png)

        # AI Executive Summary
        exec_summary = None
        if "executive_summary" in sections:
            exec_summary = self._generate_ai_summary()

        data = ReportData(
            title="MERIDIAN PORTFOLIO REPORT",
            portfolio_name="Optimized Portfolio",
            weights=self._weights,
            metrics=self._metrics,
            executive_summary=exec_summary,
            chart_images=chart_images,
            sections=sections,
        )

        html = render_html(data)
        success = generate_pdf(html, output_path)

        if success:
            reply = QMessageBox.information(
                self, "Report Generated",
                f"Report saved to:\n{output_path}\n\nOpen the file?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._open_file(output_path)
            self.accept()
        else:
            QMessageBox.critical(self, "Error", "Failed to generate PDF.")

    def _generate_ai_summary(self) -> str | None:
        """Generate an AI executive summary using a one-shot Claude call."""
        try:
            from portopt.utils.credentials import ANTHROPIC_API_KEY, get_credential
            import os

            api_key = get_credential(ANTHROPIC_API_KEY) or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                return "(AI summary unavailable — no API key configured)"

            import anthropic

            # Build context for the summary
            context_parts = []
            if self._weights:
                sorted_w = sorted(self._weights.items(), key=lambda x: -x[1])[:10]
                holdings = ", ".join(f"{s} ({w:.1%})" for s, w in sorted_w)
                context_parts.append(f"Holdings: {holdings}")

            if self._metrics:
                key_metrics = ["sharpe_ratio", "annualized_return", "annualized_volatility",
                               "max_drawdown", "cvar_95"]
                for m in key_metrics:
                    if m in self._metrics:
                        context_parts.append(f"{m}: {self._metrics[m]:.4f}")

            if not context_parts:
                return "(No portfolio data available for AI summary)"

            context_text = "\n".join(context_parts)

            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Write a brief (3-4 sentence) executive summary for this "
                        f"portfolio report. Be concise and professional.\n\n{context_text}"
                    ),
                }],
            )

            for block in response.content:
                if block.type == "text":
                    return block.text

            return None

        except Exception as e:
            logger.warning("AI summary generation failed: %s", e)
            return f"(AI summary unavailable: {e})"

    @staticmethod
    def _open_file(path: str):
        """Open the generated file with the system default viewer."""
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.run(["open", path], check=False)
            else:
                subprocess.run(["xdg-open", path], check=False)
        except Exception as e:
            logger.warning("Could not open file: %s", e)
