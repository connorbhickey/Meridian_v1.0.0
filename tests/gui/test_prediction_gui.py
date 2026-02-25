"""GUI tests for the stock prediction panel and controller.

Tests verify:
  - PredictionPanel instantiates and renders correctly
  - Panel controls work (symbol input, horizon, run button)
  - set_result() populates all tabs
  - PredictionController signal chain works end-to-end
  - set_running() / set_status() update the UI
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed")

from PySide6.QtCore import Qt

from portopt.engine.prediction.ensemble import (
    BootstrapResult,
    MethodResult,
    PredictionInterval,
    PredictionResult,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def prediction_result():
    """A synthetic PredictionResult for testing panel display."""
    methods = []
    colors = [
        "#00d4ff", "#00ff88", "#f0b429", "#a855f7", "#ec4899",
        "#06b6d4", "#f97316", "#84cc16", "#6366f1", "#f43f5e",
        "#00d4ff", "#00ff88", "#f0b429", "#a855f7", "#ec4899",
        "#06b6d4", "#f97316", "#84cc16", "#6366f1", "#f43f5e",
        "#00d4ff", "#00ff88", "#f0b429", "#a855f7", "#ec4899",
    ]
    names = [
        "MJD Monte Carlo", "Earnings Valuation", "Analyst Consensus",
        "52wk Mean Reversion", "Dividend Discount", "Regime", "Momentum",
        "Sector Rel Strength", "Vol Regime", "Institutional",
        "EPS Revision", "Size Factor", "Value", "Quality",
        "Investment", "Low Vol", "PEAD", "Seasonality",
        "Options Skew", "Insider", "Revenue Accel", "FCF Yield",
        "Leverage", "Buyback/Dilution", "Kelly Criterion",
    ]
    for i in range(25):
        methods.append(MethodResult(
            name=names[i],
            est=150.0 + i * 2,
            weight=1.0 / 25,
            color=colors[i],
            source=f"Test method {i}",
        ))

    bootstrap = BootstrapResult(
        mean=175.0, std=10.0,
        ci68=(165.0, 185.0), ci90=(155.0, 195.0), ci95=(150.0, 200.0),
    )

    pred_interval = PredictionInterval(
        model_std=10.0, market_std=15.0, total_std=18.0,
        model_pct=40.0, market_pct=60.0,
        model_90=(145.0, 205.0), market_90=(135.0, 215.0),
        total_90=(140.0, 210.0),
        fidelity=0.6,
    )

    return PredictionResult(
        symbol="AAPL",
        is_etf=False,
        vol_scale=1.2,
        ensemble_point=175.0,
        ensemble_return_pct=16.7,
        js_coeff=0.85,
        methods=methods,
        probabilities=[
            {"label": "> Current", "value": 72.0},
            {"label": "> +5%", "value": 65.0},
            {"label": "> +10%", "value": 55.0},
            {"label": "< Current", "value": 28.0},
            {"label": "< -10%", "value": 12.0},
        ],
        histogram=[{"c": 140 + i * 5, "d": 3.0} for i in range(20)],
        bootstrap=bootstrap,
        prediction_interval=pred_interval,
        mc={"est": 170.0, "mean": 172.0, "p5": 120.0, "p10": 130.0,
            "p25": 150.0, "p50": 170.0, "p75": 190.0, "p90": 210.0, "p95": 220.0},
        signals={
            "reg": {"regime": "BULL", "bW": 0.6, "mW": 0.3, "eW": 0.1, "est": 175.0},
            "mom": {"label": "BULLISH", "est": 160.0, "signal": 0.1},
            "srs": {"label": "OUTPERF", "est": 155.0},
            "volR": {"regime": "NORMAL", "est": 150.0},
            "inst": {"label": "NEUTRAL", "est": 152.0, "si": 3.0},
            "epsRev": {"label": "UPGRADING", "est": 170.0, "revision": 5.0},
            "size": {"label": "LARGE-CAP", "est": 155.0},
            "val": {"label": "FAIR VALUE", "est": 155.0, "score": 0},
            "qual": {"label": "HIGH QUALITY", "est": 165.0, "score": 2},
            "inv": {"label": "MODERATE", "est": 152.0},
            "lowV": {"label": "MODERATE", "est": 155.0},
            "pead": {"label": "BEAT", "est": 160.0, "surprise": 5.0},
            "season": {"est": 155.0, "month": 6},
            "opts": {"label": "NEUTRAL", "est": 150.0},
            "insider": {"label": "NEUTRAL", "est": 150.0},
            "revAcc": {"label": "ACCEL", "est": 158.0},
            "fcf": {"label": "HIGH", "est": 165.0, "yield": 8.0},
            "lev": {"label": "FORTRESS", "est": 160.0, "de": 0.3},
            "buyback": {"label": "BUYBACK", "est": 158.0, "change": -2.0},
            "macro": {"label": "SUPPORTIVE", "est": 160.0, "comp": 1},
            "kelly": {"confidence": 72, "label": "MODERATE", "est": 175.0,
                       "cv": 0.06, "kellyFull": 0.45, "kellyHalf": 0.225,
                       "agreement": 0.8, "mean": 175.0, "edge": 0.15},
        },
    )


# ══════════════════════════════════════════════════════════════════
# Panel Smoke Tests
# ══════════════════════════════════════════════════════════════════

class TestPredictionPanelInstantiation:
    """Verify the prediction panel can be created without errors."""

    def test_panel_creates(self, qtbot):
        from portopt.gui.panels.prediction_panel import PredictionPanel
        panel = PredictionPanel()
        qtbot.addWidget(panel)
        assert panel.panel_id == "prediction"
        assert panel.panel_title == "STOCK PREDICTOR"

    def test_panel_has_tabs(self, qtbot):
        from portopt.gui.panels.prediction_panel import PredictionPanel
        panel = PredictionPanel()
        qtbot.addWidget(panel)
        assert panel._tabs.count() == 5

    def test_panel_has_run_button(self, qtbot):
        from portopt.gui.panels.prediction_panel import PredictionPanel
        panel = PredictionPanel()
        qtbot.addWidget(panel)
        assert panel._run_btn is not None
        assert panel._run_btn.isEnabled()

    def test_panel_has_symbol_input(self, qtbot):
        from portopt.gui.panels.prediction_panel import PredictionPanel
        panel = PredictionPanel()
        qtbot.addWidget(panel)
        assert panel._sym_input is not None
        assert panel._sym_input.placeholderText() == "AAPL"

    def test_panel_has_horizon_spin(self, qtbot):
        from portopt.gui.panels.prediction_panel import PredictionPanel
        panel = PredictionPanel()
        qtbot.addWidget(panel)
        assert panel._horizon_spin is not None
        assert panel._horizon_spin.value() > 0


# ══════════════════════════════════════════════════════════════════
# Panel Data Display Tests
# ══════════════════════════════════════════════════════════════════

class TestPredictionPanelDisplay:
    """Verify set_result populates the panel without errors."""

    def test_set_result_populates(self, qtbot, prediction_result):
        from portopt.gui.panels.prediction_panel import PredictionPanel
        panel = PredictionPanel()
        qtbot.addWidget(panel)
        panel.set_result(prediction_result)
        assert panel._result is not None
        assert panel._result.symbol == "AAPL"

    def test_set_running_disables_button(self, qtbot):
        from portopt.gui.panels.prediction_panel import PredictionPanel
        panel = PredictionPanel()
        qtbot.addWidget(panel)
        panel.set_running(True)
        assert not panel._run_btn.isEnabled()
        panel.set_running(False)
        assert panel._run_btn.isEnabled()

    def test_set_status_updates_label(self, qtbot):
        from portopt.gui.panels.prediction_panel import PredictionPanel
        panel = PredictionPanel()
        qtbot.addWidget(panel)
        panel.set_status("Running prediction for AAPL...")
        assert "AAPL" in panel._status_label.text()

    def test_set_result_updates_methods_table(self, qtbot, prediction_result):
        from portopt.gui.panels.prediction_panel import PredictionPanel
        panel = PredictionPanel()
        qtbot.addWidget(panel)
        panel.set_result(prediction_result)
        # Methods table should have 25 rows
        assert panel._methods_table.rowCount() == 25

    def test_set_result_updates_signals_table(self, qtbot, prediction_result):
        from portopt.gui.panels.prediction_panel import PredictionPanel
        panel = PredictionPanel()
        qtbot.addWidget(panel)
        panel.set_result(prediction_result)
        # Signals table should be populated
        assert panel._signals_table.rowCount() > 0

    def test_set_result_with_etf(self, qtbot, prediction_result):
        from portopt.gui.panels.prediction_panel import PredictionPanel
        panel = PredictionPanel()
        qtbot.addWidget(panel)
        prediction_result.is_etf = True
        prediction_result.symbol = "SPY"
        panel.set_result(prediction_result)
        assert panel._result.is_etf is True


# ══════════════════════════════════════════════════════════════════
# Signal Emission Tests
# ══════════════════════════════════════════════════════════════════

class TestPredictionPanelSignals:
    """Verify the panel emits correct signals."""

    def test_run_requested_emitted(self, qtbot):
        from portopt.gui.panels.prediction_panel import PredictionPanel
        panel = PredictionPanel()
        qtbot.addWidget(panel)

        signals = []
        panel.run_requested.connect(lambda s, h: signals.append((s, h)))

        panel._sym_input.setText("MSFT")
        panel._on_run()

        assert len(signals) == 1
        assert signals[0][0] == "MSFT"
        assert signals[0][1] > 0

    def test_empty_symbol_no_signal(self, qtbot):
        from portopt.gui.panels.prediction_panel import PredictionPanel
        panel = PredictionPanel()
        qtbot.addWidget(panel)

        signals = []
        panel.run_requested.connect(lambda s, h: signals.append((s, h)))

        panel._sym_input.setText("")
        panel._on_run()

        assert len(signals) == 0


# ══════════════════════════════════════════════════════════════════
# Controller Tests
# ══════════════════════════════════════════════════════════════════

class TestPredictionControllerSignals:
    """Test the prediction controller signal chain."""

    def test_controller_creates(self, qtbot):
        from portopt.gui.controllers.prediction_controller import PredictionController
        ctrl = PredictionController()
        assert ctrl._running is False
        assert ctrl.last_result is None

    def test_empty_symbol_emits_error(self, qtbot):
        from portopt.gui.controllers.prediction_controller import PredictionController
        ctrl = PredictionController()

        errors = []
        ctrl.error.connect(lambda msg: errors.append(msg))

        ctrl.run_prediction("", 252)
        assert len(errors) == 1
        assert "No symbol" in errors[0]

    def test_duplicate_run_emits_error(self, qtbot):
        from portopt.gui.controllers.prediction_controller import PredictionController
        ctrl = PredictionController()
        ctrl._running = True

        errors = []
        ctrl.error.connect(lambda msg: errors.append(msg))

        ctrl.run_prediction("AAPL", 252)
        assert len(errors) == 1
        assert "already running" in errors[0]

    def test_on_prediction_done_emits_signals(self, qtbot, prediction_result):
        from portopt.gui.controllers.prediction_controller import PredictionController
        ctrl = PredictionController()
        ctrl._running = True

        results = []
        ctrl.prediction_complete.connect(lambda r: results.append(r))

        statuses = []
        ctrl.status_changed.connect(lambda s: statuses.append(s))

        running_states = []
        ctrl.running_changed.connect(lambda r: running_states.append(r))

        ctrl._on_prediction_done(prediction_result)

        assert len(results) == 1
        assert results[0].symbol == "AAPL"
        assert ctrl._running is False
        assert running_states[-1] is False
        assert len(statuses) == 1
        assert "complete" in statuses[0].lower() or "AAPL" in statuses[0]

    def test_on_error_resets_running(self, qtbot):
        from portopt.gui.controllers.prediction_controller import PredictionController
        ctrl = PredictionController()
        ctrl._running = True

        errors = []
        ctrl.error.connect(lambda msg: errors.append(msg))

        running_states = []
        ctrl.running_changed.connect(lambda r: running_states.append(r))

        ctrl._on_error("Something failed")

        assert ctrl._running is False
        assert running_states[-1] is False
        assert len(errors) == 1


# ══════════════════════════════════════════════════════════════════
# Controller → Panel Integration
# ══════════════════════════════════════════════════════════════════

class TestPredictionIntegration:
    """Test the full controller-to-panel signal chain."""

    def test_controller_result_populates_panel(self, qtbot, prediction_result):
        from portopt.gui.panels.prediction_panel import PredictionPanel
        from portopt.gui.controllers.prediction_controller import PredictionController

        panel = PredictionPanel()
        qtbot.addWidget(panel)
        ctrl = PredictionController()

        # Wire signals like MainWindow does
        ctrl.prediction_complete.connect(panel.set_result)
        ctrl.running_changed.connect(panel.set_running)
        ctrl.status_changed.connect(panel.set_status)

        # Simulate a completed prediction
        ctrl._running = True
        ctrl._on_prediction_done(prediction_result)

        assert panel._result is not None
        assert panel._result.symbol == "AAPL"
        assert panel._run_btn.isEnabled()
