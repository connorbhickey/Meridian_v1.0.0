"""Tests for CSV importers (Fidelity + generic)."""

import tempfile
from pathlib import Path

import pytest

from portopt.data.importers.fidelity_csv import parse_fidelity_csv
from portopt.data.importers.generic_csv import parse_generic_csv
from portopt.data.models import Portfolio


@pytest.fixture
def fidelity_csv(tmp_path):
    """Create a Fidelity-format CSV."""
    content = """Account Number,Account Name,Symbol,Description,Quantity,Last Price,Last Price Change,Current Value,Today's Gain/Loss Dollar,Today's Gain/Loss Percent,Total Gain/Loss Dollar,Total Gain/Loss Percent,Percent Of Account,Cost Basis,Cost Basis Per Share,Type
123456789,INDIVIDUAL,AAPL,APPLE INC,100,$175.50,+$1.20,"$17,550.00",+$120.00,+0.69%,+$3,050.00,+21.04%,35.00%,"$14,500.00",$145.00,Cash
123456789,INDIVIDUAL,MSFT,MICROSOFT CORP,50,$350.00,-$2.50,"$17,500.00",-$125.00,-0.71%,+$5,000.00,+40.00%,35.00%,"$12,500.00",$250.00,Cash
123456789,INDIVIDUAL,GOOG,ALPHABET INC CL A,30,$140.00,+$0.50,"$4,200.00",+$15.00,+0.36%,+$300.00,+7.69%,8.40%,"$3,900.00",$130.00,Cash
"""
    csv_file = tmp_path / "fidelity_positions.csv"
    csv_file.write_text(content)
    return csv_file


@pytest.fixture
def generic_csv(tmp_path):
    """Create a generic portfolio CSV."""
    content = """Symbol,Quantity,Price,CostBasis
AAPL,100,175.50,14500
MSFT,50,350.00,12500
GOOG,30,140.00,3900
"""
    csv_file = tmp_path / "portfolio.csv"
    csv_file.write_text(content)
    return csv_file


class TestFidelityCSV:
    def test_parse_basic(self, fidelity_csv):
        portfolio = parse_fidelity_csv(fidelity_csv)
        assert isinstance(portfolio, Portfolio)
        assert len(portfolio.holdings) == 3

    def test_symbols(self, fidelity_csv):
        portfolio = parse_fidelity_csv(fidelity_csv)
        symbols = portfolio.symbols
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "GOOG" in symbols

    def test_total_value(self, fidelity_csv):
        portfolio = parse_fidelity_csv(fidelity_csv)
        assert portfolio.total_value > 0


class TestGenericCSV:
    def test_parse_basic(self, generic_csv):
        portfolio = parse_generic_csv(generic_csv)
        assert isinstance(portfolio, Portfolio)
        assert len(portfolio.holdings) == 3

    def test_symbols(self, generic_csv):
        portfolio = parse_generic_csv(generic_csv)
        assert set(portfolio.symbols) == {"AAPL", "MSFT", "GOOG"}
