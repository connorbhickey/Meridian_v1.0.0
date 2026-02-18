"""Tests for brokerage CSV/OFX importers (Fidelity, Generic, Schwab, Robinhood, OFX)."""

from __future__ import annotations

from pathlib import Path

import pytest

from portopt.data.importers.fidelity_csv import parse_fidelity_csv
from portopt.data.importers.generic_csv import parse_generic_csv
from portopt.data.importers.schwab_csv import parse_schwab_csv
from portopt.data.importers.robinhood_csv import parse_robinhood_csv
from portopt.data.importers.ofx_importer import parse_ofx_file
from portopt.data.models import AssetType, Portfolio


# ── Fidelity CSV ────────────────────────────────────────────────────────


@pytest.fixture
def fidelity_csv(tmp_path):
    """Create a Fidelity-format CSV."""
    content = (
        "Account Number,Account Name,Symbol,Description,Quantity,Last Price,"
        "Last Price Change,Current Value,Today's Gain/Loss Dollar,"
        "Today's Gain/Loss Percent,Total Gain/Loss Dollar,"
        "Total Gain/Loss Percent,Percent Of Account,Cost Basis,"
        "Cost Basis Per Share,Type\n"
        '123456789,INDIVIDUAL,AAPL,APPLE INC,100,$175.50,+$1.20,'
        '"$17,550.00",+$120.00,+0.69%,+$3,050.00,+21.04%,35.00%,'
        '"$14,500.00",$145.00,Cash\n'
        '123456789,INDIVIDUAL,MSFT,MICROSOFT CORP,50,$350.00,-$2.50,'
        '"$17,500.00",-$125.00,-0.71%,+$5,000.00,+40.00%,35.00%,'
        '"$12,500.00",$250.00,Cash\n'
        '123456789,INDIVIDUAL,GOOG,ALPHABET INC CL A,30,$140.00,+$0.50,'
        '"$4,200.00",+$15.00,+0.36%,+$300.00,+7.69%,8.40%,'
        '"$3,900.00",$130.00,Cash\n'
    )
    csv_file = tmp_path / "fidelity_positions.csv"
    csv_file.write_text(content)
    return csv_file


@pytest.fixture
def generic_csv(tmp_path):
    """Create a generic portfolio CSV."""
    content = "Symbol,Quantity,Price,CostBasis\nAAPL,100,175.50,14500\nMSFT,50,350.00,12500\nGOOG,30,140.00,3900\n"
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


# ── Schwab CSV ──────────────────────────────────────────────────────────


def _schwab_multi_account_csv() -> str:
    """Two-account Schwab CSV with header rows, data rows, and summary rows."""
    return (
        '"Positions for account XXXX-1234 as of 02/15/2026"\n'
        "\n"
        "Symbol,Description,Quantity,Price,Price Change %,Price Change $,"
        "Market Value,Day Change %,Day Change $,Cost Basis,"
        "Gain/Loss %,Gain/Loss $,Ratings,Reinvest Dividends?,"
        "Capital Gains?,% Of Account,Security Type\n"
        'AAPL,"Apple Inc",100,"$175.50","+1.20%","+$2.10",'
        '"$17,550.00","+0.69%","+$120.00","$14,500.00",'
        '"+21.04%","+$3,050.00",--,--,--,35.10%,Equity\n'
        'VOO,"Vanguard S&P 500 ETF",50,"$450.00","--","--",'
        '"$22,500.00","--","--","$20,000.00",'
        '"+12.50%","+$2,500.00",--,--,--,45.00%,ETF\n'
        'Account Total,,,,,,"$40,050.00",,,,,,,,100%,\n'
        "\n"
        '"Positions for account YYYY-5678 as of 02/15/2026"\n'
        "\n"
        "Symbol,Description,Quantity,Price,Price Change %,Price Change $,"
        "Market Value,Day Change %,Day Change $,Cost Basis,"
        "Gain/Loss %,Gain/Loss $,Ratings,Reinvest Dividends?,"
        "Capital Gains?,% Of Account,Security Type\n"
        'MSFT,"Microsoft Corp",30,"$410.00","-0.50%","-$2.05",'
        '"$12,300.00","-0.17%","-$61.50","$9,000.00",'
        '"+36.67%","+$3,300.00",--,--,--,60.00%,Equity\n'
        'VFIAX,"Vanguard 500 Index Admiral",20,"$400.00","--","--",'
        '"$8,000.00","--","--","$7,000.00",'
        '"+14.29%","+$1,000.00",--,--,--,40.00%,Mutual Fund\n'
        'Account Total,,,,,,"$20,300.00",,,,,,,,100%,\n'
    )


def _schwab_single_account_csv() -> str:
    """Schwab CSV with no 'Positions for account' header."""
    return (
        "Symbol,Description,Quantity,Price,Price Change %,Price Change $,"
        "Market Value,Day Change %,Day Change $,Cost Basis,"
        "Gain/Loss %,Gain/Loss $,Ratings,Reinvest Dividends?,"
        "Capital Gains?,% Of Account,Security Type\n"
        'TSLA,"Tesla Inc",25,"$250.00","--","--",'
        '"$6,250.00","--","--","$5,000.00",'
        '"+25.00%","+$1,250.00",--,--,--,100.00%,Equity\n'
    )


@pytest.fixture
def schwab_multi_csv(tmp_path):
    csv_file = tmp_path / "schwab_multi.csv"
    csv_file.write_text(_schwab_multi_account_csv())
    return csv_file


@pytest.fixture
def schwab_single_csv(tmp_path):
    csv_file = tmp_path / "schwab_single.csv"
    csv_file.write_text(_schwab_single_account_csv())
    return csv_file


class TestSchwabCSV:
    """Tests for portopt.data.importers.schwab_csv.parse_schwab_csv."""

    # ── Basic multi-account parsing ──────────────────────────────────

    def test_multi_account_holdings_count(self, schwab_multi_csv):
        portfolio = parse_schwab_csv(schwab_multi_csv)
        assert isinstance(portfolio, Portfolio)
        # AAPL, VOO, MSFT, VFIAX = 4 holdings (Account Total rows skipped)
        assert len(portfolio.holdings) == 4

    def test_multi_account_symbols(self, schwab_multi_csv):
        portfolio = parse_schwab_csv(schwab_multi_csv)
        symbols = [h.asset.symbol for h in portfolio.holdings]
        assert set(symbols) == {"AAPL", "VOO", "MSFT", "VFIAX"}

    def test_multi_account_ids(self, schwab_multi_csv):
        portfolio = parse_schwab_csv(schwab_multi_csv)
        accounts_on_holdings = {h.account for h in portfolio.holdings}
        assert "XXXX-1234" in accounts_on_holdings
        assert "YYYY-5678" in accounts_on_holdings

    def test_multi_account_summaries(self, schwab_multi_csv):
        portfolio = parse_schwab_csv(schwab_multi_csv)
        acct_ids = {a.account_id for a in portfolio.accounts}
        assert "XXXX-1234" in acct_ids
        assert "YYYY-5678" in acct_ids

    def test_portfolio_name(self, schwab_multi_csv):
        portfolio = parse_schwab_csv(schwab_multi_csv)
        assert "Schwab" in portfolio.name

    # ── Security type mapping ────────────────────────────────────────

    @pytest.mark.parametrize(
        "security_type_str, expected_asset_type",
        [
            ("Equity", AssetType.STOCK),
            ("ETF", AssetType.ETF),
            ("Mutual Fund", AssetType.MUTUAL_FUND),
            ("Fixed Income", AssetType.BOND),
            ("Cash & Cash Equivalents", AssetType.MONEY_MARKET),
        ],
    )
    def test_security_type_mapping(self, tmp_path, security_type_str, expected_asset_type):
        content = (
            "Symbol,Description,Quantity,Price,Price Change %,Price Change $,"
            "Market Value,Day Change %,Day Change $,Cost Basis,"
            "Gain/Loss %,Gain/Loss $,Ratings,Reinvest Dividends?,"
            "Capital Gains?,% Of Account,Security Type\n"
            f'TEST,"Test Asset",10,"$100.00","--","--",'
            f'"$1,000.00","--","--","$900.00",'
            f'"--","--",--,--,--,100%,{security_type_str}\n'
        )
        csv_file = tmp_path / "schwab_type_test.csv"
        csv_file.write_text(content)
        portfolio = parse_schwab_csv(csv_file)
        assert len(portfolio.holdings) == 1
        assert portfolio.holdings[0].asset.asset_type == expected_asset_type

    # ── Summary/total row skipping ───────────────────────────────────

    def test_account_total_rows_skipped(self, schwab_multi_csv):
        portfolio = parse_schwab_csv(schwab_multi_csv)
        symbols = [h.asset.symbol for h in portfolio.holdings]
        assert "ACCOUNT TOTAL" not in symbols
        assert "Account Total" not in symbols

    def test_grand_total_row_skipped(self, tmp_path):
        content = (
            "Symbol,Description,Quantity,Price,Price Change %,Price Change $,"
            "Market Value,Day Change %,Day Change $,Cost Basis,"
            "Gain/Loss %,Gain/Loss $,Ratings,Reinvest Dividends?,"
            "Capital Gains?,% Of Account,Security Type\n"
            'AAPL,"Apple Inc",10,"$175.00","--","--",'
            '"$1,750.00","--","--","$1,500.00",'
            '"--","--",--,--,--,50%,Equity\n'
            'Grand Total,,,,,,"$3,500.00",,,,,,,,100%,\n'
        )
        csv_file = tmp_path / "schwab_grand_total.csv"
        csv_file.write_text(content)
        portfolio = parse_schwab_csv(csv_file)
        assert len(portfolio.holdings) == 1
        assert portfolio.holdings[0].asset.symbol == "AAPL"

    # ── Numeric parsing ──────────────────────────────────────────────

    def test_dollar_comma_parsing(self, schwab_multi_csv):
        portfolio = parse_schwab_csv(schwab_multi_csv)
        aapl = next(h for h in portfolio.holdings if h.asset.symbol == "AAPL")
        assert aapl.current_price == pytest.approx(175.50)
        assert aapl.cost_basis == pytest.approx(14_500.00)
        assert aapl.quantity == pytest.approx(100.0)

    def test_dash_and_na_parsed_as_zero(self, tmp_path):
        content = (
            "Symbol,Description,Quantity,Price,Price Change %,Price Change $,"
            "Market Value,Day Change %,Day Change $,Cost Basis,"
            "Gain/Loss %,Gain/Loss $,Ratings,Reinvest Dividends?,"
            "Capital Gains?,% Of Account,Security Type\n"
            'AAPL,"Apple",10,"$100.00","--","N/A",'
            '"$1,000.00","n/a","--","$900.00",'
            '"--","--",--,--,--,100%,Equity\n'
        )
        csv_file = tmp_path / "schwab_na.csv"
        csv_file.write_text(content)
        portfolio = parse_schwab_csv(csv_file)
        assert len(portfolio.holdings) == 1

    # ── Single account (no header) ───────────────────────────────────

    def test_single_account_no_header(self, schwab_single_csv):
        portfolio = parse_schwab_csv(schwab_single_csv)
        assert len(portfolio.holdings) == 1
        assert portfolio.holdings[0].asset.symbol == "TSLA"
        assert portfolio.holdings[0].quantity == pytest.approx(25.0)
        assert portfolio.holdings[0].current_price == pytest.approx(250.0)

    def test_single_account_empty_account_id(self, schwab_single_csv):
        """Without an account header, the account field on holdings should be empty."""
        portfolio = parse_schwab_csv(schwab_single_csv)
        assert portfolio.holdings[0].account == ""

    # ── Zero quantity rows skipped ───────────────────────────────────

    def test_zero_quantity_row_skipped(self, tmp_path):
        content = (
            "Symbol,Description,Quantity,Price,Price Change %,Price Change $,"
            "Market Value,Day Change %,Day Change $,Cost Basis,"
            "Gain/Loss %,Gain/Loss $,Ratings,Reinvest Dividends?,"
            "Capital Gains?,% Of Account,Security Type\n"
            'AAPL,"Apple Inc",10,"$175.00","--","--",'
            '"$1,750.00","--","--","$1,500.00",'
            '"--","--",--,--,--,50%,Equity\n'
            'DEAD,"Dead Stock",0,"$0.00","--","--",'
            '"$0.00","--","--","$0.00",'
            '"--","--",--,--,--,0%,Equity\n'
        )
        csv_file = tmp_path / "schwab_zero.csv"
        csv_file.write_text(content)
        portfolio = parse_schwab_csv(csv_file)
        assert len(portfolio.holdings) == 1
        assert portfolio.holdings[0].asset.symbol == "AAPL"

    # ── FileNotFoundError ────────────────────────────────────────────

    def test_file_not_found(self, tmp_path):
        missing = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            parse_schwab_csv(missing)

    # ── Empty file ───────────────────────────────────────────────────

    def test_empty_file_returns_empty_portfolio(self, tmp_path):
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        portfolio = parse_schwab_csv(csv_file)
        assert len(portfolio.holdings) == 0

    def test_header_only_returns_empty_portfolio(self, tmp_path):
        content = (
            "Symbol,Description,Quantity,Price,Price Change %,Price Change $,"
            "Market Value,Day Change %,Day Change $,Cost Basis,"
            "Gain/Loss %,Gain/Loss $,Ratings,Reinvest Dividends?,"
            "Capital Gains?,% Of Account,Security Type\n"
        )
        csv_file = tmp_path / "schwab_header_only.csv"
        csv_file.write_text(content)
        portfolio = parse_schwab_csv(csv_file)
        assert len(portfolio.holdings) == 0

    # ── Weight computation ───────────────────────────────────────────

    def test_weights_sum_to_one(self, schwab_multi_csv):
        portfolio = parse_schwab_csv(schwab_multi_csv)
        total_weight = sum(h.weight for h in portfolio.holdings)
        assert total_weight == pytest.approx(1.0, abs=1e-6)

    # ── Market value ─────────────────────────────────────────────────

    def test_total_value_positive(self, schwab_multi_csv):
        portfolio = parse_schwab_csv(schwab_multi_csv)
        assert portfolio.total_value > 0

    def test_individual_market_values(self, schwab_multi_csv):
        portfolio = parse_schwab_csv(schwab_multi_csv)
        aapl = next(h for h in portfolio.holdings if h.asset.symbol == "AAPL")
        assert aapl.market_value == pytest.approx(175.50 * 100)

    # ── Description preserved ────────────────────────────────────────

    def test_description_on_asset(self, schwab_multi_csv):
        portfolio = parse_schwab_csv(schwab_multi_csv)
        aapl = next(h for h in portfolio.holdings if h.asset.symbol == "AAPL")
        assert "Apple" in aapl.asset.name


# ── Robinhood CSV ───────────────────────────────────────────────────────


def _robinhood_basic_csv() -> str:
    return (
        "Symbol,Name,Type,Quantity,Average Cost,Current Price,"
        "Total Return,Equity,Percent Change\n"
        "AAPL,Apple Inc,stock,50,$145.00,$175.50,+$1525.00,$8775.00,+21.04%\n"
        "VOO,Vanguard S&P 500 ETF,etf,20,$380.00,$450.00,+$1400.00,$9000.00,+18.42%\n"
        "BTC,Bitcoin,crypto,0.5,$30000.00,$42000.00,+$6000.00,$21000.00,+40.00%\n"
    )


@pytest.fixture
def robinhood_csv(tmp_path):
    csv_file = tmp_path / "robinhood_positions.csv"
    csv_file.write_text(_robinhood_basic_csv())
    return csv_file


class TestRobinhoodCSV:
    """Tests for portopt.data.importers.robinhood_csv.parse_robinhood_csv."""

    # ── Basic parsing ────────────────────────────────────────────────

    def test_basic_holdings_count(self, robinhood_csv):
        portfolio = parse_robinhood_csv(robinhood_csv)
        assert isinstance(portfolio, Portfolio)
        assert len(portfolio.holdings) == 3

    def test_basic_symbols(self, robinhood_csv):
        portfolio = parse_robinhood_csv(robinhood_csv)
        symbols = [h.asset.symbol for h in portfolio.holdings]
        assert set(symbols) == {"AAPL", "VOO", "BTC"}

    def test_portfolio_name(self, robinhood_csv):
        portfolio = parse_robinhood_csv(robinhood_csv)
        assert "Robinhood" in portfolio.name

    # ── Cost basis calculation ───────────────────────────────────────

    def test_cost_basis_from_avg_cost_times_quantity(self, robinhood_csv):
        portfolio = parse_robinhood_csv(robinhood_csv)
        aapl = next(h for h in portfolio.holdings if h.asset.symbol == "AAPL")
        # avg_cost=145, quantity=50 -> cost_basis=7250
        assert aapl.cost_basis == pytest.approx(145.0 * 50.0)

    def test_cost_basis_crypto(self, robinhood_csv):
        portfolio = parse_robinhood_csv(robinhood_csv)
        btc = next(h for h in portfolio.holdings if h.asset.symbol == "BTC")
        # avg_cost=30000, quantity=0.5 -> cost_basis=15000
        assert btc.cost_basis == pytest.approx(30_000.0 * 0.5)

    # ── Type mapping ─────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "type_str, expected",
        [
            ("stock", AssetType.STOCK),
            ("etf", AssetType.ETF),
            ("crypto", AssetType.CRYPTO),
        ],
    )
    def test_type_mapping_basic(self, tmp_path, type_str, expected):
        content = (
            "Symbol,Name,Type,Quantity,Average Cost,Current Price,"
            "Total Return,Equity,Percent Change\n"
            f"TEST,Test Asset,{type_str},10,$100.00,$110.00,+$100.00,$1100.00,+10.00%\n"
        )
        csv_file = tmp_path / "rh_type_test.csv"
        csv_file.write_text(content)
        portfolio = parse_robinhood_csv(csv_file)
        assert portfolio.holdings[0].asset.asset_type == expected

    def test_adr_maps_to_stock(self, tmp_path):
        content = (
            "Symbol,Name,Type,Quantity,Average Cost,Current Price,"
            "Total Return,Equity,Percent Change\n"
            "TSM,Taiwan Semi ADR,adr,10,$100.00,$110.00,+$100.00,$1100.00,+10.00%\n"
        )
        csv_file = tmp_path / "rh_adr.csv"
        csv_file.write_text(content)
        portfolio = parse_robinhood_csv(csv_file)
        assert portfolio.holdings[0].asset.asset_type == AssetType.STOCK

    # ── Fallback price from Equity column ────────────────────────────

    def test_fallback_price_from_equity(self, tmp_path):
        content = (
            "Symbol,Name,Type,Quantity,Average Cost,Current Price,"
            "Total Return,Equity,Percent Change\n"
            "AAPL,Apple Inc,stock,10,$145.00,$0.00,--,$1755.00,--\n"
        )
        csv_file = tmp_path / "rh_fallback.csv"
        csv_file.write_text(content)
        portfolio = parse_robinhood_csv(csv_file)
        assert len(portfolio.holdings) == 1
        # price should be equity / quantity = 1755 / 10 = 175.5
        assert portfolio.holdings[0].current_price == pytest.approx(175.5)

    def test_no_fallback_when_price_present(self, robinhood_csv):
        portfolio = parse_robinhood_csv(robinhood_csv)
        aapl = next(h for h in portfolio.holdings if h.asset.symbol == "AAPL")
        assert aapl.current_price == pytest.approx(175.50)

    # ── Zero quantity rows skipped ───────────────────────────────────

    def test_zero_quantity_skipped(self, tmp_path):
        content = (
            "Symbol,Name,Type,Quantity,Average Cost,Current Price,"
            "Total Return,Equity,Percent Change\n"
            "AAPL,Apple Inc,stock,10,$145.00,$175.50,+$305.00,$1755.00,+21.04%\n"
            "DEAD,Dead Stock,stock,0,$50.00,$0.00,--,$0.00,--\n"
        )
        csv_file = tmp_path / "rh_zero_qty.csv"
        csv_file.write_text(content)
        portfolio = parse_robinhood_csv(csv_file)
        assert len(portfolio.holdings) == 1
        assert portfolio.holdings[0].asset.symbol == "AAPL"

    # ── All holdings have account="Robinhood" ────────────────────────

    def test_all_holdings_robinhood_account(self, robinhood_csv):
        portfolio = parse_robinhood_csv(robinhood_csv)
        for h in portfolio.holdings:
            assert h.account == "Robinhood"

    # ── FileNotFoundError ────────────────────────────────────────────

    def test_file_not_found(self, tmp_path):
        missing = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            parse_robinhood_csv(missing)

    # ── Weights sum to 1 ─────────────────────────────────────────────

    def test_weights_sum_to_one(self, robinhood_csv):
        portfolio = parse_robinhood_csv(robinhood_csv)
        total = sum(h.weight for h in portfolio.holdings)
        assert total == pytest.approx(1.0, abs=1e-6)

    # ── Quantity and price correctness ───────────────────────────────

    def test_quantity_parsed(self, robinhood_csv):
        portfolio = parse_robinhood_csv(robinhood_csv)
        aapl = next(h for h in portfolio.holdings if h.asset.symbol == "AAPL")
        assert aapl.quantity == pytest.approx(50.0)

    def test_fractional_quantity(self, robinhood_csv):
        portfolio = parse_robinhood_csv(robinhood_csv)
        btc = next(h for h in portfolio.holdings if h.asset.symbol == "BTC")
        assert btc.quantity == pytest.approx(0.5)

    # ── Empty file ───────────────────────────────────────────────────

    def test_empty_file_returns_empty_portfolio(self, tmp_path):
        csv_file = tmp_path / "rh_empty.csv"
        csv_file.write_text(
            "Symbol,Name,Type,Quantity,Average Cost,Current Price,"
            "Total Return,Equity,Percent Change\n"
        )
        portfolio = parse_robinhood_csv(csv_file)
        assert len(portfolio.holdings) == 0

    # ── Description preserved ────────────────────────────────────────

    def test_name_on_asset(self, robinhood_csv):
        portfolio = parse_robinhood_csv(robinhood_csv)
        aapl = next(h for h in portfolio.holdings if h.asset.symbol == "AAPL")
        assert "Apple" in aapl.asset.name


# ── OFX / QFX ───────────────────────────────────────────────────────────


def _basic_ofx_content() -> str:
    """Minimal valid OFX file with SECLIST, INVPOSLIST, and INVACCTFROM."""
    return (
        "OFXHEADER:100\n"
        "DATA:OFXSGML\n"
        "VERSION:102\n"
        "SECURITY:NONE\n"
        "ENCODING:USASCII\n"
        "\n"
        "<OFX>\n"
        "<SIGNONMSGSRSV1>\n"
        "<SONRS>\n"
        "<STATUS><CODE>0<SEVERITY>INFO</STATUS>\n"
        "<DTSERVER>20260215120000\n"
        "<LANGUAGE>ENG\n"
        "</SONRS>\n"
        "</SIGNONMSGSRSV1>\n"
        "\n"
        "<INVSTMTMSGSRSV1>\n"
        "<INVSTMTTRNRS>\n"
        "<TRNUID>1234\n"
        "<STATUS><CODE>0<SEVERITY>INFO</STATUS>\n"
        "<INVSTMTRS>\n"
        "<DTASOF>20260215120000\n"
        "<CURDEF>USD\n"
        "<INVACCTFROM>\n"
        "<BROKERID>schwab.com\n"
        "<ACCTID>12345678\n"
        "</INVACCTFROM>\n"
        "\n"
        "<INVPOSLIST>\n"
        "<POSSTOCK>\n"
        "<INVPOS>\n"
        "<SECID>\n"
        "<UNIQUEID>037833100\n"
        "<UNIQUEIDTYPE>CUSIP\n"
        "</SECID>\n"
        "<UNITS>100\n"
        "<UNITPRICE>175.50\n"
        "<MKTVAL>17550.00\n"
        "<DTPRICEASOF>20260215120000\n"
        "</INVPOS>\n"
        "</POSSTOCK>\n"
        "\n"
        "<POSMF>\n"
        "<INVPOS>\n"
        "<SECID>\n"
        "<UNIQUEID>922908363\n"
        "<UNIQUEIDTYPE>CUSIP\n"
        "</SECID>\n"
        "<UNITS>50\n"
        "<UNITPRICE>400.00\n"
        "<MKTVAL>20000.00\n"
        "<DTPRICEASOF>20260215120000\n"
        "</INVPOS>\n"
        "</POSMF>\n"
        "\n"
        "<POSDEBT>\n"
        "<INVPOS>\n"
        "<SECID>\n"
        "<UNIQUEID>912828YK0\n"
        "<UNIQUEIDTYPE>CUSIP\n"
        "</SECID>\n"
        "<UNITS>10\n"
        "<UNITPRICE>98.50\n"
        "<MKTVAL>985.00\n"
        "<DTPRICEASOF>20260215120000\n"
        "</INVPOS>\n"
        "</POSDEBT>\n"
        "</INVPOSLIST>\n"
        "\n"
        "</INVSTMTRS>\n"
        "</INVSTMTTRNRS>\n"
        "</INVSTMTMSGSRSV1>\n"
        "\n"
        "<SECLISTMSGSRSV1>\n"
        "<SECLIST>\n"
        "<STOCKINFO>\n"
        "<SECINFO>\n"
        "<SECID>\n"
        "<UNIQUEID>037833100\n"
        "<UNIQUEIDTYPE>CUSIP\n"
        "</SECID>\n"
        "<SECNAME>Apple Inc\n"
        "<TICKER>AAPL\n"
        "</SECINFO>\n"
        "</STOCKINFO>\n"
        "\n"
        "<MFINFO>\n"
        "<SECINFO>\n"
        "<SECID>\n"
        "<UNIQUEID>922908363\n"
        "<UNIQUEIDTYPE>CUSIP\n"
        "</SECID>\n"
        "<SECNAME>Vanguard 500 Index Admiral\n"
        "<TICKER>VFIAX\n"
        "</SECINFO>\n"
        "</MFINFO>\n"
        "\n"
        "<DEBTINFO>\n"
        "<SECINFO>\n"
        "<SECID>\n"
        "<UNIQUEID>912828YK0\n"
        "<UNIQUEIDTYPE>CUSIP\n"
        "</SECID>\n"
        "<SECNAME>US Treasury Bond 2.5%\n"
        "<TICKER>USTB\n"
        "</SECINFO>\n"
        "</DEBTINFO>\n"
        "</SECLIST>\n"
        "</SECLISTMSGSRSV1>\n"
        "</OFX>\n"
    )


@pytest.fixture
def ofx_file(tmp_path):
    f = tmp_path / "positions.ofx"
    f.write_text(_basic_ofx_content())
    return f


@pytest.fixture
def qfx_file(tmp_path):
    f = tmp_path / "positions.qfx"
    f.write_text(_basic_ofx_content())
    return f


class TestOFXImporter:
    """Tests for portopt.data.importers.ofx_importer.parse_ofx_file."""

    # ── Basic OFX parsing ────────────────────────────────────────────

    def test_basic_holdings_count(self, ofx_file):
        portfolio = parse_ofx_file(ofx_file)
        assert isinstance(portfolio, Portfolio)
        assert len(portfolio.holdings) == 3

    def test_basic_symbols(self, ofx_file):
        portfolio = parse_ofx_file(ofx_file)
        symbols = {h.asset.symbol for h in portfolio.holdings}
        assert symbols == {"AAPL", "VFIAX", "USTB"}

    def test_portfolio_name_ofx(self, ofx_file):
        portfolio = parse_ofx_file(ofx_file)
        assert "OFX" in portfolio.name

    # ── Multiple position types ──────────────────────────────────────

    def test_posstock_type(self, ofx_file):
        portfolio = parse_ofx_file(ofx_file)
        aapl = next(h for h in portfolio.holdings if h.asset.symbol == "AAPL")
        assert aapl.asset.asset_type == AssetType.STOCK

    def test_posmf_type(self, ofx_file):
        portfolio = parse_ofx_file(ofx_file)
        vfiax = next(h for h in portfolio.holdings if h.asset.symbol == "VFIAX")
        assert vfiax.asset.asset_type == AssetType.MUTUAL_FUND

    def test_posdebt_type(self, ofx_file):
        portfolio = parse_ofx_file(ofx_file)
        bond = next(h for h in portfolio.holdings if h.asset.symbol == "USTB")
        assert bond.asset.asset_type == AssetType.BOND

    # ── CUSIP-to-ticker mapping via SECLIST ──────────────────────────

    def test_cusip_mapped_to_ticker(self, ofx_file):
        portfolio = parse_ofx_file(ofx_file)
        # CUSIP 037833100 should resolve to AAPL
        aapl = next(h for h in portfolio.holdings if h.asset.symbol == "AAPL")
        assert aapl.asset.name == "Apple Inc"

    def test_cusip_mapped_to_mf_ticker(self, ofx_file):
        portfolio = parse_ofx_file(ofx_file)
        vfiax = next(h for h in portfolio.holdings if h.asset.symbol == "VFIAX")
        assert "Vanguard" in vfiax.asset.name

    # ── Fallback to CUSIP as symbol when no SECLIST entry ────────────

    def test_cusip_fallback_when_no_seclist(self, tmp_path):
        """Position with a CUSIP not in the SECLIST should use CUSIP as symbol."""
        content = (
            "OFXHEADER:100\n"
            "DATA:OFXSGML\n"
            "<OFX>\n"
            "<INVSTMTMSGSRSV1>\n"
            "<INVSTMTTRNRS>\n"
            "<INVSTMTRS>\n"
            "<INVPOSLIST>\n"
            "<POSSTOCK>\n"
            "<INVPOS>\n"
            "<SECID>\n"
            "<UNIQUEID>999999999\n"
            "<UNIQUEIDTYPE>CUSIP\n"
            "</SECID>\n"
            "<UNITS>10\n"
            "<UNITPRICE>50.00\n"
            "<MKTVAL>500.00\n"
            "<DTPRICEASOF>20260215\n"
            "</INVPOS>\n"
            "</POSSTOCK>\n"
            "</INVPOSLIST>\n"
            "</INVSTMTRS>\n"
            "</INVSTMTTRNRS>\n"
            "</INVSTMTMSGSRSV1>\n"
            "</OFX>\n"
        )
        f = tmp_path / "no_seclist.ofx"
        f.write_text(content)
        portfolio = parse_ofx_file(f)
        assert len(portfolio.holdings) == 1
        assert portfolio.holdings[0].asset.symbol == "999999999"
        assert portfolio.holdings[0].asset.asset_type == AssetType.STOCK

    # ── Price derivation from MKTVAL/UNITS when UNITPRICE missing ────

    def test_price_derived_from_mktval_over_units(self, tmp_path):
        content = (
            "OFXHEADER:100\n"
            "DATA:OFXSGML\n"
            "<OFX>\n"
            "<INVSTMTMSGSRSV1>\n"
            "<INVSTMTTRNRS>\n"
            "<INVSTMTRS>\n"
            "<INVPOSLIST>\n"
            "<POSSTOCK>\n"
            "<INVPOS>\n"
            "<SECID>\n"
            "<UNIQUEID>037833100\n"
            "<UNIQUEIDTYPE>CUSIP\n"
            "</SECID>\n"
            "<UNITS>20\n"
            "<MKTVAL>3510.00\n"
            "<DTPRICEASOF>20260215\n"
            "</INVPOS>\n"
            "</POSSTOCK>\n"
            "</INVPOSLIST>\n"
            "</INVSTMTRS>\n"
            "</INVSTMTTRNRS>\n"
            "</INVSTMTMSGSRSV1>\n"
            "<SECLISTMSGSRSV1>\n"
            "<SECLIST>\n"
            "<STOCKINFO>\n"
            "<SECINFO>\n"
            "<SECID>\n"
            "<UNIQUEID>037833100\n"
            "<UNIQUEIDTYPE>CUSIP\n"
            "</SECID>\n"
            "<SECNAME>Apple Inc\n"
            "<TICKER>AAPL\n"
            "</SECINFO>\n"
            "</STOCKINFO>\n"
            "</SECLIST>\n"
            "</SECLISTMSGSRSV1>\n"
            "</OFX>\n"
        )
        f = tmp_path / "no_unitprice.ofx"
        f.write_text(content)
        portfolio = parse_ofx_file(f)
        assert len(portfolio.holdings) == 1
        # price = 3510 / 20 = 175.5
        assert portfolio.holdings[0].current_price == pytest.approx(175.5)

    # ── Datetime parsing from DTPRICEASOF ────────────────────────────

    def test_datetime_from_dtpriceasof_full(self, ofx_file):
        portfolio = parse_ofx_file(ofx_file)
        assert portfolio.last_updated is not None
        assert portfolio.last_updated.year == 2026
        assert portfolio.last_updated.month == 2
        assert portfolio.last_updated.day == 15

    def test_datetime_from_dtpriceasof_date_only(self, tmp_path):
        content = (
            "OFXHEADER:100\n"
            "DATA:OFXSGML\n"
            "<OFX>\n"
            "<INVSTMTMSGSRSV1>\n"
            "<INVSTMTTRNRS>\n"
            "<INVSTMTRS>\n"
            "<INVPOSLIST>\n"
            "<POSSTOCK>\n"
            "<INVPOS>\n"
            "<SECID>\n"
            "<UNIQUEID>037833100\n"
            "<UNIQUEIDTYPE>CUSIP\n"
            "</SECID>\n"
            "<UNITS>10\n"
            "<UNITPRICE>175.50\n"
            "<MKTVAL>1755.00\n"
            "<DTPRICEASOF>20260301\n"
            "</INVPOS>\n"
            "</POSSTOCK>\n"
            "</INVPOSLIST>\n"
            "</INVSTMTRS>\n"
            "</INVSTMTTRNRS>\n"
            "</INVSTMTMSGSRSV1>\n"
            "</OFX>\n"
        )
        f = tmp_path / "date_only.ofx"
        f.write_text(content)
        portfolio = parse_ofx_file(f)
        assert portfolio.last_updated.year == 2026
        assert portfolio.last_updated.month == 3
        assert portfolio.last_updated.day == 1

    # ── FileNotFoundError ────────────────────────────────────────────

    def test_file_not_found(self, tmp_path):
        missing = tmp_path / "nonexistent.ofx"
        with pytest.raises(FileNotFoundError):
            parse_ofx_file(missing)

    # ── QFX extension in portfolio name ──────────────────────────────

    def test_qfx_extension_in_portfolio_name(self, qfx_file):
        portfolio = parse_ofx_file(qfx_file)
        assert "QFX" in portfolio.name

    def test_ofx_extension_in_portfolio_name(self, ofx_file):
        portfolio = parse_ofx_file(ofx_file)
        assert "OFX" in portfolio.name

    # ── Account info from INVACCTFROM ────────────────────────────────

    def test_account_extracted(self, ofx_file):
        portfolio = parse_ofx_file(ofx_file)
        assert len(portfolio.accounts) >= 1
        acct = portfolio.accounts[0]
        assert acct.account_id == "12345678"
        assert "schwab.com" in acct.account_name

    # ── Units and prices ─────────────────────────────────────────────

    def test_units_parsed(self, ofx_file):
        portfolio = parse_ofx_file(ofx_file)
        aapl = next(h for h in portfolio.holdings if h.asset.symbol == "AAPL")
        assert aapl.quantity == pytest.approx(100.0)

    def test_unit_price_parsed(self, ofx_file):
        portfolio = parse_ofx_file(ofx_file)
        aapl = next(h for h in portfolio.holdings if h.asset.symbol == "AAPL")
        assert aapl.current_price == pytest.approx(175.50)

    def test_market_value_computed(self, ofx_file):
        portfolio = parse_ofx_file(ofx_file)
        aapl = next(h for h in portfolio.holdings if h.asset.symbol == "AAPL")
        assert aapl.market_value == pytest.approx(17550.00)

    # ── Weights sum to 1 ─────────────────────────────────────────────

    def test_weights_sum_to_one(self, ofx_file):
        portfolio = parse_ofx_file(ofx_file)
        total = sum(h.weight for h in portfolio.holdings)
        assert total == pytest.approx(1.0, abs=1e-6)

    # ── Total value ──────────────────────────────────────────────────

    def test_total_value(self, ofx_file):
        portfolio = parse_ofx_file(ofx_file)
        # 17550 + 20000 + 985 = 38535
        assert portfolio.total_value == pytest.approx(38535.00)

    # ── Zero-value positions skipped ─────────────────────────────────

    def test_zero_units_zero_mktval_skipped(self, tmp_path):
        content = (
            "OFXHEADER:100\n"
            "DATA:OFXSGML\n"
            "<OFX>\n"
            "<INVSTMTMSGSRSV1>\n"
            "<INVSTMTTRNRS>\n"
            "<INVSTMTRS>\n"
            "<INVPOSLIST>\n"
            "<POSSTOCK>\n"
            "<INVPOS>\n"
            "<SECID>\n"
            "<UNIQUEID>037833100\n"
            "<UNIQUEIDTYPE>CUSIP\n"
            "</SECID>\n"
            "<UNITS>100\n"
            "<UNITPRICE>175.50\n"
            "<MKTVAL>17550.00\n"
            "<DTPRICEASOF>20260215\n"
            "</INVPOS>\n"
            "</POSSTOCK>\n"
            "<POSSTOCK>\n"
            "<INVPOS>\n"
            "<SECID>\n"
            "<UNIQUEID>000000000\n"
            "<UNIQUEIDTYPE>CUSIP\n"
            "</SECID>\n"
            "<UNITS>0\n"
            "<UNITPRICE>0\n"
            "<MKTVAL>0\n"
            "<DTPRICEASOF>20260215\n"
            "</INVPOS>\n"
            "</POSSTOCK>\n"
            "</INVPOSLIST>\n"
            "</INVSTMTRS>\n"
            "</INVSTMTTRNRS>\n"
            "</INVSTMTMSGSRSV1>\n"
            "<SECLISTMSGSRSV1>\n"
            "<SECLIST>\n"
            "<STOCKINFO>\n"
            "<SECINFO>\n"
            "<SECID>\n"
            "<UNIQUEID>037833100\n"
            "<UNIQUEIDTYPE>CUSIP\n"
            "</SECID>\n"
            "<SECNAME>Apple Inc\n"
            "<TICKER>AAPL\n"
            "</SECINFO>\n"
            "</STOCKINFO>\n"
            "</SECLIST>\n"
            "</SECLISTMSGSRSV1>\n"
            "</OFX>\n"
        )
        f = tmp_path / "zero_pos.ofx"
        f.write_text(content)
        portfolio = parse_ofx_file(f)
        assert len(portfolio.holdings) == 1
        assert portfolio.holdings[0].asset.symbol == "AAPL"

    # ── Empty OFX returns empty portfolio ────────────────────────────

    def test_empty_ofx_returns_empty(self, tmp_path):
        content = (
            "OFXHEADER:100\n"
            "DATA:OFXSGML\n"
            "<OFX>\n"
            "<INVSTMTMSGSRSV1>\n"
            "<INVSTMTTRNRS>\n"
            "<INVSTMTRS>\n"
            "<INVPOSLIST>\n"
            "</INVPOSLIST>\n"
            "</INVSTMTRS>\n"
            "</INVSTMTTRNRS>\n"
            "</INVSTMTMSGSRSV1>\n"
            "</OFX>\n"
        )
        f = tmp_path / "empty.ofx"
        f.write_text(content)
        portfolio = parse_ofx_file(f)
        assert len(portfolio.holdings) == 0

    # ── Security name preserved ──────────────────────────────────────

    def test_security_name_from_seclist(self, ofx_file):
        portfolio = parse_ofx_file(ofx_file)
        bond = next(h for h in portfolio.holdings if h.asset.symbol == "USTB")
        assert "Treasury" in bond.asset.name

    # ── Account summary populated ────────────────────────────────────

    def test_account_summary_totals(self, ofx_file):
        portfolio = parse_ofx_file(ofx_file)
        assert len(portfolio.accounts) >= 1
        acct = portfolio.accounts[0]
        assert acct.total_value > 0
        assert acct.holdings_count == 3

    # ── No DTPRICEASOF falls back to now ─────────────────────────────

    def test_no_dtpriceasof_fallback(self, tmp_path):
        content = (
            "OFXHEADER:100\n"
            "DATA:OFXSGML\n"
            "<OFX>\n"
            "<INVSTMTMSGSRSV1>\n"
            "<INVSTMTTRNRS>\n"
            "<INVSTMTRS>\n"
            "<INVPOSLIST>\n"
            "<POSSTOCK>\n"
            "<INVPOS>\n"
            "<SECID>\n"
            "<UNIQUEID>037833100\n"
            "<UNIQUEIDTYPE>CUSIP\n"
            "</SECID>\n"
            "<UNITS>10\n"
            "<UNITPRICE>100.00\n"
            "<MKTVAL>1000.00\n"
            "</INVPOS>\n"
            "</POSSTOCK>\n"
            "</INVPOSLIST>\n"
            "</INVSTMTRS>\n"
            "</INVSTMTTRNRS>\n"
            "</INVSTMTMSGSRSV1>\n"
            "</OFX>\n"
        )
        f = tmp_path / "no_date.ofx"
        f.write_text(content)
        portfolio = parse_ofx_file(f)
        # Should fall back to datetime.now() so year should be current
        assert portfolio.last_updated is not None
        assert portfolio.last_updated.year >= 2025
