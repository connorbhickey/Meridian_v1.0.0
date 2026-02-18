"""Tests for multi-account household-level portfolio optimization with tax-aware asset location."""

import pytest

from portopt.engine.account_optimizer import (
    AccountConstraints,
    AccountType,
    AssetTaxProfile,
    HouseholdOptResult,
    compute_tax_efficiency_score,
    estimate_tax_savings,
    optimal_asset_location,
    optimize_household,
)
from portopt.engine.constraints import PortfolioConstraints


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def bond_profile():
    """High-dividend, non-qualified: typical taxable bond fund."""
    return AssetTaxProfile(
        symbol="BND",
        dividend_yield=0.04,
        qualified_dividend_pct=0.0,
        turnover_rate=0.30,
        tax_cost_ratio=0.0,
    )


@pytest.fixture()
def growth_profile():
    """Zero-dividend growth stock: very tax-efficient."""
    return AssetTaxProfile(
        symbol="AMZN",
        dividend_yield=0.0,
        qualified_dividend_pct=1.0,
        turnover_rate=0.0,
        tax_cost_ratio=0.0,
    )


@pytest.fixture()
def reit_profile():
    """REIT: high non-qualified dividend, very tax-inefficient."""
    return AssetTaxProfile(
        symbol="VNQ",
        dividend_yield=0.05,
        qualified_dividend_pct=0.0,
        turnover_rate=0.10,
        tax_cost_ratio=0.0,
    )


@pytest.fixture()
def qualified_dividend_profile():
    """Blue-chip with moderate qualified dividends."""
    return AssetTaxProfile(
        symbol="JNJ",
        dividend_yield=0.03,
        qualified_dividend_pct=1.0,
        turnover_rate=0.0,
        tax_cost_ratio=0.0,
    )


@pytest.fixture()
def default_constraints():
    return PortfolioConstraints()


@pytest.fixture()
def taxable_account(default_constraints):
    return AccountConstraints(
        account_id="taxable_1",
        account_type=AccountType.TAXABLE,
        total_value=500_000.0,
        constraints=default_constraints,
    )


@pytest.fixture()
def ira_account(default_constraints):
    return AccountConstraints(
        account_id="ira_1",
        account_type=AccountType.TAX_DEFERRED,
        total_value=300_000.0,
        constraints=default_constraints,
    )


@pytest.fixture()
def roth_account(default_constraints):
    return AccountConstraints(
        account_id="roth_1",
        account_type=AccountType.TAX_EXEMPT,
        total_value=200_000.0,
        constraints=default_constraints,
    )


@pytest.fixture()
def two_account_household(taxable_account, ira_account):
    return [taxable_account, ira_account]


@pytest.fixture()
def three_account_household(taxable_account, ira_account, roth_account):
    return [taxable_account, ira_account, roth_account]


@pytest.fixture()
def sample_tax_profiles(bond_profile, growth_profile, reit_profile, qualified_dividend_profile):
    return {
        "BND": bond_profile,
        "AMZN": growth_profile,
        "VNQ": reit_profile,
        "JNJ": qualified_dividend_profile,
    }


# ── TestComputeTaxEfficiencyScore ───────────────────────────────────


class TestComputeTaxEfficiencyScore:
    def test_bond_scores_low_in_taxable(self, bond_profile):
        score = compute_tax_efficiency_score(bond_profile, AccountType.TAXABLE)
        assert 0.0 <= score <= 1.0
        assert score < 0.5, "Bond fund should score low in taxable account"

    def test_bond_scores_high_in_tax_deferred(self, bond_profile):
        score = compute_tax_efficiency_score(bond_profile, AccountType.TAX_DEFERRED)
        assert 0.0 <= score <= 1.0
        assert score > 0.7, "Bond fund should score high in tax-deferred account"

    def test_bond_scores_high_in_tax_exempt(self, bond_profile):
        score = compute_tax_efficiency_score(bond_profile, AccountType.TAX_EXEMPT)
        assert 0.0 <= score <= 1.0
        assert score > 0.7, "Bond fund should score high in tax-exempt account"

    def test_growth_stock_scores_high_in_taxable(self, growth_profile):
        score = compute_tax_efficiency_score(growth_profile, AccountType.TAXABLE)
        assert 0.0 <= score <= 1.0
        assert score > 0.8, "Growth stock should score high in taxable account"

    def test_growth_stock_scores_baseline_in_deferred(self, growth_profile):
        score = compute_tax_efficiency_score(growth_profile, AccountType.TAX_DEFERRED)
        assert 0.0 <= score <= 1.0
        # With zero inefficiency the deferred score should be the baseline 0.3
        assert score == pytest.approx(0.3, abs=0.05)

    def test_reit_scores_very_low_in_taxable(self, reit_profile):
        score = compute_tax_efficiency_score(reit_profile, AccountType.TAXABLE)
        assert 0.0 <= score <= 1.0
        assert score < 0.3, "REIT should score very low in taxable account"

    def test_reit_scores_high_in_tax_deferred(self, reit_profile):
        score = compute_tax_efficiency_score(reit_profile, AccountType.TAX_DEFERRED)
        assert score > 0.8, "REIT should score high in tax-deferred account"

    def test_reit_scores_high_in_tax_exempt(self, reit_profile):
        score = compute_tax_efficiency_score(reit_profile, AccountType.TAX_EXEMPT)
        assert score > 0.8, "REIT should score high in tax-exempt account"

    def test_tax_exempt_shelters_inefficient_assets(self, bond_profile):
        taxable = compute_tax_efficiency_score(bond_profile, AccountType.TAXABLE)
        exempt = compute_tax_efficiency_score(bond_profile, AccountType.TAX_EXEMPT)
        assert exempt > taxable, "Tax-exempt should shelter bonds better than taxable"

    def test_tax_deferred_shelters_inefficient_assets(self, reit_profile):
        taxable = compute_tax_efficiency_score(reit_profile, AccountType.TAXABLE)
        deferred = compute_tax_efficiency_score(reit_profile, AccountType.TAX_DEFERRED)
        assert deferred > taxable, "Tax-deferred should shelter REITs better than taxable"

    def test_zero_profile_scores_reasonably(self):
        zero_profile = AssetTaxProfile(symbol="XYZ")
        for acct_type in AccountType:
            score = compute_tax_efficiency_score(zero_profile, acct_type)
            assert 0.0 <= score <= 1.0

    def test_zero_profile_high_in_taxable(self):
        zero_profile = AssetTaxProfile(symbol="XYZ")
        score = compute_tax_efficiency_score(zero_profile, AccountType.TAXABLE)
        assert score >= 0.9, "Zero-drag asset should be very efficient in taxable"

    def test_score_always_in_range(self):
        """Parametric check over extreme profiles."""
        extreme_profiles = [
            AssetTaxProfile(symbol="A", dividend_yield=0.20, qualified_dividend_pct=0.0, turnover_rate=1.0),
            AssetTaxProfile(symbol="B", dividend_yield=0.0, qualified_dividend_pct=1.0, turnover_rate=0.0),
            AssetTaxProfile(symbol="C", dividend_yield=0.10, qualified_dividend_pct=0.5, turnover_rate=0.5),
            AssetTaxProfile(symbol="D", tax_cost_ratio=0.05),
            AssetTaxProfile(symbol="E", tax_cost_ratio=0.0, dividend_yield=0.0, turnover_rate=0.0),
        ]
        for profile in extreme_profiles:
            for acct_type in AccountType:
                score = compute_tax_efficiency_score(profile, acct_type)
                assert 0.0 <= score <= 1.0, (
                    f"Out of range for {profile.symbol} in {acct_type}: {score}"
                )

    def test_qualified_dividend_better_than_nonqualified_in_taxable(self):
        qualified = AssetTaxProfile(symbol="Q", dividend_yield=0.03, qualified_dividend_pct=1.0)
        nonqualified = AssetTaxProfile(symbol="NQ", dividend_yield=0.03, qualified_dividend_pct=0.0)
        q_score = compute_tax_efficiency_score(qualified, AccountType.TAXABLE)
        nq_score = compute_tax_efficiency_score(nonqualified, AccountType.TAXABLE)
        assert q_score > nq_score, "Qualified dividends should be more tax-efficient in taxable"

    def test_high_turnover_penalized_in_taxable(self):
        low_turnover = AssetTaxProfile(symbol="LT", turnover_rate=0.05)
        high_turnover = AssetTaxProfile(symbol="HT", turnover_rate=0.80)
        lt_score = compute_tax_efficiency_score(low_turnover, AccountType.TAXABLE)
        ht_score = compute_tax_efficiency_score(high_turnover, AccountType.TAXABLE)
        assert lt_score > ht_score, "High turnover should be penalized in taxable"

    def test_explicit_tax_cost_ratio_used(self):
        profile = AssetTaxProfile(symbol="TC", tax_cost_ratio=0.03)
        score = compute_tax_efficiency_score(profile, AccountType.TAXABLE)
        assert score < 0.5, "High explicit tax cost ratio should lower taxable score"


# ── TestOptimalAssetLocation ────────────────────────────────────────


class TestOptimalAssetLocation:
    def test_empty_accounts_returns_empty(self, sample_tax_profiles):
        result = optimal_asset_location(["BND", "AMZN"], sample_tax_profiles, [])
        assert result == {}

    def test_all_symbols_get_placed(self, sample_tax_profiles, three_account_household):
        symbols = list(sample_tax_profiles.keys())
        result = optimal_asset_location(symbols, sample_tax_profiles, three_account_household)
        for sym in symbols:
            assert sym in result, f"{sym} was not placed in any account"

    def test_bonds_route_to_tax_advantaged(self, sample_tax_profiles, three_account_household):
        symbols = list(sample_tax_profiles.keys())
        result = optimal_asset_location(symbols, sample_tax_profiles, three_account_household)
        bond_acct = result["BND"]
        # BND should be in the IRA or Roth, not taxable
        account_map = {a.account_id: a for a in three_account_household}
        acct_type = account_map[bond_acct].account_type
        assert acct_type in (AccountType.TAX_DEFERRED, AccountType.TAX_EXEMPT), (
            f"BND placed in {acct_type}, expected tax-advantaged"
        )

    def test_reits_route_to_tax_advantaged(self, sample_tax_profiles, three_account_household):
        symbols = list(sample_tax_profiles.keys())
        result = optimal_asset_location(symbols, sample_tax_profiles, three_account_household)
        reit_acct = result["VNQ"]
        account_map = {a.account_id: a for a in three_account_household}
        acct_type = account_map[reit_acct].account_type
        assert acct_type in (AccountType.TAX_DEFERRED, AccountType.TAX_EXEMPT), (
            f"VNQ placed in {acct_type}, expected tax-advantaged"
        )

    def test_growth_stock_routes_to_taxable(self, sample_tax_profiles, three_account_household):
        symbols = list(sample_tax_profiles.keys())
        result = optimal_asset_location(symbols, sample_tax_profiles, three_account_household)
        growth_acct = result["AMZN"]
        account_map = {a.account_id: a for a in three_account_household}
        assert account_map[growth_acct].account_type == AccountType.TAXABLE, (
            "Growth stock should route to taxable account"
        )

    def test_excluded_symbols_respected(self, default_constraints):
        """Account that excludes BND should not receive BND."""
        ira = AccountConstraints(
            account_id="ira_excl",
            account_type=AccountType.TAX_DEFERRED,
            total_value=300_000.0,
            constraints=default_constraints,
            excluded_symbols={"BND"},
        )
        taxable = AccountConstraints(
            account_id="taxable_1",
            account_type=AccountType.TAXABLE,
            total_value=500_000.0,
            constraints=default_constraints,
        )
        profiles = {
            "BND": AssetTaxProfile(symbol="BND", dividend_yield=0.04, qualified_dividend_pct=0.0),
        }
        result = optimal_asset_location(["BND"], profiles, [ira, taxable])
        assert result.get("BND") != "ira_excl", "BND should not be placed in account that excludes it"

    def test_allowed_only_symbols_respected(self, default_constraints):
        """Account with allowed_symbols set should only receive those symbols."""
        ira = AccountConstraints(
            account_id="ira_allow",
            account_type=AccountType.TAX_DEFERRED,
            total_value=300_000.0,
            constraints=default_constraints,
            allowed_symbols={"AGG"},
        )
        taxable = AccountConstraints(
            account_id="taxable_1",
            account_type=AccountType.TAXABLE,
            total_value=500_000.0,
            constraints=default_constraints,
        )
        profiles = {
            "BND": AssetTaxProfile(symbol="BND", dividend_yield=0.04, qualified_dividend_pct=0.0),
        }
        result = optimal_asset_location(["BND"], profiles, [ira, taxable])
        # BND is not in ira's allowed_symbols, so it must go to taxable
        assert result["BND"] == "taxable_1"

    def test_single_account_places_everything(self, taxable_account, sample_tax_profiles):
        symbols = list(sample_tax_profiles.keys())
        result = optimal_asset_location(symbols, sample_tax_profiles, [taxable_account])
        for sym in symbols:
            assert result[sym] == "taxable_1"

    def test_missing_profile_uses_default(self, three_account_household):
        """Symbol not in tax_profiles should still get placed using a default profile."""
        result = optimal_asset_location(["UNKNOWN"], {}, three_account_household)
        assert "UNKNOWN" in result

    def test_zero_value_accounts_return_empty(self, default_constraints):
        zero_acct = AccountConstraints(
            account_id="zero",
            account_type=AccountType.TAXABLE,
            total_value=0.0,
            constraints=default_constraints,
        )
        result = optimal_asset_location(["AAPL"], {}, [zero_acct])
        assert result == {}


# ── TestOptimizeHousehold ───────────────────────────────────────────


class TestOptimizeHousehold:
    def test_basic_two_account(self, two_account_household):
        target = {"AAPL": 0.40, "MSFT": 0.30, "BND": 0.30}
        result = optimize_household(target, two_account_household)
        assert isinstance(result, HouseholdOptResult)
        assert set(result.account_weights.keys()) == {"taxable_1", "ira_1"}

    def test_household_weights_approximate_targets(self, two_account_household):
        target = {"AAPL": 0.40, "MSFT": 0.30, "BND": 0.30}
        result = optimize_household(target, two_account_household)
        for sym, tw in target.items():
            assert result.household_weights[sym] == pytest.approx(tw, abs=0.05), (
                f"Household weight for {sym} should be close to {tw}"
            )

    def test_per_account_weights_sum_to_one(self, two_account_household):
        target = {"AAPL": 0.40, "MSFT": 0.30, "BND": 0.30}
        result = optimize_household(target, two_account_household)
        for acct_id, weights in result.account_weights.items():
            total = sum(weights.values())
            assert total == pytest.approx(1.0, abs=0.02), (
                f"Account {acct_id} weights sum to {total}, expected ~1.0"
            )

    def test_without_tax_profiles_proportional(self, two_account_household):
        target = {"AAPL": 0.50, "MSFT": 0.50}
        result = optimize_household(target, two_account_household)
        # Without tax profiles, no asset location optimization
        assert result.asset_locations == {}
        assert result.tax_savings_estimate == 0.0

    def test_with_tax_profiles_adjusts_location(self, three_account_household, sample_tax_profiles):
        target = {"BND": 0.30, "AMZN": 0.40, "VNQ": 0.15, "JNJ": 0.15}
        result = optimize_household(target, three_account_household, tax_profiles=sample_tax_profiles)
        # Should have asset locations when tax profiles provided
        assert len(result.asset_locations) > 0
        # Tax savings should be computed
        assert result.tax_savings_estimate >= 0.0

    def test_zero_total_value_returns_empty(self, default_constraints):
        zero_accounts = [
            AccountConstraints(
                account_id="z1",
                account_type=AccountType.TAXABLE,
                total_value=0.0,
                constraints=default_constraints,
            ),
            AccountConstraints(
                account_id="z2",
                account_type=AccountType.TAX_DEFERRED,
                total_value=0.0,
                constraints=default_constraints,
            ),
        ]
        target = {"AAPL": 0.50, "MSFT": 0.50}
        result = optimize_household(target, zero_accounts)
        for sym in target:
            assert result.household_weights[sym] == 0.0

    def test_single_account_household(self, taxable_account):
        target = {"AAPL": 0.60, "MSFT": 0.40}
        result = optimize_household(target, [taxable_account])
        assert len(result.account_weights) == 1
        weights = result.account_weights["taxable_1"]
        assert weights["AAPL"] == pytest.approx(0.60, abs=0.05)
        assert weights["MSFT"] == pytest.approx(0.40, abs=0.05)

    def test_per_account_results_have_method(self, two_account_household):
        target = {"AAPL": 0.50, "MSFT": 0.50}
        result = optimize_household(target, two_account_household)
        for acct_id, opt_result in result.per_account_results.items():
            assert opt_result.method == "household_proportional"

    def test_per_account_results_have_account_type_metadata(
        self, three_account_household, sample_tax_profiles,
    ):
        target = {"BND": 0.30, "AMZN": 0.40, "VNQ": 0.15, "JNJ": 0.15}
        result = optimize_household(target, three_account_household, tax_profiles=sample_tax_profiles)
        account_types = {a.account_id: a.account_type.value for a in three_account_household}
        for acct_id, opt_result in result.per_account_results.items():
            assert opt_result.metadata.get("account_type") == account_types[acct_id]

    def test_household_weights_sum_to_approximately_one(self, three_account_household, sample_tax_profiles):
        target = {"BND": 0.25, "AMZN": 0.25, "VNQ": 0.25, "JNJ": 0.25}
        result = optimize_household(target, three_account_household, tax_profiles=sample_tax_profiles)
        total = sum(result.household_weights.values())
        assert total == pytest.approx(1.0, abs=0.05)

    def test_excluded_symbols_produce_zero_weight(self, default_constraints):
        """If a symbol is excluded from all accounts, it gets zero weight everywhere."""
        accounts = [
            AccountConstraints(
                account_id="only_one",
                account_type=AccountType.TAXABLE,
                total_value=100_000.0,
                constraints=default_constraints,
                excluded_symbols={"BANNED"},
            ),
        ]
        target = {"AAPL": 0.50, "BANNED": 0.50}
        result = optimize_household(target, accounts)
        assert result.account_weights["only_one"]["BANNED"] == 0.0

    def test_empty_target_weights(self, two_account_household):
        result = optimize_household({}, two_account_household)
        assert isinstance(result, HouseholdOptResult)
        assert result.household_weights == {}

    def test_many_symbols(self, two_account_household):
        """Household optimization with many symbols should still work."""
        symbols = [f"SYM{i}" for i in range(20)]
        target = {sym: 1.0 / 20 for sym in symbols}
        result = optimize_household(target, two_account_household)
        assert len(result.household_weights) == 20


# ── TestEstimateTaxSavings ──────────────────────────────────────────


class TestEstimateTaxSavings:
    def test_no_tax_profiles_zero_savings(self, two_account_household):
        account_weights = {
            "taxable_1": {"AAPL": 0.50, "MSFT": 0.50},
            "ira_1": {"AAPL": 0.50, "MSFT": 0.50},
        }
        savings = estimate_tax_savings(account_weights, two_account_household, {})
        assert savings == 0.0

    def test_optimal_location_nonnegative(self, three_account_household, sample_tax_profiles):
        target = {"BND": 0.30, "AMZN": 0.40, "VNQ": 0.15, "JNJ": 0.15}
        result = optimize_household(target, three_account_household, tax_profiles=sample_tax_profiles)
        savings = estimate_tax_savings(
            result.account_weights, three_account_household, sample_tax_profiles,
        )
        assert savings >= 0.0, "Tax savings should be non-negative"

    def test_bonds_in_deferred_saves_vs_taxable(self, default_constraints):
        """Moving bonds from taxable to tax-deferred should produce positive savings."""
        taxable = AccountConstraints(
            account_id="tax",
            account_type=AccountType.TAXABLE,
            total_value=500_000.0,
            constraints=default_constraints,
        )
        ira = AccountConstraints(
            account_id="ira",
            account_type=AccountType.TAX_DEFERRED,
            total_value=500_000.0,
            constraints=default_constraints,
        )
        bond_profile = AssetTaxProfile(
            symbol="BND", dividend_yield=0.04, qualified_dividend_pct=0.0, turnover_rate=0.2,
        )
        growth_profile = AssetTaxProfile(
            symbol="AMZN", dividend_yield=0.0, qualified_dividend_pct=1.0, turnover_rate=0.0,
        )
        profiles = {"BND": bond_profile, "AMZN": growth_profile}

        # Optimal: bonds in IRA, growth in taxable
        optimal_weights = {
            "tax": {"BND": 0.0, "AMZN": 1.0},
            "ira": {"BND": 1.0, "AMZN": 0.0},
        }
        savings = estimate_tax_savings(optimal_weights, [taxable, ira], profiles)
        assert savings > 0.0, "Moving bonds to IRA should produce positive tax savings"

    def test_zero_value_accounts_zero_savings(self, default_constraints):
        zero_accounts = [
            AccountConstraints(
                account_id="z1",
                account_type=AccountType.TAXABLE,
                total_value=0.0,
                constraints=default_constraints,
            ),
        ]
        savings = estimate_tax_savings(
            {"z1": {"AAPL": 1.0}},
            zero_accounts,
            {"AAPL": AssetTaxProfile(symbol="AAPL", dividend_yield=0.02)},
        )
        assert savings == 0.0

    def test_custom_tax_rates(self, two_account_household):
        profiles = {
            "BND": AssetTaxProfile(symbol="BND", dividend_yield=0.04, qualified_dividend_pct=0.0),
        }
        custom_rates = {
            "ordinary_income": 0.50,
            "qualified_dividend": 0.20,
            "long_term_capital_gain": 0.20,
            "short_term_capital_gain": 0.50,
        }
        account_weights = {
            "taxable_1": {"BND": 0.0},
            "ira_1": {"BND": 1.0},
        }
        savings = estimate_tax_savings(
            account_weights, two_account_household, profiles, custom_rates,
        )
        assert savings >= 0.0

    def test_all_in_tax_exempt_no_savings_vs_self(self, default_constraints):
        """If everything is already in a single tax-exempt account, savings should be zero."""
        roth = AccountConstraints(
            account_id="roth",
            account_type=AccountType.TAX_EXEMPT,
            total_value=100_000.0,
            constraints=default_constraints,
        )
        profiles = {
            "BND": AssetTaxProfile(symbol="BND", dividend_yield=0.04, qualified_dividend_pct=0.0),
        }
        account_weights = {"roth": {"BND": 1.0}}
        savings = estimate_tax_savings(account_weights, [roth], profiles)
        assert savings == 0.0


# ── TestDataclasses ─────────────────────────────────────────────────


class TestDataclasses:
    def test_account_type_enum_values(self):
        assert AccountType.TAXABLE.value == "taxable"
        assert AccountType.TAX_DEFERRED.value == "tax_deferred"
        assert AccountType.TAX_EXEMPT.value == "tax_exempt"

    def test_account_constraints_creation(self, default_constraints):
        acct = AccountConstraints(
            account_id="test",
            account_type=AccountType.TAXABLE,
            total_value=100_000.0,
            constraints=default_constraints,
        )
        assert acct.account_id == "test"
        assert acct.account_type == AccountType.TAXABLE
        assert acct.total_value == 100_000.0
        assert acct.allowed_symbols is None
        assert acct.excluded_symbols == set()
        assert acct.max_positions is None

    def test_account_constraints_with_optional_fields(self, default_constraints):
        acct = AccountConstraints(
            account_id="ira",
            account_type=AccountType.TAX_DEFERRED,
            total_value=200_000.0,
            constraints=default_constraints,
            allowed_symbols={"AAPL", "MSFT"},
            excluded_symbols={"GME"},
            max_positions=10,
        )
        assert acct.allowed_symbols == {"AAPL", "MSFT"}
        assert "GME" in acct.excluded_symbols
        assert acct.max_positions == 10

    def test_asset_tax_profile_defaults(self):
        profile = AssetTaxProfile(symbol="TEST")
        assert profile.symbol == "TEST"
        assert profile.dividend_yield == 0.0
        assert profile.qualified_dividend_pct == 1.0
        assert profile.turnover_rate == 0.0
        assert profile.tax_cost_ratio == 0.0

    def test_asset_tax_profile_custom_values(self):
        profile = AssetTaxProfile(
            symbol="BND",
            dividend_yield=0.04,
            qualified_dividend_pct=0.0,
            turnover_rate=0.30,
            tax_cost_ratio=0.02,
        )
        assert profile.dividend_yield == 0.04
        assert profile.qualified_dividend_pct == 0.0
        assert profile.turnover_rate == 0.30
        assert profile.tax_cost_ratio == 0.02

    def test_household_opt_result_creation(self):
        from portopt.data.models import OptimizationResult

        result = HouseholdOptResult(
            account_weights={"acct1": {"AAPL": 0.5, "MSFT": 0.5}},
            household_weights={"AAPL": 0.5, "MSFT": 0.5},
            tax_savings_estimate=150.0,
            asset_locations={"AAPL": "acct1", "MSFT": "acct1"},
            per_account_results={
                "acct1": OptimizationResult(
                    method="household_proportional",
                    weights={"AAPL": 0.5, "MSFT": 0.5},
                ),
            },
        )
        assert result.tax_savings_estimate == 150.0
        assert result.household_weights["AAPL"] == 0.5
        assert result.asset_locations["MSFT"] == "acct1"
        assert result.per_account_results["acct1"].method == "household_proportional"

    def test_household_opt_result_field_access(self):
        from portopt.data.models import OptimizationResult

        result = HouseholdOptResult(
            account_weights={"a": {"X": 1.0}, "b": {"Y": 1.0}},
            household_weights={"X": 0.6, "Y": 0.4},
            tax_savings_estimate=0.0,
            asset_locations={"X": "a", "Y": "b"},
            per_account_results={
                "a": OptimizationResult(method="test", weights={"X": 1.0}),
                "b": OptimizationResult(method="test", weights={"Y": 1.0}),
            },
        )
        assert len(result.account_weights) == 2
        assert len(result.per_account_results) == 2
        assert "X" in result.asset_locations
        assert "Y" in result.asset_locations
