"""Unit tests for finance_math module."""

import pytest

from finance_math import (
    future_value_monthly_payment,
    growth_curve,
    opportunity_cost,
    scenario_table,
    total_principal,
)


def test_future_value_zero_rate():
    """When annual_rate = 0, FV should equal total principal."""
    assert future_value_monthly_payment(payment=300, annual_rate=0.0, years=18) == 64800.0


def test_future_value_standard_scenario():
    """MOP 300/mo, 8% annual, 18 years → ~MOP $144,026."""
    fv = future_value_monthly_payment(payment=300, annual_rate=0.08, years=18)
    assert 144_000 < fv < 144_100


def test_future_value_higher_rate_higher_fv():
    fv_low = future_value_monthly_payment(300, 0.04, 18)
    fv_high = future_value_monthly_payment(300, 0.12, 18)
    assert fv_high > fv_low


def test_future_value_longer_horizon_higher_fv():
    fv_5 = future_value_monthly_payment(300, 0.08, 5)
    fv_20 = future_value_monthly_payment(300, 0.08, 20)
    assert fv_20 > fv_5


def test_future_value_rejects_negative_payment():
    with pytest.raises(ValueError):
        future_value_monthly_payment(-100, 0.08, 10)


def test_future_value_rejects_negative_years():
    with pytest.raises(ValueError):
        future_value_monthly_payment(100, 0.08, -1)


def test_future_value_rejects_zero_compounds():
    with pytest.raises(ValueError):
        future_value_monthly_payment(100, 0.08, 10, compounds_per_year=0)


def test_total_principal_simple():
    assert total_principal(300, 18) == 64800


def test_opportunity_cost_basic():
    oc = opportunity_cost(300, 0.08, 18)
    assert oc["principal"] == 64800
    assert oc["future_value"] > oc["principal"]
    assert oc["multiplier"] > 2.0
    assert oc["gap"] == oc["future_value"] - oc["principal"]


def test_growth_curve_length():
    g = growth_curve(300, 0.08, 18)
    assert len(g["months"]) == 18 * 12 + 1
    assert len(g["principal"]) == len(g["months"])
    assert len(g["compound"]) == len(g["months"])


def test_growth_curve_zero_rate():
    g = growth_curve(300, 0.0, 5)
    assert g["principal"] == g["compound"]


def test_growth_curve_compound_exceeds_principal():
    g = growth_curve(300, 0.08, 18)
    assert g["compound"][-1] > g["principal"][-1]


def test_scenario_table():
    table = scenario_table([(50, 0.08, 18), (100, 0.08, 18), (300, 0.08, 18), (500, 0.08, 18)])
    assert len(table) == 4
    assert all(row["multiplier"] > 2.0 for row in table)