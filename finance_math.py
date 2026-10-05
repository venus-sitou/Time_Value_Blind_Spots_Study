"""
finance_math.py — compound interest primitives.

Pure, dependency-light module that can be imported by both the
analysis pipeline and the CIII Diagnostic Tool.
"""

from typing import Iterable


def future_value_monthly_payment(
    payment: float,
    annual_rate: float,
    years: int,
    compounds_per_year: int = 12,
) -> float:
    """Future value of a recurring monthly contribution with monthly compounding.

    Formula: FV = PMT × [((1 + r)^n − 1) / r]
    where r = annual_rate / compounds_per_year, n = years × compounds_per_year

    Args:
        payment: monthly contribution (non-negative).
        annual_rate: nominal annual interest rate as decimal (e.g. 0.08 for 8%).
        years: investment horizon in years (non-negative).
        compounds_per_year: compounding frequency per year (default 12 = monthly).

    Returns:
        Future value at the end of `years`.

    Raises:
        ValueError: if any input is invalid.
    """
    if payment < 0:
        raise ValueError("payment must be non-negative")
    if years < 0:
        raise ValueError("years must be non-negative")
    if compounds_per_year <= 0:
        raise ValueError("compounds_per_year must be positive")

    n_periods = years * compounds_per_year
    periodic_rate = annual_rate / compounds_per_year
    principal = payment * n_periods

    if periodic_rate == 0:
        return principal
    return payment * ((1 + periodic_rate) ** n_periods - 1) / periodic_rate


def total_principal(payment: float, years: int, compounds_per_year: int = 12) -> float:
    """Total principal invested over the horizon (no interest)."""
    return payment * years * compounds_per_year


def opportunity_cost(
    payment: float,
    annual_rate: float,
    years: int,
) -> dict:
    """Compute opportunity-cost scenario: principal vs compound.

    Returns dict with:
        principal: total money invested with no growth.
        future_value: total money if invested at annual_rate.
        gap: future_value - principal (the "time-value added").
        multiplier: future_value / principal (e.g. 2.22 means 2.22× the principal).
    """
    principal = total_principal(payment, years)
    fv = future_value_monthly_payment(payment, annual_rate, years)
    return {
        "principal": principal,
        "future_value": fv,
        "gap": fv - principal,
        "multiplier": fv / principal if principal > 0 else float("nan"),
    }


def growth_curve(
    payment: float,
    annual_rate: float,
    years: int,
    steps_per_year: int = 12,
) -> dict:
    """Month-by-month principal vs compound growth trajectories.

    Returns dict with:
        months: list of month indices (0 to years * steps_per_year).
        principal: cumulative principal at each month.
        compound: cumulative compound value at each month.
    """
    n_months = years * steps_per_year
    months = list(range(0, n_months + 1))
    principal = [payment * m for m in months]
    r = annual_rate / steps_per_year
    if r == 0:
        compound = list(principal)
    else:
        compound = [payment * ((1 + r) ** m - 1) / r if m > 0 else 0 for m in months]
    return {"months": months, "principal": principal, "compound": compound}


def scenario_table(
    scenarios: Iterable[tuple[float, float, int]],
) -> list[dict]:
    """Compute opportunity-cost for multiple (PMT, rate, years) tuples.

    Returns list of dicts with pmt, rate, years, principal, future_value, multiplier.
    """
    return [
        {
            "pmt": pmt,
            "rate": rate,
            "years": years,
            **opportunity_cost(pmt, rate, years),
        }
        for pmt, rate, years in scenarios
    ]


__all__ = [
    "future_value_monthly_payment",
    "total_principal",
    "opportunity_cost",
    "growth_curve",
    "scenario_table",
]