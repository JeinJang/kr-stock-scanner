from __future__ import annotations

from datetime import date

from src.dart.models import FinancialStatement
from src.fundamentals.models import FundamentalsMetrics


def _account_values_by_year(
    statements: list[FinancialStatement], account: str, quarter: int = 0,
) -> dict[int, float]:
    """Return dict of year -> value for an account (annual reports only by default)."""
    result = {}
    for s in statements:
        if s.account == account and s.quarter == quarter:
            result[s.year] = s.value
    return result


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def _cagr(start: float, end: float, years: int) -> float | None:
    """Compound Annual Growth Rate as percentage."""
    if start <= 0 or end <= 0 or years <= 0:
        return None
    return ((end / start) ** (1.0 / years) - 1.0) * 100.0


def compute_metrics(
    ticker: str,
    corp_code: str,
    statements: list[FinancialStatement],
    as_of: date,
    market_cap: float | None,
    eps: float | None,
    bps: float | None,
) -> FundamentalsMetrics:
    """Compute derived financial metrics for a single ticker."""
    revenue = _account_values_by_year(statements, "매출액")
    op_income = _account_values_by_year(statements, "영업이익")
    net_income = _account_values_by_year(statements, "당기순이익")
    equity = _account_values_by_year(statements, "자본총계")
    debt = _account_values_by_year(statements, "부채총계")
    assets = _account_values_by_year(statements, "자산총계")
    current_assets = _account_values_by_year(statements, "유동자산")
    current_liabilities = _account_values_by_year(statements, "유동부채")

    if not equity:
        # No data at all
        return FundamentalsMetrics(ticker=ticker, as_of_date=as_of)

    latest_year = max(equity.keys())
    latest_revenue = revenue.get(latest_year)
    latest_op = op_income.get(latest_year)
    latest_ni = net_income.get(latest_year)
    latest_equity = equity.get(latest_year)
    latest_debt = debt.get(latest_year)
    latest_assets = assets.get(latest_year)
    latest_ca = current_assets.get(latest_year)
    latest_cl = current_liabilities.get(latest_year)

    # Stability
    current_ratio = _safe_div(latest_ca, latest_cl)
    debt_ratio_pct = _safe_div(latest_debt, latest_equity)
    if debt_ratio_pct is not None:
        debt_ratio_pct *= 100.0

    # Profitability — 3-year average ROE per spec (B. 수익성)
    roe_avg = None
    roe_years = sorted(
        (y for y in net_income if y in equity),
        reverse=True,
    )[:3]
    roe_values = [
        net_income[y] / equity[y]
        for y in roe_years
        if equity[y] != 0
    ]
    if roe_values:
        roe_avg = (sum(roe_values) / len(roe_values)) * 100.0

    # ROIC ≈ NI / (Equity + Debt) — 3-year average per spec
    roic_avg = None
    roic_years = sorted(
        (y for y in net_income if y in equity),
        reverse=True,
    )[:3]
    roic_values = []
    for y in roic_years:
        d = debt.get(y, 0) or 0
        capital = (equity[y] or 0) + d
        if capital > 0:
            roic_values.append(net_income[y] / capital)
    if roic_values:
        roic_avg = (sum(roic_values) / len(roic_values)) * 100.0

    operating_margin = None
    if latest_op is not None and latest_revenue and latest_revenue > 0:
        operating_margin = (latest_op / latest_revenue) * 100.0

    # Growth (3y CAGR)
    revenue_cagr = None
    if len(revenue) >= 4:
        years_sorted = sorted(revenue.keys())
        start_y, end_y = years_sorted[-4], years_sorted[-1]
        revenue_cagr = _cagr(revenue[start_y], revenue[end_y], end_y - start_y)

    op_income_cagr = None
    if len(op_income) >= 4:
        years_sorted = sorted(op_income.keys())
        start_y, end_y = years_sorted[-4], years_sorted[-1]
        if op_income[start_y] > 0 and op_income[end_y] > 0:
            op_income_cagr = _cagr(op_income[start_y], op_income[end_y], end_y - start_y)

    # Valuation (using market cap)
    pe = None
    if market_cap is not None and latest_ni is not None and latest_ni > 0:
        pe = market_cap / latest_ni

    pb = None
    if market_cap is not None and latest_equity is not None and latest_equity > 0:
        pb = market_cap / latest_equity

    peg = None
    if pe is not None and op_income_cagr is not None and op_income_cagr > 0:
        peg = pe / op_income_cagr

    return FundamentalsMetrics(
        ticker=ticker,
        as_of_date=as_of,
        current_ratio=current_ratio,
        debt_ratio=debt_ratio_pct,
        roe=roe_avg,
        roic=roic_avg,
        operating_margin=operating_margin,
        revenue_cagr_3y=revenue_cagr,
        op_income_cagr_3y=op_income_cagr,
        pe=pe,
        pb=pb,
        peg=peg,
    )
