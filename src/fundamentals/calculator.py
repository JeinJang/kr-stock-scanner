from __future__ import annotations

from datetime import date

from src.dart.models import FinancialStatement
from src.fundamentals.models import FundamentalsMetrics
from src.market_data.models import MarketYearly


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
    market_yearly: list[MarketYearly],
    as_of: date,
) -> FundamentalsMetrics:
    """Compute derived financial metrics for a single ticker.

    Reads:
      - statements: pivot-by-year dart accounts.
      - market_yearly: list of yearly market snapshots (already includes the in-progress year).

    Rule:
      - LY (latest year) values come from the most recent COMPLETED annual report.
      - market_cap_now / shares_ly use:
          market_cap_now -> highest year() in market_yearly (i.e., the in-progress year row,
                            written by the most recent backfill)
          shares_ly       -> the market_yearly row whose year == latest_year (the year that
                             produced the LY financials)
    """
    # --- pivot existing accounts ---
    revenue = _account_values_by_year(statements, "매출액")
    op_income = _account_values_by_year(statements, "영업이익")
    net_income = _account_values_by_year(statements, "당기순이익")
    equity = _account_values_by_year(statements, "자본총계")
    debt = _account_values_by_year(statements, "부채총계")
    assets = _account_values_by_year(statements, "자산총계")
    current_assets = _account_values_by_year(statements, "유동자산")
    current_liabilities = _account_values_by_year(statements, "유동부채")

    if not equity:
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

    # --- market data ---
    market_now = None
    market_ly = None
    if market_yearly:
        market_now = max(market_yearly, key=lambda r: r.year)
        ly_rows = [r for r in market_yearly if r.year == latest_year]
        market_ly = ly_rows[0] if ly_rows else None
    market_cap_now = market_now.market_cap if market_now else None
    shares_ly = market_ly.shares_outstanding if market_ly else None

    # --- stability ---
    current_ratio = _safe_div(latest_ca, latest_cl)
    debt_ratio_pct = _safe_div(latest_debt, latest_equity)
    if debt_ratio_pct is not None:
        debt_ratio_pct *= 100.0

    # --- profitability (3-year averages) ---
    roe_avg = _avg_ratio(net_income, equity, years=3)
    roic_avg = _avg_roic(net_income, equity, debt, years=3)

    operating_margin = None
    if latest_op is not None and latest_revenue and latest_revenue > 0:
        operating_margin = (latest_op / latest_revenue) * 100.0

    # --- growth (3y CAGR) ---
    revenue_cagr = _three_year_cagr(revenue)
    op_income_cagr = _three_year_cagr_positive_only(op_income)

    # --- valuation (PE/PB now use market_cap_now) ---
    pe = market_cap_now / latest_ni if (market_cap_now is not None and latest_ni and latest_ni > 0) else None
    pb = market_cap_now / latest_equity if (market_cap_now is not None and latest_equity and latest_equity > 0) else None
    peg = pe / op_income_cagr if (pe is not None and op_income_cagr is not None and op_income_cagr > 0) else None

    # --- share-derived metrics (new) ---
    eps = latest_ni / shares_ly if (latest_ni is not None and shares_ly and shares_ly > 0) else None
    bps = latest_equity / shares_ly if (latest_equity is not None and shares_ly and shares_ly > 0) else None
    psr = market_cap_now / latest_revenue if (market_cap_now is not None and latest_revenue and latest_revenue > 0) else None

    return FundamentalsMetrics(
        ticker=ticker, as_of_date=as_of,
        current_ratio=current_ratio, debt_ratio=debt_ratio_pct,
        roe=roe_avg, roic=roic_avg, operating_margin=operating_margin,
        revenue_cagr_3y=revenue_cagr, op_income_cagr_3y=op_income_cagr,
        pe=pe, pb=pb, peg=peg,
        eps=eps, bps=bps, psr=psr,
    )


def _avg_ratio(numer, denom, years):
    common = sorted((y for y in numer if y in denom), reverse=True)[:years]
    vals = [numer[y] / denom[y] for y in common if denom[y] != 0]
    return (sum(vals) / len(vals)) * 100.0 if vals else None


def _avg_roic(net_income, equity, debt, years):
    common = sorted((y for y in net_income if y in equity), reverse=True)[:years]
    vals = []
    for y in common:
        capital = (equity[y] or 0) + (debt.get(y) or 0)
        if capital > 0:
            vals.append(net_income[y] / capital)
    return (sum(vals) / len(vals)) * 100.0 if vals else None


def _three_year_cagr(series):
    if len(series) < 4:
        return None
    ys = sorted(series.keys())
    return _cagr(series[ys[-4]], series[ys[-1]], ys[-1] - ys[-4])


def _three_year_cagr_positive_only(series):
    if len(series) < 4:
        return None
    ys = sorted(series.keys())
    s, e = series[ys[-4]], series[ys[-1]]
    if s <= 0 or e <= 0:
        return None
    return _cagr(s, e, ys[-1] - ys[-4])
