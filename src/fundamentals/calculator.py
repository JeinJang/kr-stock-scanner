from __future__ import annotations

from datetime import date

from src.dart.models import FinancialStatement
from src.fundamentals.basis import filter_to_basis
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
    # 연결/별도가 섞이지 않도록 계산 전에 한 기준으로 통일합니다.
    statements, fs_basis = filter_to_basis(statements)

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
        return FundamentalsMetrics(ticker=ticker, as_of_date=as_of, fs_basis=fs_basis)

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

    # --- cashflow-derived metrics ---
    ocf_by_year = _account_values_by_year(statements, "영업활동현금흐름")
    capex_by_year = _account_values_by_year(statements, "유형자산취득")
    # Note: dividend_by_year is used in Task 11 — also extract here for forward use:
    dividend_by_year = _account_values_by_year(statements, "배당총액")

    # absolute values converted to 억원 (1e8)
    ocf = ocf_by_year.get(latest_year) / 1e8 if latest_year in ocf_by_year else None
    fcf = None
    if latest_year in ocf_by_year and latest_year in capex_by_year:
        fcf = (ocf_by_year[latest_year] - capex_by_year[latest_year]) / 1e8

    capex_to_revenue = None
    if latest_year in capex_by_year and latest_revenue and latest_revenue > 0:
        capex_to_revenue = (capex_by_year[latest_year] / latest_revenue) * 100.0

    ocf_to_ni_ratio = _avg_ocf_to_ni(ocf_by_year, net_income, years=3)
    fcf_positive_years = _count_fcf_positive(ocf_by_year, capex_by_year, years=5)

    dividend_yield = None
    payout_ratio = None
    consecutive_dividend_years = None

    if dividend_by_year:
        # absolute dividend total in 원 (DART negative-sign normalize)
        latest_div = abs(dividend_by_year.get(latest_year, 0.0))

        if market_cap_now is not None and market_cap_now > 0 and latest_year in dividend_by_year:
            dividend_yield = (latest_div / market_cap_now) * 100.0

        if latest_ni is not None and latest_ni > 0 and latest_year in dividend_by_year:
            payout_ratio = (latest_div / latest_ni) * 100.0

        consecutive_dividend_years = _count_consecutive_dividends(dividend_by_year)

    return FundamentalsMetrics(
        ticker=ticker, as_of_date=as_of, fs_basis=fs_basis,
        current_ratio=current_ratio, debt_ratio=debt_ratio_pct,
        roe=roe_avg, roic=roic_avg, operating_margin=operating_margin,
        revenue_cagr_3y=revenue_cagr, op_income_cagr_3y=op_income_cagr,
        ocf_to_ni_ratio=ocf_to_ni_ratio, fcf_positive_years=fcf_positive_years,
        pe=pe, pb=pb, peg=peg,
        eps=eps, bps=bps, psr=psr,
        ocf=ocf, fcf=fcf, capex_to_revenue=capex_to_revenue,
        dividend_yield=dividend_yield, payout_ratio=payout_ratio,
        consecutive_dividend_years=consecutive_dividend_years,
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


def _avg_ocf_to_ni(ocf: dict[int, float], ni: dict[int, float], years: int) -> float | None:
    common = sorted((y for y in ocf if y in ni and ni[y] != 0), reverse=True)[:years]
    if not common:
        return None
    vals = [ocf[y] / ni[y] for y in common]
    return sum(vals) / len(vals)


def _count_fcf_positive(ocf: dict[int, float], capex: dict[int, float], years: int) -> int | None:
    common_years = sorted(set(ocf.keys()) | set(capex.keys()), reverse=True)[:years]
    if not common_years:
        return None
    count = 0
    for y in common_years:
        if y in ocf and y in capex:
            if (ocf[y] - capex[y]) > 0:
                count += 1
    return count


def _count_consecutive_dividends(div: dict[int, float]) -> int:
    """From latest year backward, count years where |dividend| > 0. Stops on first zero."""
    if not div:
        return 0
    count = 0
    for y in sorted(div.keys(), reverse=True):
        if abs(div[y]) > 0:
            count += 1
        else:
            break
    return count
