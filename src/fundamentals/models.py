from datetime import date

from pydantic import BaseModel


class FundamentalsMetrics(BaseModel):
    """Derived financial metrics for a ticker."""

    ticker: str
    as_of_date: date

    # Stability
    current_ratio: float | None = None
    interest_coverage: float | None = None
    debt_ratio: float | None = None

    # Profitability
    roe: float | None = None
    roic: float | None = None
    operating_margin: float | None = None

    # Growth (3-year CAGR)
    revenue_cagr_3y: float | None = None
    op_income_cagr_3y: float | None = None

    # Cashflow
    ocf_to_ni_ratio: float | None = None
    fcf_positive_years: int | None = None

    # Valuation
    pe: float | None = None
    pb: float | None = None
    peg: float | None = None

    # Share-based derived (new)
    eps: float | None = None
    bps: float | None = None
    psr: float | None = None

    # Cashflow absolutes & ratios (new)
    ocf: float | None = None                   # 억원
    fcf: float | None = None                   # 억원
    capex_to_revenue: float | None = None      # %

    # Dividend (new)
    dividend_yield: float | None = None        # %
    payout_ratio: float | None = None          # %
    consecutive_dividend_years: int | None = None


class ScoreCard(BaseModel):
    """4-dimension scores plus total and categories."""

    ticker: str
    as_of_date: date
    liquidity_score: float | None = None      # out of 25
    profitability_score: float | None = None  # out of 25
    growth_score: float | None = None         # out of 25
    cashflow_score: float | None = None       # out of 25
    total_score: float                        # 0-100
    grade: str                                # ★★★★★ etc.
    categories: list[str] = []                # ["Quality", "GARP"] etc.
