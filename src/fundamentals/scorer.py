from __future__ import annotations

from src.fundamentals.models import FundamentalsMetrics, ScoreCard


def _linear_score(value: float | None, max_pt: float, full: float, zero: float) -> float | None:
    """Linear scoring: full→max_pt, zero→0, clipped."""
    if value is None:
        return None
    if (full > zero and value >= full) or (full < zero and value <= full):
        return max_pt
    if (full > zero and value <= zero) or (full < zero and value >= zero):
        return 0.0
    return max_pt * (value - zero) / (full - zero)


def _liquidity(m: FundamentalsMetrics) -> float | None:
    """A. Liquidity/Stability (max 25)."""
    parts: list[float] = []
    cur = _linear_score(m.current_ratio, max_pt=25 / 3, full=1.5, zero=1.0)
    ic = _linear_score(m.interest_coverage, max_pt=25 / 3, full=5.0, zero=1.0)
    debt = _linear_score(m.debt_ratio, max_pt=25 / 3, full=100.0, zero=200.0)
    for p in (cur, ic, debt):
        if p is not None:
            parts.append(p)
    if not parts:
        return None
    return (sum(parts) / len(parts)) * 3  # scale back to /25


def _profitability(m: FundamentalsMetrics) -> float | None:
    """B. Profitability (max 25). ROE + ROIC + operating_margin."""
    parts: list[float] = []
    roe = _linear_score(m.roe, max_pt=25 / 3, full=15.0, zero=5.0)
    roic = _linear_score(m.roic, max_pt=25 / 3, full=15.0, zero=5.0)
    op = _linear_score(m.operating_margin, max_pt=25 / 3, full=20.0, zero=0.0)
    for p in (roe, roic, op):
        if p is not None:
            parts.append(p)
    if not parts:
        return None
    return (sum(parts) / len(parts)) * 3


def _growth(m: FundamentalsMetrics) -> float | None:
    """C. Growth (max 25). Revenue + Op income 3y CAGR."""
    parts: list[float] = []
    rev = _linear_score(m.revenue_cagr_3y, max_pt=25 / 2, full=15.0, zero=0.0)
    op = _linear_score(m.op_income_cagr_3y, max_pt=25 / 2, full=15.0, zero=0.0)
    for p in (rev, op):
        if p is not None:
            parts.append(p)
    if not parts:
        return None
    return (sum(parts) / len(parts)) * 2


def _cashflow(m: FundamentalsMetrics) -> float | None:
    """D. Cashflow Quality (max 25)."""
    parts: list[float] = []
    if m.ocf_to_ni_ratio is not None:
        ratio = m.ocf_to_ni_ratio
        if 1.0 <= ratio <= 1.2:
            parts.append(12.5)
        elif ratio > 1.2:
            parts.append(max(0, 12.5 - (ratio - 1.2) * 5))
        elif ratio >= 0.5:
            parts.append(12.5 * (ratio - 0.5) / 0.5)
        else:
            parts.append(0.0)
    if m.fcf_positive_years is not None:
        parts.append(min(12.5, m.fcf_positive_years * 12.5 / 3))
    if not parts:
        return None
    return (sum(parts) / len(parts)) * 2


def _grade_for(score_value: float) -> str:
    if score_value >= 90:
        return "★★★★★"
    if score_value >= 75:
        return "★★★★☆"
    if score_value >= 60:
        return "★★★☆☆"
    if score_value >= 45:
        return "★★☆☆☆"
    return "★☆☆☆☆"


def score(m: FundamentalsMetrics) -> ScoreCard:
    """Compute 4-dimension scores for a ticker."""
    liq = _liquidity(m)
    prof = _profitability(m)
    growth = _growth(m)
    cf = _cashflow(m)

    available = [s for s in (liq, prof, growth, cf) if s is not None]
    if not available:
        total = 0.0
    else:
        # Each dimension is 0-25; scale to 0-100 by averaging available, multiplying by 4
        avg = sum(available) / len(available)
        total = avg * 4  # because avg is /25, so /25 * 4 = /100
    total = round(total, 1)

    return ScoreCard(
        ticker=m.ticker,
        as_of_date=m.as_of_date,
        liquidity_score=round(liq, 1) if liq is not None else None,
        profitability_score=round(prof, 1) if prof is not None else None,
        growth_score=round(growth, 1) if growth is not None else None,
        cashflow_score=round(cf, 1) if cf is not None else None,
        total_score=total,
        grade=_grade_for(total),
        categories=[],  # filled by classifier
    )
