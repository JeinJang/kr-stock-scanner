from __future__ import annotations

from pydantic import BaseModel

from src.fundamentals.models import FundamentalsMetrics, ScoreCard


class MarketMedians(BaseModel):
    """Median P/E and P/B per market."""

    kospi_pe: float
    kospi_pb: float
    kosdaq_pe: float
    kosdaq_pb: float

    def pe_median(self, market: str) -> float:
        return self.kospi_pe if market == "KOSPI" else self.kosdaq_pe

    def pb_median(self, market: str) -> float:
        return self.kospi_pb if market == "KOSPI" else self.kosdaq_pb


def _is_quality(m: FundamentalsMetrics, s: ScoreCard) -> bool:
    return (
        s.total_score >= 75
        and m.roe is not None and m.roe >= 15
        and m.debt_ratio is not None and m.debt_ratio < 100
    )


def _is_value(
    m: FundamentalsMetrics, s: ScoreCard, market: str, medians: MarketMedians,
) -> bool:
    if m.pe is None or m.pb is None or s.liquidity_score is None:
        return False
    pe_threshold = medians.pe_median(market) * 0.7
    pb_threshold = medians.pb_median(market) * 0.7
    return (
        m.pe <= pe_threshold
        and m.pb <= pb_threshold
        and s.liquidity_score >= 18
    )


def _is_growth(m: FundamentalsMetrics) -> bool:
    if m.revenue_cagr_3y is None or m.revenue_cagr_3y < 20:
        return False
    # If op income data exists, require >= 15; else accept revenue alone
    if m.op_income_cagr_3y is not None and m.op_income_cagr_3y < 15:
        return False
    return True


def _is_garp(m: FundamentalsMetrics) -> bool:
    return (
        m.revenue_cagr_3y is not None and m.revenue_cagr_3y >= 15
        and m.peg is not None and m.peg <= 1.0
    )


def _is_caution(m: FundamentalsMetrics, s: ScoreCard) -> bool:
    if s.total_score < 45:
        return True
    if m.ocf_to_ni_ratio is not None and m.ocf_to_ni_ratio < 0.5:
        return True
    if m.interest_coverage is not None and m.interest_coverage < 1:
        return True
    return False


def classify(
    m: FundamentalsMetrics,
    s: ScoreCard,
    market: str,
    medians: MarketMedians,
) -> list[str]:
    """Assign category labels to a ticker."""
    cats: list[str] = []
    if _is_quality(m, s):
        cats.append("Quality")
    if _is_value(m, s, market, medians):
        cats.append("Value")
    if _is_growth(m):
        cats.append("Growth")
    if _is_garp(m):
        cats.append("GARP")
    if _is_caution(m, s):
        cats.append("Caution")
    return cats


def compute_market_medians(
    metrics: list[FundamentalsMetrics],
    markets: dict[str, str],
) -> MarketMedians:
    """Compute median P/E and P/B per market from a batch of metrics."""
    kospi_pe: list[float] = []
    kospi_pb: list[float] = []
    kosdaq_pe: list[float] = []
    kosdaq_pb: list[float] = []
    for m in metrics:
        market = markets.get(m.ticker)
        if market == "KOSPI":
            if m.pe is not None and m.pe > 0:
                kospi_pe.append(m.pe)
            if m.pb is not None and m.pb > 0:
                kospi_pb.append(m.pb)
        elif market == "KOSDAQ":
            if m.pe is not None and m.pe > 0:
                kosdaq_pe.append(m.pe)
            if m.pb is not None and m.pb > 0:
                kosdaq_pb.append(m.pb)

    def median(xs: list[float], default: float) -> float:
        if not xs:
            return default
        s = sorted(xs)
        n = len(s)
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2

    return MarketMedians(
        kospi_pe=median(kospi_pe, 15.0),
        kospi_pb=median(kospi_pb, 1.5),
        kosdaq_pe=median(kosdaq_pe, 20.0),
        kosdaq_pb=median(kosdaq_pb, 2.0),
    )
