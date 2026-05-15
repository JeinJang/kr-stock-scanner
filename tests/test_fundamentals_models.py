from datetime import date
from src.fundamentals.models import FundamentalsMetrics, ScoreCard


def test_metrics_creation():
    m = FundamentalsMetrics(
        ticker="005930",
        as_of_date=date(2026, 4, 17),
        roe=15.5, roic=12.0, debt_ratio=45.0,
        current_ratio=1.8, interest_coverage=8.5,
        operating_margin=20.0, revenue_cagr_3y=10.0, op_income_cagr_3y=12.0,
        ocf_to_ni_ratio=1.1, fcf_positive_years=3,
        pe=15.0, pb=1.5, peg=1.2,
    )
    assert m.ticker == "005930"
    assert m.roe == 15.5


def test_scorecard_creation():
    sc = ScoreCard(
        ticker="005930",
        as_of_date=date(2026, 4, 17),
        liquidity_score=20.0,
        profitability_score=22.0,
        growth_score=18.0,
        cashflow_score=23.0,
        total_score=83.0,
        grade="★★★★☆",
        categories=["Quality", "GARP"],
    )
    assert sc.total_score == 83.0
    assert "Quality" in sc.categories


def test_metrics_accepts_new_enrichment_fields():
    m = FundamentalsMetrics(
        ticker="000660",
        as_of_date=date(2026, 5, 15),
        eps=1900.5,
        bps=120000.0,
        psr=2.3,
        ocf=470000.0,
        fcf=320000.0,
        capex_to_revenue=15.4,
        dividend_yield=1.8,
        payout_ratio=22.5,
        consecutive_dividend_years=7,
    )
    assert m.eps == 1900.5
    assert m.consecutive_dividend_years == 7


def test_metrics_new_fields_default_to_none():
    m = FundamentalsMetrics(ticker="X", as_of_date=date(2026, 1, 1))
    assert m.eps is None
    assert m.consecutive_dividend_years is None
