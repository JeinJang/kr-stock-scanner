from datetime import date
from src.fundamentals.models import FundamentalsMetrics, ScoreCard
from src.fundamentals.classifier import classify, MarketMedians


def _metrics(**kwargs):
    return FundamentalsMetrics(ticker="X", as_of_date=date(2026, 4, 17), **kwargs)


def _score(total, **kwargs):
    return ScoreCard(
        ticker="X", as_of_date=date(2026, 4, 17),
        total_score=total, grade="★★★★☆", categories=[],
        **kwargs,
    )


def test_quality_label():
    m = _metrics(roe=20.0, debt_ratio=50.0)
    s = _score(80.0, liquidity_score=20)
    medians = MarketMedians(kospi_pe=15, kospi_pb=1.5, kosdaq_pe=20, kosdaq_pb=2.0)
    cats = classify(m, s, market="KOSPI", medians=medians)
    assert "Quality" in cats


def test_value_label_uses_market_specific_medians():
    m = _metrics(pe=8.0, pb=0.7)
    s = _score(60.0, liquidity_score=20)
    medians = MarketMedians(kospi_pe=15, kospi_pb=1.5, kosdaq_pe=20, kosdaq_pb=2.0)
    cats = classify(m, s, market="KOSPI", medians=medians)
    # 8 < 15*0.7 = 10.5 ✓; 0.7 < 1.5*0.7 = 1.05 ✓; liquidity 20 ≥ 18 ✓
    assert "Value" in cats


def test_growth_label():
    m = _metrics(revenue_cagr_3y=25.0, op_income_cagr_3y=20.0)
    s = _score(70.0)
    medians = MarketMedians(kospi_pe=15, kospi_pb=1.5, kosdaq_pe=20, kosdaq_pb=2.0)
    cats = classify(m, s, market="KOSDAQ", medians=medians)
    assert "Growth" in cats


def test_garp_label():
    m = _metrics(revenue_cagr_3y=18.0, peg=0.8)
    s = _score(70.0)
    medians = MarketMedians(kospi_pe=15, kospi_pb=1.5, kosdaq_pe=20, kosdaq_pb=2.0)
    cats = classify(m, s, market="KOSPI", medians=medians)
    assert "GARP" in cats


def test_caution_label():
    m = _metrics(interest_coverage=0.5)
    s = _score(40.0)
    medians = MarketMedians(kospi_pe=15, kospi_pb=1.5, kosdaq_pe=20, kosdaq_pb=2.0)
    cats = classify(m, s, market="KOSPI", medians=medians)
    assert "Caution" in cats


def test_multi_category():
    """A stock can have Quality + GARP simultaneously."""
    m = _metrics(roe=18.0, debt_ratio=40.0, revenue_cagr_3y=18.0, peg=0.8)
    s = _score(80.0, liquidity_score=20)
    medians = MarketMedians(kospi_pe=15, kospi_pb=1.5, kosdaq_pe=20, kosdaq_pb=2.0)
    cats = classify(m, s, market="KOSPI", medians=medians)
    assert "Quality" in cats
    assert "GARP" in cats
