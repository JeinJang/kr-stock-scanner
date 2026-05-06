from datetime import date
from src.fundamentals.models import FundamentalsMetrics
from src.fundamentals.scorer import score


def test_perfect_metrics_get_high_score():
    m = FundamentalsMetrics(
        ticker="005930", as_of_date=date(2026, 4, 17),
        current_ratio=2.0, interest_coverage=10.0, debt_ratio=50.0,
        roe=20.0, roic=18.0, operating_margin=25.0,
        revenue_cagr_3y=18.0, op_income_cagr_3y=20.0,
        ocf_to_ni_ratio=1.1, fcf_positive_years=3,
    )
    sc = score(m)
    assert sc.total_score >= 85
    assert sc.grade in ["★★★★★", "★★★★☆"]


def test_poor_metrics_get_low_score():
    m = FundamentalsMetrics(
        ticker="000000", as_of_date=date(2026, 4, 17),
        current_ratio=0.5, interest_coverage=0.5, debt_ratio=300.0,
        roe=2.0, roic=1.0, operating_margin=2.0,
        revenue_cagr_3y=-5.0, op_income_cagr_3y=-10.0,
        ocf_to_ni_ratio=0.3, fcf_positive_years=0,
    )
    sc = score(m)
    assert sc.total_score < 30


def test_partial_data_proportional_scaling():
    """Missing dimensions scale total proportionally."""
    m = FundamentalsMetrics(
        ticker="100000", as_of_date=date(2026, 4, 17),
        current_ratio=2.0, interest_coverage=10.0, debt_ratio=50.0,
        roe=20.0, roic=18.0, operating_margin=25.0,
        # No growth or cashflow data
    )
    sc = score(m)
    # Has 2/4 dimensions scored, both perfect → total scaled to 100
    assert sc.liquidity_score is not None
    assert sc.profitability_score is not None
    assert sc.growth_score is None
    assert sc.cashflow_score is None
    assert 90 <= sc.total_score <= 100
