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
    """결측 차원이 있으면 총점이 나머지 차원의 합으로만 매겨집니다.

    이전에는 산출 가능한 2개 차원(liquidity·profitability)만 평균 내고 ×4를 해
    둘 다 만점이면 총점이 100점(90~100 기대)으로 나왔습니다. 그러나 이는
    아스테라시스(450950)처럼 상장 2년 미만이라 growth·cashflow가 NULL인
    종목이 4개 차원을 모두 갖춘 종목과 동일한 만점을 받는 버그였습니다.
    새 산식에서는 결측 차원을 0점으로 두고 100점 만점으로 환산하므로,
    liquidity=25, profitability=25 만점이어도 총점은 50점(=25+25)에 그칩니다.
    """
    m = FundamentalsMetrics(
        ticker="100000", as_of_date=date(2026, 4, 17),
        current_ratio=2.0, interest_coverage=10.0, debt_ratio=50.0,
        roe=20.0, roic=18.0, operating_margin=25.0,
        # No growth or cashflow data
    )
    sc = score(m)
    # Has 2/4 dimensions scored, both perfect → total is capped at 50 (25+25), not 100
    assert sc.liquidity_score is not None
    assert sc.profitability_score is not None
    assert sc.growth_score is None
    assert sc.cashflow_score is None
    assert sc.coverage == 2
    assert 45 <= sc.total_score <= 50


def test_missing_dimensions_do_not_yield_perfect_score():
    """4개 차원 중 2개만 산출돼도 만점이 나오면 안 됩니다 (아스테라시스 사례)."""
    m = FundamentalsMetrics(
        ticker="450950", as_of_date=date(2026, 8, 8),
        current_ratio=5.0, debt_ratio=10.0,      # liquidity 만점 유도
        roe=40.0, roic=30.0, operating_margin=30.0,  # profitability 만점 유도
        # growth / cashflow 입력 없음 → None
    )
    sc = score(m)
    assert sc.growth_score is None
    assert sc.cashflow_score is None
    assert sc.total_score < 100.0
    assert sc.coverage == 2
