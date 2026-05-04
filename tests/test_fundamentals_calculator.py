from datetime import date
from src.dart.models import FinancialStatement
from src.fundamentals.calculator import compute_metrics


def _make_fs(corp_code, year, account, value, quarter=0):
    return FinancialStatement(
        corp_code=corp_code, year=year, quarter=quarter,
        account=account, value=value,
    )


def test_compute_basic_metrics():
    """Test ROE, debt ratio computation from minimal data."""
    statements = [
        _make_fs("001", 2025, "당기순이익", 100),
        _make_fs("001", 2025, "자본총계", 1000),
        _make_fs("001", 2025, "부채총계", 500),
        _make_fs("001", 2025, "자산총계", 1500),
        _make_fs("001", 2025, "유동자산", 800),
        _make_fs("001", 2025, "유동부채", 400),
        _make_fs("001", 2024, "당기순이익", 90),
        _make_fs("001", 2024, "자본총계", 950),
        _make_fs("001", 2023, "당기순이익", 80),
        _make_fs("001", 2023, "자본총계", 900),
    ]

    metrics = compute_metrics(
        ticker="000001", corp_code="001",
        statements=statements, as_of=date(2026, 4, 17),
        market_cap=None, eps=None, bps=None,
    )

    # ROE = NI / Equity = 100/1000 = 10%
    assert metrics.roe is not None
    assert abs(metrics.roe - 10.0) < 0.01
    # Debt ratio = 500/1000 = 50%
    assert abs(metrics.debt_ratio - 50.0) < 0.01
    # Current ratio = 800/400 = 2.0
    assert abs(metrics.current_ratio - 2.0) < 0.01


def test_compute_with_market_data():
    """Test P/E, P/B with market cap and EPS/BPS."""
    statements = [
        _make_fs("001", 2025, "당기순이익", 100),
        _make_fs("001", 2025, "자본총계", 1000),
        _make_fs("001", 2025, "자산총계", 1500),
        _make_fs("001", 2025, "부채총계", 500),
    ]

    metrics = compute_metrics(
        ticker="000001", corp_code="001",
        statements=statements, as_of=date(2026, 4, 17),
        market_cap=1500.0, eps=10.0, bps=100.0,
    )

    # P/E = price / EPS; market_cap=1500, NI=100 → implied price/EPS via market_cap/NI = 15
    assert metrics.pe is not None
    assert abs(metrics.pe - 15.0) < 0.01
    # P/B = market_cap / equity = 1500/1000 = 1.5
    assert abs(metrics.pb - 1.5) < 0.01


def test_revenue_cagr_3y():
    """Test 3-year CAGR computation."""
    statements = [
        _make_fs("001", 2025, "매출액", 1331),  # 33.1% CAGR vs 2022
        _make_fs("001", 2024, "매출액", 1210),
        _make_fs("001", 2023, "매출액", 1100),
        _make_fs("001", 2022, "매출액", 1000),
        _make_fs("001", 2025, "자본총계", 1000),
    ]

    metrics = compute_metrics(
        ticker="000001", corp_code="001",
        statements=statements, as_of=date(2026, 4, 17),
        market_cap=None, eps=None, bps=None,
    )

    # CAGR = (1331/1000)^(1/3) - 1 ≈ 0.10 → 10%
    assert metrics.revenue_cagr_3y is not None
    assert abs(metrics.revenue_cagr_3y - 10.0) < 0.5


def test_missing_data_returns_none():
    """Empty statements produce metrics with None fields."""
    metrics = compute_metrics(
        ticker="000001", corp_code="001",
        statements=[], as_of=date(2026, 4, 17),
        market_cap=None, eps=None, bps=None,
    )
    assert metrics.roe is None
    assert metrics.pe is None
