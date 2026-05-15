from datetime import date
from src.dart.models import FinancialStatement
from src.fundamentals.calculator import compute_metrics
from src.market_data.models import MarketYearly


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
        market_yearly=[],
    )

    # ROE = 3-year average: (100/1000 + 90/950 + 80/900) / 3 ≈ 9.45%
    assert metrics.roe is not None
    assert abs(metrics.roe - 9.45) < 0.5
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
        market_yearly=[MarketYearly(
            corp_code='001', ticker='000001', year=2025,
            as_of_date=date(2025, 12, 30),
            market_cap=1500, shares_outstanding=10,
        )],
    )

    # P/E = market_cap / NI = 1500/100 = 15
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
        market_yearly=[],
    )

    # CAGR = (1331/1000)^(1/3) - 1 ≈ 0.10 → 10%
    assert metrics.revenue_cagr_3y is not None
    assert abs(metrics.revenue_cagr_3y - 10.0) < 0.5


def test_missing_data_returns_none():
    """Empty statements produce metrics with None fields."""
    metrics = compute_metrics(
        ticker="000001", corp_code="001",
        statements=[], as_of=date(2026, 4, 17),
        market_yearly=[],
    )
    assert metrics.roe is None
    assert metrics.pe is None


# ---------------------------------------------------------------------------
# New tests for share-derived metrics (EPS, BPS, PSR) + refined PE/PB
# ---------------------------------------------------------------------------

def _stmt(year, account, value, corp="C1"):
    return FinancialStatement(corp_code=corp, year=year, quarter=0, account=account, value=value)


def _my(year, mc=None, shares=None, close=None, ticker="000660"):
    return MarketYearly(
        corp_code="C1", ticker=ticker, year=year,
        as_of_date=date(year, 12, 30),
        market_cap=mc, shares_outstanding=shares, close_price=close,
    )


def test_eps_uses_latest_year_net_income_over_shares():
    statements = [
        _stmt(2024, "당기순이익", 47_605_327_690),
        _stmt(2024, "자본총계", 800_000_000_000),
        _stmt(2024, "매출액", 1_065_000_000_000),
    ]
    market = [_my(2024, mc=2_000_000_000_000, shares=243_000_000)]
    m = compute_metrics(
        ticker="353200", corp_code="C1",
        statements=statements, market_yearly=market, as_of=date(2026, 5, 15),
    )
    assert m.eps is not None
    assert abs(m.eps - 195.9) < 1.0   # 47.6B / 243M


def test_bps_from_equity_over_shares():
    statements = [
        _stmt(2024, "자본총계", 800_000_000_000),
        _stmt(2024, "당기순이익", 1),
    ]
    market = [_my(2024, mc=1, shares=10_000_000)]
    m = compute_metrics(
        ticker="X", corp_code="C1",
        statements=statements, market_yearly=market, as_of=date(2026, 5, 15),
    )
    assert m.bps == 80_000.0


def test_psr_from_market_cap_now_over_revenue():
    statements = [
        _stmt(2024, "매출액", 1_000_000_000_000),
        _stmt(2024, "자본총계", 1),
        _stmt(2024, "당기순이익", 1),
    ]
    # market_yearly has both 2024 (LY) and 2025 (in-progress = "now")
    market = [
        _my(2024, mc=2_000_000_000_000, shares=100_000_000),
        _my(2025, mc=2_500_000_000_000, shares=100_000_000),
    ]
    m = compute_metrics(
        ticker="X", corp_code="C1",
        statements=statements, market_yearly=market, as_of=date(2026, 5, 15),
    )
    assert m.psr == 2.5    # 2.5T / 1T using market_cap_now (2025)


def test_pe_uses_market_cap_now_not_ly():
    statements = [
        _stmt(2024, "당기순이익", 1_000_000_000),
        _stmt(2024, "자본총계", 1),
    ]
    market = [
        _my(2024, mc=10_000_000_000, shares=1_000_000),
        _my(2025, mc=20_000_000_000, shares=1_000_000),   # "now"
    ]
    m = compute_metrics(
        ticker="X", corp_code="C1",
        statements=statements, market_yearly=market, as_of=date(2026, 5, 15),
    )
    assert m.pe == 20.0   # 20B / 1B using 2025 mc, not 10.0


def test_no_market_data_keeps_share_metrics_null():
    statements = [_stmt(2024, "당기순이익", 1), _stmt(2024, "자본총계", 1)]
    m = compute_metrics(
        ticker="X", corp_code="C1",
        statements=statements, market_yearly=[], as_of=date(2026, 5, 15),
    )
    assert m.eps is None
    assert m.bps is None
    assert m.psr is None
    assert m.pe is None
    assert m.pb is None


# ---------------------------------------------------------------------------
# New tests for cashflow-derived metrics (OCF, FCF, CAPEX/Rev, OCF/NI, FCF+yrs)
# ---------------------------------------------------------------------------

def test_ocf_and_fcf_use_latest_year_in_억():
    statements = [
        _stmt(2024, "당기순이익", 1), _stmt(2024, "자본총계", 1),
        _stmt(2024, "영업활동현금흐름", 50_000_000_000),   # 500억
        _stmt(2024, "유형자산취득", 30_000_000_000),       # 300억
    ]
    m = compute_metrics("X", "C1", statements, market_yearly=[], as_of=date(2026, 5, 15))
    assert m.ocf == 500.0
    assert m.fcf == 200.0


def test_capex_to_revenue_pct():
    statements = [
        _stmt(2024, "당기순이익", 1), _stmt(2024, "자본총계", 1),
        _stmt(2024, "매출액", 1_000_000_000_000),
        _stmt(2024, "유형자산취득", 100_000_000_000),
    ]
    m = compute_metrics("X", "C1", statements, market_yearly=[], as_of=date(2026, 5, 15))
    assert m.capex_to_revenue == 10.0


def test_ocf_to_ni_ratio_averages_3_years():
    statements = [
        _stmt(2024, "당기순이익", 100), _stmt(2024, "영업활동현금흐름", 110), _stmt(2024, "자본총계", 1),
        _stmt(2023, "당기순이익", 100), _stmt(2023, "영업활동현금흐름", 90),
        _stmt(2022, "당기순이익", 100), _stmt(2022, "영업활동현금흐름", 100),
    ]
    m = compute_metrics("X", "C1", statements, market_yearly=[], as_of=date(2026, 5, 15))
    # avg(1.1, 0.9, 1.0) = 1.0
    assert abs(m.ocf_to_ni_ratio - 1.0) < 1e-9


def test_fcf_positive_years_counts_last_5():
    statements = [_stmt(2024, "자본총계", 1)]
    # OCF=100 CAPEX=50  -> FCF=+50 (positive)
    # OCF=50  CAPEX=80  -> FCF=-30 (negative)
    yearly = [(2020, 100, 50), (2021, 100, 50), (2022, 50, 80), (2023, 100, 50), (2024, 100, 50)]
    for y, ocf, capex in yearly:
        statements.append(_stmt(y, "영업활동현금흐름", ocf))
        statements.append(_stmt(y, "유형자산취득", capex))
    m = compute_metrics("X", "C1", statements, market_yearly=[], as_of=date(2026, 5, 15))
    assert m.fcf_positive_years == 4   # all except 2022


def test_cashflow_metrics_null_when_missing():
    statements = [_stmt(2024, "당기순이익", 1), _stmt(2024, "자본총계", 1)]
    m = compute_metrics("X", "C1", statements, market_yearly=[], as_of=date(2026, 5, 15))
    assert m.ocf is None
    assert m.fcf is None
    assert m.capex_to_revenue is None
    assert m.ocf_to_ni_ratio is None
    assert m.fcf_positive_years is None
