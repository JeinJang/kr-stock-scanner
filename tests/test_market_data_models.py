from datetime import date
from src.market_data.models import MarketYearly


def test_market_yearly_roundtrip():
    m = MarketYearly(
        corp_code="00126380",
        ticker="000660",
        year=2025,
        as_of_date=date(2025, 12, 30),
        market_cap=240_000_000_000_000,
        shares_outstanding=727_960_000,
        close_price=330_000,
    )
    assert m.market_cap == 240_000_000_000_000
    assert m.year == 2025


def test_market_yearly_nullable_price_fields():
    m = MarketYearly(
        corp_code="00126380", ticker="000660",
        year=2020, as_of_date=date(2020, 12, 30),
    )
    assert m.market_cap is None
    assert m.shares_outstanding is None
