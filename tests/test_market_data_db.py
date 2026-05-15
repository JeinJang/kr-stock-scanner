from datetime import date
from src.market_data.db import MarketDB
from src.market_data.models import MarketYearly


def _row(year=2025, market_cap=100, shares=10, close=10):
    return MarketYearly(
        corp_code="00126380", ticker="000660",
        year=year, as_of_date=date(year, 12, 30),
        market_cap=market_cap, shares_outstanding=shares, close_price=close,
    )


def test_save_and_load_yearly(tmp_path):
    db = MarketDB(url=f"sqlite:///{tmp_path/'t.db'}")
    db.save_yearly([_row(2024), _row(2025)])
    loaded = db.load_for_corp("00126380")
    assert {r.year for r in loaded} == {2024, 2025}


def test_upsert_is_idempotent(tmp_path):
    db = MarketDB(url=f"sqlite:///{tmp_path/'t.db'}")
    db.save_yearly([_row(2025, market_cap=100)])
    db.save_yearly([_row(2025, market_cap=200)])  # same (corp_code, year) -> overwrite
    loaded = db.load_for_corp("00126380")
    assert len(loaded) == 1
    assert loaded[0].market_cap == 200


def test_load_all_groups_by_corp(tmp_path):
    db = MarketDB(url=f"sqlite:///{tmp_path/'t.db'}")
    db.save_yearly([
        _row(2024), _row(2025),
        MarketYearly(corp_code="00100001", ticker="005930",
                     year=2025, as_of_date=date(2025, 12, 30),
                     market_cap=500, shares_outstanding=50, close_price=10),
    ])
    grouped = db.load_all()
    assert "00126380" in grouped
    assert "00100001" in grouped
    assert len(grouped["00126380"]) == 2
