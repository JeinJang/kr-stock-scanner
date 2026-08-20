from datetime import date

from src.price_history.db import PriceDB
from src.price_history.loader import load_bars


def _db(tmp_path):
    return PriceDB(path=str(tmp_path / "prices.db"))


def test_returns_none_when_ticker_absent(tmp_path):
    assert load_bars(_db(tmp_path), "005930", date(2026, 8, 19)) is None


def test_returns_none_with_fewer_than_two_valid_bars(tmp_path):
    db = _db(tmp_path)
    db.save_day("20260818", "KOSPI", [("005930", 100, 100, 0)])
    assert load_bars(db, "005930", date(2026, 8, 19)) is None


def test_returns_ascending_bars(tmp_path):
    db = _db(tmp_path)
    db.save_day("20260819", "KOSPI", [("005930", 120, 115, 5)])
    db.save_day("20260818", "KOSPI", [("005930", 110, 110, 0)])
    bars = load_bars(db, "005930", date(2026, 8, 19))
    assert [b.date for b in bars] == [date(2026, 8, 18), date(2026, 8, 19)]
    assert [b.high for b in bars] == [110.0, 120.0]


def test_excludes_dates_after_as_of(tmp_path):
    db = _db(tmp_path)
    for d, h in [("20260817", 100), ("20260818", 110), ("20260819", 120)]:
        db.save_day(d, "KOSPI", [("005930", h, h, 0)])
    bars = load_bars(db, "005930", date(2026, 8, 18))
    assert [b.date for b in bars] == [date(2026, 8, 17), date(2026, 8, 18)]


def test_drops_halted_days_with_zero_high(tmp_path):
    db = _db(tmp_path)
    db.save_day("20260817", "KOSPI", [("005930", 100, 100, 0)])
    db.save_day("20260818", "KOSPI", [("005930", 0, 100, 0)])   # 거래정지
    db.save_day("20260819", "KOSPI", [("005930", 120, 115, 5)])
    bars = load_bars(db, "005930", date(2026, 8, 19))
    assert [b.date for b in bars] == [date(2026, 8, 17), date(2026, 8, 19)]


def test_applies_stored_adjustment_events(tmp_path):
    db = _db(tmp_path)
    # 5:1 액면병합: 정지 전 396 -> 재개일 기준가 1,980
    db.save_day("20260817", "KOSPI", [("005930", 400, 396, 0)])
    db.save_day("20260818", "KOSPI", [("005930", 2065, 1720, -260)])
    from src.price_history.backfill import rebuild_adjustments
    rebuild_adjustments(db)

    bars = load_bars(db, "005930", date(2026, 8, 18))
    assert bars[0].high == 2000.0    # 400 x 5
    assert bars[1].high == 2065.0    # 이벤트 당일은 그대로
