from datetime import date

from src.price_history.adjust import AdjustEvent
from src.price_history.db import PriceDB


def _db(tmp_path):
    return PriceDB(path=str(tmp_path / "prices.db"))


def test_save_and_load_rows(tmp_path):
    db = _db(tmp_path)
    db.save_day("20260105", "KOSPI", [("005930", 110, 100, 0)])
    db.save_day("20260106", "KOSPI", [("005930", 115, 105, 5)])

    rows = db.load_rows("005930", since="20260101")
    assert [r.d for r in rows] == [date(2026, 1, 5), date(2026, 1, 6)]
    assert rows[1].high == 115.0 and rows[1].chg == 5.0


def test_load_rows_respects_since(tmp_path):
    db = _db(tmp_path)
    db.save_day("20260105", "KOSPI", [("005930", 110, 100, 0)])
    db.save_day("20260106", "KOSPI", [("005930", 115, 105, 5)])
    assert [r.d for r in db.load_rows("005930", since="20260106")] == [date(2026, 1, 6)]


def test_save_day_is_idempotent(tmp_path):
    db = _db(tmp_path)
    db.save_day("20260105", "KOSPI", [("005930", 110, 100, 0)])
    db.save_day("20260105", "KOSPI", [("005930", 999, 999, 0)])
    rows = db.load_rows("005930", since="20260101")
    assert len(rows) == 1 and rows[0].high == 999.0


def test_loaded_dates_and_last_loaded(tmp_path):
    db = _db(tmp_path)
    assert db.loaded_dates("KOSPI") == set()
    assert db.last_loaded_date() is None
    db.save_day("20260105", "KOSPI", [("005930", 110, 100, 0)])
    db.save_day("20260106", "KOSDAQ", [("035720", 50, 48, 1)])
    assert db.loaded_dates("KOSPI") == {"20260105"}
    assert db.last_loaded_date() == "20260106"


def test_meta_roundtrip(tmp_path):
    db = _db(tmp_path)
    assert db.get_meta("backfill_years") is None
    db.set_meta("backfill_years", "11")
    db.set_meta("backfill_years", "12")
    assert db.get_meta("backfill_years") == "12"


def test_events_roundtrip_replaces_previous(tmp_path):
    db = _db(tmp_path)
    db.save_events("005930", [AdjustEvent(d=date(2026, 1, 6), factor=50.0)])
    db.save_events("005930", [AdjustEvent(d=date(2026, 1, 7), factor=2.0)])
    evs = db.load_events("005930")
    assert len(evs) == 1
    assert evs[0].d == date(2026, 1, 7) and evs[0].factor == 2.0


def test_tickers_lists_distinct(tmp_path):
    db = _db(tmp_path)
    db.save_day("20260105", "KOSPI", [("005930", 1, 1, 0), ("000660", 1, 1, 0)])
    db.save_day("20260106", "KOSPI", [("005930", 1, 1, 0)])
    assert sorted(db.tickers()) == ["000660", "005930"]
