from datetime import date

from src.aihw.db import AihwDB
from src.aihw.models import DailyCap

D1, D2 = date(2026, 1, 10), date(2026, 1, 11)


def _row(d, ticker, cap, source):
    return DailyCap(
        date=d, ticker=ticker, close=100.0, shares=10,
        market_cap_usd=cap, source=source,
    )


def _make_db():
    return AihwDB(url="sqlite:///:memory:")


class TestAihwDB:
    def test_save_and_load(self):
        db = _make_db()
        n = db.save_caps([_row(D1, "NVDA", 100.0, "backfill"),
                          _row(D1, "SPY", None, "backfill")])
        assert n == 2
        rows = db.load_caps(D1, D1)
        assert len(rows) == 2
        assert rows[0].ticker == "NVDA"
        assert rows[0].market_cap_usd == 100.0

    def test_load_range_filters_dates(self):
        db = _make_db()
        db.save_caps([_row(D1, "NVDA", 100.0, "backfill"),
                      _row(D2, "NVDA", 110.0, "backfill")])
        rows = db.load_caps(D2, D2)
        assert len(rows) == 1
        assert rows[0].date == D2

    def test_backfill_does_not_overwrite_snapshot(self):
        db = _make_db()
        db.save_caps([_row(D1, "NVDA", 100.0, "snapshot")])
        db.save_caps([_row(D1, "NVDA", 999.0, "backfill")])
        rows = db.load_caps(D1, D1)
        assert rows[0].market_cap_usd == 100.0
        assert rows[0].source == "snapshot"

    def test_snapshot_overwrites_backfill(self):
        db = _make_db()
        db.save_caps([_row(D1, "NVDA", 100.0, "backfill")])
        db.save_caps([_row(D1, "NVDA", 105.0, "snapshot")])
        rows = db.load_caps(D1, D1)
        assert rows[0].market_cap_usd == 105.0
        assert rows[0].source == "snapshot"

    def test_backfill_overwrites_backfill(self):
        db = _make_db()
        db.save_caps([_row(D1, "NVDA", 100.0, "backfill")])
        db.save_caps([_row(D1, "NVDA", 101.0, "backfill")])
        rows = db.load_caps(D1, D1)
        assert rows[0].market_cap_usd == 101.0
