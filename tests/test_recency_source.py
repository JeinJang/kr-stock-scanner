from datetime import date, timedelta

from src.models import StockHigh
from src.price_history.db import PriceDB
from src import recency_source


def _stock(ticker="005930", close=100.0):
    return StockHigh(
        ticker=ticker, name="테스트", market="KOSPI", sector="전기전자",
        close_price=close, high_52w=close, prev_high_52w=0.0,
        breakout_pct=0.0, volume=1, avg_volume_20d=0, change_pct=2.0,
    )


def _seeded_db(tmp_path, ticker="005930", days=400, high=100, last_high=110):
    """days일치 평탄한 이력 + 마지막 날 돌파."""
    db = PriceDB(path=str(tmp_path / "prices.db"))
    end = date(2026, 8, 19)
    for i in range(days):
        d = (end - timedelta(days=days - 1 - i)).strftime("%Y%m%d")
        h = last_high if i == days - 1 else high
        db.save_day(d, "KOSPI", [(ticker, h, h, 0)])
    return db


def test_enrich_fills_metrics_and_normalizes_breakout(tmp_path):
    db = _seeded_db(tmp_path)
    stock = _stock()
    recency_source.enrich_highs([stock], date(2026, 8, 19), db=db)

    assert stock.history_span_days == 399
    assert stock.days_since_price_above is None
    assert stock.days_since_prev_new_high == 1
    assert stock.prev_high_52w == 100.0
    assert stock.breakout_pct == 10.0
    assert stock.high_52w == 110.0


def test_enrich_leaves_stock_untouched_without_history(tmp_path):
    db = PriceDB(path=str(tmp_path / "prices.db"))
    stock = _stock()
    recency_source.enrich_highs([stock], date(2026, 8, 19), db=db)
    assert stock.history_span_days is None
    assert stock.breakout_pct == 0.0
    assert stock.change_pct == 2.0
    assert stock.high_52w == 100.0


def test_enrich_skips_stale_last_bar(tmp_path):
    # 저장소의 마지막 고가가 investing 종가보다 낮으면 그 종목은 건너뛴다
    db = _seeded_db(tmp_path, last_high=110)
    stock = _stock(close=200.0)
    recency_source.enrich_highs([stock], date(2026, 8, 19), db=db)
    assert stock.history_span_days is None


def test_enrich_treats_equal_high_and_close_as_fresh(tmp_path):
    # 상한가처럼 당일 고가와 종가가 같은 경우(경계는 <, 이므로 통과해야 한다)
    db = _seeded_db(tmp_path, last_high=110)
    stock = _stock(close=110.0)
    recency_source.enrich_highs([stock], date(2026, 8, 19), db=db)
    assert stock.history_span_days == 399


def test_enrich_isolates_per_stock_failure(tmp_path, monkeypatch):
    db = _seeded_db(tmp_path, ticker="000002")

    real = recency_source.load_bars

    def flaky(db_, ticker, as_of, **kw):
        if ticker == "000001":
            raise ValueError("boom")
        return real(db_, ticker, as_of, **kw)

    monkeypatch.setattr(recency_source, "load_bars", flaky)
    bad, ok = _stock("000001"), _stock("000002")
    recency_source.enrich_highs([bad, ok], date(2026, 8, 19), db=db)
    assert bad.history_span_days is None
    assert ok.history_span_days == 399


def test_fetch_bars_is_gone():
    """KRX 종목별 조회 경로는 삭제되었다."""
    assert not hasattr(recency_source, "fetch_bars")
    assert not hasattr(recency_source, "CHUNK_DAYS")


def test_enrich_skips_when_last_bar_date_is_not_as_of(tmp_path):
    """마지막 봉이 어제 것이면 가격 가드를 통과해도 건너뛴다(F2)."""
    db = _seeded_db(tmp_path, last_high=110)   # 마지막 봉 2026-08-19
    stock = _stock(close=100.0)                # 종가 < 마지막 봉 고가 -> 가격 가드는 통과
    recency_source.enrich_highs([stock], date(2026, 8, 20), db=db)
    assert stock.history_span_days is None
    assert stock.days_since_price_above is None
    assert stock.days_since_prev_new_high is None
    assert stock.high_52w == 100.0
    assert stock.prev_high_52w == 0.0
    assert stock.breakout_pct == 0.0


def test_enrich_fills_when_last_bar_date_matches_as_of(tmp_path):
    """날짜가 일치하면 종전대로 채운다 — 가드가 전부를 막지는 않는다."""
    db = _seeded_db(tmp_path, last_high=110)
    stock = _stock(close=100.0)
    recency_source.enrich_highs([stock], date(2026, 8, 19), db=db)
    assert stock.history_span_days == 399
    assert stock.high_52w == 110.0


def test_mismatch_warning_uses_shared_rule(tmp_path, monkeypatch):
    """불일치 판정은 breakout_recency.is_history_mismatch 한 곳에서만 나온다(F4)."""
    db = PriceDB(path=str(tmp_path / "prices.db"))
    end = date(2026, 8, 19)
    days = 400
    for i in range(days):
        d = (end - timedelta(days=days - 1 - i)).strftime("%Y%m%d")
        h = 120 if i == days - 30 else (110 if i == days - 1 else 100)
        db.save_day(d, "KOSPI", [("005930", h, h, 0)])

    seen = []

    def spy(b):
        seen.append(b)
        return True

    monkeypatch.setattr(recency_source, "is_history_mismatch", spy)
    stock = _stock(close=100.0)
    recency_source.enrich_highs([stock], end, db=db)

    assert seen == [stock.days_since_price_above]
    assert stock.days_since_price_above == 29
