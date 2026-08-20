from datetime import date

from src.price_history.adjust import AdjustEvent
from src.price_history.backfill import (
    backfill, business_days, rebuild_adjustments, sync,
)
from src.price_history.db import PriceDB


def _db(tmp_path):
    return PriceDB(path=str(tmp_path / "prices.db"))


class FakeResp:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload


def _maker(rows_by_date, calls):
    """rows_by_date: {YYYYMMDD: [(ticker,high,close,chg)]}. 없는 날짜는 휴장."""
    def fake_get(url, params, headers, timeout):
        d = params["basDd"]
        calls.append(d)
        items = [
            {"ISU_CD": tk, "TDD_HGPRC": h, "TDD_CLSPRC": c, "CMPPREVDD_PRC": ch}
            for tk, h, c, ch in rows_by_date.get(d, [])
        ]
        return FakeResp({"OutBlock_1": items})
    return fake_get


def test_business_days_excludes_weekends():
    out = business_days(date(2026, 8, 14), date(2026, 8, 18))  # 금~화
    assert out == ["20260814", "20260817", "20260818"]


def test_backfill_loads_and_reports(tmp_path):
    db = _db(tmp_path)
    calls = []
    rows = {"20260818": [("005930", 110, 100, 0)], "20260819": [("005930", 120, 115, 15)]}
    res = backfill(db, "KEY", years=1, today=date(2026, 8, 19),
                   workers=2, _get=_maker(rows, calls))
    assert res["rows"] > 0
    assert db.last_loaded_date() == "20260819"


def test_backfill_skips_already_loaded_dates(tmp_path):
    db = _db(tmp_path)
    db.save_day("20260819", "KOSPI", [("005930", 1, 1, 0)])
    db.save_day("20260819", "KOSDAQ", [("035720", 1, 1, 0)])
    calls = []
    backfill(db, "KEY", years=1, today=date(2026, 8, 19), workers=2,
             _get=_maker({}, calls))
    assert "20260819" not in calls          # 이미 적재된 날짜는 요청하지 않는다


def test_sync_without_prior_data_requests_nothing(tmp_path):
    db = _db(tmp_path)
    calls = []
    res = sync(db, "KEY", today=date(2026, 8, 19), workers=2, _get=_maker({}, calls))
    assert res["requested"] == 0
    assert calls == []


def test_sync_fills_gap_since_last_loaded(tmp_path):
    db = _db(tmp_path)
    # 두 시장 모두 과거 데이터가 있어야 sync가 동작한다 — 한 시장에 이력이
    # 아예 없으면 sync가 아니라 backfill이 필요한 상태로 본다.
    db.save_day("20260814", "KOSPI", [("005930", 100, 100, 0)])
    db.save_day("20260814", "KOSDAQ", [("035720", 50, 50, 0)])
    calls = []
    rows = {"20260818": [("005930", 110, 105, 5)], "20260819": [("005930", 120, 115, 10)]}
    sync(db, "KEY", today=date(2026, 8, 19), workers=2, _get=_maker(rows, calls))
    assert "20260817" in calls and "20260819" in calls
    assert "20260814" not in calls          # 이미 있는 날짜는 다시 받지 않는다
    assert db.last_loaded_date() == "20260819"


def test_sync_resumes_missing_market_day(tmp_path):
    """두 시장 모두 이력이 있지만, 중간에 죽은 실행 탓에 한 시장만 특정
    날짜가 빠진 경우 — 통합 MAX(d)는 그 구멍을 가리지만, sync는 시장별
    최신일 중 최솟값 기준으로 그 시장의 그 날짜를 다시 요청해야 한다."""
    db = _db(tmp_path)
    db.save_day("20260812", "KOSPI", [("005930", 100, 100, 0)])
    db.save_day("20260813", "KOSPI", [("005930", 100, 100, 0)])
    db.save_day("20260814", "KOSPI", [("005930", 100, 100, 0)])
    db.save_day("20260812", "KOSDAQ", [("035720", 50, 50, 0)])
    db.save_day("20260813", "KOSDAQ", [("035720", 50, 50, 0)])
    # KOSDAQ의 20260814만 빠졌다. 통합 last_loaded_date()는 20260814를
    # 가리켜(KOSPI 덕분에) 이 구멍을 가리지만, 시장별 최솟값은 KOSDAQ의
    # 20260813이므로 sync는 20260814를 다시 후보에 올려야 한다.
    calls = []
    rows = {"20260814": [("035720", 55, 52, 2)]}
    sync(db, "KEY", today=date(2026, 8, 14), workers=2, _get=_maker(rows, calls))
    # KOSDAQ의 20260814는 다시 요청되지만, 이미 적재된 KOSPI의 20260814는
    # 건너뛴다 — 같은 날짜라도 시장별로 한 번만 요청됨을 확인한다.
    assert calls.count("20260814") == 1
    assert "20260814" in db.loaded_dates("KOSDAQ")


def test_sync_bounded_rebuild_preserves_older_events(tmp_path):
    """sync의 이벤트 재계산은 좁은 창만 훑는다 — 창 밖의 과거 이벤트가
    지워지면 안 된다(add_events는 지우지 않고 병합하기 때문)."""
    db = _db(tmp_path)
    db.save_events("005930", [AdjustEvent(d=date(2026, 1, 6), factor=50.0)])
    db.save_day("20260812", "KOSPI", [("005930", 100, 100, 0)])
    db.save_day("20260812", "KOSDAQ", [("035720", 50, 50, 0)])
    calls = []
    rows = {"20260813": [("005930", 105, 103, 3), ("035720", 51, 51, 1)]}
    sync(db, "KEY", today=date(2026, 8, 13), workers=2, _get=_maker(rows, calls))
    evs = db.load_events("005930")
    assert any(e.d == date(2026, 1, 6) and e.factor == 50.0 for e in evs)


def test_rebuild_adjustments_persists_events(tmp_path):
    db = _db(tmp_path)
    db.save_day("20260105", "KOSPI", [("005930", 0, 2_650_000, 0)])
    db.save_day("20260106", "KOSPI", [("005930", 53_900, 51_900, -1_100)])
    n = rebuild_adjustments(db)
    assert n == 1
    evs = db.load_events("005930")
    assert len(evs) == 1 and evs[0].factor == 50.0
