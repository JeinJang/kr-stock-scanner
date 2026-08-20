from datetime import date

import pandas as pd
import pytest

from src.price_history.adjust import AdjustEvent
from src.price_history.backfill import (
    backfill, business_days, rebuild_adjustments, refetch, sync,
)
from src.price_history.db import PriceDB
from src.price_history.fetcher import KrxApiError


def _db(tmp_path):
    return PriceDB(path=str(tmp_path / "prices.db"))


class FakeResp:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
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


class FakeKrxClient:
    """get_all_market_ohlcv만 흉내내는 로그인 클라이언트 더블."""

    def __init__(self, df=None, raises=False):
        self._df = df if df is not None else pd.DataFrame()
        self._raises = raises
        self.calls: list[str] = []

    def get_all_market_ohlcv(self, date_str):
        self.calls.append(date_str)
        if self._raises:
            raise RuntimeError("로그인 클라이언트 실패(테스트)")
        return self._df


def _same_day_df():
    return pd.DataFrame(
        {"시장": ["KOSPI", "KOSDAQ"], "고가": [110, 55], "종가": [108, 54], "전일대비": [3, 1]},
        index=pd.Index(["005930", "035720"], name="티커"),
    )


def test_sync_fills_today_from_login_client_when_openapi_has_nothing(tmp_path):
    """오픈 API가 당일(20260819)에 0건을 반환해도, 로그인 클라이언트로 채운다."""
    db = _db(tmp_path)
    db.save_day("20260818", "KOSPI", [("005930", 100, 100, 0)])
    db.save_day("20260818", "KOSDAQ", [("035720", 50, 50, 0)])
    calls = []
    fake_client = FakeKrxClient(_same_day_df())

    res = sync(db, "KEY", today=date(2026, 8, 19), workers=2,
               _get=_maker({}, calls), krx_client=fake_client)

    assert fake_client.calls == ["20260819"]        # 시장별로 따로 부르지 않고 한 번만 호출
    assert res["same_day_rows"] == 2
    assert res["rows"] >= 2
    assert "20260819" in db.loaded_dates("KOSPI")
    assert "20260819" in db.loaded_dates("KOSDAQ")


def test_sync_does_not_call_login_client_when_today_already_loaded(tmp_path):
    """두 시장 모두 당일이 이미 저장돼 있으면(=오픈 API가 이미 채웠으면) 로그인 클라이언트는 호출되지 않는다."""
    db = _db(tmp_path)
    db.save_day("20260818", "KOSPI", [("005930", 100, 100, 0)])
    db.save_day("20260818", "KOSDAQ", [("035720", 50, 50, 0)])
    db.save_day("20260819", "KOSPI", [("005930", 120, 115, 15)])
    db.save_day("20260819", "KOSDAQ", [("035720", 60, 58, 2)])
    calls = []
    fake_client = FakeKrxClient(_same_day_df())

    res = sync(db, "KEY", today=date(2026, 8, 19), workers=2,
               _get=_maker({}, calls), krx_client=fake_client)

    assert fake_client.calls == []
    assert res["same_day_rows"] == 0


def test_sync_survives_login_client_error(tmp_path):
    """로그인 클라이언트가 예외를 던져도 sync는 정상 반환한다(KrxApiError와 동일한 취급)."""
    db = _db(tmp_path)
    db.save_day("20260818", "KOSPI", [("005930", 100, 100, 0)])
    db.save_day("20260818", "KOSDAQ", [("035720", 50, 50, 0)])
    calls = []
    fake_client = FakeKrxClient(raises=True)

    res = sync(db, "KEY", today=date(2026, 8, 19), workers=2,
               _get=_maker({}, calls), krx_client=fake_client)

    assert res["same_day_rows"] == 0
    assert "20260819" not in db.loaded_dates("KOSPI")


def test_sync_survives_login_client_frame_missing_price_column(tmp_path):
    """가격 컬럼이 빠진 프레임이 와도 sync는 정상 반환한다.

    get_all_market_ohlcv는 ISU_SRT_CD·MKT_NM만 필수라 KRX가 TDD_HGPRC를
    개명하면 고가 없는 프레임이 온다. 그 KeyError가 sync 밖으로 새면
    cli의 _sync_price_store_or_warn(KrxApiError만 포착)을 통과해 run이
    통째로 죽고 리포트가 나가지 않는다.
    """
    db = _db(tmp_path)
    db.save_day("20260818", "KOSPI", [("005930", 100, 100, 0)])
    db.save_day("20260818", "KOSDAQ", [("035720", 50, 50, 0)])
    no_high = pd.DataFrame(
        {"시장": ["KOSPI", "KOSDAQ"], "종가": [108, 54], "전일대비": [3, 1]},
        index=pd.Index(["005930", "035720"], name="티커"),
    )
    fake_client = FakeKrxClient(no_high)

    res = sync(db, "KEY", today=date(2026, 8, 19), workers=2,
               _get=_maker({}, []), krx_client=fake_client)

    assert res["same_day_rows"] == 0
    assert "20260819" not in db.loaded_dates("KOSPI")
    assert "20260819" not in db.loaded_dates("KOSDAQ")


def test_refetch_removes_and_readds_date(tmp_path):
    db = _db(tmp_path)
    db.save_day("20260819", "KOSPI", [("005930", 100, 90, -5)])
    db.save_day("20260819", "KOSDAQ", [("035720", 50, 48, -1)])
    calls = []
    rows = {"20260819": [("005930", 120, 115, 15), ("035720", 60, 58, 2)]}

    res = refetch(db, "KEY", "20260819", workers=2, _get=_maker(rows, calls))

    assert calls.count("20260819") == 2      # KOSPI, KOSDAQ 각각 재요청
    assert res["deleted"] == 2
    got = dict(db.con.execute(
        "SELECT ticker, high FROM daily_px WHERE d='20260819'"
    ).fetchall())
    assert got == {"005930": 120, "035720": 60}
    assert res["ok"] is True


def _existing_day(tmp_path):
    """20260819에 두 시장 한 행씩 들어 있는 저장소."""
    db = _db(tmp_path)
    db.save_day("20260819", "KOSPI", [("005930", 100, 90, -5)])
    db.save_day("20260819", "KOSDAQ", [("035720", 50, 48, -1)])
    return db


def _stored(db, d="20260819"):
    return dict(db.con.execute(
        "SELECT ticker, high FROM daily_px WHERE d=?", (d,)
    ).fetchall())


def test_refetch_keeps_existing_rows_when_reload_is_empty(tmp_path):
    """두 시장 모두 0건이면 기존 행을 지우지 않는다.

    지우고 나서 받는 순서였을 때는, 오픈 API가 아직 당일을 주지 않는
    시각에 refetch를 돌리면 그 날짜의 유일한 사본이 사라졌다.
    """
    db = _existing_day(tmp_path)

    res = refetch(db, "KEY", "20260819", workers=2, _get=_maker({}, []))

    assert res["ok"] is False
    assert res["rows"] == 0
    assert res["deleted"] == 0
    assert _stored(db) == {"005930": 100, "035720": 50}


def test_refetch_keeps_existing_rows_when_fetch_fails(tmp_path):
    """조회가 KrxApiError로 죽어도 기존 행은 그대로다(아직 아무것도 쓰지 않았다)."""
    db = _existing_day(tmp_path)

    def failing_get(url, params, headers, timeout):
        return FakeResp({}, status_code=401)

    with pytest.raises(KrxApiError):
        refetch(db, "KEY", "20260819", workers=2, _get=failing_get)

    assert _stored(db) == {"005930": 100, "035720": 50}


def test_refetch_keeps_existing_rows_when_only_one_market_fails(tmp_path):
    """한 시장만 죽어도 부분 저장은 없다 — 전부 받은 뒤에 한꺼번에 쓴다."""
    db = _existing_day(tmp_path)
    good = _maker({"20260819": [("005930", 120, 115, 15)]}, [])

    def half_failing_get(url, params, headers, timeout):
        if url.endswith("ksq_bydd_trd"):
            return FakeResp({}, status_code=401)
        return good(url, params, headers, timeout)

    with pytest.raises(KrxApiError):
        refetch(db, "KEY", "20260819", workers=1, _get=half_failing_get)

    assert _stored(db) == {"005930": 100, "035720": 50}


def test_rebuild_adjustments_persists_events(tmp_path):
    db = _db(tmp_path)
    db.save_day("20260105", "KOSPI", [("005930", 0, 2_650_000, 0)])
    db.save_day("20260106", "KOSPI", [("005930", 53_900, 51_900, -1_100)])
    n = rebuild_adjustments(db)
    assert n == 1
    evs = db.load_events("005930")
    assert len(evs) == 1 and evs[0].factor == 50.0
