"""일봉 적재 오케스트레이션 — 초기 백필과 일일 증분.

초기 백필은 5,740 요청(11년 x 2시장)으로 실측 12.1분이 걸린다(742MB). 이미
적재된 날짜를 건너뛰므로 중단돼도 다시 실행하면 이어받는다.
"""
from __future__ import annotations

from datetime import date, timedelta

from loguru import logger

from src.price_history.adjust import detect_adjustments
from src.price_history.fetcher import MARKET_ENDPOINTS, fetch_many

# sync()가 시장별 최신 적재일 중 가장 이른 날짜에서 며칠을 더 거슬러 올라가
# 재요청할지. 한 시장만 저장하고 죽은 실행 뒤에도 다음 sync가 빠진 (market,
# date)를 다시 후보에 올리게 한다. _load가 이미 있는 (market,date)는
# 걸러내므로 이 창을 두어도 요청 수는 늘지 않는다.
SYNC_SAFETY_DAYS = 10


def _to_date(s: str) -> date:
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def business_days(start: date, end: date) -> list[str]:
    """start~end(양끝 포함)의 주말 제외 날짜. 공휴일은 빈 응답으로 걸러진다."""
    out: list[str] = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return out


def _load(db, api_key: str, days: list[str], workers: int, _get) -> dict:
    """이미 적재된 (market, d)는 건너뛰고 나머지를 받아 저장한다."""
    jobs: list[tuple[str, str]] = []
    skipped = 0
    for market in MARKET_ENDPOINTS:
        done = db.loaded_dates(market)
        for d in days:
            if d in done:
                skipped += 1
            else:
                jobs.append((market, d))

    rows_total = 0
    loaded_days = 0
    if jobs:
        for i, (market, d, rows) in enumerate(fetch_many(api_key, jobs, workers, _get), 1):
            if rows:
                rows_total += db.save_day(d, market, rows)
                loaded_days += 1
            if i % 200 == 0:
                logger.info(f"일봉 적재 {i}/{len(jobs)} 요청, {rows_total:,}행")

    return {
        "requested": len(jobs),
        "loaded_days": loaded_days,
        "rows": rows_total,
        "skipped": skipped,
    }


def backfill(
    db, api_key: str, years: int = 11, workers: int = 8,
    today: date | None = None, _get=None,
) -> dict:
    """years년치를 적재한다. 이미 있는 날짜는 건너뛴다(재개 가능)."""
    end = today or date.today()
    start = end - timedelta(days=int(365.25 * years))
    days = business_days(start, end)
    logger.info(f"일봉 백필 시작: {start} ~ {end} ({len(days)} 영업일 x {len(MARKET_ENDPOINTS)} 시장)")
    res = _load(db, api_key, days, workers, _get)
    db.set_meta("backfill_years", str(years))
    rebuilt = rebuild_adjustments(db)
    res["adjust_events"] = rebuilt
    logger.info(
        f"일봉 백필 완료: {res['rows']:,}행 적재, {res['skipped']}건 건너뜀, "
        f"수정 이벤트 {rebuilt}건"
    )
    return res


def _fill_same_day(db, krx_client, end: date) -> int:
    """오픈 API가 아직 당일 데이터를 주지 않을 때 로그인 클라이언트로 채운다.

    실측: 장 마감(15:30) 후 15:42·16:53·17:46에도 오픈 API의 모든 엔드포인트가
    당일에 0건을 반환했다. 반면 로그인 클라이언트(data.krx.co.kr)는 마감
    시점에 이미 당일 데이터를 갖고 있어, mktId=ALL 통합 조회 한 번으로
    두 시장 모두를 대신 채운다. 두 시장 모두 이미 당일이 있으면 호출조차
    하지 않는다.

    조회부터 저장까지 전부를 하나의 try로 감싼다. get_all_market_ohlcv는
    가격 컬럼이 없어도 프레임을 돌려주므로(ISU_SRT_CD·MKT_NM만 필수),
    KRX가 컬럼명을 바꾸면 변환 단계에서 KeyError가 난다. 그 예외가 sync
    밖으로 새면 cli의 _sync_price_store_or_warn이 KrxApiError만 잡는 탓에
    run 전체가 중단돼 리포트가 나가지 않는다. 당일 보완 실패는 경고만
    남기고 sync는 정상 반환해야 한다.
    """
    if krx_client is None:
        return 0
    today_str = end.strftime("%Y%m%d")
    missing = [m for m in MARKET_ENDPOINTS if today_str not in db.loaded_dates(m)]
    if not missing:
        return 0

    try:
        return _do_fill_same_day(db, krx_client, today_str, missing)
    except Exception as e:
        logger.warning(f"로그인 클라이언트 당일({today_str}) 채우기 실패, 생략: {e}")
        return 0


def _do_fill_same_day(db, krx_client, today_str: str, missing: list[str]) -> int:
    """조회·변환·저장 본체. 예외는 호출자(_fill_same_day)가 삼킨다."""
    df = krx_client.get_all_market_ohlcv(today_str)
    if df is None or df.empty:
        return 0

    total = 0
    for market in missing:
        sub = df[df["시장"] == market]
        if sub.empty:
            continue
        # 정지 종목(고가 0)도 그대로 저장 — loader가 걸러내고, 보정 계산이
        # 인접 정지행을 필요로 한다.
        records = [
            (ticker, row["고가"], row["종가"], row["전일대비"])
            for ticker, row in sub.iterrows()
        ]
        saved = db.save_day(today_str, market, records)
        total += saved
        logger.info(
            f"{market} {today_str} 당일 {saved:,}행을 로그인 클라이언트로 채움 "
            f"(오픈 API가 당일 데이터 미제공)"
        )
    return total


def sync(
    db, api_key: str, workers: int = 8,
    today: date | None = None, _get=None, *, krx_client=None,
) -> dict:
    """시장별 최신 적재일 중 가장 이른 날짜 - SYNC_SAFETY_DAYS부터 오늘까지 채운다.

    저장소가 비어 있거나 시장 중 하나라도 적재 이력이 없으면 아무것도 받지
    않는다 — 12분짜리 백필을 자동으로 시작하면 안 된다. 경고만 남기고
    사용자가 'prices backfill'을 실행하게 한다.

    시작일을 두 시장 통합 MAX(d)가 아니라 시장별 최신일 중 최솟값에서
    잡는 이유: 한 시장만 저장하고 죽은 실행 뒤에는 통합 MAX(d)가 실제로는
    비어 있는 시장의 그 날짜를 가려버린다. SYNC_SAFETY_DAYS만큼 더 거슬러
    올라가 재요청해도 _load가 이미 적재된 (market,date)는 건너뛰므로 요청
    수는 늘지 않는다.

    krx_client를 주면, 오픈 API 적재 뒤에도 당일이 두 시장 중 하나라도
    빠져 있을 때 로그인 클라이언트로 보완한다(_fill_same_day). 실패해도
    KrxApiError처럼 sync 자체는 정상 반환한다.
    """
    end = today or date.today()
    lasts = [db.last_loaded_date(m) for m in MARKET_ENDPOINTS]
    if any(x is None for x in lasts):
        logger.warning(
            "일봉 저장소가 비어 있거나 일부 시장 데이터가 없습니다. "
            "'python -m src.cli prices backfill'을 먼저 실행하세요(약 12분). "
            "돌파 신선도는 이번 실행에서 생략됩니다."
        )
        return {"requested": 0, "loaded_days": 0, "rows": 0, "skipped": 0, "same_day_rows": 0}

    oldest = min(lasts)
    start = _to_date(oldest) + timedelta(days=1) - timedelta(days=SYNC_SAFETY_DAYS)
    if start > end:
        return {"requested": 0, "loaded_days": 0, "rows": 0, "skipped": 0, "same_day_rows": 0}

    days = business_days(start, end)
    res = _load(db, api_key, days, workers, _get)

    same_day_rows = _fill_same_day(db, krx_client, end)
    res["rows"] += same_day_rows
    res["same_day_rows"] = same_day_rows

    if res["rows"]:
        # 새로 받은 날짜 각각이 창 안에서 비교 대상(전일 행)을 갖도록
        # SYNC_SAFETY_DAYS보다 넉넉히 40일을 더 거슬러 올라가 재계산한다.
        # 연휴 등으로 거래일 간격이 길게 벌어질 수 있어서다. 로그인
        # 클라이언트로 채운 당일행도 이 창에 포함되므로, 오늘 발효되는
        # 기업행위도 오늘 안에 검출된다.
        window_start = (start - timedelta(days=40)).strftime("%Y%m%d")
        rebuild_adjustments(db, since=window_start, replace=False)
    return res


def refetch(
    db, api_key: str, date_str: str, workers: int = 8, _get=None,
) -> dict:
    """date_str의 저장된 행을 지우고 오픈 API에서 두 시장 다시 받는다.

    적재된 하루치가 의심스러울 때(부분 실패, 이상치 등) 쓴다. delete_date로
    지운 뒤에는 _load가 그 (market,date)를 '미적재'로 보고 다시 요청한다.
    """
    deleted = db.delete_date(date_str)
    res = _load(db, api_key, [date_str], workers, _get)
    if res["rows"]:
        d = _to_date(date_str)
        window_start = (d - timedelta(days=40)).strftime("%Y%m%d")
        rebuild_adjustments(db, since=window_start, replace=False)
    res["deleted"] = deleted
    return res


def rebuild_adjustments(
    db, tickers: list[str] | None = None, since: str = "19900101",
    replace: bool = True,
) -> int:
    """티커별로 수정 이벤트를 다시 계산해 저장한다. 반환은 총 이벤트 수.

    replace=True(기본, 백필 경로)는 티커의 이벤트를 통째로 교체한다.
    replace=False(일일 동기화 경로)는 since 이후만 재계산해 병합하고
    since 이전의 기존 이벤트는 그대로 둔다.
    """
    total = 0
    save = db.save_events if replace else db.add_events
    for ticker in (tickers if tickers is not None else db.tickers()):
        rows = db.load_rows(ticker, since=since)
        events = detect_adjustments(rows)
        save(ticker, events)
        total += len(events)
    return total
