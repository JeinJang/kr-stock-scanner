"""일봉 적재 오케스트레이션 — 초기 백필과 일일 증분.

초기 백필은 약 5,400 요청(11년 x 2시장)으로 실측 16분이 걸린다. 이미
적재된 날짜를 건너뛰므로 중단돼도 다시 실행하면 이어받는다.
"""
from __future__ import annotations

from datetime import date, timedelta

from loguru import logger

from src.price_history.adjust import detect_adjustments
from src.price_history.fetcher import MARKET_ENDPOINTS, fetch_many


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


def sync(
    db, api_key: str, workers: int = 8,
    today: date | None = None, _get=None,
) -> dict:
    """마지막 적재일 다음날부터 오늘까지 채운다.

    저장소가 비어 있으면 아무것도 받지 않는다 — 16분짜리 백필을 자동으로
    시작하면 안 된다. 경고만 남기고 사용자가 'prices backfill'을 실행하게 한다.
    """
    end = today or date.today()
    last = db.last_loaded_date()
    if last is None:
        logger.warning(
            "일봉 저장소가 비어 있습니다. 'python -m src.cli prices backfill'을 "
            "먼저 실행하세요(약 16분). 돌파 신선도는 이번 실행에서 생략됩니다."
        )
        return {"requested": 0, "loaded_days": 0, "rows": 0, "skipped": 0}

    start = date(int(last[:4]), int(last[4:6]), int(last[6:8])) + timedelta(days=1)
    if start > end:
        return {"requested": 0, "loaded_days": 0, "rows": 0, "skipped": 0}

    days = business_days(start, end)
    res = _load(db, api_key, days, workers, _get)
    if res["rows"]:
        rebuild_adjustments(db)
    return res


def rebuild_adjustments(db, tickers: list[str] | None = None, since: str = "19900101") -> int:
    """티커별로 수정 이벤트를 다시 계산해 저장한다. 반환은 총 이벤트 수."""
    total = 0
    for ticker in (tickers if tickers is not None else db.tickers()):
        rows = db.load_rows(ticker, since=since)
        events = detect_adjustments(rows)
        db.save_events(ticker, events)
        total += len(events)
    return total
