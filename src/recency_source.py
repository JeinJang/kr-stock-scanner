"""KRX 일봉 이력 취득 어댑터 — 돌파 신선도 계산의 입력을 만든다.

계산 자체는 src.breakout_recency의 순수 함수가 담당한다. 이 모듈은
'어디서 어떻게 가져오는가'만 안다.
"""
from __future__ import annotations

from datetime import date, timedelta
from math import ceil

from loguru import logger

from src.breakout_recency import Bar, compute_recency
from src.krx_login_client import KrxBlockedError
from src.models import StockHigh

# KRX 종목별 단건 조회 상한(2026-08-20 실측). 이 폭을 넘겨 요청하면 오류가
# 아니라 빈 응답(LOGOUT)이 온다 — 그래서 첫 요청부터 이 폭으로 쪼갠다.
CHUNK_DAYS = 730


def _to_bars(df) -> list[Bar]:
    """KRX 응답 DataFrame(날짜 인덱스, '고가' 컬럼) → 날짜 오름차순 Bar 리스트."""
    if df is None or df.empty or "고가" not in df.columns:
        return []
    bars: list[Bar] = []
    for raw_date, row in df.iterrows():
        try:
            d = date(int(str(raw_date)[:4]), int(str(raw_date)[4:6]), int(str(raw_date)[6:8]))
            high = float(row["고가"])
        except (ValueError, TypeError):
            continue
        if high > 0:
            bars.append(Bar(date=d, high=high))
    bars.sort(key=lambda b: b.date)
    return bars


def fetch_bars(
    client,
    ticker: str,
    as_of: date,
    years: int = 11,
    chunk_days: int = CHUNK_DAYS,
    max_calls: int | None = None,
) -> list[Bar] | None:
    """as_of 기준 years년치 수정주가 일봉을 chunk_days 폭으로 나눠 가져온다.

    KRX는 종목당 조회 폭이 chunk_days를 넘으면 잘라주는 게 아니라 빈 응답을
    준다. 그래서 첫 요청부터 [cursor_end - (chunk_days-1), cursor_end] 폭으로
    쪼개 요청하고, 성공할 때마다 커서를 그 구간 시작일 하루 전으로 물려
    역방향으로 이어간다. supports_history가 False인 클라이언트에서는 None.

    빈 응답은 곧바로 상장 시점으로 단정하지 않는다. KrxLoginClient._post는
    HTTP != 200, LOGOUT, JSON 파싱 실패에도 []를 돌려주므로 "그 앞에 데이터가
    없음"과 "일시적 오류"가 구분되지 않는다. 같은 구간을 한 번 더 요청해
    재확인하고, 재요청도 비어 있을 때만 상장 시점으로 본다. 재요청에서
    데이터가 나오면 첫 응답이 헛것이었다는 뜻이므로 경고를 남기고 그 값으로
    계속 진행한다. 재요청도 max_calls 예산을 소비한다.

    완료 판정은 "요청한 구간의 시작일이 실제로 start에 닿았는가"로 본다
    (반환된 데이터의 가장 이른 날짜가 아니라). 요청 기준이라 거래일 공백 같은
    거래소 응답의 우연에 흔들리지 않는다. 호출 수(max_calls) 소진이나 중간
    실패로 완결을 확인하지 못하면 잘린 리스트를 조용히 넘기는 대신 None을
    반환한다 — 하위 계산(history_span_days 등)은 "이력 전체 확보"를 전제하므로,
    잘린 리스트를 넘기면 "상장 이후 최고" 같은 판정이 사용자에게 틀린 확신으로
    노출된다. 대신 로그로 소진 사실을 남겨 운영자가 확인할 수 있게 한다.

    max_calls를 넘기지 않으면 필요 청크 수(ceil((as_of-start)/chunk_days)) +
    재확인 재요청 여유분(2)으로 자동 계산한다. 하드코딩하지 않는 이유는
    chunk_days·years가 바뀌어도 예산이 맞게 따라오게 하기 위함이다 — 예산을
    고정값으로 박아두는 것이 이번 결함의 원인이었다. (호출 수 자체를 함부로
    늘리지는 않는다 — 종목당 호출 수를 늘리는 것은 과거 거래소 IP 차단을
    유발한 바로 그 위험이다.)
    """
    if not getattr(client, "supports_history", False):
        return None

    start = as_of - timedelta(days=int(365.25 * years))
    if max_calls is None:
        chunks_needed = ceil((as_of - start).days / chunk_days)
        max_calls = chunks_needed + 2   # 빈 응답 재확인 재요청 여유분

    bars: list[Bar] = []
    cursor_end = as_of
    complete = False
    calls_made = 0

    _FAILED = object()   # 조회 자체가 실패한 경우 (빈 응답과 구분)

    def _request(range_start: date, range_end: date):
        nonlocal calls_made
        try:
            df = client.get_market_ohlcv_by_date(
                range_start.strftime("%Y%m%d"), range_end.strftime("%Y%m%d"),
                ticker, adjusted=True,
            )
            calls_made += 1
        except KrxBlockedError:
            raise
        except Exception as e:  # noqa: BLE001 — 개별 종목 실패는 스캔을 막지 않는다
            logger.warning(f"{ticker} 일봉 조회 실패: {type(e).__name__}: {e}")
            return _FAILED
        return _to_bars(df)

    while calls_made < max_calls:
        chunk_start = max(start, cursor_end - timedelta(days=chunk_days - 1))
        chunk = _request(chunk_start, cursor_end)
        if chunk is _FAILED:
            return None

        if not chunk:
            # 빈 응답 = 상장 시점일 수도, 거래소의 일시 오류일 수도 있다.
            # 같은 구간을 한 번 더 물어 확인한다(예산 소진 시에는 확인 불가).
            if calls_made >= max_calls:
                break
            retry = _request(chunk_start, cursor_end)
            if retry is _FAILED:
                return None
            if not retry:
                # 두 번 모두 비었음 → 그 지점을 상장 시점으로 본다.
                complete = True
                break
            logger.warning(
                f"{ticker} 빈 응답 재요청에서 {len(retry)}봉 반환 "
                f"(구간 {chunk_start}~{cursor_end}) — 거래소 일시 오류로 보고 계속 진행"
            )
            chunk = retry

        bars = chunk + bars
        # 요청 구간의 시작일이 목표 시작일에 닿았으면 완료
        if chunk_start <= start:
            complete = True
            break
        cursor_end = chunk_start - timedelta(days=1)

    if not complete:
        earliest = bars[0].date if bars else as_of
        logger.error(
            f"{ticker} 일봉 이력 미완료 — {calls_made}콜 후 중단"
            f"(cap={max_calls}), 도달 최소일 {earliest} (목표 시작일 {start})"
        )
        return None

    return bars or None


def enrich_highs(
    client,
    highs: list[StockHigh],
    as_of: date,
    window: int = 250,
) -> None:
    """highs 각 종목의 돌파 신선도를 계산해 제자리에서 채운다.

    이력을 못 가져온 종목, 그리고 마지막 봉이 오늘 것이 아닌 종목은 지표를
    None으로 남기고 기존 값을 보존한다.
    KrxBlockedError는 전파한다 — 차단 상태에서 추가 요청을 보내면 안 된다.
    """
    filled = 0
    for stock in highs:
        try:
            bars = fetch_bars(client, stock.ticker, as_of)
        except KrxBlockedError:
            logger.error(f"KRX 차단 감지 — 신선도 계산 중단 ({filled}/{len(highs)} 완료)")
            raise
        except Exception as e:  # noqa: BLE001 — 개별 종목 실패는 나머지를 막지 않는다
            logger.warning(f"{stock.ticker} 신선도 계산 실패: {type(e).__name__}: {e}")
            continue

        if not bars:
            continue

        recency = compute_recency(bars, window=window)
        if recency is None:
            continue

        # 마지막 봉이 오늘 것인지 확인. KRX가 장 마감 데이터를 아직 안 올렸으면
        # bars[-1]은 전 거래일이고, 그 고가는 오늘 종가보다 낮을 수 있다.
        # 같은 날 고가가 종가보다 낮을 수는 없으므로 이는 오래된 봉이라는 신호다.
        # bars[-1].date == as_of 같은 엄격한 비교는 주말·소급 실행에서 오탐이므로
        # 쓰지 않는다. 이 경우 지표를 비워 두는 쪽이 정직하다(배지 미표시).
        if recency.today_high < stock.close_price:
            logger.warning(
                f"{stock.name}({stock.ticker}) 최신 봉이 오늘 것이 아님 — "
                f"마지막 봉 {bars[-1].date} 고가 {recency.today_high:,.0f} "
                f"< 종가 {stock.close_price:,.0f}, 신선도 계산 생략"
            )
            continue

        stock.days_since_prev_new_high = recency.days_since_prev_new_high
        stock.days_since_price_above = recency.days_since_price_above
        stock.history_span_days = recency.history_span_days
        stock.high_52w = recency.today_high
        stock.prev_high_52w = recency.prev_high_52w
        if recency.prev_high_52w > 0:
            stock.breakout_pct = round(
                (recency.today_high - recency.prev_high_52w) / recency.prev_high_52w * 100, 2
            )

        # 52주 신고가라면 B는 최소 1년 이상이어야 한다. 아니면 investing 판정과
        # KRX 수정주가가 어긋난 것이므로 기록만 남기고 값은 그대로 쓴다.
        if recency.days_since_price_above is not None and recency.days_since_price_above < 365:
            logger.warning(
                f"{stock.name}({stock.ticker}) B={recency.days_since_price_above}일 — "
                f"52주 신고가와 불일치(수정주가 차이 가능성)"
            )
        filled += 1

    logger.info(f"돌파 신선도 산출: {filled}/{len(highs)}종목")
