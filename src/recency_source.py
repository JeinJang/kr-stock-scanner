"""KRX 일봉 이력 취득 어댑터 — 돌파 신선도 계산의 입력을 만든다.

계산 자체는 src.breakout_recency의 순수 함수가 담당한다. 이 모듈은
'어디서 어떻게 가져오는가'만 안다.
"""
from __future__ import annotations

from datetime import date, timedelta

from loguru import logger

from src.breakout_recency import Bar
from src.krx_login_client import KrxBlockedError


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
    max_calls: int = 4,
) -> list[Bar] | None:
    """as_of 기준 years년치 수정주가 일봉을 가져온다.

    한 번에 다 오면 1콜로 끝난다. 응답이 잘리면 반환된 첫 거래일 직전까지
    역방향으로 다시 요청한다. 빈 응답이 오면 그 지점을 상장 시점으로 보고 종료한다.
    supports_history가 False인 클라이언트에서는 None.
    """
    if not getattr(client, "supports_history", False):
        return None

    start = as_of - timedelta(days=int(365.25 * years))
    bars: list[Bar] = []
    cursor_end = as_of

    for _ in range(max_calls):
        if cursor_end < start:
            break
        try:
            df = client.get_market_ohlcv_by_date(
                start.strftime("%Y%m%d"), cursor_end.strftime("%Y%m%d"),
                ticker, adjusted=True,
            )
        except KrxBlockedError:
            raise
        except Exception as e:  # noqa: BLE001 — 개별 종목 실패는 스캔을 막지 않는다
            logger.warning(f"{ticker} 일봉 조회 실패: {type(e).__name__}: {e}")
            return bars or None

        chunk = _to_bars(df)
        if not chunk:
            break

        bars = chunk + bars
        # 요청 시작일 근처까지 왔으면 완료 (거래일 공백 감안해 7일 여유)
        if chunk[0].date <= start + timedelta(days=7):
            break
        cursor_end = chunk[0].date - timedelta(days=1)

    return bars or None
