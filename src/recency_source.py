"""신고가 종목에 돌파 신선도를 채운다.

가격 이력은 로컬 저장소(data/prices.db)에서 읽는다. KRX에 종목별로
조회하던 경로는 제거됐다 — 그쪽은 2년 조회 상한이 있고 종목당 6콜이 들었다.
계산 자체는 src.breakout_recency의 순수 함수가 담당한다.
"""
from __future__ import annotations

from datetime import date

from loguru import logger

from src.breakout_recency import compute_recency
from src.models import StockHigh
from src.price_history.loader import load_bars


def enrich_highs(
    highs: list[StockHigh],
    as_of: date,
    db=None,
    window_days: int = 365,
) -> None:
    """highs 각 종목의 돌파 신선도를 계산해 제자리에서 채운다.

    이력을 못 읽은 종목은 지표를 None으로 남기고 기존 값을 보존한다.
    """
    if db is None:
        from src.price_history.db import PriceDB
        db = PriceDB()

    filled = 0
    for stock in highs:
        try:
            bars = load_bars(db, stock.ticker, as_of)
        except Exception as e:  # noqa: BLE001 — 개별 종목 실패는 나머지를 막지 않는다
            logger.warning(f"{stock.ticker} 신선도 계산 실패: {type(e).__name__}: {e}")
            continue

        if not bars:
            continue

        recency = compute_recency(bars, window_days=window_days)
        if recency is None:
            continue

        # 저장소의 마지막 봉이 오늘 것이 아니면 종가보다 낮은 고가가 나온다.
        # 실제 장중 고가는 같은 날 종가보다 낮을 수 없다.
        if recency.today_high < stock.close_price:
            logger.warning(
                f"{stock.name}({stock.ticker}) 최신 봉이 오늘 것이 아님 — "
                f"마지막 봉 {bars[-1].date} 고가 {recency.today_high:,.0f} < "
                f"종가 {stock.close_price:,.0f}, 신선도 계산 생략"
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

        if recency.days_since_price_above is not None and recency.days_since_price_above < 365:
            logger.warning(
                f"{stock.name}({stock.ticker}) B={recency.days_since_price_above}일 — "
                f"52주 신고가와 불일치(액면병합 등 investing 미반영 가능성)"
            )
        filled += 1

    logger.info(f"돌파 신선도 산출: {filled}/{len(highs)}종목")
