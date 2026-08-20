"""저장소에서 티커의 수정 일봉을 꺼낸다. 네트워크를 타지 않는다."""
from __future__ import annotations

from datetime import date, timedelta

from src.breakout_recency import Bar
from src.price_history.adjust import adjusted_highs


def load_bars(db, ticker: str, as_of: date, years: int = 11) -> list[Bar] | None:
    """as_of 기준 years년치 수정 일봉(날짜 오름차순).

    유효 봉이 2개 미만이면 None. 거래정지일(고가 0)은 제외한다.
    """
    since = (as_of - timedelta(days=int(365.25 * years))).strftime("%Y%m%d")
    rows = [r for r in db.load_rows(ticker, since=since) if r.d <= as_of]
    if len(rows) < 2:
        return None

    events = [e for e in db.load_events(ticker) if e.d <= as_of]
    bars = [
        Bar(date=d, high=high)
        for d, high in adjusted_highs(rows, events)
        if high > 0
    ]
    return bars if len(bars) >= 2 else None
