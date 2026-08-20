"""52주 신고가의 '돌파 신선도' 계산 — 순수 함수.

네트워크·DB에 의존하지 않는다. 입력은 일봉 리스트, 출력은 지표 값뿐이다.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Bar:
    """하루치 일봉 — 이 계산에 필요한 것은 날짜와 고가뿐이다."""

    date: date
    high: float


@dataclass(frozen=True)
class Recency:
    days_since_prev_new_high: int | None   # A: 직전 52주 신고가일로부터의 일수
    days_since_price_above: int | None     # B: 오늘 고가 이상이었던 마지막 날로부터의 일수
    history_span_days: int                 # 확보된 이력 길이
    prev_high_52w: float                   # 직전 window 봉의 최고 고가 (0.0 = 봉 부족)
    today_high: float


def _prior_max_by_calendar(bars: list[Bar], window_days: int) -> list[float | None]:
    """각 봉 시점에서 '직전 window_days 구간(당일 제외)'의 최고 고가.

    창 전체를 덮는 데이터가 없으면 None. 단조 덱으로 O(n)에 구한다 —
    후보일마다 창을 다시 훑으면 O(n^2)이 되어 11년치에서 수 초가 걸린다.
    """
    out: list[float | None] = [None] * len(bars)
    if not bars:
        return out
    first = bars[0].date
    dq: deque[tuple] = deque()          # (date, high), high 내림차순
    for j, bar in enumerate(bars):
        win_start = bar.date - timedelta(days=window_days)
        while dq and dq[0][0] < win_start:
            dq.popleft()
        if first <= win_start and dq:
            out[j] = dq[0][1]
        # 당일은 자기 창에서 제외되므로 값을 읽은 뒤에 넣는다.
        while dq and dq[-1][1] <= bar.high:
            dq.pop()
        dq.append((bar.date, bar.high))
    return out


def compute_recency(bars: list[Bar], window_days: int = 365) -> Recency | None:
    """bars(날짜 오름차순, 마지막이 오늘)에서 A·B를 계산한다."""
    if len(bars) < 2:
        return None

    today = bars[-1]
    past = bars[:-1]

    # B: 오늘 고가 이상이었던 가장 최근 과거일. 창을 쓰지 않고 이력 전체를 본다.
    days_since_price_above: int | None = None
    for bar in reversed(past):
        if bar.high >= today.high:
            days_since_price_above = (today.date - bar.date).days
            break

    prior_max = _prior_max_by_calendar(bars, window_days)

    # A: 그날 자체가 52주 신고가였던 가장 최근 과거일.
    days_since_prev_new_high: int | None = None
    for j in range(len(past) - 1, -1, -1):
        pm = prior_max[j]
        if pm is not None and bars[j].high >= pm:
            days_since_prev_new_high = (today.date - bars[j].date).days
            break

    # 오늘 시점의 창 최고가가 곧 직전 52주 고점이다.
    prev_high_52w = prior_max[-1] if prior_max[-1] is not None else 0.0

    return Recency(
        days_since_prev_new_high=days_since_prev_new_high,
        days_since_price_above=days_since_price_above,
        history_span_days=(today.date - bars[0].date).days,
        prev_high_52w=prev_high_52w,
        today_high=today.high,
    )


MISMATCH_MAX_DAYS = 365


def is_history_mismatch(days_since_price_above: int | None) -> bool:
    """우리 이력이 52주 신고가를 반박하는가.

    B가 1년 미만이면 그 가격보다 높았던 날이 52주 안에 있다는 뜻이다.
    액면병합 등을 investing이 소급 반영하지 않아 생기는 거짓 신고가를 잡는다.
    """
    return days_since_price_above is not None and days_since_price_above < MISMATCH_MAX_DAYS
