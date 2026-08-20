"""52주 신고가의 '돌파 신선도' 계산 — 순수 함수.

네트워크·DB에 의존하지 않는다. 입력은 일봉 리스트, 출력은 지표 값뿐이다.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


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


def compute_recency(bars: list[Bar], window: int = 250) -> Recency | None:
    """bars(날짜 오름차순, 마지막이 오늘)에서 A·B를 계산한다."""
    if len(bars) < 2:
        return None

    today = bars[-1]
    past = bars[:-1]

    # B: 오늘 고가 이상이었던 가장 최근 과거일. rolling window가 필요 없으므로
    #    확보된 이력 전체를 본다.
    days_since_price_above: int | None = None
    for bar in reversed(past):
        if bar.high >= today.high:
            days_since_price_above = (today.date - bar.date).days
            break

    # A: 그날 자체가 52주 신고가였던 가장 최근 과거일.
    #    앞쪽 window개 봉은 rolling max 워밍업으로 소비된다.
    days_since_prev_new_high: int | None = None
    for j in range(len(past) - 1, window - 1, -1):
        prior_max = max(b.high for b in past[j - window:j])
        if past[j].high >= prior_max:
            days_since_prev_new_high = (today.date - past[j].date).days
            break

    prev_high_52w = max(b.high for b in past[-window:]) if len(past) >= window else 0.0

    return Recency(
        days_since_prev_new_high=days_since_prev_new_high,
        days_since_price_above=days_since_price_above,
        history_span_days=(today.date - bars[0].date).days,
        prev_high_52w=prev_high_52w,
        today_high=today.high,
    )
