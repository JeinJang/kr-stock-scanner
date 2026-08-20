"""수정주가 보정 — 순수 함수.

KRX Open API 일별매매정보는 원주가를 준다. 액면분할·병합·무상증자 등으로
KRX가 기준가를 재설정한 날을 찾아 그 이전 가격을 현재 기준으로 환산한다.

기준가 = 당일 종가 - 당일 전일대비. 이 값이 전일 실제 종가와 어긋나면
그날 기준가가 재설정된 것이다. 현금배당락은 기준가를 조정하지 않으므로
검출되지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# 11년치 6.48M행 실측: 평일 거래일은 편차가 정확히 0 — 부동소수점 잡음
# 여유를 둘 이유가 없다. 0.5~2% 구간(253건, 56%가 12월말 무상증자·주식배당
# 락일)은 실제 이벤트인데 구 임계값 2%로는 놓쳤다. 0.5% 미만(371+22건)은
# 호가 반올림 잡음 — 78%가 같은 종목이 인접일에 같은 편차를 반복하는
# 패턴으로 식별 가능. 그래서 0.5%로 낮춘다.
THRESHOLD = 0.005


@dataclass(frozen=True)
class PxRow:
    """보정 계산에 필요한 하루치 원주가."""

    d: date
    high: float
    close: float
    chg: float


@dataclass(frozen=True)
class AdjustEvent:
    """기준가가 재설정된 날과 그 계수(전일 실제 종가 / 당일 기준가)."""

    d: date
    factor: float


def detect_adjustments(rows: list[PxRow], threshold: float = THRESHOLD) -> list[AdjustEvent]:
    """rows(날짜 오름차순)에서 기준가 재설정 이벤트를 찾는다."""
    events: list[AdjustEvent] = []
    for i in range(1, len(rows)):
        base = rows[i].close - rows[i].chg
        prev = rows[i - 1].close
        if base <= 0 or prev <= 0:
            continue
        factor = prev / base
        if abs(factor - 1.0) > threshold:
            events.append(AdjustEvent(d=rows[i].d, factor=factor))
    return events


def adjusted_highs(
    rows: list[PxRow], events: list[AdjustEvent],
) -> list[tuple[date, float]]:
    """이벤트를 소급 적용한 (날짜, 수정 고가) 목록. 입력과 같은 순서."""
    factor_at = {e.d: e.factor for e in events}
    out: list[tuple[date, float]] = []
    cum = 1.0
    for row in reversed(rows):
        out.append((row.d, row.high * cum))
        # 이벤트 당일은 이미 새 기준이므로, 그 이전 날짜부터 계수를 먹인다.
        if row.d in factor_at:
            cum /= factor_at[row.d]
    out.reverse()
    return out
