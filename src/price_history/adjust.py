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

# 장기 거래정지가 풀리는 날, KRX는 시가 단일가 경매로 기준가를 다시 잡는다.
# 그러면 기준가가 정지 직전 종가와 어긋나 이벤트로 잡히지만 실제로 조정된
# 것은 없다. 실측 저장소 3,365건 중 239건이 이 유형이었고(예: 009410
# 20241031 계수 0.983, 정지 153일), 계수가 모두 1 근처(0.90~1.10)였다.
# 진짜 액면분할·병합도 정지 다음 날 재개되지만 계수가 50·10·5·1.8 등으로
# 1에서 멀다. 그래서 '직전 행이 정지일(고가 0)' + '계수가 1의 ±10% 이내'
# 두 조건이 모두 성립할 때만 경매 잡음으로 보고 버린다.
# 감수하는 위험: 10% 미만의 진짜 이벤트(예: 5% 주식배당)의 락일이 하필
# 정지 바로 다음 날이면 놓친다. 락일은 정상 거래일에 잡히므로 드물다.
HALT_RESUME_MAX = 0.1

# 고가 0인 행이 하루뿐이면 정지가 아니라 거래 없는 날일 수 있다 — 특히
# 우선주 등 저유동성 종목에서 흔하다(예: 001067 20230313, 계수 1.0281,
# 정지 1일이지만 형제 종목 001060·001065가 같은 날 같은 이벤트를 기록해
# 진짜 이벤트임이 확인됨). 진짜 장기 정지는 여러 거래일에 걸쳐 이어지므로
# 연속 정지일이 이 값 이상일 때만 경매 잡음으로 간주한다.
HALT_MIN_ROWS = 2


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


def _halt_run_length(rows: list[PxRow], i: int) -> int:
    """rows[i-1]에서 거슬러 올라가며 이어지는 연속 정지일(고가 0) 개수."""
    n = 0
    j = i - 1
    while j >= 0 and rows[j].high == 0:
        n += 1
        j -= 1
    return n


def detect_adjustments(rows: list[PxRow], threshold: float = THRESHOLD) -> list[AdjustEvent]:
    """rows(날짜 오름차순)에서 기준가 재설정 이벤트를 찾는다."""
    events: list[AdjustEvent] = []
    for i in range(1, len(rows)):
        base = rows[i].close - rows[i].chg
        prev = rows[i - 1].close
        if base <= 0 or prev <= 0:
            continue
        factor = prev / base
        if abs(factor - 1.0) <= threshold:
            continue
        # 거래정지 해제일의 시가 경매 기준가 — 조정된 것이 없다.
        if _halt_run_length(rows, i) >= HALT_MIN_ROWS and abs(factor - 1.0) < HALT_RESUME_MAX:
            continue
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
