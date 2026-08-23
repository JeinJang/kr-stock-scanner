"""src/aihw/compute.py

순수 계산: 그룹 합산, 비율, 지수화. 네트워크·DB 접근 없음.
"""
from __future__ import annotations

from datetime import date

from src.aihw.models import AihwSeries, DailyCap


def _index_100(values: list[float], base_idx: int) -> list[float]:
    base = values[base_idx]
    return [v / base * 100.0 for v in values]


def build_series(
    caps: list[DailyCap],
    ai_hw: list[str],
    big_tech: list[str],
    benchmarks: list[str],
    base_date: date,
) -> AihwSeries:
    cap_tickers = set(ai_hw) | set(big_tech)
    by_date: dict[date, dict[str, DailyCap]] = {}
    for c in caps:
        by_date.setdefault(c.date, {})[c.ticker] = c

    # cap 종목이 전부 있는 날짜만 채택 (부분 데이터로 비율 왜곡 방지)
    dates = sorted(
        d for d, row in by_date.items()
        if cap_tickers <= {t for t, c in row.items() if c.market_cap_usd is not None}
    )
    if not dates:
        raise ValueError("cap 종목 전체가 존재하는 날짜가 없습니다")

    ai_hw_total = [sum(by_date[d][t].market_cap_usd for t in ai_hw) for d in dates]
    big_tech_total = [sum(by_date[d][t].market_cap_usd for t in big_tech) for d in dates]
    ratio = [a / b for a, b in zip(ai_hw_total, big_tech_total)]

    base_idx = next((i for i, d in enumerate(dates) if d >= base_date), 0)
    indexed: dict[str, list[float]] = {
        "AI HW": _index_100(ai_hw_total, base_idx),
        "빅테크": _index_100(big_tech_total, base_idx),
    }
    for bench in benchmarks:
        closes = [by_date[d][bench].close for d in dates if bench in by_date[d]]
        if len(closes) == len(dates):
            indexed[bench] = _index_100(closes, base_idx)

    return AihwSeries(
        dates=dates,
        ai_hw_total=ai_hw_total,
        big_tech_total=big_tech_total,
        ratio=ratio,
        indexed=indexed,
    )
