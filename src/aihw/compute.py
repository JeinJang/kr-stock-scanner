"""src/aihw/compute.py

순수 계산: 그룹 합산, 비율, 지수화. 네트워크·DB 접근 없음.
"""
from __future__ import annotations

from datetime import date

from src.aihw.models import (
    AihwSeries,
    AihwSummary,
    CompanySummary,
    DailyCap,
    GroupSummary,
)


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

    company_caps = {
        "AI HW": {t: [by_date[d][t].market_cap_usd for d in dates] for t in ai_hw},
        "빅테크": {t: [by_date[d][t].market_cap_usd for d in dates] for t in big_tech},
    }

    return AihwSeries(
        dates=dates,
        ai_hw_total=ai_hw_total,
        big_tech_total=big_tech_total,
        ratio=ratio,
        indexed=indexed,
        company_caps=company_caps,
    )


def threshold_status(
    ratio_today: float, ratio_prev: float | None, threshold: float
) -> str | None:
    """임계값 기준 상태 판정: cross_up, cross_down, above, None."""
    above = ratio_today >= threshold
    if ratio_prev is None:
        return "above" if above else None
    prev_above = ratio_prev >= threshold
    if above and not prev_above:
        return "cross_up"
    if not above and prev_above:
        return "cross_down"
    return "above" if above else None


def _group_summary(
    name: str,
    tickers: dict[str, str],
    latest: dict[str, DailyCap],
    prev: dict[str, DailyCap],
) -> GroupSummary:
    """그룹별 요약 정보 생성 (시총 내림차순으로 정렬)."""
    companies = []
    for ticker, display_name in tickers.items():
        cap = latest[ticker].market_cap_usd
        change = None
        p = prev[ticker]
        if p.date != latest[ticker].date and p.market_cap_usd:
            change = (cap / p.market_cap_usd - 1.0) * 100.0
        companies.append(CompanySummary(
            ticker=ticker, name=display_name, cap_usd=cap, day_change_pct=change,
        ))
    companies.sort(key=lambda c: c.cap_usd, reverse=True)
    return GroupSummary(
        name=name, total_usd=sum(c.cap_usd for c in companies), companies=companies,
    )


def summarize(
    series: AihwSeries,
    caps: list[DailyCap],
    ai_hw: dict[str, str],
    big_tech: dict[str, str],
    threshold: float,
) -> AihwSummary:
    """AI HW 비율 지표의 전체 요약 정보 생성.

    시장별 마감 시차(한국 오늘 종가 vs 미국 어제 종가) 때문에 단일 날짜 행을
    기준으로 삼지 않고, 종목마다 자기 자신의 최신 관측치와 직전 관측치를
    비교한다. 헤드라인 비율·그룹 합계도 종목별 최신 시총의 합으로 계산한다.
    (시계열 차트는 series — 완전 거래일만 — 를 그대로 쓴다.)
    """
    cap_tickers = list(ai_hw) + list(big_tech)
    history: dict[str, list[DailyCap]] = {t: [] for t in cap_tickers}
    for c in sorted(caps, key=lambda c: c.date):
        if c.ticker in history and c.market_cap_usd is not None:
            history[c.ticker].append(c)

    latest = {t: rows[-1] for t, rows in history.items()}
    prev = {t: (rows[-2] if len(rows) >= 2 else rows[-1]) for t, rows in history.items()}
    as_of = max(c.date for c in latest.values())

    basis_dates: dict[str, date] = {}
    for t, c in latest.items():
        market = "한국" if t.endswith(".KS") else "미국"
        basis_dates[market] = max(basis_dates.get(market, c.date), c.date)

    ai_group = _group_summary("AI HW", ai_hw, latest, prev)
    big_group = _group_summary("빅테크", big_tech, latest, prev)
    ratio = ai_group.total_usd / big_group.total_usd

    # 종목별 직전 관측치의 합으로 전일 비율 계산 (직전 관측치가 하나도 없으면 None)
    has_prev = any(prev[t].date != latest[t].date for t in cap_tickers)
    ratio_prev = None
    if has_prev:
        prev_ai = sum(prev[t].market_cap_usd for t in ai_hw)
        prev_big = sum(prev[t].market_cap_usd for t in big_tech)
        ratio_prev = prev_ai / prev_big

    last_30 = series.ratio[-30:]

    return AihwSummary(
        as_of=as_of,
        ratio=ratio,
        ratio_prev=ratio_prev,
        change_pp=(ratio - ratio_prev) * 100.0 if ratio_prev is not None else None,
        high_30d=max(last_30),
        low_30d=min(last_30),
        threshold=threshold,
        status=threshold_status(ratio, ratio_prev, threshold),
        groups=[ai_group, big_group],
        basis_dates=basis_dates,
    )
