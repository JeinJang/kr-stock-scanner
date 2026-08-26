"""src/aihw/models.py

AI HW / 빅테크 시총 비율 지표의 데이터 모델.
"""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class DailyCap(BaseModel):
    """종목 1개의 일별 시총 스냅샷 (벤치마크·환율은 close만 유효)."""

    date: date
    ticker: str
    close: float  # 현지통화 종가
    shares: int | None = None
    market_cap_usd: float | None = None
    source: str = "backfill"  # "backfill" | "snapshot"


class AihwSeries(BaseModel):
    """그룹 합산·비율·지수화 시계열. 모든 리스트는 dates와 같은 길이."""

    dates: list[date]
    ai_hw_total: list[float]
    big_tech_total: list[float]
    ratio: list[float]
    indexed: dict[str, list[float]]  # "AI HW", "빅테크", 벤치마크 티커
    company_caps: dict[str, dict[str, list[float]]] = {}  # 그룹명 → 티커 → 시총(USD)


class CompanySummary(BaseModel):
    """개별 종목의 요약 정보."""

    ticker: str
    name: str
    cap_usd: float
    day_change_pct: float | None = None


class GroupSummary(BaseModel):
    """그룹(AI HW 또는 빅테크)의 요약 정보."""

    name: str  # "AI HW" | "빅테크"
    total_usd: float
    companies: list[CompanySummary]  # 시총 내림차순


class AihwSummary(BaseModel):
    """AI HW 비율 지표의 전체 요약."""

    as_of: date
    ratio: float
    ratio_prev: float | None
    change_pp: float | None  # 전일 대비 %p (ratio 차이 × 100)
    high_30d: float
    low_30d: float
    threshold: float
    status: str | None  # "cross_up" | "cross_down" | "above" | None
    groups: list[GroupSummary]
    basis_dates: dict[str, date] = {}  # 시장 라벨("미국"/"한국") → 해당 시장 기준일
