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
