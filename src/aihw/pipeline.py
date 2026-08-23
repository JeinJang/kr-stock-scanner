"""src/aihw/pipeline.py

수집 → 저장 → 계산 → 리포트 생성 오케스트레이션.
텔레그램 전송은 CLI가 담당한다 (pipeline의 부수효과는 DB·파일까지).
"""
from __future__ import annotations

from datetime import date, datetime

from loguru import logger
from pydantic import BaseModel

from src.aihw.compute import build_series, summarize
from src.aihw.db import AihwDB
from src.aihw.fetcher import fetch_all
from src.aihw.models import AihwSummary
from src.aihw.report import build_caption, generate_html, generate_png
from src.config import AihwSection


class AihwResult(BaseModel):
    summary: AihwSummary
    html_path: str
    png_path: str
    caption: str


def run_aihw(
    config: AihwSection,
    db: AihwDB | None = None,
    fetch=fetch_all,
) -> AihwResult:
    if db is None:
        db = AihwDB()
    base_date = datetime.strptime(config.base_date, "%Y-%m-%d").date()
    cap_tickers = list(config.ai_hw_tickers) + list(config.big_tech_tickers)

    logger.info(f"aihw 수집 시작: {len(cap_tickers)}종목 + {config.benchmarks}")
    fetched = fetch(cap_tickers, config.benchmarks, base_date, date.today())
    db.save_caps(fetched)

    caps = db.load_caps(base_date, date.today())
    series = build_series(
        caps,
        ai_hw=list(config.ai_hw_tickers),
        big_tech=list(config.big_tech_tickers),
        benchmarks=config.benchmarks,
        base_date=base_date,
    )
    summary = summarize(
        series, caps, config.ai_hw_tickers, config.big_tech_tickers, config.threshold,
    )
    return AihwResult(
        summary=summary,
        html_path=generate_html(series, summary, config.report_dir),
        png_path=generate_png(series, summary, config.report_dir),
        caption=build_caption(summary),
    )
