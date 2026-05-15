from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from src.krx_client import KrxClient
from src.market_data.db import MarketDB
from src.market_data.fetcher import fetch_yearly_market_data


@dataclass
class SourceReport:
    requested_years: list[int]
    requested_tickers_count: int
    successful_rows: int
    failed_items: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class MarketDataPipeline:
    """Orchestrates KRX -> corp_market_yearly refresh."""

    def __init__(self, db: MarketDB, client: KrxClient | None = None):
        self._db = db
        self._client = client

    def refresh(
        self,
        years: list[int],
        tickers: list[str],
        corp_code_map: dict[str, str],
    ) -> SourceReport:
        start = datetime.now()
        rows = fetch_yearly_market_data(
            tickers=tickers,
            years=years,
            corp_code_map=corp_code_map,
            client=self._client,
        )
        self._db.save_yearly(rows)
        duration = (datetime.now() - start).total_seconds()
        logger.info(f"market_data pipeline: {len(rows)} rows in {duration:.1f}s")
        return SourceReport(
            requested_years=years,
            requested_tickers_count=len(tickers),
            successful_rows=len(rows),
            duration_seconds=duration,
        )
