from __future__ import annotations

from datetime import date, datetime, timedelta

from loguru import logger

from src.dart.cache import DartCache
from src.dart.fetcher import DartFetcher
from src.dart.models import CorpInfo, FinancialStatement
from src.fundamentals.calculator import compute_metrics
from src.fundamentals.classifier import classify, compute_market_medians
from src.fundamentals.db import FundamentalsDB
from src.fundamentals.models import FundamentalsMetrics, ScoreCard
from src.fundamentals.scorer import score


class Pipeline:
    """Orchestrates the full fundamentals screening pipeline."""

    def __init__(
        self,
        cache: DartCache,
        fetcher: DartFetcher,
        ttl_days: int = 30,
        fundamentals_db: FundamentalsDB | None = None,
    ):
        self._cache = cache
        self._fetcher = fetcher
        self._ttl_days = ttl_days
        self._db = fundamentals_db

    async def refresh_data(
        self,
        force: bool,
        years: list[int],
        markets: list[str] | None = None,
        market_map: dict[str, str] | None = None,
    ) -> None:
        """Refresh corp universe and financial data if needed.

        Args:
            force: Skip TTL check and refresh anyway.
            years: List of years to fetch.
            markets: Markets to include.
            market_map: ticker -> market (from KRX). Required for accurate market labels.
        """
        markets = markets or ["KOSPI", "KOSDAQ"]
        last = self._cache.last_updated()
        needs_refresh = (
            force
            or last is None
            or (datetime.now() - last) > timedelta(days=self._ttl_days)
        )
        if not needs_refresh:
            logger.info(f"DART cache fresh (last update: {last}), skipping refresh")
            return

        logger.info("Refreshing DART data...")
        corps = await self._fetcher.fetch_corp_universe(
            markets=markets, market_map=market_map,
        )
        self._cache.save_corp_info(corps)

        corp_codes = [c.corp_code for c in corps]
        # Annual reports only for now
        statements = await self._fetcher.fetch_financials(
            corp_codes=corp_codes,
            years=years,
            report_codes=["11011"],
        )
        self._cache.save_financials(statements)
        self._cache.set_last_updated(datetime.now())
        logger.info(f"Refresh complete: {len(corps)} corps, {len(statements)} statements")

    def compute_all(
        self,
        market_caps: dict[str, float],
        eps_map: dict[str, float] | None = None,
        bps_map: dict[str, float] | None = None,
        markets: list[str] | None = None,
    ) -> tuple[list[FundamentalsMetrics], list[ScoreCard]]:
        """Compute metrics and scores for all cached corps."""
        markets = markets or ["KOSPI", "KOSDAQ"]
        corps = self._cache.load_corp_info(markets=markets)
        all_statements = self._cache.load_financials()
        # Group statements by corp_code
        grouped: dict[str, list[FinancialStatement]] = {}
        for s in all_statements:
            grouped.setdefault(s.corp_code, []).append(s)

        as_of = date.today()
        metrics_list: list[FundamentalsMetrics] = []
        for corp in corps:
            statements = grouped.get(corp.corp_code, [])
            mc = market_caps.get(corp.ticker)
            eps = (eps_map or {}).get(corp.ticker)
            bps = (bps_map or {}).get(corp.ticker)
            m = compute_metrics(
                ticker=corp.ticker,
                corp_code=corp.corp_code,
                statements=statements,
                as_of=as_of,
                market_cap=mc, eps=eps, bps=bps,
            )
            metrics_list.append(m)

        # Compute market medians for valuation classification
        ticker_to_market = {c.ticker: c.market for c in corps}
        medians = compute_market_medians(metrics_list, ticker_to_market)

        # Score and classify
        scores: list[ScoreCard] = []
        for m in metrics_list:
            sc = score(m)
            market = ticker_to_market.get(m.ticker, "KOSPI")
            sc.categories = classify(m, sc, market=market, medians=medians)
            scores.append(sc)

        if self._db is not None:
            self._db.save_metrics(metrics_list)
            self._db.save_scores(scores)

        return metrics_list, scores
