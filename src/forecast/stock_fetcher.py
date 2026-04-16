# src/forecast/stock_fetcher.py
from __future__ import annotations

from datetime import datetime, timedelta

from loguru import logger

from src.krx_client import KrxClient


class StockFetcher:
    """Fetches historical close prices for individual stocks via KRX."""

    def __init__(self, client: KrxClient):
        self._client = client

    def fetch_history(
        self, ticker: str, end_date: str, lookback_days: int = 250,
    ) -> tuple[list[str], list[float]]:
        """Fetch historical close prices for a single ticker.

        Returns (dates, close_prices) as parallel lists.
        """
        end = datetime.strptime(end_date, "%Y%m%d")
        start = end - timedelta(days=int(lookback_days * 1.5))
        df = self._client.get_market_ohlcv_by_date(
            start.strftime("%Y%m%d"), end_date, ticker,
        )
        if df.empty or "종가" not in df.columns:
            return [], []

        dates = [d.strftime("%Y%m%d") for d in df.index]
        values = [float(v) for v in df["종가"].values]
        return dates, values

    def fetch_histories(
        self,
        tickers: list[str],
        end_date: str,
        lookback_days: int = 250,
    ) -> dict[str, tuple[list[str], list[float]]]:
        """Fetch historical close prices for multiple tickers."""
        results: dict[str, tuple[list[str], list[float]]] = {}
        for i, ticker in enumerate(tickers):
            logger.info(f"Fetching stock history [{i+1}/{len(tickers)}]: {ticker}")
            dates, values = self.fetch_history(ticker, end_date, lookback_days)
            if values:
                results[ticker] = (dates, values)
            else:
                logger.warning(f"  -> No data for {ticker}, skipping")
        return results
