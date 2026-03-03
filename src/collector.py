# src/collector.py
from __future__ import annotations

from datetime import datetime, timedelta

from loguru import logger

from src.krx_client import KrxClient


class Collector:
    """Collects stock market data from KRX via KrxClient."""

    def __init__(self, client: KrxClient):
        self.client = client
        self._high_cache: dict[str, float] | None = None

    def collect_daily(
        self, date_str: str, markets: list[str] | None = None
    ) -> dict[str, dict]:
        if markets is None:
            markets = ["KOSPI", "KOSDAQ", "ETF"]

        result = {}
        for market in markets:
            if market == "ETF":
                df = self.client.get_etf_ohlcv_by_ticker(date_str)
                for ticker in df.index:
                    result[ticker] = {
                        "market": "ETF",
                        "open": int(df.loc[ticker, "시가"]),
                        "high": int(df.loc[ticker, "고가"]),
                        "low": int(df.loc[ticker, "저가"]),
                        "close": int(df.loc[ticker, "종가"]),
                        "volume": int(df.loc[ticker, "거래량"]),
                        "change_pct": 0.0,
                    }
            else:
                df = self.client.get_market_ohlcv_by_ticker(date_str, market=market)
                for ticker in df.index:
                    result[ticker] = {
                        "market": market,
                        "open": int(df.loc[ticker, "시가"]),
                        "high": int(df.loc[ticker, "고가"]),
                        "low": int(df.loc[ticker, "저가"]),
                        "close": int(df.loc[ticker, "종가"]),
                        "volume": int(df.loc[ticker, "거래량"]),
                        "change_pct": float(df.loc[ticker, "등락률"]) if "등락률" in df.columns else 0.0,
                    }

        logger.info(f"Collected daily data for {len(result)} tickers on {date_str}")
        return result

    # -- 52-week high (strategy depends on client.supports_history) --------

    def _prefetch_history(
        self, date_str: str, lookback: int, markets: list[str] | None = None,
    ) -> None:
        if markets is None:
            markets = ["KOSPI", "KOSDAQ"]
        end = datetime.strptime(date_str, "%Y%m%d")
        prev_day = end - timedelta(days=1)
        start = end - timedelta(days=int(lookback * 1.5))

        high_map: dict[str, float] = {}
        current = start
        call_count = 0
        logger.info(
            f"Fetching historical data for 52w high "
            f"({start.strftime('%Y%m%d')} ~ {prev_day.strftime('%Y%m%d')})"
        )
        while current <= prev_day:
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue
            d = current.strftime("%Y%m%d")
            for market in markets:
                df = self.client.get_market_ohlcv_by_ticker(d, market=market)
                call_count += 1
                if df.empty:
                    continue
                for ticker in df.index:
                    h = float(df.loc[ticker, "고가"])
                    if h > high_map.get(ticker, 0.0):
                        high_map[ticker] = h
            current += timedelta(days=1)

        logger.info(f"52w high prefetch done: {call_count} API calls, {len(high_map)} tickers")
        self._high_cache = high_map

    def get_52w_high(self, ticker: str, date_str: str, lookback: int = 250) -> float:
        if self.client.supports_history:
            end = datetime.strptime(date_str, "%Y%m%d")
            prev_day = end - timedelta(days=1)
            start = end - timedelta(days=int(lookback * 1.5))
            df = self.client.get_market_ohlcv_by_date(
                start.strftime("%Y%m%d"), prev_day.strftime("%Y%m%d"), ticker,
            )
            if not df.empty and "고가" in df.columns:
                return float(df["고가"].max())
            return 0.0
        else:
            if self._high_cache is None:
                self._prefetch_history(date_str, lookback)
            assert self._high_cache is not None
            return self._high_cache.get(ticker, 0.0)

    # -- Sector & market cap -----------------------------------------------

    def get_sector_map(self, date_str: str, market: str) -> dict[str, str]:
        df = self.client.get_market_sector_classifications(date_str, market)
        if df.empty:
            return {}
        if "티커" in df.columns and "업종명" in df.columns:
            return dict(zip(df["티커"], df["업종명"]))
        return {}

    def get_market_caps(self, date_str: str, market: str = "ALL") -> dict[str, int]:
        result: dict[str, int] = {}
        markets = ["KOSPI", "KOSDAQ"] if market == "ALL" else [market]
        for m in markets:
            df = self.client.get_market_cap_by_ticker(date_str, market=m)
            for ticker, row in df.iterrows():
                result[ticker] = int(row["시가총액"])
        return result
