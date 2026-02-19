# src/collector.py
from loguru import logger
from pykrx import stock


class Collector:
    """Collects stock market data from KRX via pykrx."""

    def collect_daily(
        self, date_str: str, markets: list[str] | None = None
    ) -> dict[str, dict]:
        """Fetch daily OHLCV for all tickers across specified markets.

        Returns dict: ticker -> {market, open, high, low, close, volume, change_pct}
        """
        if markets is None:
            markets = ["KOSPI", "KOSDAQ", "ETF"]

        result = {}

        for market in markets:
            if market == "ETF":
                df = stock.get_etf_ohlcv_by_ticker(date_str)
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
                df = stock.get_market_ohlcv_by_ticker(date_str, market=market)
                for ticker in df.index:
                    result[ticker] = {
                        "market": market,
                        "open": int(df.loc[ticker, "시가"]),
                        "high": int(df.loc[ticker, "고가"]),
                        "low": int(df.loc[ticker, "저가"]),
                        "close": int(df.loc[ticker, "종가"]),
                        "volume": int(df.loc[ticker, "거래량"]),
                        "change_pct": float(df.loc[ticker, "등락률"]),
                    }

        logger.info(f"Collected daily data for {len(result)} tickers on {date_str}")
        return result

    def get_52w_high(self, ticker: str, date_str: str, lookback: int = 250) -> float:
        """Get the 52-week high price for a ticker."""
        from datetime import datetime, timedelta

        end = datetime.strptime(date_str, "%Y%m%d")
        start = end - timedelta(days=int(lookback * 1.5))
        start_str = start.strftime("%Y%m%d")

        df = stock.get_market_ohlcv_by_date(start_str, date_str, ticker)
        if df.empty:
            return 0.0
        return float(df["고가"].max())

    def get_sector_map(self, date_str: str, market: str) -> dict[str, str]:
        """Get ticker -> sector name mapping for a market."""
        df = stock.get_market_sector_classifications(date_str, market)
        return {ticker: row["업종명"] for ticker, row in df.iterrows()}

    def get_market_caps(self, date_str: str, market: str = "ALL") -> dict[str, int]:
        """Get ticker -> market cap mapping."""
        df = stock.get_market_cap_by_ticker(date_str, market=market)
        return {ticker: int(row["시가총액"]) for ticker, row in df.iterrows()}
