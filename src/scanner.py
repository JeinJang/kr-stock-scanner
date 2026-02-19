# src/scanner.py
from collections import defaultdict
from datetime import date

from loguru import logger

from src.collector import Collector
from src.models import StockHigh, ScanResult, MarketStats


class Scanner:
    """Identifies 52-week high stocks from daily market data."""

    def __init__(self, collector: Collector | None = None):
        self.collector = collector or Collector()

    def find_new_highs(
        self,
        daily_data: dict[str, dict],
        date_str: str,
        sector_map: dict[str, str],
        name_map: dict[str, str],
        lookback: int = 250,
    ) -> list[StockHigh]:
        """Find stocks that hit 52-week highs on the given date."""
        highs = []

        for ticker, data in daily_data.items():
            today_high = data["high"]
            if today_high == 0:
                continue

            prev_52w_high = self.collector.get_52w_high(ticker, date_str, lookback=lookback)

            if today_high >= prev_52w_high and prev_52w_high > 0:
                breakout_pct = (
                    ((today_high - prev_52w_high) / prev_52w_high * 100)
                    if prev_52w_high > 0 else 0.0
                )

                highs.append(StockHigh(
                    ticker=ticker,
                    name=name_map.get(ticker, ticker),
                    market=data["market"],
                    sector=sector_map.get(ticker, "기타"),
                    close_price=data["close"],
                    high_52w=today_high,
                    prev_high_52w=prev_52w_high,
                    breakout_pct=round(breakout_pct, 2),
                    volume=data["volume"],
                    avg_volume_20d=0,
                ))

        logger.info(f"Found {len(highs)} stocks at 52-week high on {date_str}")
        return highs

    def build_scan_result(
        self,
        scan_date: date,
        highs: list[StockHigh],
        total_stocks: int,
    ) -> ScanResult:
        """Build a ScanResult with stats and sector breakdown."""
        kospi = [h for h in highs if h.market == "KOSPI"]
        kosdaq = [h for h in highs if h.market == "KOSDAQ"]
        etf = [h for h in highs if h.market == "ETF"]

        sector_breakdown: dict[str, list[StockHigh]] = defaultdict(list)
        for h in highs:
            sector_breakdown[h.sector].append(h)

        return ScanResult(
            scan_date=scan_date,
            stats=MarketStats(
                total_stocks=total_stocks,
                new_high_count=len(highs),
                kospi_count=len(kospi),
                kosdaq_count=len(kosdaq),
                etf_count=len(etf),
            ),
            highs=highs,
            sector_breakdown=dict(sector_breakdown),
        )
