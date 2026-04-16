from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import requests
from loguru import logger


# ECOS stat codes for Korean macro indicators
ECOS_INDICATORS = {
    "KOSPI": {"stat_code": "802Y001", "item_code": "0001000", "freq": "D"},
    "KOSDAQ": {"stat_code": "802Y002", "item_code": "0001000", "freq": "D"},
    "USD_KRW": {"stat_code": "731Y003", "item_code": "0000001", "freq": "D"},
    "KR_RATE": {"stat_code": "722Y001", "item_code": "0101000", "freq": "D"},
}

ECOS_BASE_URL = "https://ecos.bok.or.kr/api/StatisticSearch"

FRED_INDICATORS = {
    "SP500": "SP500",
    "NASDAQ": "NASDAQCOM",
    "US_RATE": "FEDFUNDS",
}

# Display names for indicators
INDICATOR_NAMES = {
    "KOSPI": "KOSPI",
    "KOSDAQ": "KOSDAQ",
    "USD_KRW": "USD/KRW",
    "KR_RATE": "한국 기준금리",
    "SP500": "S&P 500",
    "NASDAQ": "NASDAQ",
    "US_RATE": "미국 기준금리",
}


class MacroFetcher:
    """Fetches macro indicator data from ECOS and FRED APIs."""

    def __init__(self, ecos_api_key: str, fred_api_key: str):
        self._ecos_key = ecos_api_key
        self._fred_key = fred_api_key

    def fetch_ecos_series(
        self,
        stat_code: str,
        item_code: str,
        start_date: str,
        end_date: str,
        freq: str = "D",
    ) -> tuple[list[str], list[float]]:
        """Fetch a single time series from ECOS API.

        Returns (dates, values) as parallel lists.
        """
        url = (
            f"{ECOS_BASE_URL}/{self._ecos_key}/json/kr/1/1000"
            f"/{stat_code}/{freq}/{start_date}/{end_date}/{item_code}"
        )
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()

        rows = data.get("StatisticSearch", {}).get("row", [])
        dates = []
        values = []
        for row in rows:
            val_str = row.get("DATA_VALUE", "")
            if val_str == "" or val_str == "-":
                continue
            dates.append(row["TIME"])
            values.append(float(val_str))
        return dates, values

    def fetch_all_ecos(self, lookback_days: int = 400) -> dict[str, tuple[list[str], list[float]]]:
        """Fetch all ECOS indicators for the given lookback period."""
        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")

        results: dict[str, tuple[list[str], list[float]]] = {}
        for name, params in ECOS_INDICATORS.items():
            logger.info(f"Fetching ECOS: {name}")
            dates, values = self.fetch_ecos_series(
                stat_code=params["stat_code"],
                item_code=params["item_code"],
                start_date=start_str,
                end_date=end_str,
                freq=params["freq"],
            )
            results[name] = (dates, values)
            logger.info(f"  -> {len(values)} data points")
        return results

    def fetch_fred_series(
        self, series_id: str, lookback_days: int = 400,
    ) -> tuple[list[str], list[float]]:
        """Fetch a single time series from FRED API."""
        from fredapi import Fred

        fred = Fred(api_key=self._fred_key)
        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        series: pd.Series = fred.get_series(series_id, start, end)
        series = series.dropna()

        dates = [d.strftime("%Y%m%d") for d in series.index]
        values = [float(v) for v in series.values]
        return dates, values

    def fetch_all_fred(self, lookback_days: int = 400) -> dict[str, tuple[list[str], list[float]]]:
        """Fetch all FRED indicators."""
        results: dict[str, tuple[list[str], list[float]]] = {}
        for name, series_id in FRED_INDICATORS.items():
            logger.info(f"Fetching FRED: {name} ({series_id})")
            dates, values = self.fetch_fred_series(series_id, lookback_days)
            results[name] = (dates, values)
            logger.info(f"  -> {len(values)} data points")
        return results

    def fetch_all(self, lookback_days: int = 400) -> dict[str, tuple[list[str], list[float]]]:
        """Fetch all macro indicators from both ECOS and FRED."""
        results = self.fetch_all_ecos(lookback_days)
        results.update(self.fetch_all_fred(lookback_days))
        return results
