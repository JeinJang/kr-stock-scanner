"""src/market_data/fetcher.py

Fetch yearly market snapshot data (market_cap, shares, close) from KRX.
Uses the project's existing KrxClient — one call per year covers all tickers.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from src.krx_client import KrxClient, create_krx_client
from src.market_data.models import MarketYearly

if TYPE_CHECKING:
    pass


def _step_back_for_business_day(
    client: KrxClient,
    year: int,
    today: datetime,
    max_steps: int = 10,
) -> tuple[str | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Find the most recent trading day for the given year by stepping back from year-end.

    For completed years: starts at YYYY-12-30 and steps back.
    For in-progress years: starts at today and steps back.

    Exceptions from client calls are NOT caught here — they propagate to the caller
    so the outer retry loop can handle them.

    Returns:
        (date_str, kospi_df, kosdaq_df) when a valid trading day is found,
        (None, None, None) if no trading day with nonzero 시가총액 is found within max_steps.
    """
    if year < today.year:
        start = datetime(year, 12, 30)
    else:
        start = today

    for step in range(max_steps):
        d = start - timedelta(days=step)
        date_str = d.strftime("%Y%m%d")

        kospi = client.get_market_cap_by_ticker(date_str, market="KOSPI")
        kosdaq = client.get_market_cap_by_ticker(date_str, market="KOSDAQ")

        frames = [df for df in (kospi, kosdaq) if df is not None and not df.empty]
        if frames:
            combined = pd.concat(frames)
            if (combined["시가총액"] > 0).any():
                return date_str, kospi, kosdaq

    return None, None, None


def fetch_yearly_market_data(
    tickers: list[str],
    years: list[int],
    corp_code_map: dict[str, str],
    client: KrxClient | None = None,
    max_retries: int = 3,
) -> list[MarketYearly]:
    """Fetch (ticker, year) yearly market_cap / shares / close from KRX.

    Strategy: one KRX call per year (returns ALL tickers in one DataFrame).
    Failures on a year are logged; other years continue.

    Args:
        tickers: filter to these tickers (others in KRX response are dropped).
        years: list of years to fetch.
        corp_code_map: ticker -> corp_code (for storing alongside).
        client: optional KrxClient; if None, built from .env via create_krx_client.
        max_retries: per-year retry count with exponential backoff.

    Returns:
        list[MarketYearly]. Failed (ticker, year) tuples are simply absent.
    """
    if client is None:
        from src.config import Settings
        s = Settings()
        client = create_krx_client(
            krx_id=s.krx_id,
            krx_pw=s.krx_pw,
            krx_api_key=s.krx_api_key,
        )

    out: list[MarketYearly] = []
    ticker_set = set(tickers)
    today = datetime.now()

    for year in years:
        logger.info(f"market_data: fetching year={year}")
        last_err: Exception | None = None
        succeeded = False

        for attempt in range(max_retries):
            try:
                date_str, kospi, kosdaq = _step_back_for_business_day(client, year, today)

                if date_str is None:
                    logger.warning(
                        f"market_data: year={year} — no trading day found in step-back"
                    )
                    break  # no point retrying — KRX returned data but all zero

                as_of_d = datetime.strptime(date_str, "%Y%m%d").date()

                for df in (kospi, kosdaq):
                    if df is None or df.empty:
                        continue
                    for tkr in df.index:
                        if tkr not in ticker_set:
                            continue
                        row = df.loc[tkr]
                        try:
                            mc = int(row["시가총액"])
                            shares = int(row["상장주식수"])
                            close = int(row["종가"])
                        except (KeyError, ValueError, TypeError):
                            continue
                        if mc <= 0:
                            continue
                        cc = corp_code_map.get(tkr)
                        if cc is None:
                            continue
                        out.append(MarketYearly(
                            corp_code=cc,
                            ticker=tkr,
                            year=year,
                            as_of_date=as_of_d,
                            market_cap=mc,
                            shares_outstanding=shares,
                            close_price=close,
                        ))

                succeeded = True
                break  # year complete

            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        if not succeeded and last_err is not None:
            logger.warning(
                f"market_data: year={year} failed after {max_retries} retries: {last_err}"
            )

    return out
