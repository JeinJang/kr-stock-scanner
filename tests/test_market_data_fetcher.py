from datetime import date
from unittest.mock import MagicMock
import pandas as pd
import pytest

from src.market_data.fetcher import fetch_yearly_market_data
from src.market_data.models import MarketYearly


def _fake_df(rows):
    """rows: list of (ticker, close, market_cap, volume, value, shares)."""
    df = pd.DataFrame(
        rows,
        columns=["티커", "종가", "시가총액", "거래량", "거래대금", "상장주식수"],
    ).set_index("티커")
    return df


def _client_returning(kospi_df, kosdaq_df=None):
    c = MagicMock()
    kosdaq_df = kosdaq_df if kosdaq_df is not None else _fake_df([])
    def side(date, market="KOSPI"):
        return kospi_df if market == "KOSPI" else kosdaq_df
    c.get_market_cap_by_ticker.side_effect = side
    return c


def test_fetcher_returns_one_row_per_ticker_per_year():
    client = _client_returning(_fake_df([
        ("000660", 173_900, 126_599_611_273_500, 1_434_202, 250_367_749_400, 728_002_365),
        ("353200", 26_000, 1_300_000_000_000, 100_000, 2_600_000_000, 50_000_000),
    ]))
    out = fetch_yearly_market_data(
        tickers=["000660", "353200"], years=[2024],
        corp_code_map={"000660": "00126380", "353200": "00120182"},
        client=client,
    )
    assert len(out) == 2
    by_t = {m.ticker: m for m in out}
    assert by_t["000660"].market_cap == 126_599_611_273_500
    assert by_t["000660"].year == 2024
    assert by_t["353200"].shares_outstanding == 50_000_000


def test_fetcher_skips_tickers_not_requested():
    client = _client_returning(_fake_df([
        ("000660", 1, 1_000_000_000, 1, 1, 100),
        ("999999", 1, 999_000_000, 1, 1, 100),  # not requested
    ]))
    out = fetch_yearly_market_data(
        tickers=["000660"], years=[2024],
        corp_code_map={"000660": "00126380"},
        client=client,
    )
    assert len(out) == 1
    assert out[0].ticker == "000660"


def test_fetcher_skips_zero_market_cap():
    client = _client_returning(_fake_df([
        ("000660", 0, 0, 0, 0, 100),  # halted/holiday
    ]))
    out = fetch_yearly_market_data(
        tickers=["000660"], years=[2024],
        corp_code_map={"000660": "00126380"},
        client=client,
    )
    # zero market_cap -> skipped, but the year-stepper will also try other dates...
    # Since our mock returns 0 for every call, all attempts yield zero. Result: empty.
    assert out == []


def test_fetcher_continues_when_year_raises():
    client = MagicMock()
    # First call (year 2024) raises 3x, second year (2025) succeeds.
    client.get_market_cap_by_ticker.side_effect = [
        Exception("KRX timeout"),
        Exception("KRX timeout"),
        Exception("KRX timeout"),
        Exception("KRX timeout"),  # KOSPI year 2025 attempt 1, retry, etc.
        # Then year 2025 KOSPI succeeds
        _fake_df([("000660", 1, 100_000_000, 1, 1, 100)]),
        _fake_df([]),  # KOSDAQ 2025
    ]
    out = fetch_yearly_market_data(
        tickers=["000660"], years=[2024, 2025],
        corp_code_map={"000660": "00126380"},
        client=client,
    )
    # 2024 always fails (3 retries exhausted across attempts), 2025 succeeds for one ticker
    tickers_returned = {(r.ticker, r.year) for r in out}
    assert ("000660", 2025) in tickers_returned
    assert all(y != 2024 for _, y in tickers_returned)
