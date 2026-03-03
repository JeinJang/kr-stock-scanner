# tests/test_collector.py
from unittest.mock import MagicMock
import pandas as pd
import pytest


def _make_collector(client=None):
    from src.collector import Collector
    return Collector(client=client or MagicMock())


def test_collect_daily_ohlcv():
    kospi_df = pd.DataFrame(
        {"시가": [70000], "고가": [72000], "저가": [69000], "종가": [71500],
         "거래량": [15000000], "거래대금": [1000000000], "등락률": [1.5],
         "시가총액": [450000000000000]},
        index=["005930"],
    )
    kosdaq_df = pd.DataFrame(
        {"시가": [50000], "고가": [52000], "저가": [49000], "종가": [51000],
         "거래량": [3000000], "거래대금": [150000000], "등락률": [2.0],
         "시가총액": [36000000000000]},
        index=["035420"],
    )
    etf_df = pd.DataFrame(
        {"NAV": [50000.0], "시가": [49800], "고가": [50500], "저가": [49500],
         "종가": [50200], "거래량": [500000], "거래대금": [25000000],
         "기초지수": [2500.0]},
        index=["069500"],
    )
    client = MagicMock()
    client.supports_history = False
    client.get_market_ohlcv_by_ticker.side_effect = [kospi_df, kosdaq_df]
    client.get_etf_ohlcv_by_ticker.return_value = etf_df

    collector = _make_collector(client)
    result = collector.collect_daily("20260219", markets=["KOSPI", "KOSDAQ", "ETF"])

    assert result["005930"]["market"] == "KOSPI"
    assert result["005930"]["close"] == 71500
    assert result["069500"]["market"] == "ETF"


def test_52w_high_batch_strategy():
    """supports_history=False: batch prefetch via get_market_ohlcv_by_ticker."""
    from src.collector import Collector

    day1_df = pd.DataFrame(
        {"시가": [65000], "고가": [66000], "저가": [64000], "종가": [65500],
         "거래량": [10000000], "거래대금": [100000000], "등락률": [0.5],
         "시가총액": [400000000000000]},
        index=["005930"],
    )
    day2_df = pd.DataFrame(
        {"시가": [70000], "고가": [72000], "저가": [69000], "종가": [71500],
         "거래량": [15000000], "거래대금": [150000000], "등락률": [1.5],
         "시가총액": [450000000000000]},
        index=["005930"],
    )
    call_count = 0

    def mock_ohlcv(date, market="KOSPI"):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return day1_df
        elif call_count == 3:
            return day2_df
        return pd.DataFrame()

    client = MagicMock()
    client.supports_history = False
    client.get_market_ohlcv_by_ticker.side_effect = mock_ohlcv
    assert Collector(client=client).get_52w_high("005930", "20260219", lookback=250) == 72000


def test_52w_high_per_ticker_strategy():
    """supports_history=True: per-ticker query via get_market_ohlcv_by_date."""
    from src.collector import Collector

    history_df = pd.DataFrame(
        {"시가": [65000, 70000, 68000], "고가": [66000, 72000, 69000],
         "저가": [64000, 69000, 67000], "종가": [65500, 71500, 68500],
         "거래량": [10000000, 15000000, 12000000],
         "거래대금": [100000000, 150000000, 120000000]},
        index=["20250301", "20250302", "20250303"],
    )
    client = MagicMock()
    client.supports_history = True
    client.get_market_ohlcv_by_date.return_value = history_df
    assert Collector(client=client).get_52w_high("005930", "20260219", lookback=250) == 72000
    client.get_market_ohlcv_by_date.assert_called_once()


def test_52w_high_per_ticker_empty():
    from src.collector import Collector
    client = MagicMock()
    client.supports_history = True
    client.get_market_ohlcv_by_date.return_value = pd.DataFrame()
    assert Collector(client=client).get_52w_high("005930", "20260219") == 0.0


def test_sector_map_from_client():
    sector_df = pd.DataFrame({
        "티커": ["005930", "035420"],
        "종목명": ["삼성전자", "NAVER"],
        "업종명": ["전기전자", "서비스업"],
    })
    client = MagicMock()
    client.get_market_sector_classifications.return_value = sector_df
    sectors = _make_collector(client).get_sector_map("20260219", "KOSPI")
    assert sectors["005930"] == "전기전자"


def test_sector_map_returns_empty():
    client = MagicMock()
    client.get_market_sector_classifications.return_value = pd.DataFrame()
    assert _make_collector(client).get_sector_map("20260219", "KOSPI") == {}


def test_market_cap():
    cap_df = pd.DataFrame(
        {"시가총액": [450000000000000], "거래량": [15000000],
         "거래대금": [1000000000], "상장주식수": [5969782550], "종가": [71500]},
        index=["005930"],
    )
    client = MagicMock()
    client.get_market_cap_by_ticker.return_value = cap_df
    assert _make_collector(client).get_market_caps("20260219", market="KOSPI")["005930"] == 450000000000000


def test_create_krx_client_login():
    from src.krx_client import create_krx_client
    from src.krx_login_client import KrxLoginClient
    client = create_krx_client(krx_id="test", krx_pw="test")
    assert isinstance(client, KrxLoginClient)
    assert client.supports_history is True


def test_create_krx_client_openapi():
    from src.krx_client import create_krx_client
    from src.krx_openapi_client import KrxOpenApiClient
    client = create_krx_client(krx_api_key="test-key")
    assert isinstance(client, KrxOpenApiClient)
    assert client.supports_history is False


def test_create_krx_client_prefers_login():
    from src.krx_client import create_krx_client
    from src.krx_login_client import KrxLoginClient
    assert isinstance(create_krx_client(krx_id="t", krx_pw="t", krx_api_key="k"), KrxLoginClient)


def test_create_krx_client_no_credentials():
    from src.krx_client import create_krx_client
    with pytest.raises(ValueError, match="KRX 인증 정보가 없습니다"):
        create_krx_client()
