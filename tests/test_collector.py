# tests/test_collector.py
from unittest.mock import patch
import pandas as pd


def test_collect_daily_ohlcv():
    """Collector should fetch OHLCV for KOSPI, KOSDAQ, and ETF."""
    from src.collector import Collector

    kospi_df = pd.DataFrame(
        {"시가": [70000], "고가": [72000], "저가": [69000], "종가": [71500],
         "거래량": [15000000], "거래대금": [1000000000], "등락률": [1.5]},
        index=["005930"],
    )
    kosdaq_df = pd.DataFrame(
        {"시가": [50000], "고가": [52000], "저가": [49000], "종가": [51000],
         "거래량": [3000000], "거래대금": [150000000], "등락률": [2.0]},
        index=["035420"],
    )
    etf_df = pd.DataFrame(
        {"NAV": [50000], "시가": [49800], "고가": [50500], "저가": [49500],
         "종가": [50200], "거래량": [500000], "거래대금": [25000000], "기초지수": [2500]},
        index=["069500"],
    )

    with patch("src.collector.stock") as mock_stock:
        mock_stock.get_market_ohlcv_by_ticker.side_effect = [kospi_df, kosdaq_df]
        mock_stock.get_etf_ohlcv_by_ticker.return_value = etf_df

        collector = Collector()
        result = collector.collect_daily("20260219", markets=["KOSPI", "KOSDAQ", "ETF"])

    assert "005930" in result
    assert result["005930"]["market"] == "KOSPI"
    assert result["005930"]["close"] == 71500
    assert result["005930"]["high"] == 72000
    assert "069500" in result
    assert result["069500"]["market"] == "ETF"


def test_collect_historical_high():
    """Collector should fetch 52-week high for a ticker."""
    from src.collector import Collector

    hist_df = pd.DataFrame(
        {"시가": [65000, 68000, 70000], "고가": [66000, 69000, 72000],
         "저가": [64000, 67000, 69000], "종가": [65500, 68500, 71500],
         "거래량": [10000000, 12000000, 15000000]},
        index=pd.to_datetime(["2025-06-01", "2025-12-01", "2026-02-19"]),
    )

    with patch("src.collector.stock") as mock_stock:
        mock_stock.get_market_ohlcv_by_date.return_value = hist_df

        collector = Collector()
        high_52w = collector.get_52w_high("005930", "20260219", lookback=250)

    assert high_52w == 72000


def test_collect_sector_info():
    """Collector should fetch sector classifications."""
    from src.collector import Collector

    sector_df = pd.DataFrame(
        {"종목명": ["삼성전자", "NAVER"], "업종명": ["전기전자", "서비스업"],
         "종가": [71500, 220000], "대비": [1000, 3000],
         "등락률": [1.42, 1.38], "시가총액": [450000000000000, 36000000000000]},
        index=["005930", "035420"],
    )

    with patch("src.collector.stock") as mock_stock:
        mock_stock.get_market_sector_classifications.return_value = sector_df

        collector = Collector()
        sectors = collector.get_sector_map("20260219", "KOSPI")

    assert sectors["005930"] == "전기전자"
    assert sectors["035420"] == "서비스업"


def test_collect_market_cap():
    """Collector should fetch market cap data."""
    from src.collector import Collector

    cap_df = pd.DataFrame(
        {"시가총액": [450000000000000, 36000000000000],
         "거래량": [15000000, 3000000], "거래대금": [1000000000, 600000000],
         "상장주식수": [5969782550, 163750000], "외국인보유주식수": [0, 0]},
        index=["005930", "035420"],
    )

    with patch("src.collector.stock") as mock_stock:
        mock_stock.get_market_cap_by_ticker.return_value = cap_df

        collector = Collector()
        caps = collector.get_market_caps("20260219")

    assert caps["005930"] == 450000000000000
