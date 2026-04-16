from unittest.mock import MagicMock
import pandas as pd
from src.forecast.stock_fetcher import StockFetcher


def test_fetch_stock_history():
    mock_client = MagicMock()
    mock_client.supports_history = True
    mock_df = pd.DataFrame(
        {"종가": [50000, 51000, 52000]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
    )
    mock_client.get_market_ohlcv_by_date.return_value = mock_df

    fetcher = StockFetcher(client=mock_client)
    dates, values = fetcher.fetch_history("005930", end_date="20260103", lookback_days=250)

    assert len(values) == 3
    assert values == [50000.0, 51000.0, 52000.0]
    mock_client.get_market_ohlcv_by_date.assert_called_once()


def test_fetch_multiple_stocks():
    mock_client = MagicMock()
    mock_client.supports_history = True

    def make_df(prices):
        return pd.DataFrame(
            {"종가": prices},
            index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
        )

    mock_client.get_market_ohlcv_by_date.side_effect = [
        make_df([50000, 51000]),
        make_df([10000, 10500]),
    ]

    fetcher = StockFetcher(client=mock_client)
    results = fetcher.fetch_histories(
        tickers=["005930", "035720"],
        end_date="20260103",
        lookback_days=250,
    )
    assert "005930" in results
    assert "035720" in results
    assert len(results["005930"][1]) == 2
