from unittest.mock import MagicMock
import numpy as np

from src.forecast.predictor import Predictor
from src.forecast.models import ForecastResult


def test_predict_single():
    """Test prediction for a single time series."""
    predictor = Predictor.__new__(Predictor)
    predictor._model = MagicMock()
    predictor._horizon = 5

    # Mock TimesFM forecast output
    # point_forecast shape: (1, horizon)
    # quantile_forecast shape: (1, horizon, 10) — mean + 9 quantile levels
    point = np.array([[105.0, 106.0, 107.0, 108.0, 109.0]])
    quantiles = np.zeros((1, 5, 10))
    quantiles[0, :, 0] = [105.0, 106.0, 107.0, 108.0, 109.0]  # mean
    quantiles[0, :, 1] = [102.0, 103.0, 104.0, 105.0, 106.0]  # 10th pct
    quantiles[0, :, 9] = [108.0, 109.0, 110.0, 111.0, 112.0]  # 90th pct

    predictor._model.forecast.return_value = (point, quantiles)

    history = [100.0, 101.0, 102.0, 103.0, 104.0]
    dates_hist = ["20260101", "20260102", "20260103", "20260106", "20260107"]

    result = predictor.predict_single(
        ticker="005930",
        name="Samsung",
        category="stock",
        history=history,
        dates_history=dates_hist,
    )

    assert isinstance(result, ForecastResult)
    assert result.ticker == "005930"
    assert len(result.forecast) == 5
    assert len(result.quantile_low) == 5
    assert len(result.quantile_high) == 5
    assert result.predicted_return > 0  # price went up


def test_predict_batch():
    """Test batch prediction for multiple series."""
    predictor = Predictor.__new__(Predictor)
    predictor._model = MagicMock()
    predictor._horizon = 3

    point = np.array([
        [105.0, 106.0, 107.0],
        [2850.0, 2860.0, 2870.0],
    ])
    quantiles = np.zeros((2, 3, 10))
    # Series 1
    quantiles[0, :, 0] = [105.0, 106.0, 107.0]
    quantiles[0, :, 1] = [103.0, 104.0, 105.0]
    quantiles[0, :, 9] = [107.0, 108.0, 109.0]
    # Series 2
    quantiles[1, :, 0] = [2850.0, 2860.0, 2870.0]
    quantiles[1, :, 1] = [2830.0, 2840.0, 2850.0]
    quantiles[1, :, 9] = [2870.0, 2880.0, 2890.0]

    predictor._model.forecast.return_value = (point, quantiles)

    items = [
        {
            "ticker": "005930", "name": "Samsung", "category": "stock",
            "history": [100.0, 101.0, 102.0, 103.0, 104.0],
            "dates_history": ["20260101", "20260102", "20260103", "20260106", "20260107"],
        },
        {
            "ticker": "KOSPI", "name": "KOSPI", "category": "macro",
            "history": [2800.0, 2810.0, 2820.0, 2830.0, 2840.0],
            "dates_history": ["20260101", "20260102", "20260103", "20260106", "20260107"],
        },
    ]

    results = predictor.predict_batch(items)
    assert len(results) == 2
    assert results[0].ticker == "005930"
    assert results[1].ticker == "KOSPI"
