from unittest.mock import MagicMock
import numpy as np

from src.forecast.predictor import Predictor
from src.forecast.models import ForecastResult


def test_predict_single_via_batch():
    """Test prediction for a single time series via predict_batch."""
    predictor = Predictor.__new__(Predictor)
    predictor._model = MagicMock()
    predictor._horizon = 5

    point = np.array([[105.0, 106.0, 107.0, 108.0, 109.0]])
    quantiles = np.zeros((1, 5, 10))
    quantiles[0, :, 1] = [102.0, 103.0, 104.0, 105.0, 106.0]  # 10th pct
    quantiles[0, :, 9] = [108.0, 109.0, 110.0, 111.0, 112.0]  # 90th pct

    predictor._model.forecast.return_value = (point, quantiles)

    items = [{
        "ticker": "005930", "name": "Samsung", "category": "stock",
        "history": [100.0, 101.0, 102.0, 103.0, 104.0],
        "dates_history": ["20260101", "20260102", "20260103", "20260106", "20260107"],
    }]

    results = predictor.predict_batch(items)
    result = results[0]

    assert isinstance(result, ForecastResult)
    assert result.ticker == "005930"
    assert len(result.forecast) == 5
    assert len(result.quantile_low) == 5
    assert result.predicted_return > 0
    assert all("-" in d for d in result.dates_forecast)  # YYYY-MM-DD format


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


def test_predict_with_covariates():
    """Test prediction with macro covariates."""
    predictor = Predictor.__new__(Predictor)
    predictor._model = MagicMock()
    predictor._horizon = 3

    # Mock covariate forecast output: (cov_forecast, xreg_forecast)
    cov_point = [np.array([106.0, 107.0, 108.0])]
    predictor._model.forecast_with_covariates.return_value = (cov_point, cov_point)

    # Mock regular forecast for quantiles
    quantiles = np.zeros((1, 3, 10))
    quantiles[0, :, 1] = [103.0, 104.0, 105.0]
    quantiles[0, :, 9] = [109.0, 110.0, 111.0]
    predictor._model.forecast.return_value = (np.array([[106.0, 107.0, 108.0]]), quantiles)

    items = [
        {
            "ticker": "005930", "name": "Samsung", "category": "stock",
            "history": [100.0, 101.0, 102.0, 103.0, 104.0],
            "dates_history": ["20260101", "20260102", "20260103", "20260106", "20260107"],
        },
    ]

    macro_cov = {
        "KOSPI": ([2800.0, 2810.0, 2820.0, 2830.0, 2840.0], [2850.0, 2860.0, 2870.0]),
        "USD_KRW": ([1300.0, 1305.0, 1310.0, 1315.0, 1320.0], [1325.0, 1330.0, 1335.0]),
    }

    results = predictor.predict_with_covariates(items, macro_cov)

    assert len(results) == 1
    assert results[0].ticker == "005930"
    assert results[0].forecast == [106.0, 107.0, 108.0]
    # Covariates were passed
    predictor._model.forecast_with_covariates.assert_called_once()
    call_kwargs = predictor._model.forecast_with_covariates.call_args
    assert "KOSPI" in call_kwargs.kwargs["dynamic_numerical_covariates"]
    assert "USD_KRW" in call_kwargs.kwargs["dynamic_numerical_covariates"]
