from src.forecast.models import ForecastResult


def test_forecast_result_creation():
    result = ForecastResult(
        ticker="005930",
        name="Samsung Electronics",
        category="stock",
        history=[100.0, 101.0, 102.0],
        dates_history=["2026-01-01", "2026-01-02", "2026-01-03"],
        forecast=[103.0, 104.0],
        dates_forecast=["2026-01-04", "2026-01-05"],
        quantile_low=[101.0, 102.0],
        quantile_high=[105.0, 106.0],
        predicted_return=3.0,
        uncertainty=4.0,
    )
    assert result.ticker == "005930"
    assert result.category == "stock"
    assert len(result.forecast) == 2
    assert result.predicted_return == 3.0


def test_forecast_result_macro():
    result = ForecastResult(
        ticker="KOSPI",
        name="KOSPI",
        category="macro",
        history=[2800.0, 2810.0],
        dates_history=["2026-01-01", "2026-01-02"],
        forecast=[2820.0],
        dates_forecast=["2026-01-03"],
        quantile_low=[2790.0],
        quantile_high=[2850.0],
        predicted_return=0.71,
        uncertainty=2.14,
    )
    assert result.category == "macro"
