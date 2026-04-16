from pydantic import BaseModel


class ForecastResult(BaseModel):
    """Prediction result for a single ticker or macro indicator."""

    ticker: str
    name: str
    category: str  # "macro" | "stock"
    sector: str = ""
    history: list[float]
    dates_history: list[str]
    forecast: list[float]
    dates_forecast: list[str]
    quantile_low: list[float]
    quantile_high: list[float]
    predicted_return: float
    uncertainty: float
