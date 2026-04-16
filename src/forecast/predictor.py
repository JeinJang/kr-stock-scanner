from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from loguru import logger

from src.forecast.models import ForecastResult


class Predictor:
    """Wraps TimesFM model for time-series forecasting."""

    def __init__(self, model_name: str, horizon: int = 60):
        import timesfm

        self._horizon = horizon
        logger.info(f"Loading TimesFM model: {model_name}")
        self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_name)
        self._model.compile(timesfm.ForecastConfig(
            max_context=512,
            max_horizon=horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
        ))
        logger.info("TimesFM model loaded and compiled")

    def _generate_forecast_dates(self, last_date_str: str, n: int) -> list[str]:
        """Generate n future business dates starting from last_date."""
        fmt = "%Y-%m-%d" if "-" in last_date_str else "%Y%m%d"
        last = datetime.strptime(last_date_str, fmt)
        dates = []
        current = last
        while len(dates) < n:
            current += timedelta(days=1)
            if current.weekday() < 5:  # skip weekends
                dates.append(current.strftime("%Y-%m-%d"))
        return dates

    def predict_single(
        self,
        ticker: str,
        name: str,
        category: str,
        history: list[float],
        dates_history: list[str],
    ) -> ForecastResult:
        """Run prediction for a single time series."""
        inputs = [np.array(history)]
        point_forecast, quantile_forecast = self._model.forecast(
            horizon=self._horizon,
            inputs=inputs,
        )

        forecast_values = point_forecast[0].tolist()
        q_low = quantile_forecast[0, :, 1].tolist()   # 10th percentile
        q_high = quantile_forecast[0, :, 9].tolist()   # 90th percentile

        last_price = history[-1]
        final_price = forecast_values[-1]
        predicted_return = ((final_price - last_price) / last_price) * 100

        avg_spread = np.mean(np.array(q_high) - np.array(q_low))
        uncertainty = (avg_spread / last_price) * 100

        dates_forecast = self._generate_forecast_dates(dates_history[-1], self._horizon)

        return ForecastResult(
            ticker=ticker,
            name=name,
            category=category,
            history=history,
            dates_history=dates_history,
            forecast=forecast_values,
            dates_forecast=dates_forecast,
            quantile_low=q_low,
            quantile_high=q_high,
            predicted_return=round(predicted_return, 2),
            uncertainty=round(uncertainty, 2),
        )

    def predict_batch(
        self, items: list[dict],
    ) -> list[ForecastResult]:
        """Run batch prediction for multiple time series at once."""
        if not items:
            return []

        inputs = [np.array(item["history"]) for item in items]
        point_forecast, quantile_forecast = self._model.forecast(
            horizon=self._horizon,
            inputs=inputs,
        )

        results = []
        for i, item in enumerate(items):
            forecast_values = point_forecast[i].tolist()
            q_low = quantile_forecast[i, :, 1].tolist()
            q_high = quantile_forecast[i, :, 9].tolist()

            last_price = item["history"][-1]
            final_price = forecast_values[-1]
            predicted_return = ((final_price - last_price) / last_price) * 100

            avg_spread = np.mean(np.array(q_high) - np.array(q_low))
            uncertainty = (avg_spread / last_price) * 100

            dates_forecast = self._generate_forecast_dates(
                item["dates_history"][-1], self._horizon,
            )

            results.append(ForecastResult(
                ticker=item["ticker"],
                name=item["name"],
                category=item["category"],
                sector=item.get("sector", ""),
                history=item["history"],
                dates_history=item["dates_history"],
                forecast=forecast_values,
                dates_forecast=dates_forecast,
                quantile_low=q_low,
                quantile_high=q_high,
                predicted_return=round(predicted_return, 2),
                uncertainty=round(uncertainty, 2),
            ))

        return results
