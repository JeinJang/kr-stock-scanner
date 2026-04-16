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
            return_backcast=True,
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

    def _build_result(
        self,
        item: dict,
        forecast_values: list[float],
        q_low: list[float],
        q_high: list[float],
    ) -> ForecastResult:
        """Build a ForecastResult from prediction outputs."""
        last_price = item["history"][-1]
        final_price = forecast_values[-1]
        predicted_return = ((final_price - last_price) / last_price) * 100

        avg_spread = np.mean(np.array(q_high) - np.array(q_low))
        uncertainty = (avg_spread / last_price) * 100

        dates_forecast = self._generate_forecast_dates(
            item["dates_history"][-1], self._horizon,
        )

        return ForecastResult(
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
        )

    def predict_batch(
        self, items: list[dict],
    ) -> list[ForecastResult]:
        """Run batch prediction for multiple time series (no covariates)."""
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
            results.append(self._build_result(item, forecast_values, q_low, q_high))

        return results

    def predict_with_covariates(
        self,
        items: list[dict],
        macro_covariates: dict[str, tuple[list[float], list[float]]],
    ) -> list[ForecastResult]:
        """Run prediction with macro indicators as covariates.

        Args:
            items: List of stock items with keys: ticker, name, category,
                   sector, history, dates_history
            macro_covariates: Dict of indicator_name -> (history_values, forecast_values).
                              Each indicator provides historical + forecasted values
                              to be used as dynamic covariates.
        """
        if not items:
            return []

        inputs = [np.array(item["history"]) for item in items]

        # Build dynamic covariates: each must be len(history) + horizon per series
        # We align macro data to each stock's history length by taking the last N points
        dynamic_covs: dict[str, list[np.ndarray]] = {}

        for cov_name, (cov_hist, cov_forecast) in macro_covariates.items():
            cov_arrays = []
            for item in items:
                stock_len = len(item["history"])
                # Take last stock_len points from covariate history
                if len(cov_hist) >= stock_len:
                    aligned_hist = cov_hist[-stock_len:]
                else:
                    # Pad with first value if covariate history is shorter
                    pad_len = stock_len - len(cov_hist)
                    aligned_hist = [cov_hist[0]] * pad_len + list(cov_hist)

                # Concat history + forecast for this covariate
                full = np.array(aligned_hist + list(cov_forecast[:self._horizon]))
                cov_arrays.append(full)

            dynamic_covs[cov_name] = cov_arrays

        logger.info(
            f"Predicting {len(items)} stocks with {len(dynamic_covs)} covariates: "
            f"{list(dynamic_covs.keys())}"
        )

        # Run covariate forecast for point estimates
        cov_forecast, _ = self._model.forecast_with_covariates(
            inputs=inputs,
            dynamic_numerical_covariates=dynamic_covs,
            xreg_mode="xreg + timesfm",
            ridge=0.01,
            normalize_xreg_target_per_input=True,
        )

        # Run regular forecast for quantile estimates (covariates API doesn't return quantiles)
        _, quantile_forecast = self._model.forecast(
            horizon=self._horizon,
            inputs=inputs,
        )

        results = []
        for i, item in enumerate(items):
            forecast_values = cov_forecast[i].tolist()
            q_low = quantile_forecast[i, :, 1].tolist()
            q_high = quantile_forecast[i, :, 9].tolist()
            results.append(self._build_result(item, forecast_values, q_low, q_high))

        return results
