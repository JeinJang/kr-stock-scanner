from __future__ import annotations

import os
from pathlib import Path

import plotly
from jinja2 import Environment, FileSystemLoader
from loguru import logger

from src.forecast.models import ForecastResult
from src.forecast.macro_fetcher import INDICATOR_NAMES


class ReportGenerator:
    """Generates an interactive HTML forecast report."""

    def __init__(self):
        template_dir = Path(__file__).parent / "templates"
        self._env = Environment(loader=FileSystemLoader(str(template_dir)))

    def generate(
        self,
        macro_results: list[ForecastResult],
        stock_results: list[ForecastResult],
        ai_analyses: dict[str, dict],
        scan_date: str,
        output_dir: str = "reports",
        horizon: int = 60,
    ) -> str:
        """Generate HTML report and return the file path."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"forecast-{scan_date}.html")

        ranked_stocks = sorted(stock_results, key=lambda r: r.predicted_return, reverse=True)

        # Stocks that have AI analysis but aren't in the stock forecast list
        forecasted_tickers = {s.ticker for s in stock_results}
        ai_only_stocks = {
            ticker: data for ticker, data in ai_analyses.items()
            if ticker not in forecasted_tickers
        }

        # Inline plotly.js for offline use
        plotly_js = plotly.offline.get_plotlyjs()

        template = self._env.get_template("report.html")
        html = template.render(
            scan_date=scan_date,
            horizon=horizon,
            macro_results=macro_results,
            stock_results=stock_results,
            ranked_stocks=ranked_stocks,
            ai_analyses=ai_analyses,
            ai_only_stocks=ai_only_stocks,
            indicator_names=INDICATOR_NAMES,
            plotly_js=plotly_js,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"Report generated: {output_path}")
        return output_path
