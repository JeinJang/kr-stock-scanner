from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import plotly
from jinja2 import Environment, FileSystemLoader
from loguru import logger

from src.fundamentals.models import FundamentalsMetrics, ScoreCard


def _tojson_unicode(value: object) -> str:
    """Jinja2 filter: serialize to JSON with non-ASCII characters preserved."""
    return json.dumps(value, ensure_ascii=False)


class ReportGenerator:
    """Generates the fundamentals HTML report."""

    def __init__(self):
        template_dir = Path(__file__).parent / "templates"
        self._env = Environment(loader=FileSystemLoader(str(template_dir)))
        self._env.filters["tojson"] = _tojson_unicode

    def generate(
        self,
        metrics: list[FundamentalsMetrics],
        scores: list[ScoreCard],
        name_map: dict[str, str],
        market_map: dict[str, str],
        as_of_date: str,
        output_dir: str = "reports",
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"fundamentals-{as_of_date}.html")

        metrics_by_ticker = {m.ticker: m for m in metrics}

        # Build flat dict per stock for template
        stock_data = []
        for s in scores:
            m = metrics_by_ticker.get(s.ticker)
            stock_data.append({
                "ticker": s.ticker,
                "name": name_map.get(s.ticker, s.ticker),
                "market": market_map.get(s.ticker, ""),
                "grade": s.grade,
                "total_score": s.total_score,
                "liquidity_score": s.liquidity_score,
                "profitability_score": s.profitability_score,
                "growth_score": s.growth_score,
                "cashflow_score": s.cashflow_score,
                "categories": s.categories,
                "pe": m.pe if m else None,
                "pb": m.pb if m else None,
                "roe": m.roe if m else None,
            })

        # Category counts (count distinct memberships, multi-label)
        cat_counter: Counter = Counter()
        for s in scores:
            for c in s.categories:
                cat_counter[c] += 1

        plotly_js = plotly.offline.get_plotlyjs()
        template = self._env.get_template("report.html")
        html = template.render(
            as_of_date=as_of_date,
            total_count=len(stock_data),
            stock_data=stock_data,
            category_counts=dict(cat_counter),
            plotly_js=plotly_js,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Report written: {output_path}")
        return output_path
