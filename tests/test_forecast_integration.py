"""End-to-end integration test with all external services mocked."""
from unittest.mock import patch, MagicMock
from datetime import date

import numpy as np
import pandas as pd

from src.forecast.models import ForecastResult
from src.forecast.macro_fetcher import MacroFetcher
from src.forecast.stock_fetcher import StockFetcher
from src.forecast.report import ReportGenerator


def test_full_pipeline_mocked(tmp_path):
    """Test the full pipeline: fetch -> predict -> report."""
    # 1. Mock macro data
    macro_fetcher = MacroFetcher(ecos_api_key="test", fred_api_key="test")

    mock_ecos_resp = MagicMock()
    mock_ecos_resp.json.return_value = {
        "StatisticSearch": {
            "row": [
                {"TIME": f"2026010{i}", "DATA_VALUE": str(2800 + i * 10)}
                for i in range(1, 6)
            ]
        }
    }
    mock_ecos_resp.raise_for_status = MagicMock()

    mock_fred_series = pd.Series(
        [4500.0 + i * 10 for i in range(5)],
        index=pd.to_datetime([f"2026-01-0{i}" for i in range(1, 6)]),
    )

    with patch("requests.get", return_value=mock_ecos_resp):
        ecos_data = macro_fetcher.fetch_all_ecos(lookback_days=400)

    with patch("fredapi.Fred") as MockFred:
        mock_fred = MagicMock()
        mock_fred.get_series.return_value = mock_fred_series
        MockFred.return_value = mock_fred
        fred_data = macro_fetcher.fetch_all_fred(lookback_days=400)

    all_macro = {**ecos_data, **fred_data}

    # 2. Build ForecastResults directly (skip actual TimesFM)
    macro_results = []
    for name, (dates, values) in all_macro.items():
        if not values:
            continue
        macro_results.append(ForecastResult(
            ticker=name,
            name=name,
            category="macro",
            history=values,
            dates_history=dates,
            forecast=[values[-1] + i for i in range(1, 4)],
            dates_forecast=["20260106", "20260107", "20260108"],
            quantile_low=[values[-1] - 10 + i for i in range(1, 4)],
            quantile_high=[values[-1] + 10 + i for i in range(1, 4)],
            predicted_return=1.5,
            uncertainty=3.0,
        ))

    stock_results = [
        ForecastResult(
            ticker="005930", name="삼성전자", category="stock",
            history=[50000.0, 51000.0, 52000.0],
            dates_history=["20260101", "20260102", "20260103"],
            forecast=[53000.0, 54000.0, 55000.0],
            dates_forecast=["20260106", "20260107", "20260108"],
            quantile_low=[51000.0, 52000.0, 53000.0],
            quantile_high=[55000.0, 56000.0, 57000.0],
            predicted_return=5.77,
            uncertainty=7.69,
        ),
    ]

    # 3. Generate report
    ai_analyses = {
        "005930": {
            "ai_analysis": "[상승 원인] 반도체 호황",
            "news_summary": "삼성전자 실적 발표",
        }
    }

    generator = ReportGenerator()
    path = generator.generate(
        macro_results=macro_results,
        stock_results=stock_results,
        ai_analyses=ai_analyses,
        scan_date="2026-04-16",
        output_dir=str(tmp_path),
    )

    assert path.endswith(".html")
    with open(path) as f:
        html = f.read()
    assert "삼성전자" in html
    assert "반도체 호황" in html
    assert len(html) > 1000  # should have substantial content
