import os
from src.forecast.report import ReportGenerator
from src.forecast.models import ForecastResult


def _make_result(ticker: str, name: str, category: str) -> ForecastResult:
    return ForecastResult(
        ticker=ticker,
        name=name,
        category=category,
        history=[100.0, 101.0, 102.0],
        dates_history=["20260101", "20260102", "20260103"],
        forecast=[103.0, 104.0, 105.0],
        dates_forecast=["20260106", "20260107", "20260108"],
        quantile_low=[101.0, 102.0, 103.0],
        quantile_high=[105.0, 106.0, 107.0],
        predicted_return=2.94,
        uncertainty=3.92,
    )


def test_generate_report(tmp_path):
    macro_results = [
        _make_result("KOSPI", "KOSPI", "macro"),
        _make_result("SP500", "S&P 500", "macro"),
    ]
    stock_results = [
        _make_result("005930", "삼성전자", "stock"),
        _make_result("035720", "카카오", "stock"),
    ]

    generator = ReportGenerator()
    path = generator.generate(
        macro_results=macro_results,
        stock_results=stock_results,
        ai_analyses={},
        scan_date="2026-04-16",
        output_dir=str(tmp_path),
    )

    assert os.path.exists(path)
    with open(path) as f:
        html = f.read()
    assert "KOSPI" in html
    assert "삼성전자" in html
    assert "plotly" in html.lower() or "Plotly" in html


def test_report_with_ai_analysis(tmp_path):
    stock_results = [_make_result("005930", "삼성전자", "stock")]
    ai_analyses = {
        "005930": {
            "ai_analysis": "[상승 원인] 반도체 호황\n[핵심 뉴스] HBM 수주 확대\n[투자 포인트] AI 수요",
            "news_summary": "삼성전자 실적 호조",
        }
    }

    generator = ReportGenerator()
    path = generator.generate(
        macro_results=[],
        stock_results=stock_results,
        ai_analyses=ai_analyses,
        scan_date="2026-04-16",
        output_dir=str(tmp_path),
    )

    with open(path) as f:
        html = f.read()
    assert "반도체 호황" in html
