import os
from datetime import date

from src.fundamentals.models import FundamentalsMetrics, ScoreCard
from src.fundamentals.report import ReportGenerator


def _metrics(ticker, **kwargs):
    return FundamentalsMetrics(ticker=ticker, as_of_date=date(2026, 4, 17), **kwargs)


def _score(ticker, total, cats, **kwargs):
    return ScoreCard(
        ticker=ticker, as_of_date=date(2026, 4, 17),
        total_score=total, grade="★★★★☆", categories=cats,
        **kwargs,
    )


def test_generate_report_creates_html(tmp_path):
    metrics = [
        _metrics("005930", roe=15.0, pe=12.0, pb=1.4),
        _metrics("000660", roe=20.0, pe=8.0, pb=0.9),
    ]
    scores = [
        _score("005930", 75.0, ["Quality"], liquidity_score=20),
        _score("000660", 80.0, ["Quality", "Value"], liquidity_score=22),
    ]
    name_map = {"005930": "삼성전자", "000660": "SK하이닉스"}
    market_map = {"005930": "KOSPI", "000660": "KOSPI"}

    gen = ReportGenerator()
    path = gen.generate(
        metrics=metrics, scores=scores,
        name_map=name_map, market_map=market_map,
        as_of_date="2026-04-17",
        output_dir=str(tmp_path),
    )

    assert os.path.exists(path)
    with open(path) as f:
        html = f.read()
    assert "삼성전자" in html
    assert "SK하이닉스" in html
    assert "Quality" in html
    assert "Plotly" in html or "plotly" in html
