import os
from datetime import datetime, date

from src.related.db import StoredEdge
from src.related.graph import build_graph, expand
from src.related.report import ReportGenerator


def _e(src, tgt, name, rel="Customer"):
    return StoredEdge(
        source_ticker=src, target_ticker=tgt, target_name=name,
        relation=rel, evidence=f"{rel} link {src}→{tgt}",
        extracted_at=datetime(2026, 5, 6),
    )


def test_generate_report_creates_html(tmp_path):
    edges = [
        _e("042700", "000660", "SK하이닉스", "Customer"),
        _e("042700", "005930", "삼성전자", "Supplier"),
    ]
    g = build_graph(edges)
    sub = expand(g, "042700", depth=1)

    name_map = {"042700": "한미반도체", "000660": "SK하이닉스", "005930": "삼성전자"}
    market_map = {"042700": "KOSDAQ", "000660": "KOSPI", "005930": "KOSPI"}

    gen = ReportGenerator()
    path = gen.generate(
        root_ticker="042700",
        graph=sub,
        edges=edges,
        name_map=name_map,
        market_map=market_map,
        scores={},
        depth=1,
        rcept_no="20250318000123",
        as_of_date=str(date(2026, 5, 6)),
        output_dir=str(tmp_path),
    )
    assert os.path.exists(path)
    with open(path) as f:
        html = f.read()
    assert "한미반도체" in html
    assert "SK하이닉스" in html
    assert "삼성전자" in html
    assert "Customer" in html or "고객" in html
    assert "Plotly" in html or "plotly" in html
