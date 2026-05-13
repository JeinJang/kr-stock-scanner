from datetime import datetime
from src.related.db import StoredEdge
from src.related.graph import build_graph, expand


def _e(src, tgt, name, rel="Customer"):
    return StoredEdge(
        source_ticker=src, target_ticker=tgt, target_name=name,
        relation=rel, evidence="ev", extracted_at=datetime(2026, 5, 6),
    )


def test_build_graph_basic():
    edges = [
        _e("042700", "000660", "SK하이닉스", "Customer"),
        _e("042700", "005930", "삼성전자", "Supplier"),
    ]
    g = build_graph(edges)
    assert "042700" in g.nodes
    assert g.has_edge("042700", "000660")
    assert g.edges["042700", "000660"]["relation"] == "Customer"


def test_build_graph_unlisted_target_uses_virtual_node():
    edges = [
        _e("042700", None, "ASML", "Supplier"),
    ]
    g = build_graph(edges)
    virtual = [n for n in g.nodes if n.startswith("_unlisted_")]
    assert len(virtual) == 1
    assert "ASML" in virtual[0]


def test_expand_1_hop():
    edges = [
        _e("042700", "000660", "SK하이닉스", "Customer"),
        _e("042700", "005930", "삼성전자", "Supplier"),
        _e("000660", "999999", "Other", "Competitor"),  # 2-hop neighbor
    ]
    g = build_graph(edges)
    sub = expand(g, "042700", depth=1)
    assert set(sub.nodes) == {"042700", "000660", "005930"}


def test_expand_2_hop():
    edges = [
        _e("042700", "000660", "SK하이닉스", "Customer"),
        _e("000660", "999999", "Other", "Competitor"),
    ]
    g = build_graph(edges)
    sub = expand(g, "042700", depth=2)
    assert "999999" in sub.nodes


def test_expand_includes_inbound_edges():
    """Expand should follow edges in both directions."""
    edges = [
        _e("005930", "042700", "한미반도체", "Customer"),  # 005930 → 042700
    ]
    g = build_graph(edges)
    sub = expand(g, "042700", depth=1)
    assert "005930" in sub.nodes
