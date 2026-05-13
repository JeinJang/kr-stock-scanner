from datetime import datetime
import pytest

from src.related.db import RelatedDB
from src.related.models import ExtractedEdge


@pytest.fixture
def db(tmp_path):
    return RelatedDB(url=f"sqlite:///{tmp_path}/test.db")


def test_meta_save_and_load(db):
    assert db.get_meta("005930") is None
    db.set_meta("005930", "20250318000123", datetime(2026, 5, 6))
    meta = db.get_meta("005930")
    assert meta is not None
    assert meta.rcept_no == "20250318000123"


def test_needs_refresh(db):
    assert db.needs_refresh("005930", "20260101000001") is True
    db.set_meta("005930", "20260101000001", datetime(2026, 5, 6))
    assert db.needs_refresh("005930", "20260101000001") is False
    assert db.needs_refresh("005930", "20260301000002") is True  # new rcept_no


def test_save_and_load_edges(db):
    edges = [
        ExtractedEdge(name="SK하이닉스", ticker="000660",
                      relation="Customer", evidence="HBM 장비 납품"),
        ExtractedEdge(name="ASML", ticker=None,
                      relation="Supplier", evidence="EUV 도입"),
    ]
    db.save_edges("042700", edges, extracted_at=datetime(2026, 5, 6))

    loaded = db.load_edges(source_tickers=["042700"])
    assert len(loaded) == 2
    by_name = {e.target_name: e for e in loaded}
    assert by_name["SK하이닉스"].target_ticker == "000660"
    assert by_name["ASML"].target_ticker is None


def test_save_edges_replaces_old(db):
    """Re-saving for same source replaces previous edges."""
    e1 = ExtractedEdge(name="A", ticker="111111", relation="Customer", evidence="old")
    db.save_edges("042700", [e1], datetime(2026, 1, 1))

    e2 = ExtractedEdge(name="B", ticker="222222", relation="Supplier", evidence="new")
    db.save_edges("042700", [e2], datetime(2026, 5, 6))

    loaded = db.load_edges(source_tickers=["042700"])
    assert len(loaded) == 1
    assert loaded[0].target_name == "B"


def test_stats(db):
    db.save_edges("042700", [
        ExtractedEdge(name="X", ticker="111111", relation="Customer", evidence="a"),
        ExtractedEdge(name="Y", ticker="222222", relation="Supplier", evidence="b"),
    ], datetime(2026, 5, 6))
    db.save_edges("005930", [
        ExtractedEdge(name="Z", ticker="333333", relation="Affiliate", evidence="c"),
    ], datetime(2026, 5, 6))

    stats = db.stats()
    assert stats["sources"] == 2
    assert stats["edges"] == 3
    assert stats["by_relation"]["Customer"] == 1
    assert stats["by_relation"]["Supplier"] == 1
    assert stats["by_relation"]["Affiliate"] == 1
