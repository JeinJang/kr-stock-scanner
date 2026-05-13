from datetime import datetime
from src.related.models import ReportSections, ExtractedEdge


def test_report_sections_creation():
    rs = ReportSections(
        corp_code="00126380",
        rcept_no="20250318000123",
        business_content="당사는 반도체를 제조한다...",
        affiliates="삼성디스플레이, 삼성SDI...",
        related_party="대주주: 이재용...",
        related_party_notes="특수관계자 거래: 삼성디스플레이...",
    )
    assert rs.corp_code == "00126380"
    assert "반도체" in rs.business_content


def test_extracted_edge_creation():
    e = ExtractedEdge(
        name="SK하이닉스",
        ticker="000660",
        relation="Customer",
        evidence="HBM 후공정 장비 납품",
    )
    assert e.name == "SK하이닉스"
    assert e.relation == "Customer"
    assert e.ticker == "000660"


def test_extracted_edge_unlisted():
    """ticker can be None for unlisted / foreign companies."""
    e = ExtractedEdge(
        name="ASML",
        ticker=None,
        relation="Supplier",
        evidence="EUV 노광장비 도입",
    )
    assert e.ticker is None
