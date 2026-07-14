import io
import zipfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.related.report_fetcher import ReportFetcher


@pytest.mark.asyncio
async def test_latest_rcept_no_picks_annual_report():
    client = MagicMock()
    client.get = AsyncMock(return_value={
        "status": "000",
        "list": [
            {"rcept_no": "20250318000123", "report_nm": "사업보고서 (2024.12)"},
            {"rcept_no": "20250101000050", "report_nm": "분기보고서 (2024.09)"},
        ],
    })
    fetcher = ReportFetcher(client=client)

    rcept_no = await fetcher.latest_rcept_no("00126380")
    assert rcept_no == "20250318000123"


@pytest.mark.asyncio
async def test_latest_rcept_no_none_when_empty():
    client = MagicMock()
    client.get = AsyncMock(return_value={"status": "013", "list": []})
    fetcher = ReportFetcher(client=client)
    assert await fetcher.latest_rcept_no("00000000") is None


def test_parse_sections_extracts_four_blocks():
    """Section parser splits on Korean section headers."""
    xml_text = """<?xml version="1.0"?>
<doc>
  <p>I. 회사의 개요</p>
  <p>설립일: 1969...</p>
  <p>II. 사업의 내용</p>
  <p>당사의 주요 제품은 메모리 반도체이며 주요 고객사는 Apple, Google입니다.</p>
  <p>III. 재무에 관한 사항</p>
  <p>매출액 300조원...</p>
  <p>특수관계자 거래</p>
  <p>삼성디스플레이로부터 패널 매입.</p>
  <p>다음 절</p>
  <p>IX. 계열회사 등에 관한 사항</p>
  <p>삼성SDI, 삼성디스플레이, 삼성SDS</p>
  <p>X. 대주주 등과의 거래내용</p>
  <p>이재용 회장 보유 지분 ...</p>
  <p>XI. 추가</p>
  <p>기타</p>
</doc>"""
    fetcher = ReportFetcher(client=MagicMock())
    sections = fetcher.parse_sections(
        xml_text, corp_code="00126380", rcept_no="20250318000123",
    )
    assert "메모리 반도체" in sections.business_content
    assert "삼성SDI" in sections.affiliates
    assert "이재용" in sections.related_party
    assert "삼성디스플레이로부터 패널" in sections.related_party_notes


@pytest.mark.asyncio
async def test_download_document_picks_main_entry_from_multifile_zip():
    """대기업 보고서 zip은 부속문서가 앞에 오므로 본문 `{rcept_no}.xml` 을 골라야 한다."""
    rcept_no = "20260320000859"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        # 첫 항목은 부속문서 — 예전 코드는 이걸 읽어 섹션이 비었다.
        zf.writestr(f"{rcept_no}_00760.xml", "부속문서 감사보고서")
        zf.writestr(f"{rcept_no}_00761.xml", "부속문서 내부회계")
        zf.writestr(f"{rcept_no}.xml", "본문: 사업의 내용 및 계열회사 등")
    zip_bytes = buf.getvalue()

    resp = MagicMock()
    resp.content = zip_bytes
    resp.raise_for_status = MagicMock()
    http = AsyncMock()
    http.get = AsyncMock(return_value=resp)
    http.__aenter__ = AsyncMock(return_value=http)
    http.__aexit__ = AsyncMock(return_value=False)

    fetcher = ReportFetcher(client=MagicMock(_api_key="k"))
    with patch("src.related.report_fetcher.httpx.AsyncClient", return_value=http):
        text = await fetcher.download_document(rcept_no)

    assert "본문" in text
    assert "부속문서" not in text
