from __future__ import annotations

import io
import re
import zipfile

import httpx
from bs4 import BeautifulSoup
from loguru import logger

from src.dart.client import DartClient
from src.related.models import ReportSections


# Section header patterns. DART numbering can drift (II vs III), so match
# the Korean title with optional roman-numeral prefix.
_HEADER_BUSINESS = re.compile(r"(?:[IVX]+\.\s*)?사업의\s*내용")
_HEADER_AFFILIATES = re.compile(r"(?:[IVX]+\.\s*)?계열회사\s*등")
_HEADER_RELATED_PARTY = re.compile(r"(?:[IVX]+\.\s*)?대주주\s*등과의\s*거래")
_HEADER_RP_NOTES = re.compile(r"특수관계자\s*거래")
# Any roman-numeral section header — used as the terminator.
_HEADER_ANY = re.compile(r"^[IVX]+\.\s")


class ReportFetcher:
    """Fetches and parses DART business report sections."""

    def __init__(self, client: DartClient):
        self._client = client

    async def latest_rcept_no(self, corp_code: str) -> str | None:
        """Return rcept_no of the latest 사업보고서 for this corp, or None.

        DART list.json requires bgn_de/end_de; without them it returns no data.
        We search the last 2 years to safely cover the latest annual report
        (filed within 90 days of fiscal year end).
        """
        from datetime import datetime, timedelta
        today = datetime.now()
        bgn = (today - timedelta(days=730)).strftime("%Y%m%d")
        end = today.strftime("%Y%m%d")
        data = await self._client.get(
            "/api/list.json",
            params={
                "corp_code": corp_code,
                "bgn_de": bgn,
                "end_de": end,
                "pblntf_detail_ty": "A001",
                "page_count": "20",
                "sort": "date",
                "sort_mth": "desc",
            },
        )
        if data.get("status") != "000":
            return None
        for row in data.get("list", []):
            name = row.get("report_nm", "")
            if "사업보고서" in name:
                return row.get("rcept_no")
        return None

    async def download_document(self, rcept_no: str) -> str:
        """Download document XML for rcept_no. Returns decoded UTF-8 XML text."""
        url = "https://opendart.fss.or.kr/api/document.xml"
        async with httpx.AsyncClient(timeout=120.0) as http:
            resp = await http.get(
                url, params={"crtfc_key": self._client._api_key, "rcept_no": rcept_no},
            )
            resp.raise_for_status()
            content = resp.content
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as f:
                raw = f.read()
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("euc-kr", errors="ignore")

    def parse_sections(
        self, xml_text: str, corp_code: str, rcept_no: str,
    ) -> ReportSections:
        """Split XML text into 4 relevant sections."""
        soup = BeautifulSoup(xml_text, "html.parser")
        lines: list[str] = []
        for el in soup.find_all(["p", "td", "li", "h1", "h2", "h3", "title"]):
            t = el.get_text(strip=True)
            if t:
                lines.append(t)

        buckets: dict[str, list[str]] = {
            "business": [],
            "affiliates": [],
            "related_party": [],
            "rp_notes": [],
        }
        current: str | None = None

        for line in lines:
            if _HEADER_BUSINESS.search(line) and _HEADER_ANY.match(line):
                current = "business"
                continue
            if _HEADER_AFFILIATES.search(line) and _HEADER_ANY.match(line):
                current = "affiliates"
                continue
            if _HEADER_RELATED_PARTY.search(line) and _HEADER_ANY.match(line):
                current = "related_party"
                continue
            if _HEADER_RP_NOTES.search(line):
                current = "rp_notes"
                continue
            if current == "rp_notes" and _HEADER_ANY.match(line):
                current = None
            elif current in {"business", "affiliates", "related_party"} and _HEADER_ANY.match(line):
                current = None

            if current is not None:
                buckets[current].append(line)

        def join(name: str) -> str:
            return "\n".join(buckets[name]).strip()

        return ReportSections(
            corp_code=corp_code,
            rcept_no=rcept_no,
            business_content=join("business"),
            affiliates=join("affiliates"),
            related_party=join("related_party"),
            related_party_notes=join("rp_notes"),
        )
