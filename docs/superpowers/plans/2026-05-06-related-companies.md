# Related Companies Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI module that extracts supply-chain / customer / competitor / affiliate / subsidiary relationships from DART business reports using GPT, persists them in SQLite, and renders an interactive HTML network graph.

**Architecture:** New `src/related/` package that orchestrates report download (via existing `src/dart/` client) → text section parsing → GPT JSON extraction → SQLite persistence (cached by 사업보고서 접수번호) → NetworkX graph build → Plotly HTML report.

**Tech Stack:** OpenAI SDK (gpt-5-nano), DART API, NetworkX, Plotly, Jinja2, BeautifulSoup4, SQLAlchemy, Typer, httpx

**Spec:** `docs/superpowers/specs/2026-05-06-related-companies-design.md`

---

### Task 1: Dependencies & Configuration

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/config.py`
- Modify: `config.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Add networkx dependency to pyproject.toml**

In the main `dependencies = [...]` list, add:
```
"networkx>=3.0",
```
(All other libraries already present.)

- [ ] **Step 2: Add RelatedSection to src/config.py**

After `FundamentalsSection`, add:
```python
class RelatedSection(BaseModel):
    model: str = "gpt-5-nano"
    report_dir: str = "reports"
    max_tokens_per_section: int = 8000
```

In `ScannerConfig`, add:
```python
    related: RelatedSection = RelatedSection()
```

- [ ] **Step 3: Add related section to config.yaml**

Append:
```yaml
related:
  model: "gpt-5-nano"
  report_dir: "reports"
  max_tokens_per_section: 8000
```

- [ ] **Step 4: Add config test**

Append to `tests/test_config.py`:
```python
def test_related_config_defaults():
    from src.config import RelatedSection, ScannerConfig
    config = ScannerConfig()
    assert config.related.model == "gpt-5-nano"
    assert config.related.report_dir == "reports"
    assert config.related.max_tokens_per_section == 8000
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/config.py config.yaml tests/test_config.py
git commit -m "feat(related): add config for related companies module"
```

---

### Task 2: Data Models

**Files:**
- Create: `src/related/__init__.py`
- Create: `src/related/models.py`
- Create: `tests/test_related_models.py`

- [ ] **Step 1: Create package init**

Create empty file:
```python
# src/related/__init__.py
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_related_models.py
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
```

- [ ] **Step 3: Run tests to verify failure**

Run: `pytest tests/test_related_models.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement models**

```python
# src/related/models.py
from typing import Literal
from pydantic import BaseModel


RelationType = Literal["Supplier", "Customer", "Competitor", "Affiliate", "Subsidiary"]


class ReportSections(BaseModel):
    """Parsed sections of a DART business report (사업보고서)."""

    corp_code: str
    rcept_no: str                # 접수번호 — cache invalidation key
    business_content: str        # II. 사업의 내용
    affiliates: str              # IX. 계열회사 등에 관한 사항
    related_party: str           # X. 대주주 등과의 거래내용
    related_party_notes: str     # 재무제표 주석 — 특수관계자 거래


class ExtractedEdge(BaseModel):
    """A single relationship extracted by GPT."""

    name: str                    # company name as written in report
    ticker: str | None           # null = unlisted / foreign / unresolved
    relation: RelationType
    evidence: str                # quoted excerpt from report
```

- [ ] **Step 5: Run tests to verify pass**

Run: `pytest tests/test_related_models.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/related/__init__.py src/related/models.py tests/test_related_models.py
git commit -m "feat(related): add ReportSections and ExtractedEdge models"
```

---

### Task 3: DB Layer

**Files:**
- Create: `src/related/db.py`
- Create: `tests/test_related_db.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_related_db.py
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_related_db.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement RelatedDB**

```python
# src/related/db.py
from __future__ import annotations

from collections import Counter
from datetime import datetime
from dataclasses import dataclass

from sqlalchemy import (
    Column, Integer, String, Text, DateTime,
    create_engine, delete, select,
)
from sqlalchemy.orm import DeclarativeBase, Session

from src.related.models import ExtractedEdge


class RelatedBase(DeclarativeBase):
    pass


class RelatedReportMetaRow(RelatedBase):
    __tablename__ = "related_report_meta"
    ticker = Column(String(10), primary_key=True)
    rcept_no = Column(String(20), nullable=False)
    extracted_at = Column(DateTime, nullable=False)


class RelatedEdgeRow(RelatedBase):
    __tablename__ = "related_edges"
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_ticker = Column(String(10), nullable=False, index=True)
    target_ticker = Column(String(10), nullable=True, index=True)
    target_name = Column(String(100), nullable=False)
    relation = Column(String(20), nullable=False)
    evidence = Column(Text, nullable=False)
    extracted_at = Column(DateTime, nullable=False)


@dataclass
class StoredEdge:
    source_ticker: str
    target_ticker: str | None
    target_name: str
    relation: str
    evidence: str
    extracted_at: datetime


@dataclass
class ReportMeta:
    ticker: str
    rcept_no: str
    extracted_at: datetime


class RelatedDB:
    """SQLite persistence for related-company edges and report meta."""

    def __init__(self, url: str = "sqlite:///data/scanner.db"):
        self.engine = create_engine(url)
        RelatedBase.metadata.create_all(self.engine)

    def get_meta(self, ticker: str) -> ReportMeta | None:
        with Session(self.engine) as s:
            row = s.execute(
                select(RelatedReportMetaRow).where(RelatedReportMetaRow.ticker == ticker)
            ).scalar_one_or_none()
            if row is None:
                return None
            return ReportMeta(ticker=row.ticker, rcept_no=row.rcept_no, extracted_at=row.extracted_at)

    def set_meta(self, ticker: str, rcept_no: str, extracted_at: datetime) -> None:
        with Session(self.engine) as s:
            s.execute(delete(RelatedReportMetaRow).where(RelatedReportMetaRow.ticker == ticker))
            s.add(RelatedReportMetaRow(ticker=ticker, rcept_no=rcept_no, extracted_at=extracted_at))
            s.commit()

    def needs_refresh(self, ticker: str, current_rcept_no: str) -> bool:
        meta = self.get_meta(ticker)
        if meta is None:
            return True
        return meta.rcept_no != current_rcept_no

    def save_edges(
        self, source_ticker: str, edges: list[ExtractedEdge], extracted_at: datetime,
    ) -> None:
        with Session(self.engine) as s:
            s.execute(delete(RelatedEdgeRow).where(RelatedEdgeRow.source_ticker == source_ticker))
            for e in edges:
                s.add(RelatedEdgeRow(
                    source_ticker=source_ticker,
                    target_ticker=e.ticker,
                    target_name=e.name,
                    relation=e.relation,
                    evidence=e.evidence,
                    extracted_at=extracted_at,
                ))
            s.commit()

    def load_edges(self, source_tickers: list[str] | None = None) -> list[StoredEdge]:
        with Session(self.engine) as s:
            stmt = select(RelatedEdgeRow)
            if source_tickers is not None:
                stmt = stmt.where(RelatedEdgeRow.source_ticker.in_(source_tickers))
            rows = s.execute(stmt).scalars().all()
            return [
                StoredEdge(
                    source_ticker=r.source_ticker,
                    target_ticker=r.target_ticker,
                    target_name=r.target_name,
                    relation=r.relation,
                    evidence=r.evidence,
                    extracted_at=r.extracted_at,
                )
                for r in rows
            ]

    def stats(self) -> dict:
        with Session(self.engine) as s:
            rows = s.execute(select(RelatedEdgeRow)).scalars().all()
            counter: Counter = Counter(r.relation for r in rows)
            sources = len({r.source_ticker for r in rows})
            return {
                "sources": sources,
                "edges": len(rows),
                "by_relation": dict(counter),
            }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_related_db.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/related/db.py tests/test_related_db.py
git commit -m "feat(related): add DB layer with rcept_no-based cache invalidation"
```

---

### Task 4: Report Fetcher

**Files:**
- Create: `src/related/report_fetcher.py`
- Create: `tests/test_related_report_fetcher.py`

The DART report flow:
1. `GET /api/list.json?corp_code=...&pblntf_detail_ty=A001&page_count=1&sort=date&sort_mth=desc` returns recent annual reports. Pick first row with `report_nm` containing "사업보고서" → `rcept_no`.
2. `GET /api/document.xml?rcept_no=...` returns a ZIP whose first file is a UTF-8 XML business report.
3. Parse XML; split by section headers `II. 사업의 내용`, `IX. 계열회사 등에 관한 사항`, `X. 대주주 등과의 거래내용`, plus the 특수관계자 거래 footnote subsection inside `III. 재무에 관한 사항`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_related_report_fetcher.py
from unittest.mock import AsyncMock, MagicMock
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_related_report_fetcher.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement ReportFetcher**

```python
# src/related/report_fetcher.py
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
        """Return rcept_no of the latest 사업보고서 for this corp, or None."""
        data = await self._client.get(
            "/api/list.json",
            params={
                "corp_code": corp_code,
                "pblntf_detail_ty": "A001",  # 정기공시 - 사업보고서/반기/분기 한묶음
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
        url = f"https://opendart.fss.or.kr/api/document.xml"
        async with httpx.AsyncClient(timeout=120.0) as http:
            resp = await http.get(
                url, params={"crtfc_key": self._client._api_key, "rcept_no": rcept_no},
            )
            resp.raise_for_status()
            content = resp.content
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            # The first file inside is the XML business report
            name = zf.namelist()[0]
            with zf.open(name) as f:
                raw = f.read()
        # DART XML is typically EUC-KR or UTF-8; try UTF-8 first
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("euc-kr", errors="ignore")

    def parse_sections(
        self, xml_text: str, corp_code: str, rcept_no: str,
    ) -> ReportSections:
        """Split XML text into 4 relevant sections.

        Strategy: convert XML to plain text (line per paragraph), then walk
        through lines tagging which section each belongs to based on headers.
        """
        soup = BeautifulSoup(xml_text, "html.parser")
        # Each paragraph/cell becomes a line
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
            # Detect section transitions
            if _HEADER_BUSINESS.search(line) and _HEADER_ANY.match(line):
                current = "business"; continue
            if _HEADER_AFFILIATES.search(line) and _HEADER_ANY.match(line):
                current = "affiliates"; continue
            if _HEADER_RELATED_PARTY.search(line) and _HEADER_ANY.match(line):
                current = "related_party"; continue
            if _HEADER_RP_NOTES.search(line):
                current = "rp_notes"; continue
            # A new top-level section terminates rp_notes (which is a sub-section)
            if current == "rp_notes" and _HEADER_ANY.match(line):
                current = None
            # A new top-level section terminates the current bucket too
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_related_report_fetcher.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/related/report_fetcher.py tests/test_related_report_fetcher.py
git commit -m "feat(related): add ReportFetcher with section parsing"
```

---

### Task 5: GPT Extractor

**Files:**
- Create: `src/related/extractor.py`
- Create: `tests/test_related_extractor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_related_extractor.py
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.related.extractor import RelationExtractor, normalize_name, resolve_ticker
from src.related.models import ReportSections


def test_normalize_name_strips_corporate_suffixes():
    assert normalize_name("(주)삼성전자") == "삼성전자"
    assert normalize_name("삼성전자(주)") == "삼성전자"
    assert normalize_name("주식회사 한미반도체") == "한미반도체"
    assert normalize_name("SK하이닉스주식회사") == "SK하이닉스"


def test_resolve_ticker_exact_match():
    name_to_ticker = {"삼성전자": "005930", "SK하이닉스": "000660"}
    assert resolve_ticker("삼성전자", name_to_ticker) == "005930"
    assert resolve_ticker("(주)삼성전자", name_to_ticker) == "005930"


def test_resolve_ticker_unknown_returns_none():
    name_to_ticker = {"삼성전자": "005930"}
    assert resolve_ticker("ASML", name_to_ticker) is None


@pytest.mark.asyncio
async def test_extract_calls_openai_with_json_mode():
    sections = ReportSections(
        corp_code="00126380", rcept_no="20250318",
        business_content="당사는 메모리 반도체를 제조한다. 주요 고객사는 Apple입니다.",
        affiliates="", related_party="", related_party_notes="",
    )

    mock_message = MagicMock()
    mock_message.content = json.dumps({
        "edges": [
            {"name": "Apple", "ticker": None, "relation": "Customer",
             "evidence": "주요 고객사는 Apple입니다."},
        ]
    })
    mock_choice = MagicMock(); mock_choice.message = mock_message
    mock_resp = MagicMock(); mock_resp.choices = [mock_choice]

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_resp)

    with patch("openai.AsyncOpenAI", return_value=mock_openai):
        extractor = RelationExtractor(api_key="test-key", model="gpt-5-nano")
        edges = await extractor.extract(
            target_name="삼성전자", target_ticker="005930",
            sections=sections, name_to_ticker={"삼성전자": "005930"},
        )

    assert len(edges) == 1
    assert edges[0].name == "Apple"
    assert edges[0].relation == "Customer"
    # JSON mode requested
    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs.get("response_format") == {"type": "json_object"}


@pytest.mark.asyncio
async def test_extract_resolves_unknown_tickers_from_name_map():
    """If GPT returns ticker=None but name is in the map, resolve it."""
    sections = ReportSections(
        corp_code="00", rcept_no="0",
        business_content="x", affiliates="", related_party="", related_party_notes="",
    )

    mock_message = MagicMock()
    mock_message.content = json.dumps({
        "edges": [
            {"name": "SK하이닉스", "ticker": None, "relation": "Customer",
             "evidence": "HBM 후공정"},
        ]
    })
    mock_choice = MagicMock(); mock_choice.message = mock_message
    mock_resp = MagicMock(); mock_resp.choices = [mock_choice]

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_resp)

    with patch("openai.AsyncOpenAI", return_value=mock_openai):
        extractor = RelationExtractor(api_key="key", model="gpt-5-nano")
        edges = await extractor.extract(
            target_name="한미반도체", target_ticker="042700",
            sections=sections,
            name_to_ticker={"SK하이닉스": "000660"},
        )

    assert edges[0].ticker == "000660"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_related_extractor.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement RelationExtractor**

```python
# src/related/extractor.py
from __future__ import annotations

import json
import re

import openai
from loguru import logger

from src.related.models import ExtractedEdge, ReportSections


_CORPORATE_SUFFIXES = [
    "(주)", "(주식회사)", "주식회사", "(유)", "유한회사",
    "Co., Ltd.", "Co.,Ltd.", "Inc.", "Ltd.", "Corp.",
]


def normalize_name(name: str) -> str:
    """Strip whitespace and corporate suffixes for matching."""
    out = name.strip()
    # First, strip parens occurrences anywhere
    out = re.sub(r"\(주식회사\)|\(주\)|\(유\)", "", out)
    # Then strip trailing/leading "주식회사" / "유한회사" words
    for suffix in ["주식회사", "유한회사"]:
        if out.startswith(suffix):
            out = out[len(suffix):]
        if out.endswith(suffix):
            out = out[: -len(suffix)]
    # English suffixes
    for suffix in ["Co., Ltd.", "Co.,Ltd.", "Inc.", "Ltd.", "Corp."]:
        if out.endswith(suffix):
            out = out[: -len(suffix)].strip(",.")
    return out.strip()


def resolve_ticker(name: str, name_to_ticker: dict[str, str]) -> str | None:
    """Resolve a company name to a Korean ticker, or None."""
    if name in name_to_ticker:
        return name_to_ticker[name]
    norm = normalize_name(name)
    if norm in name_to_ticker:
        return name_to_ticker[norm]
    # Build a normalized lookup once we have a miss
    norm_map = {normalize_name(k): v for k, v in name_to_ticker.items()}
    return norm_map.get(norm)


_SYSTEM_PROMPT = """당신은 한국 기업의 사업보고서를 분석하는 전문가입니다.
주어진 텍스트에서 본 기업과 다른 기업 간의 관계를 추출해 JSON으로 출력하세요.

관계 타입:
- Supplier: 공급업체 (본 기업이 매입하는 거래처)
- Customer: 고객사 (본 기업이 매출을 일으키는 거래처)
- Competitor: 경쟁사
- Affiliate: 같은 그룹의 계열사
- Subsidiary: 자회사 또는 관계회사

규칙:
- 본 기업 자체나 자기 자신은 포함하지 마세요.
- 각 항목에 evidence(원문에서 인용한 짧은 근거 문장)를 반드시 포함하세요.
- ticker는 6자리 한국 종목코드. 모르거나 비상장/외국기업이면 null.
- 출력 스키마: {"edges": [{"name": str, "ticker": str|null, "relation": str, "evidence": str}]}
"""


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


class RelationExtractor:
    """Extract company relationships from report sections using GPT."""

    def __init__(self, api_key: str, model: str = "gpt-5-nano",
                 max_chars_per_section: int = 24000):
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model
        self._max_chars = max_chars_per_section

    async def extract(
        self,
        target_name: str,
        target_ticker: str,
        sections: ReportSections,
        name_to_ticker: dict[str, str],
    ) -> list[ExtractedEdge]:
        body = (
            f"대상 기업: {target_name} ({target_ticker})\n\n"
            f"=== 사업의 내용 ===\n{_truncate(sections.business_content, self._max_chars)}\n\n"
            f"=== 계열회사 등 ===\n{_truncate(sections.affiliates, self._max_chars)}\n\n"
            f"=== 대주주 등과의 거래 ===\n{_truncate(sections.related_party, self._max_chars)}\n\n"
            f"=== 특수관계자 거래 ===\n{_truncate(sections.related_party_notes, self._max_chars)}\n"
        )

        resp = await self._client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": body},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed for {target_ticker}: {e}")
            return []

        edges_raw = data.get("edges", [])
        result: list[ExtractedEdge] = []
        for item in edges_raw:
            name = (item.get("name") or "").strip()
            if not name:
                continue
            relation = item.get("relation")
            if relation not in ("Supplier", "Customer", "Competitor",
                                "Affiliate", "Subsidiary"):
                continue
            ticker = item.get("ticker")
            if not ticker:
                ticker = resolve_ticker(name, name_to_ticker)
            else:
                # GPT may hallucinate a ticker; verify it exists in our map
                if ticker not in set(name_to_ticker.values()):
                    ticker = resolve_ticker(name, name_to_ticker)
            evidence = (item.get("evidence") or "").strip()
            result.append(ExtractedEdge(
                name=name, ticker=ticker, relation=relation, evidence=evidence,
            ))
        return result
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_related_extractor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/related/extractor.py tests/test_related_extractor.py
git commit -m "feat(related): add GPT-based relationship extractor with ticker resolution"
```

---

### Task 6: Graph Builder

**Files:**
- Create: `src/related/graph.py`
- Create: `tests/test_related_graph.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_related_graph.py
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
    # Virtual node format defined by build_graph
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_related_graph.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement graph**

```python
# src/related/graph.py
from __future__ import annotations

import networkx as nx

from src.related.db import StoredEdge


def virtual_node(name: str) -> str:
    """Stable id for an unlisted/foreign company."""
    return f"_unlisted_{name}"


def build_graph(edges: list[StoredEdge]) -> nx.DiGraph:
    """Build a DiGraph from stored edges."""
    g = nx.DiGraph()
    for e in edges:
        target = e.target_ticker if e.target_ticker else virtual_node(e.target_name)
        g.add_node(e.source_ticker)
        g.add_node(target, display_name=e.target_name)
        g.add_edge(
            e.source_ticker, target,
            relation=e.relation, evidence=e.evidence,
        )
    return g


def expand(graph: nx.DiGraph, root: str, depth: int = 1) -> nx.DiGraph:
    """Return subgraph reachable from root within `depth` hops (both directions)."""
    if root not in graph:
        return graph.subgraph([]).copy()
    visited = {root}
    frontier = {root}
    for _ in range(depth):
        next_frontier = set()
        for node in frontier:
            next_frontier |= set(graph.successors(node))
            next_frontier |= set(graph.predecessors(node))
        next_frontier -= visited
        visited |= next_frontier
        frontier = next_frontier
        if not frontier:
            break
    return graph.subgraph(visited).copy()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_related_graph.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/related/graph.py tests/test_related_graph.py
git commit -m "feat(related): add NetworkX graph builder and multi-hop expand"
```

---

### Task 7: HTML Report

**Files:**
- Create: `src/related/templates/report.html`
- Create: `src/related/report.py`
- Create: `tests/test_related_report.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_related_report.py
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_related_report.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create the HTML template**

```html
<!-- src/related/templates/report.html -->
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>연관 기업 — {{ root_name }} ({{ root_ticker }})</title>
  <script>{{ plotly_js }}</script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#f5f5f5; color:#333; }
    header { background: linear-gradient(135deg,#2c3e50,#3498db); color:#fff; padding:1.5rem; text-align:center; }
    header h1 { font-size:1.6rem; }
    .container { max-width:1400px; margin:1.5rem auto; padding:0 1.5rem; }
    .card { background:#fff; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,0.08); padding:1.5rem; margin-bottom:1.5rem; }
    h2 { font-size:1.25rem; color:#2c3e50; border-left:4px solid #3498db; padding-left:0.75rem; margin-bottom:1rem; }
    #graph { width:100%; height:600px; }
    table { width:100%; border-collapse:collapse; font-size:0.88rem; }
    th { background:#eaf0fb; padding:0.55rem; text-align:left; }
    td { padding:0.5rem 0.6rem; border-bottom:1px solid #eceff1; vertical-align:top; }
    .grade { color:#f39c12; font-size:0.85rem; }
    .ev { color:#666; font-size:0.82rem; }
    .rel-Supplier { color:#1e8449; font-weight:600; }
    .rel-Customer { color:#1864c0; font-weight:600; }
    .rel-Competitor { color:#c0392b; font-weight:600; }
    .rel-Affiliate { color:#6a1ec0; font-weight:600; }
    .rel-Subsidiary { color:#d35400; font-weight:600; }
    .meta { font-size:0.8rem; color:#888; }
  </style>
</head>
<body>
<header>
  <h1>🔗 연관 기업 — {{ root_name }} <span style="opacity:.7">({{ root_ticker }})</span></h1>
  <p>{{ root_market }} &nbsp;|&nbsp; depth={{ depth }} &nbsp;|&nbsp; 노드 {{ node_count }}개 / 엣지 {{ edge_count }}개</p>
</header>
<div class="container">

  <div class="card">
    <h2>네트워크 그래프</h2>
    <div id="graph"></div>
  </div>

  {% for rel, items in grouped.items() %}
  <div class="card">
    <h2><span class="rel-{{ rel }}">{{ rel }}</span> ({{ items|length }})</h2>
    <table>
      <thead><tr><th>종목명</th><th>티커</th><th>등급</th><th>근거</th></tr></thead>
      <tbody>
        {% for it in items %}
        <tr>
          <td>{{ it.name }}</td>
          <td>{{ it.ticker or '-' }}</td>
          <td class="grade">{{ it.grade or '' }}</td>
          <td class="ev">{{ it.evidence }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% endfor %}

  <div class="card meta">
    <div>데이터: DART 사업보고서 (접수번호 {{ rcept_no }})</div>
    <div>생성일: {{ as_of_date }}</div>
  </div>

</div>

<script>
const graphData = {{ graph_data | tojson }};
(function(){
  const nodes = graphData.nodes;
  const edges = graphData.edges;
  // Build edge traces per relation type for color legend
  const relColors = {Supplier:'#1e8449', Customer:'#1864c0', Competitor:'#c0392b', Affiliate:'#6a1ec0', Subsidiary:'#d35400'};
  const edgeTraces = {};
  edges.forEach(e => {
    if (!edgeTraces[e.relation]) {
      edgeTraces[e.relation] = {x:[], y:[], mode:'lines', name:e.relation, line:{color:relColors[e.relation]||'#999', width:1.5}, hoverinfo:'text', text:[]};
    }
    const t = edgeTraces[e.relation];
    t.x.push(e.x0, e.x1, null);
    t.y.push(e.y0, e.y1, null);
    t.text.push(e.evidence);
  });
  const nodeTrace = {
    x: nodes.map(n=>n.x), y: nodes.map(n=>n.y),
    mode:'markers+text', text: nodes.map(n=>n.label), textposition:'top center',
    marker:{
      color: nodes.map(n=>n.color),
      size: nodes.map(n=>n.size),
      line:{color:'#fff', width:1.5},
    },
    hoverinfo:'text',
    hovertext: nodes.map(n=>n.hover),
    name:'기업',
  };
  Plotly.newPlot('graph', [...Object.values(edgeTraces), nodeTrace], {
    showlegend:true,
    margin:{t:20,b:20,l:20,r:20},
    xaxis:{visible:false}, yaxis:{visible:false},
    plot_bgcolor:'#fafafa',
  }, {responsive:true, displayModeBar:false});
})();
</script>

</body>
</html>
```

- [ ] **Step 4: Implement ReportGenerator**

```python
# src/related/report.py
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import networkx as nx
import plotly
from jinja2 import Environment, FileSystemLoader

from src.related.db import StoredEdge


_REL_COLORS = {
    "Supplier": "#1e8449",
    "Customer": "#1864c0",
    "Competitor": "#c0392b",
    "Affiliate": "#6a1ec0",
    "Subsidiary": "#d35400",
}


class ReportGenerator:
    """Generates an interactive HTML report for a single ticker."""

    def __init__(self):
        template_dir = Path(__file__).parent / "templates"
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False)
        self._env = env

    def generate(
        self,
        root_ticker: str,
        graph: nx.DiGraph,
        edges: list[StoredEdge],
        name_map: dict[str, str],
        market_map: dict[str, str],
        scores: dict[str, dict],
        depth: int,
        rcept_no: str,
        as_of_date: str,
        output_dir: str,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"related-{root_ticker}-{as_of_date}.html")

        # Layout
        pos = nx.spring_layout(graph, seed=42) if len(graph) else {}

        # Build node payload
        nodes_payload = []
        for node in graph.nodes:
            x, y = pos.get(node, (0, 0))
            attrs = graph.nodes[node]
            if node.startswith("_unlisted_"):
                label = attrs.get("display_name", node[10:])
                color = "#aaaaaa"
                hover = f"{label} (비상장/외국)"
                size = 14
            else:
                label = name_map.get(node, node)
                if node == root_ticker:
                    color = "#e74c3c"
                else:
                    color = "#3498db"
                sc = scores.get(node, {})
                grade = sc.get("grade", "")
                cats = ", ".join(sc.get("categories", []))
                hover = f"{label} ({node})<br>{market_map.get(node, '')}<br>{grade} {cats}".strip()
                base_size = 12
                if sc.get("total_score") is not None:
                    base_size = 8 + sc["total_score"] / 10
                size = 22 if node == root_ticker else base_size
            nodes_payload.append({
                "id": node, "x": float(x), "y": float(y),
                "label": label, "color": color, "size": size, "hover": hover,
            })

        # Build edge payload
        edges_payload = []
        for u, v, attrs in graph.edges(data=True):
            x0, y0 = pos.get(u, (0, 0))
            x1, y1 = pos.get(v, (0, 0))
            edges_payload.append({
                "x0": float(x0), "y0": float(y0),
                "x1": float(x1), "y1": float(y1),
                "relation": attrs.get("relation", ""),
                "evidence": attrs.get("evidence", ""),
            })

        graph_data = {"nodes": nodes_payload, "edges": edges_payload}

        # Group edges by relation type — only show edges whose source is root_ticker
        # (the user's primary interest is "X's direct relationships")
        root_edges = [e for e in edges if e.source_ticker == root_ticker]
        grouped: dict[str, list[dict]] = defaultdict(list)
        for e in root_edges:
            sc = scores.get(e.target_ticker or "", {}) if e.target_ticker else {}
            grouped[e.relation].append({
                "name": e.target_name,
                "ticker": e.target_ticker,
                "grade": sc.get("grade", ""),
                "evidence": e.evidence,
            })

        plotly_js = plotly.offline.get_plotlyjs()
        template = self._env.get_template("report.html")
        html = template.render(
            root_ticker=root_ticker,
            root_name=name_map.get(root_ticker, root_ticker),
            root_market=market_map.get(root_ticker, ""),
            depth=depth,
            node_count=len(graph.nodes),
            edge_count=len(graph.edges),
            grouped=dict(grouped),
            rcept_no=rcept_no,
            as_of_date=as_of_date,
            graph_data=graph_data,
            plotly_js=plotly_js,
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_related_report.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/related/templates/report.html src/related/report.py tests/test_related_report.py
git commit -m "feat(related): add HTML report with network graph and relation table"
```

---

### Task 8: CLI Orchestration

**Files:**
- Create: `src/related/cli.py`
- Create: `tests/test_related_cli.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_related_cli.py
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from src.related.cli import app


def test_stats_command_empty(tmp_path):
    runner = CliRunner()
    with patch("src.related.cli.RelatedDB") as MockDB:
        m = MagicMock()
        m.stats.return_value = {"sources": 0, "edges": 0, "by_relation": {}}
        MockDB.return_value = m
        result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "0" in result.stdout


def test_show_missing_ticker_in_corp_info():
    """show with unknown ticker prints helpful message."""
    runner = CliRunner()
    with patch("src.related.cli.DartCache") as MockCache:
        c = MagicMock()
        c.load_corp_info.return_value = []
        MockCache.return_value = c
        result = runner.invoke(app, ["show", "999999"])
    assert result.exit_code == 0
    assert "999999" in result.stdout
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_related_cli.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement CLI**

```python
# src/related/cli.py
from __future__ import annotations

import asyncio
import json
import os
import webbrowser
from datetime import date, datetime

import typer
from loguru import logger
from rich.console import Console

from src.config import Settings, load_scanner_config
from src.dart.cache import DartCache
from src.related.db import RelatedDB

app = typer.Typer(help="Related companies discovery via DART business reports",
                  no_args_is_help=True)
console = Console()


async def _process_one(
    ticker: str,
    refresh: bool,
    settings: Settings,
    config,
    cache: DartCache,
    db: RelatedDB,
) -> bool:
    """Fetch + extract + save edges for a single ticker. Returns True on success."""
    from src.dart.client import DartClient
    from src.related.extractor import RelationExtractor
    from src.related.report_fetcher import ReportFetcher

    corps = cache.load_corp_info()
    by_ticker = {c.ticker: c for c in corps}
    name_to_ticker = {c.name: c.ticker for c in corps}
    if ticker not in by_ticker:
        console.print(
            f"[yellow]{ticker} 가 DART corp_info에 없습니다. "
            f"먼저 `python -m src.fundamentals.cli refresh` 를 실행하세요.[/yellow]"
        )
        return False
    corp = by_ticker[ticker]

    client = DartClient(api_key=settings.opendart_api_key)
    fetcher = ReportFetcher(client=client)

    rcept_no = await fetcher.latest_rcept_no(corp.corp_code)
    if rcept_no is None:
        console.print(f"[yellow]{ticker} ({corp.name}): 사업보고서 없음[/yellow]")
        return False

    if not refresh and not db.needs_refresh(ticker, rcept_no):
        console.print(f"[dim]  {ticker} ({corp.name}): 캐시 사용 (rcept_no={rcept_no})[/dim]")
        return True

    console.print(f"[dim]  {ticker} ({corp.name}): 사업보고서 다운로드 중...[/dim]")
    xml_text = await fetcher.download_document(rcept_no)
    sections = fetcher.parse_sections(xml_text, corp.corp_code, rcept_no)

    console.print(f"[dim]  {ticker}: GPT 추출 중...[/dim]")
    extractor = RelationExtractor(
        api_key=settings.openai_api_key, model=config.related.model,
    )
    edges = await extractor.extract(
        target_name=corp.name, target_ticker=ticker,
        sections=sections, name_to_ticker=name_to_ticker,
    )
    now = datetime.now()
    db.save_edges(ticker, edges, now)
    db.set_meta(ticker, rcept_no, now)
    console.print(f"[green]  {ticker}: {len(edges)}개 관계 추출[/green]")
    return True


def _load_scores(db_url: str = "sqlite:///data/scanner.db") -> dict[str, dict]:
    """Load fundamentals scores keyed by ticker. Returns empty if module/data missing."""
    try:
        from src.fundamentals.db import FundamentalsDB
        fdb = FundamentalsDB(url=db_url)
        scores = fdb.load_scores(date.today())
        return {
            s.ticker: {
                "total_score": s.total_score, "grade": s.grade,
                "categories": s.categories,
            } for s in scores
        }
    except Exception:
        return {}


@app.command()
def show(
    ticker: str,
    depth: int = typer.Option(1, "--depth", "-d", help="Multi-hop expansion depth"),
    refresh: bool = typer.Option(False, "--refresh", help="Force re-extraction"),
):
    """Build related-companies graph for one ticker and open HTML report."""
    from src.dart.client import DartClient
    from src.related.graph import build_graph, expand
    from src.related.report import ReportGenerator

    settings = Settings()
    config = load_scanner_config()
    cache = DartCache()
    db = RelatedDB()

    corps = cache.load_corp_info()
    by_ticker = {c.ticker: c for c in corps}
    if ticker not in by_ticker:
        console.print(f"[yellow]{ticker} 가 DART corp_info에 없습니다.[/yellow]")
        return

    console.print(f"[bold]연관 기업 발굴: {by_ticker[ticker].name} ({ticker})[/bold]")
    ok = asyncio.run(_process_one(ticker, refresh, settings, config, cache, db))
    if not ok:
        return

    # If depth>1, also process direct neighbors (so we have their edges to follow)
    if depth > 1:
        first_edges = db.load_edges(source_tickers=[ticker])
        neighbors = [e.target_ticker for e in first_edges
                     if e.target_ticker and e.target_ticker != ticker]
        for n in neighbors:
            asyncio.run(_process_one(n, refresh=False, settings=settings,
                                      config=config, cache=cache, db=db))

    # Build subgraph
    seed = [ticker]
    if depth > 1:
        seed += [e.target_ticker for e in db.load_edges(source_tickers=[ticker])
                 if e.target_ticker]
    full_edges = db.load_edges(source_tickers=list(set(seed)))
    g = build_graph(full_edges)
    sub = expand(g, ticker, depth=depth)

    # Get rcept_no for header
    meta = db.get_meta(ticker)
    rcept_no = meta.rcept_no if meta else "unknown"

    name_map = {c.ticker: c.name for c in corps}
    market_map = {c.ticker: c.market for c in corps}
    scores = _load_scores()

    gen = ReportGenerator()
    path = gen.generate(
        root_ticker=ticker, graph=sub, edges=full_edges,
        name_map=name_map, market_map=market_map, scores=scores,
        depth=depth, rcept_no=rcept_no,
        as_of_date=str(date.today()),
        output_dir=config.related.report_dir,
    )
    console.print(f"[bold green]완료! 리포트: {path}[/bold green]")
    webbrowser.open(f"file://{os.path.abspath(path)}")


@app.command()
def batch(
    tickers: str = typer.Option(..., "--tickers", help="Comma-separated tickers"),
    refresh: bool = typer.Option(False, "--refresh", help="Force re-extraction"),
):
    """Run extraction (no report) for an explicit list of tickers."""
    settings = Settings()
    config = load_scanner_config()
    cache = DartCache()
    db = RelatedDB()
    ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
    console.print(f"[bold]배치 추출: {len(ticker_list)}개 종목[/bold]")
    for t in ticker_list:
        asyncio.run(_process_one(t, refresh, settings, config, cache, db))


@app.command()
def stats():
    """Print stored edge stats."""
    db = RelatedDB()
    s = db.stats()
    console.print(f"[bold]저장된 관계 통계[/bold]")
    console.print(f"  source 종목: {s['sources']}")
    console.print(f"  전체 엣지: {s['edges']}")
    for rel, cnt in s.get("by_relation", {}).items():
        console.print(f"    {rel}: {cnt}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_related_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/related/cli.py tests/test_related_cli.py
git commit -m "feat(related): add CLI (show/batch/stats) with multi-hop pipeline"
```

---

### Task 9: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read existing README.md**

Find the location after the "펀더멘털 스크리너 (Fundamentals)" section.

- [ ] **Step 2: Append new section**

Insert this section right after the fundamentals section:

```markdown
## 연관 기업 발굴 (Related)

DART 사업보고서를 GPT로 분석하여 공급망/고객사/경쟁사/계열사/자회사 관계를 추출하고, 인터랙티브 네트워크 그래프 HTML 리포트로 시각화합니다.

### 사전 조건

DART 데이터(corp_info)가 먼저 적재되어 있어야 합니다:

```bash
python -m src.fundamentals.cli refresh
```

### 사용법

```bash
# 단일 종목 (1-hop 기본)
python -m src.related.cli show 042700

# 2-hop 확장
python -m src.related.cli show 042700 --depth 2

# 강제 재추출 (사업보고서 미변경에도 GPT 재호출)
python -m src.related.cli show 042700 --refresh

# 명시적 티커 리스트 배치 추출
python -m src.related.cli batch --tickers 005930,000660,042700

# 저장된 관계 통계
python -m src.related.cli stats
```

리포트는 `reports/related-<ticker>-YYYY-MM-DD.html`에 생성되며 브라우저에서 자동으로 열립니다. 사업보고서 접수번호(`rcept_no`) 기준으로 캐싱되므로 새 보고서가 나올 때까지 GPT 재호출이 없습니다.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add related companies discovery usage to README"
```

---

### Task 10: Run All Tests

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: all PASS (existing 80 + new ~20 = ~100 tests)

- [ ] **Step 2: If any failures, fix and re-run**

Inspect failures, fix the affected files, re-run until green.
