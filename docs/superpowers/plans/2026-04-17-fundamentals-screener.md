# Open DART Fundamentals Screener Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone CLI module that fetches financial data for all KOSPI/KOSDAQ stocks via Open DART, computes 4-dimension scores and 5 category labels, and outputs an interactive HTML report.

**Architecture:** Two-package structure: `src/dart/` is a shared data layer (Open DART API client, fetcher, cache) reusable by future modules; `src/fundamentals/` is the analysis layer (scorer, classifier, report). DB persistence in 3 tiers (raw / derived / scores) for cross-module reuse.

**Tech Stack:** Open DART REST API, SQLAlchemy (SQLite), Pydantic, Plotly, Jinja2, Typer, asyncio + httpx for async HTTP

**Spec:** `docs/superpowers/specs/2026-04-17-fundamentals-screener-design.md`

---

### Task 1: Dependencies & Configuration

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/config.py`
- Modify: `config.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: No new pyproject.toml dependencies needed**

All required libraries (httpx, sqlalchemy, pydantic, plotly, jinja2, typer) are already in dependencies. Skip pyproject.toml changes.

- [ ] **Step 2: Add OpenDART API key and FundamentalsSection to src/config.py**

In `Settings` class, add:
```python
    opendart_api_key: str = ""
```

After `ForecastSection`, add new section:
```python
class FundamentalsSection(BaseModel):
    years_lookback: int = 10
    cache_ttl_days: int = 30
    report_dir: str = "reports"
    market_filter: list[str] = ["KOSPI", "KOSDAQ"]
```

In `ScannerConfig`, add:
```python
    fundamentals: FundamentalsSection = FundamentalsSection()
```

- [ ] **Step 3: Add fundamentals section to config.yaml**

Append:
```yaml
fundamentals:
  years_lookback: 10
  cache_ttl_days: 30
  report_dir: "reports"
  market_filter: ["KOSPI", "KOSDAQ"]
```

- [ ] **Step 4: Add config tests**

In `tests/test_config.py`, append:
```python
def test_fundamentals_config_defaults():
    from src.config import FundamentalsSection, ScannerConfig
    config = ScannerConfig()
    assert config.fundamentals.years_lookback == 10
    assert config.fundamentals.cache_ttl_days == 30
    assert config.fundamentals.report_dir == "reports"
    assert config.fundamentals.market_filter == ["KOSPI", "KOSDAQ"]


def test_settings_has_opendart_key():
    import os
    os.environ["OPENDART_API_KEY"] = "test-dart-key"
    from src.config import Settings
    s = Settings()
    assert s.opendart_api_key == "test-dart-key"
    os.environ.pop("OPENDART_API_KEY", None)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/config.py config.yaml tests/test_config.py
git commit -m "feat(fundamentals): add config for Open DART fundamentals screener"
```

---

### Task 2: DART Data Models

**Files:**
- Create: `src/dart/__init__.py`
- Create: `src/dart/models.py`
- Create: `tests/test_dart_models.py`

- [ ] **Step 1: Create package init**

Create empty file:
```python
# src/dart/__init__.py
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_dart_models.py
from src.dart.models import CorpInfo, FinancialStatement


def test_corp_info_creation():
    corp = CorpInfo(
        corp_code="00126380",
        ticker="005930",
        name="삼성전자",
        market="KOSPI",
    )
    assert corp.corp_code == "00126380"
    assert corp.ticker == "005930"
    assert corp.market == "KOSPI"


def test_financial_statement_creation():
    fs = FinancialStatement(
        corp_code="00126380",
        year=2025,
        quarter=0,
        account="매출액",
        value=300_000_000_000_000.0,
    )
    assert fs.year == 2025
    assert fs.quarter == 0  # annual report
    assert fs.account == "매출액"
```

- [ ] **Step 3: Run tests to verify failure**

Run: `pytest tests/test_dart_models.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement models**

```python
# src/dart/models.py
from pydantic import BaseModel


class CorpInfo(BaseModel):
    """Listed corporation master data from Open DART."""

    corp_code: str        # 8-digit DART internal code
    ticker: str           # 6-digit stock code
    name: str
    market: str           # "KOSPI" | "KOSDAQ"


class FinancialStatement(BaseModel):
    """A single account value from a financial report."""

    corp_code: str
    year: int
    quarter: int          # 0 = annual report, 1/2/3 = quarter reports
    account: str          # e.g. "매출액", "영업이익", "당기순이익"
    value: float
```

- [ ] **Step 5: Run tests to verify pass**

Run: `pytest tests/test_dart_models.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/dart/__init__.py src/dart/models.py tests/test_dart_models.py
git commit -m "feat(dart): add CorpInfo and FinancialStatement models"
```

---

### Task 3: DART API Client

**Files:**
- Create: `src/dart/client.py`
- Create: `tests/test_dart_client.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dart_client.py
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from src.dart.client import DartClient


@pytest.mark.asyncio
async def test_get_returns_json():
    """Test that DartClient.get parses JSON response."""
    client = DartClient(api_key="test-key")

    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "000", "list": [{"a": 1}]}
    mock_response.raise_for_status = MagicMock()

    mock_async_client = MagicMock()
    mock_async_client.get = AsyncMock(return_value=mock_response)
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        result = await client.get("/api/foo.json", params={"p": "v"})

    assert result == {"status": "000", "list": [{"a": 1}]}
    call_args = mock_async_client.get.call_args
    assert "crtfc_key" in call_args.kwargs["params"]
    assert call_args.kwargs["params"]["crtfc_key"] == "test-key"


@pytest.mark.asyncio
async def test_get_handles_dart_error_status():
    """DART returns status='013' meaning no data; should not raise but return empty."""
    client = DartClient(api_key="test-key")

    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "013", "message": "조회된 데이타가 없습니다."}
    mock_response.raise_for_status = MagicMock()

    mock_async_client = MagicMock()
    mock_async_client.get = AsyncMock(return_value=mock_response)
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        result = await client.get("/api/foo.json", params={})

    assert result == {"status": "013", "message": "조회된 데이타가 없습니다."}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_dart_client.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement DartClient**

```python
# src/dart/client.py
from __future__ import annotations

import asyncio

import httpx
from loguru import logger


DART_BASE_URL = "https://opendart.fss.or.kr"
RATE_LIMIT_PER_MINUTE = 1000


class DartClient:
    """Open DART API client with rate limiting and retries."""

    def __init__(self, api_key: str, max_concurrency: int = 10):
        if not api_key:
            raise ValueError("OPENDART_API_KEY is required")
        self._api_key = api_key
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def get(self, path: str, params: dict, max_retries: int = 3) -> dict:
        """GET request to DART API. Adds crtfc_key automatically."""
        url = f"{DART_BASE_URL}{path}"
        full_params = {**params, "crtfc_key": self._api_key}

        async with self._semaphore:
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        resp = await client.get(url, params=full_params)
                        resp.raise_for_status()
                        return resp.json()
                except httpx.HTTPStatusError as e:
                    if e.response.status_code >= 500 and attempt < max_retries - 1:
                        wait = 2 ** attempt
                        logger.warning(f"DART {e.response.status_code}, retry in {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    raise
                except httpx.RequestError as e:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt
                        logger.warning(f"DART request error: {e}, retry in {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    raise

        raise RuntimeError("Unreachable")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_dart_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dart/client.py tests/test_dart_client.py
git commit -m "feat(dart): add async API client with rate limit and retries"
```

---

### Task 4: DART Cache (DB tables + read/write)

**Files:**
- Create: `src/dart/cache.py`
- Create: `tests/test_dart_cache.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dart_cache.py
from datetime import datetime
import pytest

from src.dart.cache import DartCache
from src.dart.models import CorpInfo, FinancialStatement


@pytest.fixture
def cache(tmp_path):
    db_url = f"sqlite:///{tmp_path}/test.db"
    return DartCache(url=db_url)


def test_save_and_load_corp_info(cache):
    corps = [
        CorpInfo(corp_code="00000001", ticker="005930", name="삼성전자", market="KOSPI"),
        CorpInfo(corp_code="00000002", ticker="000660", name="SK하이닉스", market="KOSPI"),
    ]
    cache.save_corp_info(corps)

    loaded = cache.load_corp_info()
    assert len(loaded) == 2
    assert loaded[0].ticker == "005930"


def test_save_and_load_financials(cache):
    statements = [
        FinancialStatement(corp_code="00000001", year=2025, quarter=0, account="매출액", value=300_000_000_000_000.0),
        FinancialStatement(corp_code="00000001", year=2025, quarter=0, account="영업이익", value=30_000_000_000_000.0),
    ]
    cache.save_financials(statements)

    loaded = cache.load_financials(corp_codes=["00000001"])
    assert len(loaded) == 2
    accounts = {s.account for s in loaded}
    assert "매출액" in accounts
    assert "영업이익" in accounts


def test_meta_last_updated(cache):
    assert cache.last_updated() is None
    cache.set_last_updated(datetime(2026, 4, 1, 12, 0))
    ts = cache.last_updated()
    assert ts is not None
    assert ts.year == 2026
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_dart_cache.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement DartCache**

```python
# src/dart/cache.py
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, DateTime,
    create_engine, delete, select,
)
from sqlalchemy.orm import DeclarativeBase, Session

from src.dart.models import CorpInfo, FinancialStatement


class DartBase(DeclarativeBase):
    pass


class DartCorpInfoRow(DartBase):
    __tablename__ = "dart_corp_info"
    corp_code = Column(String(8), primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    market = Column(String(10), nullable=False)


class DartFinancialRow(DartBase):
    __tablename__ = "dart_financials"
    id = Column(Integer, primary_key=True, autoincrement=True)
    corp_code = Column(String(8), nullable=False, index=True)
    year = Column(Integer, nullable=False)
    quarter = Column(Integer, nullable=False)
    account = Column(String(50), nullable=False)
    value = Column(Float, nullable=False)


class DartMetaRow(DartBase):
    __tablename__ = "dart_meta"
    key = Column(String(50), primary_key=True)
    value = Column(String(100), nullable=False)


class DartCache:
    """SQLite-based cache for Open DART data."""

    def __init__(self, url: str = "sqlite:///data/scanner.db"):
        self.engine = create_engine(url)
        DartBase.metadata.create_all(self.engine)

    def save_corp_info(self, corps: list[CorpInfo]) -> None:
        with Session(self.engine) as session:
            session.execute(delete(DartCorpInfoRow))
            for c in corps:
                session.add(DartCorpInfoRow(
                    corp_code=c.corp_code, ticker=c.ticker,
                    name=c.name, market=c.market,
                ))
            session.commit()

    def load_corp_info(self, markets: list[str] | None = None) -> list[CorpInfo]:
        with Session(self.engine) as session:
            stmt = select(DartCorpInfoRow)
            if markets:
                stmt = stmt.where(DartCorpInfoRow.market.in_(markets))
            rows = session.execute(stmt).scalars().all()
            return [
                CorpInfo(corp_code=r.corp_code, ticker=r.ticker, name=r.name, market=r.market)
                for r in rows
            ]

    def save_financials(self, statements: list[FinancialStatement]) -> None:
        with Session(self.engine) as session:
            corp_codes = {s.corp_code for s in statements}
            for cc in corp_codes:
                session.execute(
                    delete(DartFinancialRow).where(DartFinancialRow.corp_code == cc)
                )
            for s in statements:
                session.add(DartFinancialRow(
                    corp_code=s.corp_code, year=s.year, quarter=s.quarter,
                    account=s.account, value=s.value,
                ))
            session.commit()

    def load_financials(
        self, corp_codes: list[str] | None = None,
    ) -> list[FinancialStatement]:
        with Session(self.engine) as session:
            stmt = select(DartFinancialRow)
            if corp_codes:
                stmt = stmt.where(DartFinancialRow.corp_code.in_(corp_codes))
            rows = session.execute(stmt).scalars().all()
            return [
                FinancialStatement(
                    corp_code=r.corp_code, year=r.year, quarter=r.quarter,
                    account=r.account, value=r.value,
                )
                for r in rows
            ]

    def last_updated(self) -> datetime | None:
        with Session(self.engine) as session:
            row = session.execute(
                select(DartMetaRow).where(DartMetaRow.key == "last_updated")
            ).scalar_one_or_none()
            if row is None:
                return None
            return datetime.fromisoformat(row.value)

    def set_last_updated(self, ts: datetime) -> None:
        with Session(self.engine) as session:
            session.execute(delete(DartMetaRow).where(DartMetaRow.key == "last_updated"))
            session.add(DartMetaRow(key="last_updated", value=ts.isoformat()))
            session.commit()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_dart_cache.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dart/cache.py tests/test_dart_cache.py
git commit -m "feat(dart): add DartCache for SQLite-based DART data persistence"
```

---

### Task 5: DART Fetcher (Universe + Financials)

**Files:**
- Create: `src/dart/fetcher.py`
- Create: `tests/test_dart_fetcher.py`

The Open DART corp universe comes from `corpCode.xml` ZIP file at `/api/corpCode.xml`. Financial statements come from `fnlttMultiAcnt.json` (multi-corp main accounts, up to 100 corps per call).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dart_fetcher.py
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.dart.fetcher import DartFetcher
from src.dart.models import CorpInfo


@pytest.mark.asyncio
async def test_fetch_corp_universe_filters_listed():
    """Only corps with non-empty stock_code (listed) and target markets are returned."""
    mock_client = MagicMock()
    fetcher = DartFetcher(client=mock_client)

    # Mock the corp universe XML download
    fake_xml_zip = b"fake-zip-bytes"
    fake_corps_xml = """<?xml version="1.0" encoding="UTF-8"?>
<result>
  <list>
    <corp_code>00126380</corp_code>
    <corp_name>삼성전자</corp_name>
    <stock_code>005930</stock_code>
    <modify_date>20250101</modify_date>
  </list>
  <list>
    <corp_code>00264529</corp_code>
    <corp_name>비상장기업</corp_name>
    <stock_code></stock_code>
    <modify_date>20250101</modify_date>
  </list>
</result>"""

    async def mock_download(*args, **kwargs):
        return fake_xml_zip

    market_map = {"005930": "KOSPI"}

    with patch.object(fetcher, "_download_corp_zip", side_effect=mock_download), \
         patch.object(fetcher, "_extract_xml", return_value=fake_corps_xml):
        corps = await fetcher.fetch_corp_universe(
            markets=["KOSPI", "KOSDAQ"], market_map=market_map,
        )

    assert len(corps) == 1
    assert corps[0].ticker == "005930"
    assert corps[0].market == "KOSPI"


@pytest.mark.asyncio
async def test_fetch_financials_batches_by_100():
    """Multi-account API supports up to 100 corp_codes per call."""
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value={
        "status": "000",
        "list": [
            {
                "corp_code": "00000001", "bsns_year": "2025", "reprt_code": "11011",
                "account_nm": "매출액", "thstrm_amount": "1,000,000",
            },
        ],
    })
    fetcher = DartFetcher(client=mock_client)

    corp_codes = [f"{i:08d}" for i in range(1, 251)]  # 250 corps
    statements = await fetcher.fetch_financials(corp_codes, years=[2025], report_codes=["11011"])

    # 250 corps / 100 batch * 1 year * 1 report = 3 calls
    assert mock_client.get.call_count == 3
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_dart_fetcher.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement DartFetcher**

```python
# src/dart/fetcher.py
from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from typing import Iterable

import httpx
from loguru import logger

from src.dart.client import DartClient
from src.dart.models import CorpInfo, FinancialStatement


# DART report codes
REPORT_CODES = {
    "11011": "사업보고서",       # annual (Q4)
    "11012": "반기보고서",       # H1
    "11013": "1분기보고서",
    "11014": "3분기보고서",
}

# Map DART account_nm → our internal account names (use as-is for now)
ACCOUNT_WHITELIST = {
    "매출액", "영업이익", "당기순이익", "자산총계", "부채총계",
    "자본총계", "이익잉여금", "유동자산", "유동부채",
}


class DartFetcher:
    """Fetches corp universe and financial statements via DART API."""

    def __init__(self, client: DartClient):
        self._client = client

    async def _download_corp_zip(self) -> bytes:
        """Download the corpCode.xml ZIP file."""
        url = "https://opendart.fss.or.kr/api/corpCode.xml"
        async with httpx.AsyncClient(timeout=60.0) as http:
            resp = await http.get(url, params={"crtfc_key": self._client._api_key})
            resp.raise_for_status()
            return resp.content

    def _extract_xml(self, zip_bytes: bytes) -> str:
        """Extract CORPCODE.xml from ZIP."""
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            with zf.open("CORPCODE.xml") as f:
                return f.read().decode("utf-8")

    async def fetch_corp_universe(
        self,
        markets: list[str],
        market_map: dict[str, str] | None = None,
    ) -> list[CorpInfo]:
        """Fetch all listed corporations matching markets filter.

        Args:
            markets: Market codes to keep (e.g., ["KOSPI", "KOSDAQ"]).
            market_map: Optional ticker -> market mapping (from KRX).
                        If provided, only tickers in this map are kept.
                        If None, all listed corps are returned with market="UNKNOWN".
        """
        logger.info("Downloading DART corp universe...")
        zip_bytes = await self._download_corp_zip()
        xml_str = self._extract_xml(zip_bytes)
        root = ET.fromstring(xml_str)

        corps: list[CorpInfo] = []
        for item in root.findall("list"):
            stock_code = (item.findtext("stock_code") or "").strip()
            if not stock_code:
                continue  # skip unlisted

            corp_code = (item.findtext("corp_code") or "").strip()
            name = (item.findtext("corp_name") or "").strip()

            if market_map is not None:
                market = market_map.get(stock_code)
                if market is None or market not in markets:
                    continue
            else:
                market = "UNKNOWN"

            corps.append(CorpInfo(
                corp_code=corp_code, ticker=stock_code, name=name, market=market,
            ))

        logger.info(f"Fetched {len(corps)} listed corporations")
        return corps

    async def fetch_financials(
        self,
        corp_codes: list[str],
        years: list[int],
        report_codes: list[str],
    ) -> list[FinancialStatement]:
        """Fetch financial statements via fnlttMultiAcnt.json (batch up to 100 corps)."""
        statements: list[FinancialStatement] = []

        # Quarter mapping
        report_to_quarter = {"11011": 0, "11012": 2, "11013": 1, "11014": 3}

        for year in years:
            for report_code in report_codes:
                for batch_start in range(0, len(corp_codes), 100):
                    batch = corp_codes[batch_start:batch_start + 100]
                    params = {
                        "corp_code": ",".join(batch),
                        "bsns_year": str(year),
                        "reprt_code": report_code,
                    }
                    data = await self._client.get("/api/fnlttMultiAcnt.json", params=params)

                    if data.get("status") != "000":
                        continue  # no data or error

                    for row in data.get("list", []):
                        account_nm = row.get("account_nm", "")
                        if account_nm not in ACCOUNT_WHITELIST:
                            continue

                        amount_str = row.get("thstrm_amount", "0").replace(",", "")
                        try:
                            value = float(amount_str)
                        except ValueError:
                            continue

                        statements.append(FinancialStatement(
                            corp_code=row.get("corp_code", ""),
                            year=year,
                            quarter=report_to_quarter[report_code],
                            account=account_nm,
                            value=value,
                        ))

        logger.info(f"Fetched {len(statements)} financial statement entries")
        return statements
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_dart_fetcher.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dart/fetcher.py tests/test_dart_fetcher.py
git commit -m "feat(dart): add DartFetcher for corp universe and financials"
```

---

### Task 6: Fundamentals Models + DB Schema

**Files:**
- Create: `src/fundamentals/__init__.py`
- Create: `src/fundamentals/models.py`
- Create: `src/fundamentals/db.py`
- Create: `tests/test_fundamentals_models.py`

- [ ] **Step 1: Create package init**

```python
# src/fundamentals/__init__.py
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_fundamentals_models.py
from datetime import date
from src.fundamentals.models import FundamentalsMetrics, ScoreCard


def test_metrics_creation():
    m = FundamentalsMetrics(
        ticker="005930",
        as_of_date=date(2026, 4, 17),
        roe=15.5, roic=12.0, debt_ratio=45.0,
        current_ratio=1.8, interest_coverage=8.5,
        operating_margin=20.0, revenue_cagr_3y=10.0, op_income_cagr_3y=12.0,
        ocf_to_ni_ratio=1.1, fcf_positive_years=3,
        pe=15.0, pb=1.5, peg=1.2,
    )
    assert m.ticker == "005930"
    assert m.roe == 15.5


def test_scorecard_creation():
    sc = ScoreCard(
        ticker="005930",
        as_of_date=date(2026, 4, 17),
        liquidity_score=20.0,
        profitability_score=22.0,
        growth_score=18.0,
        cashflow_score=23.0,
        total_score=83.0,
        grade="★★★★☆",
        categories=["Quality", "GARP"],
    )
    assert sc.total_score == 83.0
    assert "Quality" in sc.categories
```

- [ ] **Step 3: Run tests to verify failure**

Run: `pytest tests/test_fundamentals_models.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement models**

```python
# src/fundamentals/models.py
from datetime import date

from pydantic import BaseModel


class FundamentalsMetrics(BaseModel):
    """Derived financial metrics for a ticker."""

    ticker: str
    as_of_date: date

    # Stability
    current_ratio: float | None = None
    interest_coverage: float | None = None
    debt_ratio: float | None = None

    # Profitability
    roe: float | None = None
    roic: float | None = None
    operating_margin: float | None = None

    # Growth (3-year CAGR)
    revenue_cagr_3y: float | None = None
    op_income_cagr_3y: float | None = None

    # Cashflow
    ocf_to_ni_ratio: float | None = None
    fcf_positive_years: int | None = None

    # Valuation
    pe: float | None = None
    pb: float | None = None
    peg: float | None = None


class ScoreCard(BaseModel):
    """4-dimension scores plus total and categories."""

    ticker: str
    as_of_date: date
    liquidity_score: float | None = None      # out of 25
    profitability_score: float | None = None  # out of 25
    growth_score: float | None = None         # out of 25
    cashflow_score: float | None = None       # out of 25
    total_score: float                        # 0-100
    grade: str                                # ★★★★★ etc.
    categories: list[str] = []                # ["Quality", "GARP"] etc.
```

- [ ] **Step 5: Implement DB tables**

```python
# src/fundamentals/db.py
from __future__ import annotations

import json
from datetime import date

from sqlalchemy import (
    Column, Integer, String, Float, Date,
    create_engine, delete, select,
)
from sqlalchemy.orm import DeclarativeBase, Session

from src.fundamentals.models import FundamentalsMetrics, ScoreCard


class FundamentalsBase(DeclarativeBase):
    pass


class MetricsRow(FundamentalsBase):
    __tablename__ = "fundamentals_metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    as_of_date = Column(Date, nullable=False, index=True)
    current_ratio = Column(Float, nullable=True)
    interest_coverage = Column(Float, nullable=True)
    debt_ratio = Column(Float, nullable=True)
    roe = Column(Float, nullable=True)
    roic = Column(Float, nullable=True)
    operating_margin = Column(Float, nullable=True)
    revenue_cagr_3y = Column(Float, nullable=True)
    op_income_cagr_3y = Column(Float, nullable=True)
    ocf_to_ni_ratio = Column(Float, nullable=True)
    fcf_positive_years = Column(Integer, nullable=True)
    pe = Column(Float, nullable=True)
    pb = Column(Float, nullable=True)
    peg = Column(Float, nullable=True)


class ScoreRow(FundamentalsBase):
    __tablename__ = "fundamentals_scores"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    as_of_date = Column(Date, nullable=False, index=True)
    liquidity_score = Column(Float, nullable=True)
    profitability_score = Column(Float, nullable=True)
    growth_score = Column(Float, nullable=True)
    cashflow_score = Column(Float, nullable=True)
    total_score = Column(Float, nullable=False)
    grade = Column(String(10), nullable=False)
    categories = Column(String(200), nullable=False)  # JSON-encoded list


class FundamentalsDB:
    """Persistence for derived metrics and scores."""

    def __init__(self, url: str = "sqlite:///data/scanner.db"):
        self.engine = create_engine(url)
        FundamentalsBase.metadata.create_all(self.engine)

    def save_metrics(self, metrics: list[FundamentalsMetrics]) -> None:
        if not metrics:
            return
        with Session(self.engine) as session:
            as_of = metrics[0].as_of_date
            session.execute(delete(MetricsRow).where(MetricsRow.as_of_date == as_of))
            for m in metrics:
                session.add(MetricsRow(
                    ticker=m.ticker, as_of_date=m.as_of_date,
                    current_ratio=m.current_ratio, interest_coverage=m.interest_coverage,
                    debt_ratio=m.debt_ratio, roe=m.roe, roic=m.roic,
                    operating_margin=m.operating_margin,
                    revenue_cagr_3y=m.revenue_cagr_3y, op_income_cagr_3y=m.op_income_cagr_3y,
                    ocf_to_ni_ratio=m.ocf_to_ni_ratio, fcf_positive_years=m.fcf_positive_years,
                    pe=m.pe, pb=m.pb, peg=m.peg,
                ))
            session.commit()

    def save_scores(self, scores: list[ScoreCard]) -> None:
        if not scores:
            return
        with Session(self.engine) as session:
            as_of = scores[0].as_of_date
            session.execute(delete(ScoreRow).where(ScoreRow.as_of_date == as_of))
            for s in scores:
                session.add(ScoreRow(
                    ticker=s.ticker, as_of_date=s.as_of_date,
                    liquidity_score=s.liquidity_score,
                    profitability_score=s.profitability_score,
                    growth_score=s.growth_score, cashflow_score=s.cashflow_score,
                    total_score=s.total_score, grade=s.grade,
                    categories=json.dumps(s.categories, ensure_ascii=False),
                ))
            session.commit()

    def load_scores(self, as_of_date: date) -> list[ScoreCard]:
        with Session(self.engine) as session:
            rows = session.execute(
                select(ScoreRow).where(ScoreRow.as_of_date == as_of_date)
            ).scalars().all()
            return [
                ScoreCard(
                    ticker=r.ticker, as_of_date=r.as_of_date,
                    liquidity_score=r.liquidity_score,
                    profitability_score=r.profitability_score,
                    growth_score=r.growth_score, cashflow_score=r.cashflow_score,
                    total_score=r.total_score, grade=r.grade,
                    categories=json.loads(r.categories),
                )
                for r in rows
            ]

    def load_metrics(self, as_of_date: date) -> list[FundamentalsMetrics]:
        with Session(self.engine) as session:
            rows = session.execute(
                select(MetricsRow).where(MetricsRow.as_of_date == as_of_date)
            ).scalars().all()
            return [
                FundamentalsMetrics(
                    ticker=r.ticker, as_of_date=r.as_of_date,
                    current_ratio=r.current_ratio, interest_coverage=r.interest_coverage,
                    debt_ratio=r.debt_ratio, roe=r.roe, roic=r.roic,
                    operating_margin=r.operating_margin,
                    revenue_cagr_3y=r.revenue_cagr_3y,
                    op_income_cagr_3y=r.op_income_cagr_3y,
                    ocf_to_ni_ratio=r.ocf_to_ni_ratio,
                    fcf_positive_years=r.fcf_positive_years,
                    pe=r.pe, pb=r.pb, peg=r.peg,
                )
                for r in rows
            ]
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_fundamentals_models.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/fundamentals/__init__.py src/fundamentals/models.py src/fundamentals/db.py tests/test_fundamentals_models.py
git commit -m "feat(fundamentals): add models and DB persistence layer"
```

---

### Task 7: Metrics Calculator

**Files:**
- Create: `src/fundamentals/calculator.py`
- Create: `tests/test_fundamentals_calculator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fundamentals_calculator.py
from datetime import date
from src.dart.models import FinancialStatement
from src.fundamentals.calculator import compute_metrics


def _make_fs(corp_code, year, account, value, quarter=0):
    return FinancialStatement(
        corp_code=corp_code, year=year, quarter=quarter,
        account=account, value=value,
    )


def test_compute_basic_metrics():
    """Test ROE, debt ratio computation from minimal data."""
    statements = [
        _make_fs("001", 2025, "당기순이익", 100),
        _make_fs("001", 2025, "자본총계", 1000),
        _make_fs("001", 2025, "부채총계", 500),
        _make_fs("001", 2025, "자산총계", 1500),
        _make_fs("001", 2025, "유동자산", 800),
        _make_fs("001", 2025, "유동부채", 400),
        _make_fs("001", 2024, "당기순이익", 90),
        _make_fs("001", 2024, "자본총계", 950),
        _make_fs("001", 2023, "당기순이익", 80),
        _make_fs("001", 2023, "자본총계", 900),
    ]

    metrics = compute_metrics(
        ticker="000001", corp_code="001",
        statements=statements, as_of=date(2026, 4, 17),
        market_cap=None, eps=None, bps=None,
    )

    # ROE = NI / Equity = 100/1000 = 10%
    assert metrics.roe is not None
    assert abs(metrics.roe - 10.0) < 0.01
    # Debt ratio = 500/1000 = 50%
    assert abs(metrics.debt_ratio - 50.0) < 0.01
    # Current ratio = 800/400 = 2.0
    assert abs(metrics.current_ratio - 2.0) < 0.01


def test_compute_with_market_data():
    """Test P/E, P/B with market cap and EPS/BPS."""
    statements = [
        _make_fs("001", 2025, "당기순이익", 100),
        _make_fs("001", 2025, "자본총계", 1000),
        _make_fs("001", 2025, "자산총계", 1500),
        _make_fs("001", 2025, "부채총계", 500),
    ]

    metrics = compute_metrics(
        ticker="000001", corp_code="001",
        statements=statements, as_of=date(2026, 4, 17),
        market_cap=1500.0, eps=10.0, bps=100.0,
    )

    # P/E = price / EPS; market_cap=1500, NI=100 → implied price/EPS via market_cap/NI = 15
    assert metrics.pe is not None
    assert abs(metrics.pe - 15.0) < 0.01
    # P/B = market_cap / equity = 1500/1000 = 1.5
    assert abs(metrics.pb - 1.5) < 0.01


def test_revenue_cagr_3y():
    """Test 3-year CAGR computation."""
    statements = [
        _make_fs("001", 2025, "매출액", 1331),  # 33.1% CAGR vs 2022
        _make_fs("001", 2024, "매출액", 1210),
        _make_fs("001", 2023, "매출액", 1100),
        _make_fs("001", 2022, "매출액", 1000),
        _make_fs("001", 2025, "자본총계", 1000),
    ]

    metrics = compute_metrics(
        ticker="000001", corp_code="001",
        statements=statements, as_of=date(2026, 4, 17),
        market_cap=None, eps=None, bps=None,
    )

    # CAGR = (1331/1000)^(1/3) - 1 ≈ 0.10 → 10%
    assert metrics.revenue_cagr_3y is not None
    assert abs(metrics.revenue_cagr_3y - 10.0) < 0.5


def test_missing_data_returns_none():
    """Empty statements produce metrics with None fields."""
    metrics = compute_metrics(
        ticker="000001", corp_code="001",
        statements=[], as_of=date(2026, 4, 17),
        market_cap=None, eps=None, bps=None,
    )
    assert metrics.roe is None
    assert metrics.pe is None
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_fundamentals_calculator.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement calculator**

```python
# src/fundamentals/calculator.py
from __future__ import annotations

from datetime import date

from src.dart.models import FinancialStatement
from src.fundamentals.models import FundamentalsMetrics


def _account_values_by_year(
    statements: list[FinancialStatement], account: str, quarter: int = 0,
) -> dict[int, float]:
    """Return dict of year -> value for an account (annual reports only by default)."""
    result = {}
    for s in statements:
        if s.account == account and s.quarter == quarter:
            result[s.year] = s.value
    return result


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def _cagr(start: float, end: float, years: int) -> float | None:
    """Compound Annual Growth Rate as percentage."""
    if start <= 0 or end <= 0 or years <= 0:
        return None
    return ((end / start) ** (1.0 / years) - 1.0) * 100.0


def compute_metrics(
    ticker: str,
    corp_code: str,
    statements: list[FinancialStatement],
    as_of: date,
    market_cap: float | None,
    eps: float | None,
    bps: float | None,
) -> FundamentalsMetrics:
    """Compute derived financial metrics for a single ticker."""
    revenue = _account_values_by_year(statements, "매출액")
    op_income = _account_values_by_year(statements, "영업이익")
    net_income = _account_values_by_year(statements, "당기순이익")
    equity = _account_values_by_year(statements, "자본총계")
    debt = _account_values_by_year(statements, "부채총계")
    assets = _account_values_by_year(statements, "자산총계")
    current_assets = _account_values_by_year(statements, "유동자산")
    current_liabilities = _account_values_by_year(statements, "유동부채")

    if not equity:
        # No data at all
        return FundamentalsMetrics(ticker=ticker, as_of_date=as_of)

    latest_year = max(equity.keys())
    latest_revenue = revenue.get(latest_year)
    latest_op = op_income.get(latest_year)
    latest_ni = net_income.get(latest_year)
    latest_equity = equity.get(latest_year)
    latest_debt = debt.get(latest_year)
    latest_assets = assets.get(latest_year)
    latest_ca = current_assets.get(latest_year)
    latest_cl = current_liabilities.get(latest_year)

    # Stability
    current_ratio = _safe_div(latest_ca, latest_cl)
    debt_ratio_pct = _safe_div(latest_debt, latest_equity)
    if debt_ratio_pct is not None:
        debt_ratio_pct *= 100.0

    # Profitability — 3-year average ROE/ROIC
    roe_values: list[float] = []
    for y in sorted(equity.keys())[-3:]:
        ni = net_income.get(y)
        eq = equity.get(y)
        r = _safe_div(ni, eq)
        if r is not None:
            roe_values.append(r * 100.0)
    roe_avg = sum(roe_values) / len(roe_values) if roe_values else None

    # ROIC ≈ NI / (Equity + Debt) — simplified
    roic_values: list[float] = []
    for y in sorted(equity.keys())[-3:]:
        ni = net_income.get(y)
        eq = equity.get(y)
        d = debt.get(y, 0)
        if eq is None or ni is None:
            continue
        capital = (eq or 0) + (d or 0)
        if capital > 0:
            roic_values.append(ni / capital * 100.0)
    roic_avg = sum(roic_values) / len(roic_values) if roic_values else None

    operating_margin = None
    if latest_op is not None and latest_revenue and latest_revenue > 0:
        operating_margin = (latest_op / latest_revenue) * 100.0

    # Growth (3y CAGR)
    revenue_cagr = None
    if len(revenue) >= 4:
        years_sorted = sorted(revenue.keys())
        start_y, end_y = years_sorted[-4], years_sorted[-1]
        revenue_cagr = _cagr(revenue[start_y], revenue[end_y], end_y - start_y)

    op_income_cagr = None
    if len(op_income) >= 4:
        years_sorted = sorted(op_income.keys())
        start_y, end_y = years_sorted[-4], years_sorted[-1]
        if op_income[start_y] > 0 and op_income[end_y] > 0:
            op_income_cagr = _cagr(op_income[start_y], op_income[end_y], end_y - start_y)

    # Valuation (using market cap)
    pe = None
    if market_cap is not None and latest_ni is not None and latest_ni > 0:
        pe = market_cap / latest_ni

    pb = None
    if market_cap is not None and latest_equity is not None and latest_equity > 0:
        pb = market_cap / latest_equity

    peg = None
    if pe is not None and op_income_cagr is not None and op_income_cagr > 0:
        peg = pe / op_income_cagr

    return FundamentalsMetrics(
        ticker=ticker,
        as_of_date=as_of,
        current_ratio=current_ratio,
        debt_ratio=debt_ratio_pct,
        roe=roe_avg,
        roic=roic_avg,
        operating_margin=operating_margin,
        revenue_cagr_3y=revenue_cagr,
        op_income_cagr_3y=op_income_cagr,
        pe=pe,
        pb=pb,
        peg=peg,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fundamentals_calculator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fundamentals/calculator.py tests/test_fundamentals_calculator.py
git commit -m "feat(fundamentals): add metrics calculator (ROE/ROIC/CAGR/PE/PB)"
```

---

### Task 8: Scorer (4-Dimension Scoring)

**Files:**
- Create: `src/fundamentals/scorer.py`
- Create: `tests/test_fundamentals_scorer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fundamentals_scorer.py
from datetime import date
from src.fundamentals.models import FundamentalsMetrics
from src.fundamentals.scorer import score


def test_perfect_metrics_get_high_score():
    m = FundamentalsMetrics(
        ticker="005930", as_of_date=date(2026, 4, 17),
        current_ratio=2.0, interest_coverage=10.0, debt_ratio=50.0,
        roe=20.0, roic=18.0, operating_margin=25.0,
        revenue_cagr_3y=18.0, op_income_cagr_3y=20.0,
        ocf_to_ni_ratio=1.1, fcf_positive_years=3,
    )
    sc = score(m)
    assert sc.total_score >= 85
    assert sc.grade in ["★★★★★", "★★★★☆"]


def test_poor_metrics_get_low_score():
    m = FundamentalsMetrics(
        ticker="000000", as_of_date=date(2026, 4, 17),
        current_ratio=0.5, interest_coverage=0.5, debt_ratio=300.0,
        roe=2.0, roic=1.0, operating_margin=2.0,
        revenue_cagr_3y=-5.0, op_income_cagr_3y=-10.0,
        ocf_to_ni_ratio=0.3, fcf_positive_years=0,
    )
    sc = score(m)
    assert sc.total_score < 30


def test_partial_data_proportional_scaling():
    """Missing dimensions scale total proportionally."""
    m = FundamentalsMetrics(
        ticker="100000", as_of_date=date(2026, 4, 17),
        current_ratio=2.0, interest_coverage=10.0, debt_ratio=50.0,
        roe=20.0, roic=18.0, operating_margin=25.0,
        # No growth or cashflow data
    )
    sc = score(m)
    # Has 2/4 dimensions scored, both perfect → total scaled to 100
    assert sc.liquidity_score is not None
    assert sc.profitability_score is not None
    assert sc.growth_score is None
    assert sc.cashflow_score is None
    assert 90 <= sc.total_score <= 100
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_fundamentals_scorer.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement scorer**

```python
# src/fundamentals/scorer.py
from __future__ import annotations

from src.fundamentals.models import FundamentalsMetrics, ScoreCard


def _linear_score(value: float | None, max_pt: float, full: float, zero: float) -> float | None:
    """Linear scoring: full→max_pt, zero→0, clipped."""
    if value is None:
        return None
    if (full > zero and value >= full) or (full < zero and value <= full):
        return max_pt
    if (full > zero and value <= zero) or (full < zero and value >= zero):
        return 0.0
    return max_pt * (value - zero) / (full - zero)


def _liquidity(m: FundamentalsMetrics) -> float | None:
    """A. Liquidity/Stability (max 25)."""
    parts: list[float] = []
    cur = _linear_score(m.current_ratio, max_pt=25/3, full=1.5, zero=1.0)
    ic = _linear_score(m.interest_coverage, max_pt=25/3, full=5.0, zero=1.0)
    debt = _linear_score(m.debt_ratio, max_pt=25/3, full=100.0, zero=200.0)
    for p in (cur, ic, debt):
        if p is not None:
            parts.append(p)
    if not parts:
        return None
    return (sum(parts) / len(parts)) * 3  # scale back to /25


def _profitability(m: FundamentalsMetrics) -> float | None:
    """B. Profitability (max 25). ROE + ROIC + operating_margin."""
    parts: list[float] = []
    roe = _linear_score(m.roe, max_pt=25/3, full=15.0, zero=5.0)
    roic = _linear_score(m.roic, max_pt=25/3, full=15.0, zero=5.0)
    op = _linear_score(m.operating_margin, max_pt=25/3, full=20.0, zero=0.0)
    for p in (roe, roic, op):
        if p is not None:
            parts.append(p)
    if not parts:
        return None
    return (sum(parts) / len(parts)) * 3


def _growth(m: FundamentalsMetrics) -> float | None:
    """C. Growth (max 25). Revenue + Op income 3y CAGR."""
    parts: list[float] = []
    rev = _linear_score(m.revenue_cagr_3y, max_pt=25/2, full=15.0, zero=0.0)
    op = _linear_score(m.op_income_cagr_3y, max_pt=25/2, full=15.0, zero=0.0)
    for p in (rev, op):
        if p is not None:
            parts.append(p)
    if not parts:
        return None
    return (sum(parts) / len(parts)) * 2


def _cashflow(m: FundamentalsMetrics) -> float | None:
    """D. Cashflow Quality (max 25)."""
    parts: list[float] = []
    if m.ocf_to_ni_ratio is not None:
        ratio = m.ocf_to_ni_ratio
        if 1.0 <= ratio <= 1.2:
            parts.append(12.5)
        elif ratio > 1.2:
            parts.append(max(0, 12.5 - (ratio - 1.2) * 5))
        elif ratio >= 0.5:
            parts.append(12.5 * (ratio - 0.5) / 0.5)
        else:
            parts.append(0.0)
    if m.fcf_positive_years is not None:
        parts.append(min(12.5, m.fcf_positive_years * 12.5 / 3))
    if not parts:
        return None
    return (sum(parts) / len(parts)) * 2


def _grade_for(score_value: float) -> str:
    if score_value >= 90:
        return "★★★★★"
    if score_value >= 75:
        return "★★★★☆"
    if score_value >= 60:
        return "★★★☆☆"
    if score_value >= 45:
        return "★★☆☆☆"
    return "★☆☆☆☆"


def score(m: FundamentalsMetrics) -> ScoreCard:
    """Compute 4-dimension scores for a ticker."""
    liq = _liquidity(m)
    prof = _profitability(m)
    growth = _growth(m)
    cf = _cashflow(m)

    available = [s for s in (liq, prof, growth, cf) if s is not None]
    if not available:
        total = 0.0
    else:
        # Each dimension is 0-25; scale to 0-100 by averaging available, multiplying by 4
        avg = sum(available) / len(available)
        total = avg * 4  # because avg is /25, so /25 * 4 = /100
    total = round(total, 1)

    return ScoreCard(
        ticker=m.ticker,
        as_of_date=m.as_of_date,
        liquidity_score=round(liq, 1) if liq is not None else None,
        profitability_score=round(prof, 1) if prof is not None else None,
        growth_score=round(growth, 1) if growth is not None else None,
        cashflow_score=round(cf, 1) if cf is not None else None,
        total_score=total,
        grade=_grade_for(total),
        categories=[],  # filled by classifier
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fundamentals_scorer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fundamentals/scorer.py tests/test_fundamentals_scorer.py
git commit -m "feat(fundamentals): add 4-dimension scorer with proportional scaling"
```

---

### Task 9: Classifier (5 Categories)

**Files:**
- Create: `src/fundamentals/classifier.py`
- Create: `tests/test_fundamentals_classifier.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fundamentals_classifier.py
from datetime import date
from src.fundamentals.models import FundamentalsMetrics, ScoreCard
from src.fundamentals.classifier import classify, MarketMedians


def _metrics(**kwargs):
    return FundamentalsMetrics(ticker="X", as_of_date=date(2026, 4, 17), **kwargs)


def _score(total, **kwargs):
    return ScoreCard(
        ticker="X", as_of_date=date(2026, 4, 17),
        total_score=total, grade="★★★★☆", categories=[],
        **kwargs,
    )


def test_quality_label():
    m = _metrics(roe=20.0, debt_ratio=50.0)
    s = _score(80.0, liquidity_score=20)
    medians = MarketMedians(kospi_pe=15, kospi_pb=1.5, kosdaq_pe=20, kosdaq_pb=2.0)
    cats = classify(m, s, market="KOSPI", medians=medians)
    assert "Quality" in cats


def test_value_label_uses_market_specific_medians():
    m = _metrics(pe=8.0, pb=0.7)
    s = _score(60.0, liquidity_score=20)
    medians = MarketMedians(kospi_pe=15, kospi_pb=1.5, kosdaq_pe=20, kosdaq_pb=2.0)
    cats = classify(m, s, market="KOSPI", medians=medians)
    # 8 < 15*0.7 = 10.5 ✓; 0.7 < 1.5*0.7 = 1.05 ✓; liquidity 20 ≥ 18 ✓
    assert "Value" in cats


def test_growth_label():
    m = _metrics(revenue_cagr_3y=25.0, op_income_cagr_3y=20.0)
    s = _score(70.0)
    medians = MarketMedians(kospi_pe=15, kospi_pb=1.5, kosdaq_pe=20, kosdaq_pb=2.0)
    cats = classify(m, s, market="KOSDAQ", medians=medians)
    assert "Growth" in cats


def test_garp_label():
    m = _metrics(revenue_cagr_3y=18.0, peg=0.8)
    s = _score(70.0)
    medians = MarketMedians(kospi_pe=15, kospi_pb=1.5, kosdaq_pe=20, kosdaq_pb=2.0)
    cats = classify(m, s, market="KOSPI", medians=medians)
    assert "GARP" in cats


def test_caution_label():
    m = _metrics(interest_coverage=0.5)
    s = _score(40.0)
    medians = MarketMedians(kospi_pe=15, kospi_pb=1.5, kosdaq_pe=20, kosdaq_pb=2.0)
    cats = classify(m, s, market="KOSPI", medians=medians)
    assert "Caution" in cats


def test_multi_category():
    """A stock can have Quality + GARP simultaneously."""
    m = _metrics(roe=18.0, debt_ratio=40.0, revenue_cagr_3y=18.0, peg=0.8)
    s = _score(80.0, liquidity_score=20)
    medians = MarketMedians(kospi_pe=15, kospi_pb=1.5, kosdaq_pe=20, kosdaq_pb=2.0)
    cats = classify(m, s, market="KOSPI", medians=medians)
    assert "Quality" in cats
    assert "GARP" in cats
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_fundamentals_classifier.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement classifier**

```python
# src/fundamentals/classifier.py
from __future__ import annotations

from pydantic import BaseModel

from src.fundamentals.models import FundamentalsMetrics, ScoreCard


class MarketMedians(BaseModel):
    """Median P/E and P/B per market."""

    kospi_pe: float
    kospi_pb: float
    kosdaq_pe: float
    kosdaq_pb: float

    def pe_median(self, market: str) -> float:
        return self.kospi_pe if market == "KOSPI" else self.kosdaq_pe

    def pb_median(self, market: str) -> float:
        return self.kospi_pb if market == "KOSPI" else self.kosdaq_pb


def _is_quality(m: FundamentalsMetrics, s: ScoreCard) -> bool:
    return (
        s.total_score >= 75
        and m.roe is not None and m.roe >= 15
        and m.debt_ratio is not None and m.debt_ratio < 100
    )


def _is_value(
    m: FundamentalsMetrics, s: ScoreCard, market: str, medians: MarketMedians,
) -> bool:
    if m.pe is None or m.pb is None or s.liquidity_score is None:
        return False
    pe_threshold = medians.pe_median(market) * 0.7
    pb_threshold = medians.pb_median(market) * 0.7
    return (
        m.pe <= pe_threshold
        and m.pb <= pb_threshold
        and s.liquidity_score >= 18
    )


def _is_growth(m: FundamentalsMetrics) -> bool:
    if m.revenue_cagr_3y is None or m.revenue_cagr_3y < 20:
        return False
    # If op income data exists, require >= 15; else accept revenue alone
    if m.op_income_cagr_3y is not None and m.op_income_cagr_3y < 15:
        return False
    return True


def _is_garp(m: FundamentalsMetrics) -> bool:
    return (
        m.revenue_cagr_3y is not None and m.revenue_cagr_3y >= 15
        and m.peg is not None and m.peg <= 1.0
    )


def _is_caution(m: FundamentalsMetrics, s: ScoreCard) -> bool:
    if s.total_score < 45:
        return True
    if m.ocf_to_ni_ratio is not None and m.ocf_to_ni_ratio < 0.5:
        return True
    if m.interest_coverage is not None and m.interest_coverage < 1:
        return True
    return False


def classify(
    m: FundamentalsMetrics,
    s: ScoreCard,
    market: str,
    medians: MarketMedians,
) -> list[str]:
    """Assign category labels to a ticker."""
    cats: list[str] = []
    if _is_quality(m, s):
        cats.append("Quality")
    if _is_value(m, s, market, medians):
        cats.append("Value")
    if _is_growth(m):
        cats.append("Growth")
    if _is_garp(m):
        cats.append("GARP")
    if _is_caution(m, s):
        cats.append("Caution")
    return cats


def compute_market_medians(
    metrics: list[FundamentalsMetrics],
    markets: dict[str, str],
) -> MarketMedians:
    """Compute median P/E and P/B per market from a batch of metrics."""
    kospi_pe: list[float] = []
    kospi_pb: list[float] = []
    kosdaq_pe: list[float] = []
    kosdaq_pb: list[float] = []
    for m in metrics:
        market = markets.get(m.ticker)
        if market == "KOSPI":
            if m.pe is not None and m.pe > 0:
                kospi_pe.append(m.pe)
            if m.pb is not None and m.pb > 0:
                kospi_pb.append(m.pb)
        elif market == "KOSDAQ":
            if m.pe is not None and m.pe > 0:
                kosdaq_pe.append(m.pe)
            if m.pb is not None and m.pb > 0:
                kosdaq_pb.append(m.pb)

    def median(xs: list[float], default: float) -> float:
        if not xs:
            return default
        s = sorted(xs)
        n = len(s)
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2

    return MarketMedians(
        kospi_pe=median(kospi_pe, 15.0),
        kospi_pb=median(kospi_pb, 1.5),
        kosdaq_pe=median(kosdaq_pe, 20.0),
        kosdaq_pb=median(kosdaq_pb, 2.0),
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fundamentals_classifier.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fundamentals/classifier.py tests/test_fundamentals_classifier.py
git commit -m "feat(fundamentals): add 5-category classifier with market-specific medians"
```

---

### Task 10: Pipeline Orchestrator

**Files:**
- Create: `src/fundamentals/pipeline.py`
- Create: `tests/test_fundamentals_pipeline.py`

The pipeline ties together: cache → fetch (if needed) → metrics → score → classify → save.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fundamentals_pipeline.py
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, AsyncMock
import pytest

from src.dart.models import CorpInfo, FinancialStatement
from src.fundamentals.pipeline import Pipeline


@pytest.mark.asyncio
async def test_pipeline_uses_cache_when_fresh():
    """If last_updated is recent, skip fetch."""
    cache = MagicMock()
    cache.last_updated.return_value = datetime.now() - timedelta(days=5)
    cache.load_corp_info.return_value = [
        CorpInfo(corp_code="001", ticker="005930", name="삼성전자", market="KOSPI"),
    ]
    cache.load_financials.return_value = [
        FinancialStatement(corp_code="001", year=2025, quarter=0, account="자본총계", value=1000),
    ]
    fetcher = MagicMock()
    fetcher.fetch_corp_universe = AsyncMock()
    fetcher.fetch_financials = AsyncMock()

    pipeline = Pipeline(cache=cache, fetcher=fetcher, ttl_days=30)

    await pipeline.refresh_data(force=False, years=[2025])

    fetcher.fetch_corp_universe.assert_not_called()
    fetcher.fetch_financials.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_refreshes_when_force():
    cache = MagicMock()
    cache.last_updated.return_value = datetime.now()
    fetcher = MagicMock()
    fetcher.fetch_corp_universe = AsyncMock(return_value=[
        CorpInfo(corp_code="001", ticker="005930", name="삼성전자", market="KOSPI"),
    ])
    fetcher.fetch_financials = AsyncMock(return_value=[])

    pipeline = Pipeline(cache=cache, fetcher=fetcher, ttl_days=30)

    await pipeline.refresh_data(force=True, years=[2025])

    fetcher.fetch_corp_universe.assert_called_once()
    fetcher.fetch_financials.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_refreshes_when_ttl_expired():
    cache = MagicMock()
    cache.last_updated.return_value = datetime.now() - timedelta(days=60)
    fetcher = MagicMock()
    fetcher.fetch_corp_universe = AsyncMock(return_value=[
        CorpInfo(corp_code="001", ticker="005930", name="삼성전자", market="KOSPI"),
    ])
    fetcher.fetch_financials = AsyncMock(return_value=[])

    pipeline = Pipeline(cache=cache, fetcher=fetcher, ttl_days=30)

    await pipeline.refresh_data(force=False, years=[2025])

    fetcher.fetch_corp_universe.assert_called_once()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_fundamentals_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement Pipeline**

```python
# src/fundamentals/pipeline.py
from __future__ import annotations

from datetime import date, datetime, timedelta

from loguru import logger

from src.dart.cache import DartCache
from src.dart.fetcher import DartFetcher
from src.dart.models import CorpInfo, FinancialStatement
from src.fundamentals.calculator import compute_metrics
from src.fundamentals.classifier import classify, compute_market_medians
from src.fundamentals.db import FundamentalsDB
from src.fundamentals.models import FundamentalsMetrics, ScoreCard
from src.fundamentals.scorer import score


class Pipeline:
    """Orchestrates the full fundamentals screening pipeline."""

    def __init__(
        self,
        cache: DartCache,
        fetcher: DartFetcher,
        ttl_days: int = 30,
        fundamentals_db: FundamentalsDB | None = None,
    ):
        self._cache = cache
        self._fetcher = fetcher
        self._ttl_days = ttl_days
        self._db = fundamentals_db

    async def refresh_data(
        self,
        force: bool,
        years: list[int],
        markets: list[str] | None = None,
        market_map: dict[str, str] | None = None,
    ) -> None:
        """Refresh corp universe and financial data if needed.

        Args:
            force: Skip TTL check and refresh anyway.
            years: List of years to fetch.
            markets: Markets to include.
            market_map: ticker -> market (from KRX). Required for accurate market labels.
        """
        markets = markets or ["KOSPI", "KOSDAQ"]
        last = self._cache.last_updated()
        needs_refresh = (
            force
            or last is None
            or (datetime.now() - last) > timedelta(days=self._ttl_days)
        )
        if not needs_refresh:
            logger.info(f"DART cache fresh (last update: {last}), skipping refresh")
            return

        logger.info("Refreshing DART data...")
        corps = await self._fetcher.fetch_corp_universe(
            markets=markets, market_map=market_map,
        )
        self._cache.save_corp_info(corps)

        corp_codes = [c.corp_code for c in corps]
        # Annual reports only for now
        statements = await self._fetcher.fetch_financials(
            corp_codes=corp_codes,
            years=years,
            report_codes=["11011"],
        )
        self._cache.save_financials(statements)
        self._cache.set_last_updated(datetime.now())
        logger.info(f"Refresh complete: {len(corps)} corps, {len(statements)} statements")

    def compute_all(
        self,
        market_caps: dict[str, float],
        eps_map: dict[str, float] | None = None,
        bps_map: dict[str, float] | None = None,
        markets: list[str] | None = None,
    ) -> tuple[list[FundamentalsMetrics], list[ScoreCard]]:
        """Compute metrics and scores for all cached corps."""
        markets = markets or ["KOSPI", "KOSDAQ"]
        corps = self._cache.load_corp_info(markets=markets)
        all_statements = self._cache.load_financials()
        # Group statements by corp_code
        grouped: dict[str, list[FinancialStatement]] = {}
        for s in all_statements:
            grouped.setdefault(s.corp_code, []).append(s)

        as_of = date.today()
        metrics_list: list[FundamentalsMetrics] = []
        for corp in corps:
            statements = grouped.get(corp.corp_code, [])
            mc = market_caps.get(corp.ticker)
            eps = (eps_map or {}).get(corp.ticker)
            bps = (bps_map or {}).get(corp.ticker)
            m = compute_metrics(
                ticker=corp.ticker,
                corp_code=corp.corp_code,
                statements=statements,
                as_of=as_of,
                market_cap=mc, eps=eps, bps=bps,
            )
            metrics_list.append(m)

        # Compute market medians for valuation classification
        ticker_to_market = {c.ticker: c.market for c in corps}
        medians = compute_market_medians(metrics_list, ticker_to_market)

        # Score and classify
        scores: list[ScoreCard] = []
        for m in metrics_list:
            sc = score(m)
            market = ticker_to_market.get(m.ticker, "KOSPI")
            sc.categories = classify(m, sc, market=market, medians=medians)
            scores.append(sc)

        if self._db is not None:
            self._db.save_metrics(metrics_list)
            self._db.save_scores(scores)

        return metrics_list, scores
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fundamentals_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fundamentals/pipeline.py tests/test_fundamentals_pipeline.py
git commit -m "feat(fundamentals): add pipeline orchestrator"
```

---

### Task 11: HTML Template + Report Generator

**Files:**
- Create: `src/fundamentals/templates/report.html`
- Create: `src/fundamentals/report.py`
- Create: `tests/test_fundamentals_report.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fundamentals_report.py
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_fundamentals_report.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create the HTML template**

Create `src/fundamentals/templates/report.html`:

```html
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>펀더멘털 스크리너 — {{ as_of_date }}</title>
  <script>{{ plotly_js }}</script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; color: #333; }
    header { background: linear-gradient(135deg, #2c3e50, #3498db); color: #fff; padding: 2rem; text-align: center; }
    header h1 { font-size: 2rem; margin-bottom: 0.3rem; }
    .container { max-width: 1400px; margin: 2rem auto; padding: 0 1.5rem; }
    section { margin-bottom: 3rem; }
    h2 { font-size: 1.4rem; color: #2c3e50; border-left: 4px solid #3498db; padding-left: 0.75rem; margin-bottom: 1.5rem; }
    .card { background: #fff; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 1.5rem; margin-bottom: 1.5rem; }
    .filter-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
    .filter-row input, .filter-row select { padding: 0.5rem; border: 1px solid #ccc; border-radius: 6px; font-size: 0.9rem; }
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    th { background: #eaf0fb; padding: 0.6rem; text-align: left; cursor: pointer; user-select: none; }
    th:hover { background: #d6e2f5; }
    td { padding: 0.5rem 0.6rem; border-bottom: 1px solid #eceff1; }
    tr:hover td { background: #f8fafc; }
    .badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 10px; font-size: 0.7rem; font-weight: 600; margin-right: 0.2rem; }
    .badge-Quality { background: #d6e9ff; color: #1864c0; }
    .badge-Value { background: #d4f4dd; color: #1e7e3a; }
    .badge-Growth { background: #ffd6d6; color: #c0392b; }
    .badge-GARP { background: #e6d6ff; color: #6a1ec0; }
    .badge-Caution { background: #e0e0e0; color: #555; }
    .grade { font-size: 0.9rem; color: #f39c12; }
    .top-card { background: #fff; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
    .top-card .name { font-weight: 600; color: #2c3e50; }
    .top-card .ticker { color: #888; font-size: 0.8rem; margin-left: 0.3rem; }
    .top-card .score { float: right; color: #3498db; font-weight: 600; }
    .grid-2x2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem; }
    .chart-container { width: 100%; height: 400px; }
    @media (max-width: 800px) { .grid-2x2 { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<header>
  <h1>📊 펀더멘털 스크리너</h1>
  <p>분석 기준일: {{ as_of_date }} &nbsp;|&nbsp; 분석 종목: {{ total_count }}개</p>
</header>

<div class="container">

  <!-- Section 1: Market Overview -->
  <section>
    <h2>1. 시장 개요</h2>
    <div class="card">
      <div id="category-donut" class="chart-container" style="height:300px;"></div>
    </div>
  </section>

  <!-- Section 2: Integrated Ranking (all stocks) -->
  <section>
    <h2>2. 통합 랭킹 (전체 {{ total_count }}개)</h2>
    <div class="card">
      <div class="filter-row">
        <input type="text" id="search-box" placeholder="종목명/티커 검색..." />
        <select id="market-filter">
          <option value="">전체 시장</option>
          <option value="KOSPI">KOSPI</option>
          <option value="KOSDAQ">KOSDAQ</option>
        </select>
        <select id="category-filter">
          <option value="">전체 카테고리</option>
          <option value="Quality">Quality</option>
          <option value="Value">Value</option>
          <option value="Growth">Growth</option>
          <option value="GARP">GARP</option>
          <option value="Caution">Caution</option>
        </select>
        <input type="number" id="min-score" placeholder="최소 점수" min="0" max="100" />
      </div>
      <div style="overflow-x:auto; max-height:600px; overflow-y:auto;">
        <table id="ranking-table">
          <thead>
            <tr>
              <th data-sort="rank">#</th>
              <th data-sort="ticker">티커</th>
              <th data-sort="name">종목명</th>
              <th data-sort="market">시장</th>
              <th data-sort="grade">등급</th>
              <th data-sort="total_score">종합점수</th>
              <th>카테고리</th>
              <th data-sort="pe">P/E</th>
              <th data-sort="pb">P/B</th>
              <th data-sort="roe">ROE</th>
            </tr>
          </thead>
          <tbody id="ranking-tbody"></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- Section 3: Category Top 10 -->
  <section>
    <h2>3. 카테고리별 Top 10</h2>
    <div class="grid-2x2">
      {% for cat in ['Quality', 'Value', 'Growth', 'GARP'] %}
      <div class="card">
        <h3>{{ cat }}</h3>
        <div id="top-{{ cat }}"></div>
      </div>
      {% endfor %}
    </div>
  </section>

  <!-- Section 4: Risk Scatter -->
  <section>
    <h2>4. 4차원 점수 분포</h2>
    <div class="card">
      <div id="dist-scatter" class="chart-container"></div>
    </div>
  </section>
</div>

<script>
const stockData = {{ stock_data | tojson }};
const categoryCounts = {{ category_counts | tojson }};

// Section 1: Donut chart
Plotly.newPlot('category-donut', [{
  values: Object.values(categoryCounts),
  labels: Object.keys(categoryCounts),
  type: 'pie',
  hole: 0.5,
  marker: {colors: ['#1864c0', '#1e7e3a', '#c0392b', '#6a1ec0', '#999']},
}], {margin: {t: 10, b: 10, l: 10, r: 10}, height: 280}, {responsive: true});

// Section 2: Ranking table with filters
function renderTable(data) {
  const tbody = document.getElementById('ranking-tbody');
  tbody.innerHTML = '';
  data.forEach((s, i) => {
    const cats = (s.categories || []).map(c => `<span class="badge badge-${c}">${c}</span>`).join('');
    const fmt = (v, d=2) => v == null ? '-' : Number(v).toFixed(d);
    const row = `<tr>
      <td>${i + 1}</td>
      <td>${s.ticker}</td>
      <td>${s.name}</td>
      <td>${s.market}</td>
      <td class="grade">${s.grade}</td>
      <td><b>${fmt(s.total_score, 1)}</b></td>
      <td>${cats}</td>
      <td>${fmt(s.pe)}</td>
      <td>${fmt(s.pb)}</td>
      <td>${fmt(s.roe, 1)}%</td>
    </tr>`;
    tbody.insertAdjacentHTML('beforeend', row);
  });
}

let sortedData = [...stockData].sort((a, b) => b.total_score - a.total_score);
renderTable(sortedData);

function applyFilters() {
  const search = document.getElementById('search-box').value.toLowerCase();
  const market = document.getElementById('market-filter').value;
  const cat = document.getElementById('category-filter').value;
  const minScore = parseFloat(document.getElementById('min-score').value) || 0;
  const filtered = sortedData.filter(s => {
    if (search && !(s.name.toLowerCase().includes(search) || s.ticker.includes(search))) return false;
    if (market && s.market !== market) return false;
    if (cat && !(s.categories || []).includes(cat)) return false;
    if (s.total_score < minScore) return false;
    return true;
  });
  renderTable(filtered);
}

['search-box', 'market-filter', 'category-filter', 'min-score'].forEach(id => {
  document.getElementById(id).addEventListener('input', applyFilters);
});

// Section 3: Top 10 per category
['Quality', 'Value', 'Growth', 'GARP'].forEach(cat => {
  const top = stockData
    .filter(s => (s.categories || []).includes(cat))
    .sort((a, b) => b.total_score - a.total_score)
    .slice(0, 10);
  const html = top.map(s =>
    `<div class="top-card"><span class="name">${s.name}</span><span class="ticker">${s.ticker}</span><span class="score">${s.total_score.toFixed(1)}</span></div>`
  ).join('');
  document.getElementById('top-' + cat).innerHTML = html || '<div style="color:#999;">해당 종목 없음</div>';
});

// Section 4: Distribution scatter
const stab_prof = stockData.map(s => (s.liquidity_score || 0) + (s.profitability_score || 0));
const growth = stockData.map(s => s.growth_score || 0);
const colors = stockData.map(s => {
  const c = (s.categories || [])[0];
  return {Quality: '#1864c0', Value: '#1e7e3a', Growth: '#c0392b', GARP: '#6a1ec0', Caution: '#999'}[c] || '#aaa';
});
const labels = stockData.map(s => s.name + ' (' + s.ticker + ')');
Plotly.newPlot('dist-scatter', [{
  x: stab_prof, y: growth,
  mode: 'markers', text: labels,
  marker: {size: 8, color: colors, opacity: 0.7},
  type: 'scatter',
}], {
  xaxis: {title: '안정성 + 수익성'},
  yaxis: {title: '성장성'},
  margin: {t: 20, b: 50, l: 60, r: 20},
  height: 400,
}, {responsive: true});
</script>
</body>
</html>
```

- [ ] **Step 4: Implement ReportGenerator**

```python
# src/fundamentals/report.py
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import plotly
from jinja2 import Environment, FileSystemLoader
from loguru import logger

from src.fundamentals.models import FundamentalsMetrics, ScoreCard


class ReportGenerator:
    """Generates the fundamentals HTML report."""

    def __init__(self):
        template_dir = Path(__file__).parent / "templates"
        self._env = Environment(loader=FileSystemLoader(str(template_dir)))

    def generate(
        self,
        metrics: list[FundamentalsMetrics],
        scores: list[ScoreCard],
        name_map: dict[str, str],
        market_map: dict[str, str],
        as_of_date: str,
        output_dir: str = "reports",
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"fundamentals-{as_of_date}.html")

        metrics_by_ticker = {m.ticker: m for m in metrics}

        # Build flat dict per stock for template
        stock_data = []
        for s in scores:
            m = metrics_by_ticker.get(s.ticker)
            stock_data.append({
                "ticker": s.ticker,
                "name": name_map.get(s.ticker, s.ticker),
                "market": market_map.get(s.ticker, ""),
                "grade": s.grade,
                "total_score": s.total_score,
                "liquidity_score": s.liquidity_score,
                "profitability_score": s.profitability_score,
                "growth_score": s.growth_score,
                "cashflow_score": s.cashflow_score,
                "categories": s.categories,
                "pe": m.pe if m else None,
                "pb": m.pb if m else None,
                "roe": m.roe if m else None,
            })

        # Category counts (count distinct memberships, multi-label)
        cat_counter: Counter = Counter()
        for s in scores:
            for c in s.categories:
                cat_counter[c] += 1

        plotly_js = plotly.offline.get_plotlyjs()
        template = self._env.get_template("report.html")
        html = template.render(
            as_of_date=as_of_date,
            total_count=len(stock_data),
            stock_data=stock_data,
            category_counts=dict(cat_counter),
            plotly_js=plotly_js,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Report written: {output_path}")
        return output_path
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_fundamentals_report.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/fundamentals/templates/report.html src/fundamentals/report.py tests/test_fundamentals_report.py
git commit -m "feat(fundamentals): add HTML report with filters and category sections"
```

---

### Task 12: Fundamentals CLI

**Files:**
- Create: `src/fundamentals/cli.py`
- Create: `tests/test_fundamentals_cli.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_fundamentals_cli.py
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from src.fundamentals.cli import app

runner = CliRunner()


def test_show_command_no_data():
    """show command on missing ticker prints message."""
    with patch("src.fundamentals.cli.FundamentalsDB") as MockDB:
        mock = MagicMock()
        mock.load_scores.return_value = []
        MockDB.return_value = mock

        result = runner.invoke(app, ["show", "999999"])
    assert result.exit_code == 0
    assert "999999" in result.stdout
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_fundamentals_cli.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement CLI**

```python
# src/fundamentals/cli.py
from __future__ import annotations

import asyncio
import os
import webbrowser
from datetime import date

import typer
from rich.console import Console

from src.config import Settings, load_scanner_config

app = typer.Typer(help="Open DART fundamentals screener", no_args_is_help=True)
console = Console()


def _make_pipeline(settings: Settings, ttl_days: int):
    from src.dart.cache import DartCache
    from src.dart.client import DartClient
    from src.dart.fetcher import DartFetcher
    from src.fundamentals.db import FundamentalsDB
    from src.fundamentals.pipeline import Pipeline

    cache = DartCache()
    client = DartClient(api_key=settings.opendart_api_key)
    fetcher = DartFetcher(client=client)
    db = FundamentalsDB()
    return Pipeline(cache=cache, fetcher=fetcher, ttl_days=ttl_days, fundamentals_db=db)


def _load_market_data(settings: Settings) -> tuple[dict[str, float], dict[str, str]]:
    """Load latest market caps and market info (KOSPI/KOSDAQ) from KRX.

    Returns:
        (market_caps: ticker -> 시가총액, market_map: ticker -> "KOSPI"|"KOSDAQ")
    """
    from src.krx_client import create_krx_client
    from datetime import datetime, timedelta

    client = create_krx_client(
        krx_id=settings.krx_id, krx_pw=settings.krx_pw, krx_api_key=settings.krx_api_key,
    )
    end = datetime.now()
    target_date = (end - timedelta(days=1)).strftime("%Y%m%d")
    market_caps: dict[str, float] = {}
    market_map: dict[str, str] = {}
    for market in ["KOSPI", "KOSDAQ"]:
        try:
            df = client.get_market_cap_by_ticker(target_date, market=market)
            for ticker, row in df.iterrows():
                market_caps[ticker] = float(row["시가총액"])
                market_map[ticker] = market
        except Exception:
            continue
    return market_caps, market_map


@app.command()
def run(
    refresh: bool = typer.Option(False, "--refresh", help="Force refresh of DART data"),
):
    """Run the full screening pipeline and generate report."""
    settings = Settings()
    config = load_scanner_config()

    console.print(f"[bold]펀더멘털 스크리너 시작[/bold]")

    pipeline = _make_pipeline(settings, ttl_days=config.fundamentals.cache_ttl_days)

    # Step 1: Load KRX market data (caps + market labels)
    console.print("[dim]1/4 KRX 시장 데이터 수집 중...[/dim]")
    market_caps, market_map = _load_market_data(settings)
    console.print(f"[dim]   {len(market_caps)}개 종목 시가총액/시장 정보 수집[/dim]")

    # Step 2: Refresh DART data (using KRX market_map for accurate market labels)
    console.print("[dim]2/4 DART 데이터 확인/수집 중...[/dim]")
    years = list(range(date.today().year - config.fundamentals.years_lookback, date.today().year))
    asyncio.run(pipeline.refresh_data(
        force=refresh, years=years,
        markets=config.fundamentals.market_filter,
        market_map=market_map,
    ))

    # Step 3: Compute metrics + scores
    console.print("[dim]3/4 지표 계산 + 점수 산정 중...[/dim]")
    metrics, scores = pipeline.compute_all(
        market_caps=market_caps,
        markets=config.fundamentals.market_filter,
    )
    console.print(f"[dim]   완료: {len(scores)}개 종목 점수 산정[/dim]")

    # Step 4: Generate report
    console.print("[dim]4/4 HTML 리포트 생성 중...[/dim]")
    name_map = {c.ticker: c.name for c in corps}
    market_map = {c.ticker: c.market for c in corps}
    from src.fundamentals.report import ReportGenerator
    gen = ReportGenerator()
    path = gen.generate(
        metrics=metrics, scores=scores,
        name_map=name_map, market_map=market_map,
        as_of_date=str(date.today()),
        output_dir=config.fundamentals.report_dir,
    )

    console.print(f"[bold green]완료! 리포트: {path}[/bold green]")
    webbrowser.open(f"file://{os.path.abspath(path)}")


@app.command()
def refresh():
    """Refresh DART data only (no analysis)."""
    settings = Settings()
    config = load_scanner_config()

    console.print(f"[bold]DART 데이터 갱신[/bold]")
    _, market_map = _load_market_data(settings)
    pipeline = _make_pipeline(settings, ttl_days=config.fundamentals.cache_ttl_days)
    years = list(range(date.today().year - config.fundamentals.years_lookback, date.today().year))
    asyncio.run(pipeline.refresh_data(
        force=True, years=years,
        markets=config.fundamentals.market_filter,
        market_map=market_map,
    ))
    console.print("[bold green]완료![/bold green]")


@app.command()
def show(ticker: str):
    """Show scorecard for a single ticker."""
    from src.fundamentals.db import FundamentalsDB
    db = FundamentalsDB()
    today = date.today()
    scores = db.load_scores(today)
    found = [s for s in scores if s.ticker == ticker]
    if not found:
        console.print(f"[yellow]{ticker} 점수 없음. 먼저 `python -m src.fundamentals.cli run` 실행하세요.[/yellow]")
        return
    s = found[0]
    console.print(f"[bold]{ticker}[/bold] — 등급: {s.grade}, 종합: {s.total_score}")
    console.print(f"  안정성: {s.liquidity_score}, 수익성: {s.profitability_score}")
    console.print(f"  성장성: {s.growth_score}, 현금흐름: {s.cashflow_score}")
    console.print(f"  카테고리: {', '.join(s.categories) if s.categories else '없음'}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fundamentals_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fundamentals/cli.py tests/test_fundamentals_cli.py
git commit -m "feat(fundamentals): add CLI (run/refresh/show)"
```

---

### Task 13: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read existing README.md**

Read the current content to find the right insertion point (after the forecast section).

- [ ] **Step 2: Add fundamentals section**

Append after the forecast section in README.md:

```markdown
## 펀더멘털 스크리너 (Fundamentals)

Open DART API로 KOSPI + KOSDAQ 전 상장사(~2,772개)의 재무 데이터를 수집하고, 4차원 점수와 5개 카테고리(Quality/Value/Growth/GARP/Caution)로 분류하는 스크리너.

### 추가 설정

`.env` 파일에 추가:
```
OPENDART_API_KEY=your_dart_api_key
```

[opendart.fss.or.kr](https://opendart.fss.or.kr)에서 무료 발급.

### 사용법

```bash
# 캐시 사용해서 빠른 스크리닝 + HTML 리포트
python -m src.fundamentals.cli run

# 데이터 강제 갱신 후 스크리닝
python -m src.fundamentals.cli run --refresh

# 데이터 갱신만 (분석 없이)
python -m src.fundamentals.cli refresh

# 특정 종목 스코어카드 조회
python -m src.fundamentals.cli show 005930
```

리포트는 `reports/fundamentals-YYYY-MM-DD.html`에 생성되며 브라우저에서 자동으로 열립니다.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add fundamentals screener usage to README"
```

---

### Task 14: Run All Tests

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: all PASS (existing 49 + new ~30 = ~80 tests)

- [ ] **Step 2: If any failures, fix and re-run**

Inspect failures, fix the affected files, re-run until green.
