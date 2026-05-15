# Fundamentals Data Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the NULL slots in `fundamentals_metrics` (OCF/NI ratio, FCF positive years, cashflow_score) and add 9 new derived columns (EPS, BPS, PSR, OCF, FCF, CAPEX/Revenue, dividend_yield, payout_ratio, consecutive_dividend_years) by extending DART account collection and adding pykrx-based historical market data.

**Architecture:** DART fetcher gains 4 new accounts (영업활동현금흐름 / 유형자산취득 / 배당총액 / 주당배당금). New `src/market_data/` package fetches yearly market_cap·shares·close from pykrx into a new `corp_market_yearly` table. Calculator derives all new metrics from the unified `(corp_code, year)` key. `/analyze-stock` refreshes its target ticker's current market cap on demand.

**Tech Stack:** Python 3.11, SQLAlchemy 2.0, pydantic 2.x, pykrx (new), httpx, pytest + pytest-asyncio, loguru, typer.

**Spec:** [docs/superpowers/specs/2026-05-15-fundamentals-data-enrichment-design.md](../specs/2026-05-15-fundamentals-data-enrichment-design.md)

---

## Task 1: Extend `FundamentalsMetrics` model with 9 new fields

**Files:**
- Modify: `src/fundamentals/models.py:1-37`
- Modify: `tests/test_fundamentals_models.py` (extend)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fundamentals_models.py`:

```python
from datetime import date
from src.fundamentals.models import FundamentalsMetrics


def test_metrics_accepts_new_enrichment_fields():
    m = FundamentalsMetrics(
        ticker="000660",
        as_of_date=date(2026, 5, 15),
        eps=1900.5,
        bps=120000.0,
        psr=2.3,
        ocf=470000.0,
        fcf=320000.0,
        capex_to_revenue=15.4,
        dividend_yield=1.8,
        payout_ratio=22.5,
        consecutive_dividend_years=7,
    )
    assert m.eps == 1900.5
    assert m.consecutive_dividend_years == 7


def test_metrics_new_fields_default_to_none():
    m = FundamentalsMetrics(ticker="X", as_of_date=date(2026, 1, 1))
    assert m.eps is None
    assert m.consecutive_dividend_years is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fundamentals_models.py::test_metrics_accepts_new_enrichment_fields -v`
Expected: FAIL (unknown fields for FundamentalsMetrics).

- [ ] **Step 3: Add 9 new optional fields to `FundamentalsMetrics`**

In `src/fundamentals/models.py`, add inside the `FundamentalsMetrics` class right after the `Valuation` block:

```python
    # Share-based derived (new)
    eps: float | None = None
    bps: float | None = None
    psr: float | None = None

    # Cashflow absolutes & ratios (new)
    ocf: float | None = None                   # 억원
    fcf: float | None = None                   # 억원
    capex_to_revenue: float | None = None      # %

    # Dividend (new)
    dividend_yield: float | None = None        # %
    payout_ratio: float | None = None          # %
    consecutive_dividend_years: int | None = None
```

- [ ] **Step 4: Run both tests to verify they pass**

Run: `pytest tests/test_fundamentals_models.py -v`
Expected: all PASS, including pre-existing tests.

- [ ] **Step 5: Commit**

```bash
git add src/fundamentals/models.py tests/test_fundamentals_models.py
git commit -m "feat(fundamentals): add 9 enrichment fields to FundamentalsMetrics"
```

---

## Task 2: ALTER `fundamentals_metrics` table + extend MetricsRow / save_metrics / load_metrics

**Files:**
- Modify: `src/fundamentals/db.py:18-39` (MetricsRow), `:62-84` (save_metrics), `:104-122` (load_metrics)
- Modify: `tests/test_db.py` or create `tests/test_fundamentals_db_enrichment.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fundamentals_db_enrichment.py`:

```python
from datetime import date
from src.fundamentals.db import FundamentalsDB
from src.fundamentals.models import FundamentalsMetrics


def test_save_load_round_trips_new_fields(tmp_path):
    db = FundamentalsDB(url=f"sqlite:///{tmp_path/'t.db'}")
    m = FundamentalsMetrics(
        ticker="000660", as_of_date=date(2026, 5, 15),
        eps=1900.5, bps=120000.0, psr=2.3,
        ocf=470000.0, fcf=320000.0, capex_to_revenue=15.4,
        dividend_yield=1.8, payout_ratio=22.5, consecutive_dividend_years=7,
    )
    db.save_metrics([m])
    loaded = db.load_metrics(date(2026, 5, 15))
    assert len(loaded) == 1
    r = loaded[0]
    assert r.eps == 1900.5
    assert r.bps == 120000.0
    assert r.psr == 2.3
    assert r.ocf == 470000.0
    assert r.fcf == 320000.0
    assert r.capex_to_revenue == 15.4
    assert r.dividend_yield == 1.8
    assert r.payout_ratio == 22.5
    assert r.consecutive_dividend_years == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fundamentals_db_enrichment.py -v`
Expected: FAIL (column does not exist or save_metrics ignores fields).

- [ ] **Step 3: Add 9 columns to `MetricsRow`**

In `src/fundamentals/db.py`, append inside `class MetricsRow` after `peg = Column(Float, nullable=True)`:

```python
    eps = Column(Float, nullable=True)
    bps = Column(Float, nullable=True)
    psr = Column(Float, nullable=True)
    ocf = Column(Float, nullable=True)
    fcf = Column(Float, nullable=True)
    capex_to_revenue = Column(Float, nullable=True)
    dividend_yield = Column(Float, nullable=True)
    payout_ratio = Column(Float, nullable=True)
    consecutive_dividend_years = Column(Integer, nullable=True)
```

- [ ] **Step 4: Update `save_metrics` to persist the new fields**

In `save_metrics`, extend the `MetricsRow(...)` constructor with the new kwargs:

```python
session.add(MetricsRow(
    ticker=m.ticker, as_of_date=m.as_of_date,
    current_ratio=m.current_ratio, interest_coverage=m.interest_coverage,
    debt_ratio=m.debt_ratio, roe=m.roe, roic=m.roic,
    operating_margin=m.operating_margin,
    revenue_cagr_3y=m.revenue_cagr_3y, op_income_cagr_3y=m.op_income_cagr_3y,
    ocf_to_ni_ratio=m.ocf_to_ni_ratio, fcf_positive_years=m.fcf_positive_years,
    pe=m.pe, pb=m.pb, peg=m.peg,
    eps=m.eps, bps=m.bps, psr=m.psr,
    ocf=m.ocf, fcf=m.fcf, capex_to_revenue=m.capex_to_revenue,
    dividend_yield=m.dividend_yield, payout_ratio=m.payout_ratio,
    consecutive_dividend_years=m.consecutive_dividend_years,
))
```

- [ ] **Step 5: Update `load_metrics` to populate the new fields**

Extend the `FundamentalsMetrics(...)` ctor inside the list comp:

```python
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
    eps=r.eps, bps=r.bps, psr=r.psr,
    ocf=r.ocf, fcf=r.fcf, capex_to_revenue=r.capex_to_revenue,
    dividend_yield=r.dividend_yield, payout_ratio=r.payout_ratio,
    consecutive_dividend_years=r.consecutive_dividend_years,
)
```

- [ ] **Step 6: Add `ALTER TABLE` migration for the existing prod DB**

Existing prod `data/scanner.db` already has rows. `FundamentalsBase.metadata.create_all` creates new tables but **does NOT add columns to existing tables**. Add a migration helper at the bottom of `db.py`:

```python
def _migrate_add_enrichment_columns(engine) -> None:
    """Idempotent ALTER for the 9 enrichment columns. Safe to run repeatedly."""
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    existing = {col["name"] for col in insp.get_columns("fundamentals_metrics")}
    to_add = [
        ("eps", "FLOAT"),
        ("bps", "FLOAT"),
        ("psr", "FLOAT"),
        ("ocf", "FLOAT"),
        ("fcf", "FLOAT"),
        ("capex_to_revenue", "FLOAT"),
        ("dividend_yield", "FLOAT"),
        ("payout_ratio", "FLOAT"),
        ("consecutive_dividend_years", "INTEGER"),
    ]
    with engine.begin() as conn:
        for name, sqltype in to_add:
            if name not in existing:
                conn.execute(text(f"ALTER TABLE fundamentals_metrics ADD COLUMN {name} {sqltype}"))
```

Call it in `FundamentalsDB.__init__` right after `create_all`:

```python
def __init__(self, url: str = "sqlite:///data/scanner.db"):
    self.engine = create_engine(url)
    FundamentalsBase.metadata.create_all(self.engine)
    _migrate_add_enrichment_columns(self.engine)
```

- [ ] **Step 7: Run the round-trip test + full fundamentals_db tests**

Run: `pytest tests/test_fundamentals_db_enrichment.py tests/test_db.py -v`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/fundamentals/db.py tests/test_fundamentals_db_enrichment.py
git commit -m "feat(fundamentals): add 9 enrichment columns + idempotent migration"
```

---

## Task 3: Create `MarketYearly` model

**Files:**
- Create: `src/market_data/__init__.py` (empty)
- Create: `src/market_data/models.py`
- Create: `tests/test_market_data_models.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_market_data_models.py`:

```python
from datetime import date
from src.market_data.models import MarketYearly


def test_market_yearly_roundtrip():
    m = MarketYearly(
        corp_code="00126380",
        ticker="000660",
        year=2025,
        as_of_date=date(2025, 12, 30),
        market_cap=240_000_000_000_000,
        shares_outstanding=727_960_000,
        close_price=330_000,
    )
    assert m.market_cap == 240_000_000_000_000
    assert m.year == 2025


def test_market_yearly_nullable_price_fields():
    m = MarketYearly(
        corp_code="00126380", ticker="000660",
        year=2020, as_of_date=date(2020, 12, 30),
    )
    assert m.market_cap is None
    assert m.shares_outstanding is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_market_data_models.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Create `src/market_data/__init__.py` (empty file)**

```bash
touch src/market_data/__init__.py
```

- [ ] **Step 4: Create `src/market_data/models.py`**

```python
from datetime import date
from pydantic import BaseModel


class MarketYearly(BaseModel):
    """Yearly market data snapshot for a single ticker.

    For completed years: as_of_date = last trading day of that year.
    For the in-progress year: as_of_date = backfill execution date (latest business day).
    """

    corp_code: str
    ticker: str
    year: int
    as_of_date: date
    market_cap: int | None = None        # 원
    shares_outstanding: int | None = None  # 주
    close_price: int | None = None        # 원 (verification)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_market_data_models.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/market_data/__init__.py src/market_data/models.py tests/test_market_data_models.py
git commit -m "feat(market_data): add MarketYearly model"
```

---

## Task 4: Create `corp_market_yearly` table + `MarketDB` CRUD with idempotent upsert

**Files:**
- Create: `src/market_data/db.py`
- Create: `tests/test_market_data_db.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_market_data_db.py`:

```python
from datetime import date
from src.market_data.db import MarketDB
from src.market_data.models import MarketYearly


def _row(year=2025, market_cap=100, shares=10, close=10):
    return MarketYearly(
        corp_code="00126380", ticker="000660",
        year=year, as_of_date=date(year, 12, 30),
        market_cap=market_cap, shares_outstanding=shares, close_price=close,
    )


def test_save_and_load_yearly(tmp_path):
    db = MarketDB(url=f"sqlite:///{tmp_path/'t.db'}")
    db.save_yearly([_row(2024), _row(2025)])
    loaded = db.load_for_corp("00126380")
    assert {r.year for r in loaded} == {2024, 2025}


def test_upsert_is_idempotent(tmp_path):
    db = MarketDB(url=f"sqlite:///{tmp_path/'t.db'}")
    db.save_yearly([_row(2025, market_cap=100)])
    db.save_yearly([_row(2025, market_cap=200)])  # same (corp_code, year) -> overwrite
    loaded = db.load_for_corp("00126380")
    assert len(loaded) == 1
    assert loaded[0].market_cap == 200


def test_load_all_groups_by_corp(tmp_path):
    db = MarketDB(url=f"sqlite:///{tmp_path/'t.db'}")
    db.save_yearly([
        _row(2024), _row(2025),
        MarketYearly(corp_code="00100001", ticker="005930",
                     year=2025, as_of_date=date(2025, 12, 30),
                     market_cap=500, shares_outstanding=50, close_price=10),
    ])
    grouped = db.load_all()
    assert "00126380" in grouped
    assert "00100001" in grouped
    assert len(grouped["00126380"]) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_market_data_db.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Create `src/market_data/db.py`**

```python
from __future__ import annotations
from datetime import date
from sqlalchemy import (
    Column, Integer, BigInteger, String, Date,
    create_engine, select,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import DeclarativeBase, Session

from src.market_data.models import MarketYearly


class MarketBase(DeclarativeBase):
    pass


class MarketYearlyRow(MarketBase):
    __tablename__ = "corp_market_yearly"
    corp_code = Column(String(8), primary_key=True)
    year = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    as_of_date = Column(Date, nullable=False)
    market_cap = Column(BigInteger, nullable=True)
    shares_outstanding = Column(BigInteger, nullable=True)
    close_price = Column(Integer, nullable=True)


class MarketDB:
    """Persistence for yearly market data (corp_market_yearly)."""

    def __init__(self, url: str = "sqlite:///data/scanner.db"):
        self.engine = create_engine(url)
        MarketBase.metadata.create_all(self.engine)

    def save_yearly(self, rows: list[MarketYearly]) -> None:
        if not rows:
            return
        with Session(self.engine) as session:
            for r in rows:
                stmt = sqlite_insert(MarketYearlyRow).values(
                    corp_code=r.corp_code, year=r.year, ticker=r.ticker,
                    as_of_date=r.as_of_date,
                    market_cap=r.market_cap,
                    shares_outstanding=r.shares_outstanding,
                    close_price=r.close_price,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=["corp_code", "year"],
                    set_={
                        "ticker": stmt.excluded.ticker,
                        "as_of_date": stmt.excluded.as_of_date,
                        "market_cap": stmt.excluded.market_cap,
                        "shares_outstanding": stmt.excluded.shares_outstanding,
                        "close_price": stmt.excluded.close_price,
                    },
                )
                session.execute(stmt)
            session.commit()

    def load_for_corp(self, corp_code: str) -> list[MarketYearly]:
        with Session(self.engine) as session:
            rows = session.execute(
                select(MarketYearlyRow).where(MarketYearlyRow.corp_code == corp_code)
            ).scalars().all()
            return [_row_to_model(r) for r in rows]

    def load_all(self) -> dict[str, list[MarketYearly]]:
        with Session(self.engine) as session:
            rows = session.execute(select(MarketYearlyRow)).scalars().all()
            out: dict[str, list[MarketYearly]] = {}
            for r in rows:
                out.setdefault(r.corp_code, []).append(_row_to_model(r))
            return out


def _row_to_model(r: MarketYearlyRow) -> MarketYearly:
    return MarketYearly(
        corp_code=r.corp_code, ticker=r.ticker,
        year=r.year, as_of_date=r.as_of_date,
        market_cap=r.market_cap,
        shares_outstanding=r.shares_outstanding,
        close_price=r.close_price,
    )
```

- [ ] **Step 4: Run all three tests**

Run: `pytest tests/test_market_data_db.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/market_data/db.py tests/test_market_data_db.py
git commit -m "feat(market_data): add MarketDB with idempotent upsert"
```

---

## Task 5: Extend DART `ACCOUNT_NORMALIZE` with 4 new accounts

**Files:**
- Modify: `src/dart/fetcher.py:18-44` (`ACCOUNT_NORMALIZE`)
- Modify: `tests/test_dart_fetcher.py` (add normalization test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dart_fetcher.py`:

```python
from src.dart.fetcher import ACCOUNT_NORMALIZE


def test_account_normalize_includes_cashflow_and_dividend():
    # OCF
    assert ACCOUNT_NORMALIZE.get("영업활동 현금흐름") == "영업활동현금흐름"
    assert ACCOUNT_NORMALIZE.get("영업활동으로 인한 현금흐름") == "영업활동현금흐름"
    # CAPEX
    assert ACCOUNT_NORMALIZE.get("유형자산의 취득") == "유형자산취득"
    # Dividend (paid)
    assert ACCOUNT_NORMALIZE.get("배당금지급") == "배당총액"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dart_fetcher.py::test_account_normalize_includes_cashflow_and_dividend -v`
Expected: FAIL (returns None for unknown keys).

- [ ] **Step 3: Add 4 new account groups to `ACCOUNT_NORMALIZE`**

In `src/dart/fetcher.py`, extend the `ACCOUNT_NORMALIZE` dict (after `"유동부채": "유동부채",`):

```python
    # 영업활동현금흐름 (OCF) - DART variants
    "영업활동현금흐름": "영업활동현금흐름",
    "영업활동 현금흐름": "영업활동현금흐름",
    "영업활동으로 인한 현금흐름": "영업활동현금흐름",
    "영업활동에서 창출된 현금흐름": "영업활동현금흐름",
    # 유형자산 취득 (CAPEX) - 투자활동 sub-line
    "유형자산의 취득": "유형자산취득",
    "유형자산 취득": "유형자산취득",
    # 배당금 지급 (Dividend paid) - 재무활동 sub-line, sign typically negative
    "배당금지급": "배당총액",
    "배당금 지급": "배당총액",
    "배당금의 지급": "배당총액",
```

- [ ] **Step 4: Run the new test plus full dart_fetcher test suite**

Run: `pytest tests/test_dart_fetcher.py -v`
Expected: all PASS, including any pre-existing tests.

- [ ] **Step 5: Commit**

```bash
git add src/dart/fetcher.py tests/test_dart_fetcher.py
git commit -m "feat(dart): normalize 4 new accounts (OCF, CAPEX, dividend paid)"
```

---

## Task 6: Add `pykrx` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add pykrx to dependencies**

In `pyproject.toml`, inside `dependencies = [...]`, add right after `"pandas>=2.0.0",`:

```toml
    "pykrx>=1.0.45",
```

- [ ] **Step 2: Install**

```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

Expected output: `Successfully installed pykrx-...`

- [ ] **Step 3: Verify import works**

```bash
python -c "from pykrx import stock; print(stock.get_previous_business_day('2025-12-31'))"
```

Expected: prints a date like `'20251230'`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add pykrx dependency for historical market data"
```

---

## Task 7: Create `src/market_data/fetcher.py` (pykrx wrapper)

**Files:**
- Create: `src/market_data/fetcher.py`
- Create: `tests/test_market_data_fetcher.py`

- [ ] **Step 1: Write the failing test (mock-based)**

Create `tests/test_market_data_fetcher.py`:

```python
from datetime import date
from unittest.mock import patch
import pandas as pd

from src.market_data.fetcher import fetch_yearly_market_data
from src.market_data.models import MarketYearly


def _fake_df(rows):
    """rows: list of (ticker, market_cap, shares, close)."""
    df = pd.DataFrame(rows, columns=["티커", "시가총액", "상장주식수", "종가"]).set_index("티커")
    return df


@patch("src.market_data.fetcher.stock.get_market_cap_by_ticker")
@patch("src.market_data.fetcher.stock.get_previous_business_day")
def test_fetcher_returns_one_row_per_ticker_per_year(prev_bday, get_cap):
    prev_bday.side_effect = lambda d: d.replace("-", "")[:8]   # passthrough
    get_cap.return_value = _fake_df([
        ("000660", 240_000_000_000_000, 727_960_000, 330_000),
        ("353200", 1_300_000_000_000, 50_000_000, 26_000),
    ])

    out = fetch_yearly_market_data(
        tickers=["000660", "353200"],
        years=[2024],
        corp_code_map={"000660": "00126380", "353200": "00120182"},
    )

    assert len(out) == 2
    by_ticker = {m.ticker: m for m in out}
    assert by_ticker["000660"].market_cap == 240_000_000_000_000
    assert by_ticker["000660"].year == 2024
    assert by_ticker["353200"].shares_outstanding == 50_000_000


@patch("src.market_data.fetcher.stock.get_market_cap_by_ticker")
@patch("src.market_data.fetcher.stock.get_previous_business_day")
def test_fetcher_skips_tickers_not_in_response(prev_bday, get_cap):
    prev_bday.side_effect = lambda d: d.replace("-", "")[:8]
    get_cap.return_value = _fake_df([
        ("000660", 240e12, 727_960_000, 330_000),
    ])

    out = fetch_yearly_market_data(
        tickers=["000660", "999999"],   # 999999 missing
        years=[2024],
        corp_code_map={"000660": "00126380", "999999": "99999999"},
    )
    assert len(out) == 1
    assert out[0].ticker == "000660"


@patch("src.market_data.fetcher.stock.get_market_cap_by_ticker")
@patch("src.market_data.fetcher.stock.get_previous_business_day")
def test_fetcher_continues_on_year_error(prev_bday, get_cap):
    prev_bday.side_effect = lambda d: d.replace("-", "")[:8]
    # year 2024 raises, year 2025 succeeds
    get_cap.side_effect = [
        Exception("KRX timeout"),
        _fake_df([("000660", 240e12, 727_960_000, 330_000)]),
    ]
    out = fetch_yearly_market_data(
        tickers=["000660"], years=[2024, 2025],
        corp_code_map={"000660": "00126380"},
    )
    assert len(out) == 1
    assert out[0].year == 2025
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_market_data_fetcher.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement `fetch_yearly_market_data`**

Create `src/market_data/fetcher.py`:

```python
from __future__ import annotations
import time
from datetime import date, datetime
from loguru import logger
from pykrx import stock

from src.market_data.models import MarketYearly


def _resolve_as_of(year: int) -> str:
    """Return the 'YYYYMMDD' string to query for the given year.

    Completed year -> last business day of Dec. In-progress year -> latest business day today.
    """
    current = datetime.now().year
    if year < current:
        return stock.get_previous_business_day(f"{year}-12-31")
    return stock.get_previous_business_day(datetime.now().strftime("%Y-%m-%d"))


def fetch_yearly_market_data(
    tickers: list[str],
    years: list[int],
    corp_code_map: dict[str, str],
    max_retries: int = 3,
) -> list[MarketYearly]:
    """Fetch (ticker, year) yearly market_cap / shares / close from pykrx.

    Strategy: one pykrx call per year (returns ALL tickers in one DataFrame).
    Failures on a year are logged; other years continue.

    Args:
        tickers: filter to these tickers (others in pykrx response are dropped).
        years: list of years to fetch.
        corp_code_map: ticker -> corp_code (for storing alongside).
        max_retries: per-year retry count with exponential backoff.

    Returns:
        list[MarketYearly]. Failed (ticker, year) tuples are simply absent.
    """
    out: list[MarketYearly] = []
    ticker_set = set(tickers)

    for year in years:
        target = _resolve_as_of(year)
        as_of_date = datetime.strptime(target, "%Y%m%d").date()
        df = None
        for attempt in range(max_retries):
            try:
                df = stock.get_market_cap_by_ticker(target, market="ALL")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"market_data: year={year} failed after {max_retries} retries: {e}")
                else:
                    time.sleep(2 ** attempt)
        if df is None:
            continue

        for ticker in df.index:
            if ticker not in ticker_set:
                continue
            corp_code = corp_code_map.get(ticker)
            if corp_code is None:
                continue
            row = df.loc[ticker]
            try:
                mc = int(row["시가총액"])
                shares = int(row["상장주식수"])
                close = int(row["종가"])
            except (KeyError, ValueError, TypeError):
                continue
            if mc <= 0:
                continue
            out.append(MarketYearly(
                corp_code=corp_code, ticker=ticker, year=year,
                as_of_date=as_of_date,
                market_cap=mc, shares_outstanding=shares, close_price=close,
            ))

        logger.info(f"market_data: year={year} fetched {sum(1 for m in out if m.year == year)} tickers")

    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_market_data_fetcher.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/market_data/fetcher.py tests/test_market_data_fetcher.py
git commit -m "feat(market_data): add pykrx-based yearly fetcher with retry"
```

---

## Task 8: Create `MarketDataPipeline` + `RefreshReport` dataclass

**Files:**
- Create: `src/market_data/pipeline.py`
- Create: `tests/test_market_data_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_market_data_pipeline.py`:

```python
from datetime import date
from unittest.mock import patch
import pandas as pd

from src.market_data.db import MarketDB
from src.market_data.models import MarketYearly
from src.market_data.pipeline import MarketDataPipeline


def _fake_df(rows):
    df = pd.DataFrame(rows, columns=["티커", "시가총액", "상장주식수", "종가"]).set_index("티커")
    return df


@patch("src.market_data.fetcher.stock.get_market_cap_by_ticker")
@patch("src.market_data.fetcher.stock.get_previous_business_day")
def test_refresh_persists_rows_and_returns_report(prev_bday, get_cap, tmp_path):
    prev_bday.side_effect = lambda d: d.replace("-", "")[:8]
    get_cap.return_value = _fake_df([
        ("000660", 240e12, 727_960_000, 330_000),
    ])
    db = MarketDB(url=f"sqlite:///{tmp_path/'t.db'}")
    pipeline = MarketDataPipeline(db=db)

    report = pipeline.refresh(
        years=[2024],
        tickers=["000660"],
        corp_code_map={"000660": "00126380"},
    )

    assert report.successful_rows == 1
    assert db.load_for_corp("00126380")[0].market_cap == 240_000_000_000_000


@patch("src.market_data.fetcher.stock.get_market_cap_by_ticker")
@patch("src.market_data.fetcher.stock.get_previous_business_day")
def test_refresh_is_idempotent(prev_bday, get_cap, tmp_path):
    prev_bday.side_effect = lambda d: d.replace("-", "")[:8]
    get_cap.return_value = _fake_df([
        ("000660", 240e12, 727_960_000, 330_000),
    ])
    db = MarketDB(url=f"sqlite:///{tmp_path/'t.db'}")
    pipeline = MarketDataPipeline(db=db)

    pipeline.refresh(years=[2024], tickers=["000660"], corp_code_map={"000660": "00126380"})
    pipeline.refresh(years=[2024], tickers=["000660"], corp_code_map={"000660": "00126380"})
    rows = db.load_for_corp("00126380")
    assert len(rows) == 1   # not duplicated
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_market_data_pipeline.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement pipeline and report dataclasses**

Create `src/market_data/pipeline.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from src.market_data.db import MarketDB
from src.market_data.fetcher import fetch_yearly_market_data


@dataclass
class SourceReport:
    requested_years: list[int]
    requested_tickers_count: int
    successful_rows: int
    failed_items: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class MarketDataPipeline:
    """Orchestrates pykrx -> corp_market_yearly refresh."""

    def __init__(self, db: MarketDB):
        self._db = db

    def refresh(
        self,
        years: list[int],
        tickers: list[str],
        corp_code_map: dict[str, str],
    ) -> SourceReport:
        start = datetime.now()
        rows = fetch_yearly_market_data(
            tickers=tickers, years=years, corp_code_map=corp_code_map,
        )
        self._db.save_yearly(rows)
        duration = (datetime.now() - start).total_seconds()
        logger.info(f"market_data pipeline: {len(rows)} rows in {duration:.1f}s")
        return SourceReport(
            requested_years=years,
            requested_tickers_count=len(tickers),
            successful_rows=len(rows),
            duration_seconds=duration,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_market_data_pipeline.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/market_data/pipeline.py tests/test_market_data_pipeline.py
git commit -m "feat(market_data): add pipeline orchestration with refresh report"
```

---

## Task 9: Calculator — share-derived metrics (EPS, BPS, PSR) + refined PE/PB

**Files:**
- Modify: `src/fundamentals/calculator.py` (replace signature + add derive)
- Modify: `tests/test_fundamentals_calculator.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_fundamentals_calculator.py`:

```python
from datetime import date
from src.dart.models import FinancialStatement
from src.market_data.models import MarketYearly
from src.fundamentals.calculator import compute_metrics


def _stmt(year, account, value, corp="C1"):
    return FinancialStatement(corp_code=corp, year=year, quarter=0, account=account, value=value)


def _my(year, mc=None, shares=None, close=None, ticker="000660"):
    return MarketYearly(
        corp_code="C1", ticker=ticker, year=year,
        as_of_date=date(year, 12, 30),
        market_cap=mc, shares_outstanding=shares, close_price=close,
    )


def test_eps_uses_latest_year_net_income_over_shares():
    statements = [
        _stmt(2024, "당기순이익", 47_605_327_690),
        _stmt(2024, "자본총계", 800_000_000_000),
        _stmt(2024, "매출액", 1_065_000_000_000),
    ]
    market = [_my(2024, mc=2_000_000_000_000, shares=243_000_000)]
    m = compute_metrics(
        ticker="353200", corp_code="C1",
        statements=statements, market_yearly=market, as_of=date(2026, 5, 15),
    )
    assert m.eps is not None
    assert abs(m.eps - 195.9) < 1.0   # 47.6B / 243M


def test_bps_from_equity_over_shares():
    statements = [
        _stmt(2024, "자본총계", 800_000_000_000),
        _stmt(2024, "당기순이익", 1),
    ]
    market = [_my(2024, mc=1, shares=10_000_000)]
    m = compute_metrics(
        ticker="X", corp_code="C1",
        statements=statements, market_yearly=market, as_of=date(2026, 5, 15),
    )
    assert m.bps == 80_000.0


def test_psr_from_market_cap_now_over_revenue():
    statements = [
        _stmt(2024, "매출액", 1_000_000_000_000),
        _stmt(2024, "자본총계", 1),
        _stmt(2024, "당기순이익", 1),
    ]
    # market_yearly has both 2024 (LY) and 2025 (in-progress = "now")
    market = [
        _my(2024, mc=2_000_000_000_000, shares=100_000_000),
        _my(2025, mc=2_500_000_000_000, shares=100_000_000),
    ]
    m = compute_metrics(
        ticker="X", corp_code="C1",
        statements=statements, market_yearly=market, as_of=date(2026, 5, 15),
    )
    assert m.psr == 2.5    # 2.5T / 1T using market_cap_now (2025)


def test_pe_uses_market_cap_now_not_ly():
    statements = [
        _stmt(2024, "당기순이익", 1_000_000_000),
        _stmt(2024, "자본총계", 1),
    ]
    market = [
        _my(2024, mc=10_000_000_000, shares=1_000_000),
        _my(2025, mc=20_000_000_000, shares=1_000_000),   # "now"
    ]
    m = compute_metrics(
        ticker="X", corp_code="C1",
        statements=statements, market_yearly=market, as_of=date(2026, 5, 15),
    )
    assert m.pe == 20.0   # 20B / 1B using 2025 mc, not 10.0


def test_no_market_data_keeps_share_metrics_null():
    statements = [_stmt(2024, "당기순이익", 1), _stmt(2024, "자본총계", 1)]
    m = compute_metrics(
        ticker="X", corp_code="C1",
        statements=statements, market_yearly=[], as_of=date(2026, 5, 15),
    )
    assert m.eps is None
    assert m.bps is None
    assert m.psr is None
    assert m.pe is None
    assert m.pb is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fundamentals_calculator.py -v`
Expected: FAIL (signature mismatch).

- [ ] **Step 3: Refactor `compute_metrics` signature + add share-derived logic**

In `src/fundamentals/calculator.py`, **replace** the `compute_metrics` function. Keep helpers `_account_values_by_year`, `_safe_div`, `_cagr` unchanged.

```python
def compute_metrics(
    ticker: str,
    corp_code: str,
    statements: list[FinancialStatement],
    market_yearly: list[MarketYearly],
    as_of: date,
) -> FundamentalsMetrics:
    """Compute derived financial metrics for a single ticker.

    Reads:
      - statements: pivot-by-year dart accounts.
      - market_yearly: list of yearly market snapshots (already includes the in-progress year).

    Rule:
      - LY (latest year) values come from the most recent COMPLETED annual report.
      - market_cap_now / shares_now use the highest year() in market_yearly (i.e., the
        in-progress year row, written by the most recent backfill).
    """
    # --- pivot existing accounts ---
    revenue = _account_values_by_year(statements, "매출액")
    op_income = _account_values_by_year(statements, "영업이익")
    net_income = _account_values_by_year(statements, "당기순이익")
    equity = _account_values_by_year(statements, "자본총계")
    debt = _account_values_by_year(statements, "부채총계")
    assets = _account_values_by_year(statements, "자산총계")
    current_assets = _account_values_by_year(statements, "유동자산")
    current_liabilities = _account_values_by_year(statements, "유동부채")

    if not equity:
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

    # --- market data ---
    market_now: MarketYearly | None = None
    market_ly: MarketYearly | None = None
    if market_yearly:
        market_now = max(market_yearly, key=lambda r: r.year)
        ly_rows = [r for r in market_yearly if r.year == latest_year]
        market_ly = ly_rows[0] if ly_rows else None
    market_cap_now = market_now.market_cap if market_now else None
    shares_ly = market_ly.shares_outstanding if market_ly else None

    # --- existing stability / profitability / growth (unchanged) ---
    current_ratio = _safe_div(latest_ca, latest_cl)
    debt_ratio_pct = _safe_div(latest_debt, latest_equity)
    if debt_ratio_pct is not None:
        debt_ratio_pct *= 100.0

    roe_avg = _avg_ratio(net_income, equity, years=3)
    roic_avg = _avg_roic(net_income, equity, debt, years=3)

    operating_margin = None
    if latest_op is not None and latest_revenue and latest_revenue > 0:
        operating_margin = (latest_op / latest_revenue) * 100.0

    revenue_cagr = _three_year_cagr(revenue)
    op_income_cagr = _three_year_cagr_positive_only(op_income)

    # --- valuation (PE/PB now use market_cap_now) ---
    pe = market_cap_now / latest_ni if (market_cap_now is not None and latest_ni and latest_ni > 0) else None
    pb = market_cap_now / latest_equity if (market_cap_now is not None and latest_equity and latest_equity > 0) else None
    peg = pe / op_income_cagr if (pe is not None and op_income_cagr is not None and op_income_cagr > 0) else None

    # --- share-derived metrics (new) ---
    eps = latest_ni / shares_ly if (latest_ni is not None and shares_ly and shares_ly > 0) else None
    bps = latest_equity / shares_ly if (latest_equity is not None and shares_ly and shares_ly > 0) else None
    psr = market_cap_now / latest_revenue if (market_cap_now is not None and latest_revenue and latest_revenue > 0) else None

    return FundamentalsMetrics(
        ticker=ticker, as_of_date=as_of,
        current_ratio=current_ratio, debt_ratio=debt_ratio_pct,
        roe=roe_avg, roic=roic_avg, operating_margin=operating_margin,
        revenue_cagr_3y=revenue_cagr, op_income_cagr_3y=op_income_cagr,
        pe=pe, pb=pb, peg=peg,
        eps=eps, bps=bps, psr=psr,
    )


def _avg_ratio(numer: dict[int, float], denom: dict[int, float], years: int) -> float | None:
    """Avg of numer[y]/denom[y] over the most recent N years where both exist."""
    common = sorted((y for y in numer if y in denom), reverse=True)[:years]
    vals = [numer[y] / denom[y] for y in common if denom[y] != 0]
    return (sum(vals) / len(vals)) * 100.0 if vals else None


def _avg_roic(
    net_income: dict[int, float], equity: dict[int, float], debt: dict[int, float], years: int,
) -> float | None:
    common = sorted((y for y in net_income if y in equity), reverse=True)[:years]
    vals = []
    for y in common:
        capital = (equity[y] or 0) + (debt.get(y) or 0)
        if capital > 0:
            vals.append(net_income[y] / capital)
    return (sum(vals) / len(vals)) * 100.0 if vals else None


def _three_year_cagr(series: dict[int, float]) -> float | None:
    if len(series) < 4:
        return None
    ys = sorted(series.keys())
    return _cagr(series[ys[-4]], series[ys[-1]], ys[-1] - ys[-4])


def _three_year_cagr_positive_only(series: dict[int, float]) -> float | None:
    if len(series) < 4:
        return None
    ys = sorted(series.keys())
    s, e = series[ys[-4]], series[ys[-1]]
    if s <= 0 or e <= 0:
        return None
    return _cagr(s, e, ys[-1] - ys[-4])
```

Also add the import at the top:

```python
from src.market_data.models import MarketYearly
```

- [ ] **Step 4: Run all calculator tests**

Run: `pytest tests/test_fundamentals_calculator.py -v`
Expected: all PASS (including pre-existing tests adapted to new signature — if pre-existing tests still use old signature, **update them to pass `market_yearly=[]` or `[MarketYearly(...)]`**).

- [ ] **Step 5: Adapt pre-existing tests if any break**

Run again, and for each failing pre-existing test, replace its `compute_metrics(... market_cap=X, eps=Y, bps=Z)` call with `compute_metrics(... market_yearly=[MarketYearly(corp_code="...", ticker="...", year=Y, as_of_date=..., market_cap=X)])`.

Run: `pytest tests/test_fundamentals_calculator.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/fundamentals/calculator.py tests/test_fundamentals_calculator.py
git commit -m "feat(fundamentals): derive EPS/BPS/PSR + refine PE/PB to use market_cap_now"
```

---

## Task 10: Calculator — cashflow derived (OCF, FCF, CAPEX/Rev, OCF/NI 3yr avg, FCF positive years)

**Files:**
- Modify: `src/fundamentals/calculator.py` (extend `compute_metrics`)
- Modify: `tests/test_fundamentals_calculator.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_fundamentals_calculator.py`:

```python
def test_ocf_and_fcf_use_latest_year_in_억():
    statements = [
        _stmt(2024, "당기순이익", 1), _stmt(2024, "자본총계", 1),
        _stmt(2024, "영업활동현금흐름", 50_000_000_000),   # 500억
        _stmt(2024, "유형자산취득", 30_000_000_000),       # 300억
    ]
    m = compute_metrics("X", "C1", statements, [], date(2026, 5, 15))
    assert m.ocf == 500.0   # 500억
    assert m.fcf == 200.0   # 500 - 300 = 200억


def test_capex_to_revenue_pct():
    statements = [
        _stmt(2024, "당기순이익", 1), _stmt(2024, "자본총계", 1),
        _stmt(2024, "매출액", 1_000_000_000_000),
        _stmt(2024, "유형자산취득", 100_000_000_000),
    ]
    m = compute_metrics("X", "C1", statements, [], date(2026, 5, 15))
    assert m.capex_to_revenue == 10.0   # 100B / 1T * 100


def test_ocf_to_ni_ratio_averages_3_years():
    statements = [
        _stmt(2024, "당기순이익", 100), _stmt(2024, "영업활동현금흐름", 110), _stmt(2024, "자본총계", 1),
        _stmt(2023, "당기순이익", 100), _stmt(2023, "영업활동현금흐름", 90),
        _stmt(2022, "당기순이익", 100), _stmt(2022, "영업활동현금흐름", 100),
    ]
    m = compute_metrics("X", "C1", statements, [], date(2026, 5, 15))
    # avg(1.1, 0.9, 1.0) = 1.0
    assert abs(m.ocf_to_ni_ratio - 1.0) < 1e-9


def test_fcf_positive_years_counts_last_5():
    statements = [_stmt(2024, "자본총계", 1)]
    # OCF=100 CAPEX=50  -> FCF=+50 (positive)
    # OCF=50  CAPEX=80  -> FCF=-30 (negative)
    yearly = [(2020, 100, 50), (2021, 100, 50), (2022, 50, 80), (2023, 100, 50), (2024, 100, 50)]
    for y, ocf, capex in yearly:
        statements.append(_stmt(y, "영업활동현금흐름", ocf))
        statements.append(_stmt(y, "유형자산취득", capex))
    m = compute_metrics("X", "C1", statements, [], date(2026, 5, 15))
    assert m.fcf_positive_years == 4   # all except 2022


def test_cashflow_metrics_null_when_missing():
    statements = [_stmt(2024, "당기순이익", 1), _stmt(2024, "자본총계", 1)]
    m = compute_metrics("X", "C1", statements, [], date(2026, 5, 15))
    assert m.ocf is None
    assert m.fcf is None
    assert m.capex_to_revenue is None
    assert m.ocf_to_ni_ratio is None
    assert m.fcf_positive_years is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fundamentals_calculator.py -v -k "ocf or fcf or capex"`
Expected: FAIL.

- [ ] **Step 3: Add cashflow-derived logic to `compute_metrics`**

In `src/fundamentals/calculator.py`, inside `compute_metrics`, **before** the final `return FundamentalsMetrics(...)`:

```python
    ocf_by_year = _account_values_by_year(statements, "영업활동현금흐름")
    capex_by_year = _account_values_by_year(statements, "유형자산취득")
    dividend_by_year = _account_values_by_year(statements, "배당총액")

    # absolute values converted to 억원 (1e8)
    ocf = ocf_by_year.get(latest_year) / 1e8 if latest_year in ocf_by_year else None
    fcf = None
    if latest_year in ocf_by_year and latest_year in capex_by_year:
        fcf = (ocf_by_year[latest_year] - capex_by_year[latest_year]) / 1e8

    capex_to_revenue = None
    if latest_year in capex_by_year and latest_revenue and latest_revenue > 0:
        capex_to_revenue = (capex_by_year[latest_year] / latest_revenue) * 100.0

    ocf_to_ni_ratio = _avg_ocf_to_ni(ocf_by_year, net_income, years=3)
    fcf_positive_years = _count_fcf_positive(ocf_by_year, capex_by_year, years=5)
```

Add the `dividend_by_year` line (we'll use it in Task 11). Add helpers below the main function:

```python
def _avg_ocf_to_ni(ocf: dict[int, float], ni: dict[int, float], years: int) -> float | None:
    common = sorted((y for y in ocf if y in ni and ni[y] != 0), reverse=True)[:years]
    if not common:
        return None
    vals = [ocf[y] / ni[y] for y in common]
    return sum(vals) / len(vals)


def _count_fcf_positive(ocf: dict[int, float], capex: dict[int, float], years: int) -> int | None:
    common_years = sorted(set(ocf.keys()) | set(capex.keys()), reverse=True)[:years]
    if not common_years:
        return None
    count = 0
    for y in common_years:
        if y in ocf and y in capex:
            if (ocf[y] - capex[y]) > 0:
                count += 1
    return count
```

Then extend the final return:

```python
    return FundamentalsMetrics(
        ticker=ticker, as_of_date=as_of,
        current_ratio=current_ratio, debt_ratio=debt_ratio_pct,
        roe=roe_avg, roic=roic_avg, operating_margin=operating_margin,
        revenue_cagr_3y=revenue_cagr, op_income_cagr_3y=op_income_cagr,
        ocf_to_ni_ratio=ocf_to_ni_ratio, fcf_positive_years=fcf_positive_years,
        pe=pe, pb=pb, peg=peg,
        eps=eps, bps=bps, psr=psr,
        ocf=ocf, fcf=fcf, capex_to_revenue=capex_to_revenue,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fundamentals_calculator.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fundamentals/calculator.py tests/test_fundamentals_calculator.py
git commit -m "feat(fundamentals): derive OCF/FCF + 3yr OCF/NI avg + FCF positive years"
```

---

## Task 11: Calculator — dividend metrics (yield, payout, consecutive years)

**Files:**
- Modify: `src/fundamentals/calculator.py`
- Modify: `tests/test_fundamentals_calculator.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_fundamentals_calculator.py`:

```python
def test_dividend_yield_uses_market_cap_now():
    statements = [
        _stmt(2024, "당기순이익", 100_000_000_000),
        _stmt(2024, "자본총계", 1),
        _stmt(2024, "배당총액", 20_000_000_000),
    ]
    market = [_my(2025, mc=2_000_000_000_000, shares=10)]
    m = compute_metrics("X", "C1", statements, market, date(2026, 5, 15))
    assert m.dividend_yield == 1.0   # 20B / 2T * 100


def test_payout_ratio_uses_latest_year_ni():
    statements = [
        _stmt(2024, "당기순이익", 100_000_000_000),
        _stmt(2024, "자본총계", 1),
        _stmt(2024, "배당총액", 25_000_000_000),
    ]
    m = compute_metrics("X", "C1", statements, [], date(2026, 5, 15))
    assert m.payout_ratio == 25.0


def test_payout_ratio_null_when_ni_negative():
    statements = [
        _stmt(2024, "당기순이익", -100_000_000_000),
        _stmt(2024, "자본총계", 1),
        _stmt(2024, "배당총액", 5_000_000_000),
    ]
    m = compute_metrics("X", "C1", statements, [], date(2026, 5, 15))
    assert m.payout_ratio is None


def test_consecutive_dividend_years_counts_from_latest_backward():
    statements = [_stmt(2024, "자본총계", 1)]
    # 2020: paid, 2021: skipped, 2022/2023/2024: paid -> consecutive = 3
    for y, total in [(2020, 100), (2021, 0), (2022, 100), (2023, 100), (2024, 100)]:
        statements.append(_stmt(y, "배당총액", total))
    m = compute_metrics("X", "C1", statements, [], date(2026, 5, 15))
    assert m.consecutive_dividend_years == 3


def test_consecutive_dividend_years_zero_when_no_dividends():
    statements = [_stmt(2024, "자본총계", 1), _stmt(2024, "배당총액", 0)]
    m = compute_metrics("X", "C1", statements, [], date(2026, 5, 15))
    assert m.consecutive_dividend_years == 0


def test_consecutive_dividend_years_null_when_no_dividend_data():
    statements = [_stmt(2024, "자본총계", 1)]
    m = compute_metrics("X", "C1", statements, [], date(2026, 5, 15))
    assert m.consecutive_dividend_years is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fundamentals_calculator.py -v -k "dividend"`
Expected: FAIL.

- [ ] **Step 3: Add dividend-derived logic**

In `compute_metrics`, after the cashflow block:

```python
    dividend_yield = None
    payout_ratio = None
    consecutive_dividend_years = None

    if dividend_by_year:
        # absolute dividend total in 원 (DART negative-sign normalize)
        latest_div = abs(dividend_by_year.get(latest_year, 0.0))

        if market_cap_now is not None and market_cap_now > 0 and latest_year in dividend_by_year:
            dividend_yield = (latest_div / market_cap_now) * 100.0

        if latest_ni is not None and latest_ni > 0 and latest_year in dividend_by_year:
            payout_ratio = (latest_div / latest_ni) * 100.0

        consecutive_dividend_years = _count_consecutive_dividends(dividend_by_year)
```

Add helper:

```python
def _count_consecutive_dividends(div: dict[int, float]) -> int:
    """From latest year backward, count years where |dividend| > 0. Stops on first zero."""
    if not div:
        return 0
    count = 0
    for y in sorted(div.keys(), reverse=True):
        if abs(div[y]) > 0:
            count += 1
        else:
            break
    return count
```

Extend final return with the new fields:

```python
    return FundamentalsMetrics(
        ...
        dividend_yield=dividend_yield,
        payout_ratio=payout_ratio,
        consecutive_dividend_years=consecutive_dividend_years,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fundamentals_calculator.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fundamentals/calculator.py tests/test_fundamentals_calculator.py
git commit -m "feat(fundamentals): derive dividend yield, payout ratio, consecutive years"
```

---

## Task 12: Pipeline integration — wire `market_data` into `fundamentals/pipeline.py`

**Files:**
- Modify: `src/fundamentals/pipeline.py:17-123`
- Modify: `tests/test_fundamentals_pipeline.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_fundamentals_pipeline.py` (or create if needed). This test stubs the heavy parts.

```python
from datetime import date
from unittest.mock import MagicMock

from src.market_data.models import MarketYearly
from src.fundamentals.pipeline import Pipeline


def test_compute_all_new_signature_consumes_market_yearly_dict():
    cache = MagicMock()
    cache.load_corp_info.return_value = []   # empty universe -> skip body
    cache.load_financials.return_value = []
    p = Pipeline(cache=cache, fetcher=MagicMock(), ttl_days=30, fundamentals_db=None)
    # Should not raise; signature accepts a dict and no extra positional args.
    metrics, scores = p.compute_all(market_yearly={})
    assert metrics == []
    assert scores == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fundamentals_pipeline.py::test_compute_all_new_signature_consumes_market_yearly_dict -v`
Expected: FAIL (current signature requires `market_caps` positional).

- [ ] **Step 3: Refactor `Pipeline`**

In `src/fundamentals/pipeline.py`:

1. Change imports to add MarketYearly:

```python
from src.market_data.models import MarketYearly
```

2. Replace `compute_all` signature:

```python
def compute_all(
    self,
    market_yearly: dict[str, list[MarketYearly]],
    markets: list[str] | None = None,
) -> tuple[list[FundamentalsMetrics], list[ScoreCard]]:
    """Compute metrics and scores for all cached corps.

    Args:
        market_yearly: corp_code -> list of MarketYearly snapshots (includes
                       in-progress year for current PE/PB calc).
    """
    markets = markets or ["KOSPI", "KOSDAQ"]
    corps = self._cache.load_corp_info(markets=markets)
    all_statements = self._cache.load_financials()
    grouped: dict[str, list[FinancialStatement]] = {}
    for s in all_statements:
        grouped.setdefault(s.corp_code, []).append(s)

    as_of = date.today()
    metrics_list: list[FundamentalsMetrics] = []
    for corp in corps:
        statements = grouped.get(corp.corp_code, [])
        my = market_yearly.get(corp.corp_code, [])
        m = compute_metrics(
            ticker=corp.ticker, corp_code=corp.corp_code,
            statements=statements, market_yearly=my, as_of=as_of,
        )
        metrics_list.append(m)

    ticker_to_market = {c.ticker: c.market for c in corps}
    medians = compute_market_medians(metrics_list, ticker_to_market)
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

3. (No changes needed to `refresh_data` here — DART already auto-includes the 4 new accounts because we widened `ACCOUNT_NORMALIZE` in Task 5.)

- [ ] **Step 4: Run pipeline tests**

Run: `pytest tests/test_fundamentals_pipeline.py -v`
Expected: all PASS, including pre-existing tests. **Pre-existing tests may break if they call `compute_all` with old args — update them to pass `market_yearly={}`** as needed.

- [ ] **Step 5: Commit**

```bash
git add src/fundamentals/pipeline.py tests/test_fundamentals_pipeline.py
git commit -m "refactor(fundamentals): compute_all takes market_yearly dict (drops eps/bps maps)"
```

---

## Task 13: CLI integration — wire `market_data` into `fundamentals/cli.py run`

**Files:**
- Modify: `src/fundamentals/cli.py` (the `run` command + helpers)
- Modify: `tests/test_fundamentals_cli.py`

- [ ] **Step 1: Read the full existing `run` command first**

Run: `sed -n '80,200p' src/fundamentals/cli.py`

(This step exists so the engineer reads the existing command body before editing — its exact length depends on current code, but the `@app.command()` decorator follows the helpers we already saw.)

- [ ] **Step 2: Write a failing test for `--skip-market` flag wiring**

Append to `tests/test_fundamentals_cli.py`:

```python
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from src.fundamentals.cli import app


def test_run_with_skip_market_does_not_call_market_pipeline():
    runner = CliRunner()
    with patch("src.fundamentals.cli._make_pipeline") as make_pipe, \
         patch("src.fundamentals.cli._build_market_pipeline") as build_market:
        fake = MagicMock()
        make_pipe.return_value = fake
        result = runner.invoke(app, ["run", "--skip-market", "--skip-dart"])
        assert result.exit_code == 0
        build_market.assert_not_called()
```

(`--skip-dart` is paired so the DART path also short-circuits — this test only validates the wiring.)

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_fundamentals_cli.py::test_run_with_skip_market_does_not_call_market_pipeline -v`
Expected: FAIL (`--skip-market` flag doesn't exist).

- [ ] **Step 4: Add a `_build_market_pipeline` helper and update `run`**

In `src/fundamentals/cli.py`, near `_make_pipeline`:

```python
def _build_market_pipeline():
    from src.market_data.db import MarketDB
    from src.market_data.pipeline import MarketDataPipeline
    db = MarketDB()
    return MarketDataPipeline(db=db), db
```

Then in the existing `run` command, add typer options and wiring. The exact body depends on the current file — replace the metrics-compute call site with:

```python
@app.command()
def run(
    years: list[int] | None = typer.Option(None, "--years"),
    force: bool = typer.Option(False, "--force"),
    skip_dart: bool = typer.Option(False, "--skip-dart"),
    skip_market: bool = typer.Option(False, "--skip-market"),
):
    settings = load_scanner_config()
    pipeline = _make_pipeline(settings, ttl_days=30)
    market_caps, market_map = _load_market_data(settings)   # existing — current snapshot
    years_to_use = years or list(range(date.today().year - 6, date.today().year + 1))

    # DART refresh
    if not skip_dart:
        asyncio.run(pipeline.refresh_data(
            force=force, years=years_to_use,
            markets=["KOSPI", "KOSDAQ"], market_map=market_map,
        ))

    # pykrx market_data refresh (historical years)
    market_yearly_dict: dict[str, list] = {}
    if not skip_market:
        market_pipeline, market_db = _build_market_pipeline()
        from src.dart.cache import DartCache
        cache = DartCache()
        corps = cache.load_corp_info(markets=["KOSPI", "KOSDAQ"])
        corp_code_map = {c.ticker: c.corp_code for c in corps}
        report = market_pipeline.refresh(
            years=years_to_use,
            tickers=list(corp_code_map.keys()),
            corp_code_map=corp_code_map,
        )
        console.print(f"[bold]market_data:[/bold] {report.successful_rows} rows in {report.duration_seconds:.1f}s")
        market_yearly_dict = market_db.load_all()

    metrics, scores = pipeline.compute_all(market_yearly=market_yearly_dict)
    console.print(f"[bold]fundamentals:[/bold] {len(metrics)} metrics, {len(scores)} scores")
```

(Where the existing code calls `compute_all(market_caps, eps_map, bps_map)`, **delete that call** and use the new `compute_all(market_yearly=market_yearly_dict)`.)

- [ ] **Step 5: Run the new test plus regression on cli tests**

Run: `pytest tests/test_fundamentals_cli.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/fundamentals/cli.py tests/test_fundamentals_cli.py
git commit -m "feat(fundamentals): cli wires market_data + --skip-market/--skip-dart flags"
```

---

## Task 14: Update `.claude/commands/analyze-stock.md` — freshness check + new metrics

**Files:**
- Modify: `.claude/commands/analyze-stock.md`

- [ ] **Step 1: Add freshness check block to "1. 데이터 수집"**

After the SQL query block in section 1, append:

```markdown
### 1-0. 데이터 신선도 확인 (자동 안내)

분석 시작 전, `fundamentals_metrics.as_of_date` 가 30일 이상 오래됐는지 확인합니다.

```bash
sqlite3 data/scanner.db "SELECT julianday('now') - julianday(MAX(as_of_date)) AS days_old FROM fundamentals_metrics WHERE ticker = '$ARGUMENTS';"
```

`days_old > 30` 또는 결과 없음이면 보고서 상단에 다음 경고를 출력:

> ⚠ **데이터 신선도 경고:** 이 종목의 펀더멘털 데이터가 `N`일 됐습니다 (또는 미존재). 정확도를 위해 다음 명령을 실행해 주세요:
> ```bash
> python -m src.fundamentals.cli run
> ```
```

- [ ] **Step 2: Add per-ticker fresh-fetch step**

Before "3. 분석 작성" add:

```markdown
### 2-4. 해당 종목 시총 1건 fresh 갱신 (자동)

DB 의 진행 연도 시총이 며칠 됐을 수 있으므로, 분석 시작 직전에 pykrx 로 1건만 fresh:

```bash
python -c "
from src.market_data.fetcher import fetch_yearly_market_data
from src.market_data.db import MarketDB
from datetime import datetime
rows = fetch_yearly_market_data(
    tickers=['$ARGUMENTS'],
    years=[datetime.now().year],
    corp_code_map={'$ARGUMENTS': '$(sqlite3 data/scanner.db \"SELECT corp_code FROM dart_corp_info WHERE ticker='\\\''$ARGUMENTS'\\\''\")'},
)
MarketDB().save_yearly(rows)
print(f'updated: {len(rows)} row')
"
```

DART 데이터는 갱신하지 않습니다 (연 1회 사업보고서 단위라 변동 없음).
```

- [ ] **Step 3: Add new metrics to the report template (§ 4)**

In the "## 가치 / 성장 / 품질 포지셔닝" table, add rows after `영업이익률`:

```markdown
| EPS | ... | ... | ... | ... |
| 배당수익률 | ...% | ...% | ...% | 섹터·시장 대비 위치 |
| 배당성향 | ...% | — | — | 순이익의 N% 환원 |
| OCF/NI 비율 | ... | — | — | 1.0 근처면 이익의 질 양호 |
| FCF 양수 연수 | N/5 | — | — | 5/5 면 안정적 캐시 창출 |
```

And add a new subsection after "약점/리스크":

```markdown
## 이익의 질 코멘트

- `ocf_to_ni_ratio` 와 `fcf_positive_years` 가 같이 약하면 회계이익과 현금이익 괴리 가능성. 둘 다 강하면 발생주의 회계가 실제 현금흐름과 잘 매칭됨.
- `capex_to_revenue` 가 섹터 평균 대비 높으면 자본집약 단계(투자기), 낮으면 회수기.
- `consecutive_dividend_years` ≥ 5 면 배당 정책 안정성 시사.
```

- [ ] **Step 4: Verify command file parses cleanly**

Run: `head -50 .claude/commands/analyze-stock.md`
Expected: front-matter intact, no markdown errors.

- [ ] **Step 5: Commit**

```bash
git add .claude/commands/analyze-stock.md
git commit -m "feat(analyze-stock): freshness check + per-ticker fresh refresh + new metrics in template"
```

---

## Task 15: End-to-end smoke verification (manual)

**Files:**
- Run-only (no edits)

This task does **not** modify code. It is the spec's § 11-5 "수동 검증 시나리오".

- [ ] **Step 1: Backfill**

```bash
python -m src.fundamentals.cli run --years 2019 2020 2021 2022 2023 2024 2025
```

Expected output (within tolerance):

```
market_data: ~2,600 rows in ~30s
fundamentals: ~2,600 metrics, ~2,600 scores
```

- [ ] **Step 2: Verify new columns are populated for SK하이닉스 + 대덕전자**

```bash
sqlite3 data/scanner.db <<SQL
.headers on
SELECT ticker, eps, bps, psr, ocf, fcf, dividend_yield, payout_ratio,
       consecutive_dividend_years, ocf_to_ni_ratio, fcf_positive_years, cashflow_score
FROM fundamentals_metrics m
JOIN fundamentals_scores s USING (ticker, as_of_date)
WHERE ticker IN ('000660', '353200')
ORDER BY ticker, as_of_date DESC LIMIT 2;
SQL
```

Expected: non-NULL values for `eps`, `bps`, `ocf`, `fcf`, `cashflow_score`. For SK하이닉스 specifically: `fcf_positive_years` ≥ 4 (only 2023 was negative).

- [ ] **Step 3: Verify `cashflow_score` is no longer universally NULL**

```bash
sqlite3 data/scanner.db "SELECT COUNT(*) AS total, COUNT(cashflow_score) AS with_cf FROM fundamentals_scores WHERE as_of_date = (SELECT MAX(as_of_date) FROM fundamentals_scores);"
```

Expected: `with_cf` ≫ 0 (previously was 0).

- [ ] **Step 4: Re-run analyze-stock on both reference tickers**

```bash
# Invoke /analyze-stock 000660  (via Claude Code)
# Invoke /analyze-stock 353200
```

Verify the generated reports contain:
- Non-NULL EPS, OCF/NI, dividend_yield rows in the positioning table.
- "이익의 질 코멘트" subsection populated.

- [ ] **Step 5: Check total_score distribution shifted reasonably**

```bash
sqlite3 data/scanner.db "SELECT grade, COUNT(*) FROM fundamentals_scores WHERE as_of_date=(SELECT MAX(as_of_date) FROM fundamentals_scores) GROUP BY grade ORDER BY grade DESC;"
```

If distribution looks lopsided (e.g., 80%+ at ★★★★★), file a follow-up issue for star cutoff recalibration (per spec § 12 open question).

- [ ] **Step 6: Commit verification notes (optional)**

If reports differ materially from prior runs, save updated reports:

```bash
git add docs/analysis/   # if user wants to retain
git commit -m "chore: verification reports after enrichment backfill"
```

(Note: `docs/` is gitignored — use `git add -f` if intentionally checking in.)

---

## Out-of-scope reminders (deferred — see spec § 2 Non-Goals)

These are NOT in this plan. Do not implement here:
- 일별 시세 / 베타 / 변동성 / 모멘텀 (D).
- `interest_coverage` — needs DART 이자비용 account; separate spec.
- 우선주 별도 처리.
- 신규 카테고리 (Cash Cow, Dividend Payer).
- ★ 컷오프 재조정 — happens after Task 15 if needed, separate PR.

### Deviations from spec (intentional, documented here)

- **`derivation_audit` dict return** (spec § 6) — NOT implemented as a tuple return. Instead, `compute_metrics` uses `logger.debug(f"... reason=...")` to emit per-null reasons. Rationale: keeps the function signature simple; per-ticker debug can be enabled by adjusting loguru level. If a structured audit is needed later (e.g., for a UI), revisit with a separate refactor that returns `tuple[FundamentalsMetrics, AuditDict]`.

---

## Final verification

Before merging:

- [ ] `pytest` — all tests pass
- [ ] `python -m src.fundamentals.cli run --skip-market --skip-dart` — runs without error
- [ ] `python -m src.fundamentals.cli run --years 2025` — single-year refresh works end-to-end
- [ ] Spec § 11-5 verification scenarios all pass
