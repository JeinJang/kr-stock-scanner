# 로컬 일봉 이력 저장소 구현 플랜

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 돌파 신선도의 입력을 KRX 종목별 조회(종목당 6콜)에서 KRX Open API 일자별 벌크로 만든 로컬 일봉 저장소(하루 2콜)로 교체한다.

**Architecture:** 새 패키지 `src/price_history/`가 `data/prices.db`에 원주가 일봉과 수정 이벤트를 적재하고, 티커별 수정 일봉을 제공한다. 보정 계산은 네트워크·DB에 의존하지 않는 순수 함수(`adjust.py`)로 분리한다. `src/recency_source.py`는 KRX 종목별 조회를 버리고 이 저장소를 읽는다.

**Tech Stack:** Python, requests(+ThreadPoolExecutor), SQLite(sqlite3 직접 — 이 패키지는 SQLAlchemy를 쓰지 않는다), typer(cli), pytest.

**Spec:** [docs/superpowers/specs/2026-08-20-price-history-backfill-design.md](../specs/2026-08-20-price-history-backfill-design.md)

## Global Constraints

- 파이썬 실행: `.venv/bin/python`. 기준 스위트는 **248 passed**.
- KRX Open API: `https://data-dbg.krx.co.kr/svc/apis/sto/{stk_bydd_trd|ksq_bydd_trd}`, 파라미터 `basDd=YYYYMMDD`, 헤더 `AUTH_KEY`, 응답은 `OutBlock_1` 배열.
- 조회 깊이 `years = 11`. 동시 요청 `workers = 8` (실측: 콜당 0.24초, 716건 실패 0). **workers를 올리지 않는다** — 이 프로젝트는 과다요청으로 KRX에 차단된 이력이 있다.
- 수정계수 임계값 `THRESHOLD = 0.02`. 기준가 역산 = `당일 close - 당일 chg`.
- 모든 신선도 지표는 **달력 일수** 기준. 창 `window_days = 365`.
- 가격 이력은 `data/prices.db`에 둔다. `data/scanner.db`에 넣지 않는다.
- 거래정지일은 `high = 0`으로 온다. 0은 유효 고가가 아니다.
- 코드 주석은 이 저장소의 기존 개조식/문어체를 따른다.
- 네트워크 없는 단위테스트만 작성한다. 실제 KRX 호출은 Task 4·8의 수동 확인뿐이며, 자격증명이 없으면 건너뛰고 그 사실을 커밋 메시지에 남긴다.

---

### Task 1: 순수 보정 모듈 — `adjust.py`

**Files:**
- Create: `src/price_history/__init__.py` (빈 파일)
- Create: `src/price_history/adjust.py`
- Test: `tests/test_price_adjust.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class PxRow`: `d: date`, `high: float`, `close: float`, `chg: float`
  - `@dataclass(frozen=True) class AdjustEvent`: `d: date`, `factor: float`
  - `THRESHOLD: float = 0.02`
  - `def detect_adjustments(rows: list[PxRow], threshold: float = THRESHOLD) -> list[AdjustEvent]`
  - `def adjusted_highs(rows: list[PxRow], events: list[AdjustEvent]) -> list[tuple[date, float]]`

- [ ] **Step 1: 실패 테스트 작성** (`tests/test_price_adjust.py`)

```python
from datetime import date

from src.price_history.adjust import (
    PxRow, AdjustEvent, detect_adjustments, adjusted_highs,
)


def _row(day, high, close, chg):
    return PxRow(d=date(2026, 1, day), high=high, close=close, chg=chg)


def test_no_event_when_base_matches_prev_close():
    rows = [_row(5, 110, 100, 0), _row(6, 115, 105, 5), _row(7, 108, 102, -3)]
    assert detect_adjustments(rows) == []


def test_detects_split_factor_ten():
    # 액면분할: 정지 전 종가 2,650,000 -> 재개일 기준가 53,000
    rows = [
        _row(5, 0, 2_650_000, 0),
        _row(6, 53_900, 51_900, -1_100),   # 기준가 53,000, 계수 50
    ]
    evs = detect_adjustments(rows)
    assert len(evs) == 1
    assert evs[0].d == date(2026, 1, 6)
    assert evs[0].factor == 50.0


def test_detects_reverse_split_factor_below_one():
    # 액면병합 5:1 — 정지 전 396 -> 재개일 기준가 1,980
    rows = [_row(5, 0, 396, 0), _row(6, 2_065, 1_720, -260)]
    evs = detect_adjustments(rows)
    assert len(evs) == 1
    assert round(evs[0].factor, 4) == 0.2


def test_ignores_moves_below_threshold():
    # 기준가와 전일종가가 1% 어긋나면 이벤트가 아니다
    rows = [_row(5, 110, 1_000, 0), _row(6, 1_020, 1_010, 20)]  # 기준가 990
    assert detect_adjustments(rows) == []


def test_skips_rows_with_nonpositive_prices():
    rows = [_row(5, 0, 0, 0), _row(6, 100, 100, 0)]
    assert detect_adjustments(rows) == []


def test_adjusted_highs_scales_only_dates_before_event():
    rows = [_row(5, 400, 396, 0), _row(6, 2_065, 1_720, -260), _row(7, 1_870, 1_691, -29)]
    evs = detect_adjustments(rows)
    out = dict(adjusted_highs(rows, evs))
    assert out[date(2026, 1, 5)] == 2_000.0   # 400 x 5
    assert out[date(2026, 1, 6)] == 2_065.0   # 이벤트 당일은 그대로
    assert out[date(2026, 1, 7)] == 1_870.0


def test_adjusted_highs_accumulates_multiple_events():
    # 1/6에 계수 2, 1/8에 계수 5 -> 1/5 이전 가격은 10배
    rows = [
        _row(5, 100, 100, 0),
        _row(6, 60, 55, 5),      # 기준가 50, 계수 2
        _row(8, 300, 260, 10),   # 기준가 250, 계수 55/250 = 0.22 (별도 계산)
    ]
    evs = detect_adjustments(rows)
    out = dict(adjusted_highs(rows, evs))
    # 마지막 봉은 언제나 무보정
    assert out[date(2026, 1, 8)] == 300.0
    # 이벤트가 2건이면 가장 오래된 봉은 두 계수가 모두 적용된다
    assert len(evs) == 2
    assert out[date(2026, 1, 5)] == 100.0 / evs[0].factor / evs[1].factor


def test_adjusted_highs_without_events_is_identity():
    rows = [_row(5, 110, 100, 0), _row(6, 115, 105, 5)]
    out = dict(adjusted_highs(rows, []))
    assert out == {date(2026, 1, 5): 110.0, date(2026, 1, 6): 115.0}
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_price_adjust.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.price_history'`

- [ ] **Step 3: 최소 구현**

`src/price_history/__init__.py` 는 빈 파일로 만든다.

`src/price_history/adjust.py`:

```python
"""수정주가 보정 — 순수 함수.

KRX Open API 일별매매정보는 원주가를 준다. 액면분할·병합·무상증자 등으로
KRX가 기준가를 재설정한 날을 찾아 그 이전 가격을 현재 기준으로 환산한다.

기준가 = 당일 종가 - 당일 전일대비. 이 값이 전일 실제 종가와 어긋나면
그날 기준가가 재설정된 것이다. 현금배당락은 기준가를 조정하지 않으므로
검출되지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

THRESHOLD = 0.02


@dataclass(frozen=True)
class PxRow:
    """보정 계산에 필요한 하루치 원주가."""

    d: date
    high: float
    close: float
    chg: float


@dataclass(frozen=True)
class AdjustEvent:
    """기준가가 재설정된 날과 그 계수(전일 실제 종가 / 당일 기준가)."""

    d: date
    factor: float


def detect_adjustments(rows: list[PxRow], threshold: float = THRESHOLD) -> list[AdjustEvent]:
    """rows(날짜 오름차순)에서 기준가 재설정 이벤트를 찾는다."""
    events: list[AdjustEvent] = []
    for i in range(1, len(rows)):
        base = rows[i].close - rows[i].chg
        prev = rows[i - 1].close
        if base <= 0 or prev <= 0:
            continue
        factor = prev / base
        if abs(factor - 1.0) > threshold:
            events.append(AdjustEvent(d=rows[i].d, factor=factor))
    return events


def adjusted_highs(
    rows: list[PxRow], events: list[AdjustEvent],
) -> list[tuple[date, float]]:
    """이벤트를 소급 적용한 (날짜, 수정 고가) 목록. 입력과 같은 순서."""
    factor_at = {e.d: e.factor for e in events}
    out: list[tuple[date, float]] = []
    cum = 1.0
    for row in reversed(rows):
        out.append((row.d, row.high * cum))
        # 이벤트 당일은 이미 새 기준이므로, 그 이전 날짜부터 계수를 먹인다.
        if row.d in factor_at:
            cum /= factor_at[row.d]
    out.reverse()
    return out
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_price_adjust.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/price_history/__init__.py src/price_history/adjust.py tests/test_price_adjust.py
git commit -m "feat(prices): 수정주가 보정 순수 함수 추가"
```

---

### Task 2: 저장소 — `db.py`

**Files:**
- Create: `src/price_history/db.py`
- Test: `tests/test_price_db.py`

**Interfaces:**
- Consumes: Task 1의 `PxRow`, `AdjustEvent`
- Produces:
  - `class PriceDB`: `__init__(path: str = "data/prices.db")`
  - `save_day(d: str, market: str, records: list[tuple]) -> int` — `records`는 `(ticker, high, close, chg)`
  - `loaded_dates(market: str) -> set[str]`
  - `last_loaded_date() -> str | None`
  - `set_meta(key: str, value: str) -> None` / `get_meta(key: str) -> str | None`
  - `load_rows(ticker: str, since: str) -> list[PxRow]`
  - `save_events(ticker: str, events: list[AdjustEvent]) -> None`
  - `load_events(ticker: str) -> list[AdjustEvent]`
  - `tickers() -> list[str]`

- [ ] **Step 1: 실패 테스트 작성** (`tests/test_price_db.py`)

```python
from datetime import date

from src.price_history.adjust import AdjustEvent
from src.price_history.db import PriceDB


def _db(tmp_path):
    return PriceDB(path=str(tmp_path / "prices.db"))


def test_save_and_load_rows(tmp_path):
    db = _db(tmp_path)
    db.save_day("20260105", "KOSPI", [("005930", 110, 100, 0)])
    db.save_day("20260106", "KOSPI", [("005930", 115, 105, 5)])

    rows = db.load_rows("005930", since="20260101")
    assert [r.d for r in rows] == [date(2026, 1, 5), date(2026, 1, 6)]
    assert rows[1].high == 115.0 and rows[1].chg == 5.0


def test_load_rows_respects_since(tmp_path):
    db = _db(tmp_path)
    db.save_day("20260105", "KOSPI", [("005930", 110, 100, 0)])
    db.save_day("20260106", "KOSPI", [("005930", 115, 105, 5)])
    assert [r.d for r in db.load_rows("005930", since="20260106")] == [date(2026, 1, 6)]


def test_save_day_is_idempotent(tmp_path):
    db = _db(tmp_path)
    db.save_day("20260105", "KOSPI", [("005930", 110, 100, 0)])
    db.save_day("20260105", "KOSPI", [("005930", 999, 999, 0)])
    rows = db.load_rows("005930", since="20260101")
    assert len(rows) == 1 and rows[0].high == 999.0


def test_loaded_dates_and_last_loaded(tmp_path):
    db = _db(tmp_path)
    assert db.loaded_dates("KOSPI") == set()
    assert db.last_loaded_date() is None
    db.save_day("20260105", "KOSPI", [("005930", 110, 100, 0)])
    db.save_day("20260106", "KOSDAQ", [("035720", 50, 48, 1)])
    assert db.loaded_dates("KOSPI") == {"20260105"}
    assert db.last_loaded_date() == "20260106"


def test_meta_roundtrip(tmp_path):
    db = _db(tmp_path)
    assert db.get_meta("backfill_years") is None
    db.set_meta("backfill_years", "11")
    db.set_meta("backfill_years", "12")
    assert db.get_meta("backfill_years") == "12"


def test_events_roundtrip_replaces_previous(tmp_path):
    db = _db(tmp_path)
    db.save_events("005930", [AdjustEvent(d=date(2026, 1, 6), factor=50.0)])
    db.save_events("005930", [AdjustEvent(d=date(2026, 1, 7), factor=2.0)])
    evs = db.load_events("005930")
    assert len(evs) == 1
    assert evs[0].d == date(2026, 1, 7) and evs[0].factor == 2.0


def test_tickers_lists_distinct(tmp_path):
    db = _db(tmp_path)
    db.save_day("20260105", "KOSPI", [("005930", 1, 1, 0), ("000660", 1, 1, 0)])
    db.save_day("20260106", "KOSPI", [("005930", 1, 1, 0)])
    assert sorted(db.tickers()) == ["000660", "005930"]
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_price_db.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.price_history.db'`

- [ ] **Step 3: 최소 구현** (`src/price_history/db.py`)

```python
"""data/prices.db — 원주가 일봉과 수정 이벤트 저장소.

이 패키지는 sqlite3를 직접 쓴다. 700만 행 규모의 단순 적재·범위 조회라
SQLAlchemy 계층이 이득 없이 비용만 된다(저장소의 다른 DB는 SQLAlchemy 사용).
"""
from __future__ import annotations

import os
import sqlite3
from datetime import date

from src.price_history.adjust import AdjustEvent, PxRow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_px (
    d      TEXT NOT NULL,
    ticker TEXT NOT NULL,
    market TEXT NOT NULL,
    high   INTEGER NOT NULL,
    close  INTEGER NOT NULL,
    chg    INTEGER NOT NULL,
    PRIMARY KEY (d, ticker)
);
CREATE INDEX IF NOT EXISTS idx_px_ticker_d ON daily_px(ticker, d);
CREATE TABLE IF NOT EXISTS px_adjust (
    ticker TEXT NOT NULL,
    d      TEXT NOT NULL,
    factor REAL NOT NULL,
    PRIMARY KEY (ticker, d)
);
CREATE TABLE IF NOT EXISTS px_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _to_date(s: str) -> date:
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


class PriceDB:
    """원주가 일봉 저장소. 스키마는 생성 시 보장한다."""

    def __init__(self, path: str = "data/prices.db"):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.path = path
        self.con = sqlite3.connect(path)
        self.con.executescript(_SCHEMA)
        self.con.commit()

    # -- 적재 ---------------------------------------------------------------

    def save_day(self, d: str, market: str, records: list[tuple]) -> int:
        """records = [(ticker, high, close, chg)]. 같은 (d,ticker)는 덮어쓴다."""
        rows = [(d, tk, market, int(h), int(c), int(ch)) for tk, h, c, ch in records]
        self.con.executemany(
            "INSERT OR REPLACE INTO daily_px (d,ticker,market,high,close,chg) "
            "VALUES (?,?,?,?,?,?)",
            rows,
        )
        self.con.commit()
        return len(rows)

    def loaded_dates(self, market: str) -> set[str]:
        cur = self.con.execute(
            "SELECT DISTINCT d FROM daily_px WHERE market = ?", (market,)
        )
        return {r[0] for r in cur}

    def last_loaded_date(self) -> str | None:
        r = self.con.execute("SELECT MAX(d) FROM daily_px").fetchone()
        return r[0] if r and r[0] else None

    # -- 메타 ---------------------------------------------------------------

    def set_meta(self, key: str, value: str) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO px_meta (key,value) VALUES (?,?)", (key, value)
        )
        self.con.commit()

    def get_meta(self, key: str) -> str | None:
        r = self.con.execute("SELECT value FROM px_meta WHERE key = ?", (key,)).fetchone()
        return r[0] if r else None

    # -- 조회 ---------------------------------------------------------------

    def load_rows(self, ticker: str, since: str) -> list[PxRow]:
        cur = self.con.execute(
            "SELECT d, high, close, chg FROM daily_px "
            "WHERE ticker = ? AND d >= ? ORDER BY d",
            (ticker, since),
        )
        return [
            PxRow(d=_to_date(d), high=float(h), close=float(c), chg=float(ch))
            for d, h, c, ch in cur
        ]

    def tickers(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT DISTINCT ticker FROM daily_px")]

    # -- 수정 이벤트 --------------------------------------------------------

    def save_events(self, ticker: str, events: list[AdjustEvent]) -> None:
        """해당 티커의 이벤트를 통째로 교체한다(재계산 결과로 덮어쓰기)."""
        self.con.execute("DELETE FROM px_adjust WHERE ticker = ?", (ticker,))
        self.con.executemany(
            "INSERT INTO px_adjust (ticker,d,factor) VALUES (?,?,?)",
            [(ticker, e.d.strftime("%Y%m%d"), e.factor) for e in events],
        )
        self.con.commit()

    def load_events(self, ticker: str) -> list[AdjustEvent]:
        cur = self.con.execute(
            "SELECT d, factor FROM px_adjust WHERE ticker = ? ORDER BY d", (ticker,)
        )
        return [AdjustEvent(d=_to_date(d), factor=float(f)) for d, f in cur]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_price_db.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/price_history/db.py tests/test_price_db.py
git commit -m "feat(prices): prices.db 저장소 계층 추가"
```

---

### Task 3: 취득 — `fetcher.py`

**Files:**
- Create: `src/price_history/fetcher.py`
- Test: `tests/test_price_fetcher.py`

**Interfaces:**
- Produces:
  - `MARKET_ENDPOINTS: dict[str, str]` = `{"KOSPI": "stk_bydd_trd", "KOSDAQ": "ksq_bydd_trd"}`
  - `BASE_URL: str` = `"https://data-dbg.krx.co.kr/svc/apis/sto"`
  - `class KrxApiError(RuntimeError)`
  - `def parse_rows(payload: dict) -> list[tuple]` — `OutBlock_1` → `[(ticker, high, close, chg)]`
  - `def fetch_day(api_key: str, market: str, d: str, _get=None) -> list[tuple]`
  - `def fetch_many(api_key, jobs: list[tuple[str, str]], workers: int = 8, _get=None) -> Iterator[tuple[str, str, list[tuple]]]` — `jobs`는 `[(market, d)]`, 산출은 `(market, d, rows)`

- [ ] **Step 1: 실패 테스트 작성** (`tests/test_price_fetcher.py`)

```python
import pytest

from src.price_history.fetcher import (
    KrxApiError, fetch_day, fetch_many, parse_rows,
)


class FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload


def _payload(*items):
    return {"OutBlock_1": [
        {"ISU_CD": tk, "TDD_HGPRC": h, "TDD_CLSPRC": c, "CMPPREVDD_PRC": ch}
        for tk, h, c, ch in items
    ]}


def test_parse_rows_strips_commas_and_casts():
    rows = parse_rows(_payload(("005930", "53,900", "51,900", "-1,100")))
    assert rows == [("005930", 53900, 51900, -1100)]


def test_parse_rows_skips_rows_without_ticker():
    rows = parse_rows({"OutBlock_1": [{"TDD_HGPRC": "1", "TDD_CLSPRC": "1"}]})
    assert rows == []


def test_parse_rows_on_empty_payload():
    assert parse_rows({}) == []


def test_fetch_day_passes_auth_key_and_date():
    seen = {}

    def fake_get(url, params, headers, timeout):
        seen.update(url=url, params=params, headers=headers)
        return FakeResp(200, _payload(("005930", "1", "1", "0")))

    rows = fetch_day("KEY123", "KOSPI", "20260819", _get=fake_get)
    assert rows == [("005930", 1, 1, 0)]
    assert seen["url"].endswith("/stk_bydd_trd")
    assert seen["params"] == {"basDd": "20260819"}
    assert seen["headers"]["AUTH_KEY"] == "KEY123"


def test_fetch_day_raises_on_401():
    def fake_get(url, params, headers, timeout):
        return FakeResp(401)

    with pytest.raises(KrxApiError):
        fetch_day("BAD", "KOSPI", "20260819", _get=fake_get)


def test_fetch_day_returns_empty_on_holiday():
    def fake_get(url, params, headers, timeout):
        return FakeResp(200, {"OutBlock_1": []})

    assert fetch_day("KEY", "KOSPI", "20260101", _get=fake_get) == []


def test_fetch_many_covers_every_job():
    def fake_get(url, params, headers, timeout):
        return FakeResp(200, _payload(("005930", "1", "1", "0")))

    jobs = [("KOSPI", "20260818"), ("KOSDAQ", "20260818"), ("KOSPI", "20260819")]
    out = list(fetch_many("KEY", jobs, workers=2, _get=fake_get))
    assert len(out) == 3
    assert {(m, d) for m, d, _ in out} == set(jobs)


def test_fetch_many_propagates_auth_error():
    def fake_get(url, params, headers, timeout):
        return FakeResp(401)

    with pytest.raises(KrxApiError):
        list(fetch_many("BAD", [("KOSPI", "20260819")], workers=2, _get=fake_get))
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_price_fetcher.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.price_history.fetcher'`

- [ ] **Step 3: 최소 구현** (`src/price_history/fetcher.py`)

```python
"""KRX Open API 일별매매정보 취득.

일자별 전 종목 벌크만 쓴다. 종목별 기간 조회는 이 API에 없고, 로그인
클라이언트 쪽은 2년 상한이 있어 쓰지 않는다.
"""
from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import requests

BASE_URL = "https://data-dbg.krx.co.kr/svc/apis/sto"
MARKET_ENDPOINTS = {"KOSPI": "stk_bydd_trd", "KOSDAQ": "ksq_bydd_trd"}


class KrxApiError(RuntimeError):
    """KRX Open API 인증 실패 또는 비정상 응답."""


def _default_get(url, params, headers, timeout):
    return requests.get(url, params=params, headers=headers, timeout=timeout)


def _num(v) -> int:
    s = str(v or "").replace(",", "")
    try:
        return int(float(s))
    except ValueError:
        return 0


def parse_rows(payload: dict) -> list[tuple]:
    """OutBlock_1 -> [(ticker, high, close, chg)]. 티커 없는 행은 버린다."""
    out: list[tuple] = []
    for item in (payload or {}).get("OutBlock_1", []):
        ticker = item.get("ISU_CD", "")
        if not ticker:
            continue
        out.append((
            ticker,
            _num(item.get("TDD_HGPRC")),
            _num(item.get("TDD_CLSPRC")),
            _num(item.get("CMPPREVDD_PRC")),
        ))
    return out


def fetch_day(api_key: str, market: str, d: str, _get=None) -> list[tuple]:
    """하루치 전 종목. 휴장일은 빈 리스트."""
    get = _get or _default_get
    resp = get(
        f"{BASE_URL}/{MARKET_ENDPOINTS[market]}",
        params={"basDd": d},
        headers={"AUTH_KEY": api_key},
        timeout=60,
    )
    if resp.status_code == 401:
        raise KrxApiError(
            "KRX Open API 인증 실패(401). KRX_API_KEY와 "
            "openapi.krx.co.kr의 서비스 이용 신청 상태를 확인하세요."
        )
    if resp.status_code != 200:
        raise KrxApiError(f"KRX Open API 응답 이상: status={resp.status_code} ({market} {d})")
    return parse_rows(resp.json())


def fetch_many(
    api_key: str,
    jobs: list[tuple[str, str]],
    workers: int = 8,
    _get=None,
) -> Iterator[tuple[str, str, list[tuple]]]:
    """jobs = [(market, d)]. 완료 순서와 무관하게 (market, d, rows)를 흘려보낸다.

    workers를 올리지 않는다 — 과다요청은 이 프로젝트가 차단당한 원인이다.
    """
    def one(job):
        market, d = job
        return market, d, fetch_day(api_key, market, d, _get=_get)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        yield from ex.map(one, jobs)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_price_fetcher.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/price_history/fetcher.py tests/test_price_fetcher.py
git commit -m "feat(prices): KRX Open API 일자별 벌크 취득 모듈 추가"
```

---

### Task 4: 적재 오케스트레이션 — `backfill.py`

**Files:**
- Create: `src/price_history/backfill.py`
- Test: `tests/test_price_backfill.py`

**Interfaces:**
- Consumes: Task 1~3 전부
- Produces:
  - `def business_days(start: date, end: date) -> list[str]` — 주말 제외 YYYYMMDD 오름차순
  - `def backfill(db, api_key, years: int = 11, workers: int = 8, today: date | None = None, _get=None) -> dict` — 반환 `{"requested": n, "loaded_days": n, "rows": n, "skipped": n}`
  - `def sync(db, api_key, workers: int = 8, today: date | None = None, _get=None) -> dict` — 반환 동일 형태. `last_loaded_date`가 없으면 아무것도 받지 않고 `{"requested": 0, ...}`
  - `def rebuild_adjustments(db, tickers: list[str] | None = None, since: str = "19900101") -> int`

- [ ] **Step 1: 실패 테스트 작성** (`tests/test_price_backfill.py`)

```python
from datetime import date

from src.price_history.backfill import (
    backfill, business_days, rebuild_adjustments, sync,
)
from src.price_history.db import PriceDB


def _db(tmp_path):
    return PriceDB(path=str(tmp_path / "prices.db"))


class FakeResp:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload


def _maker(rows_by_date, calls):
    """rows_by_date: {YYYYMMDD: [(ticker,high,close,chg)]}. 없는 날짜는 휴장."""
    def fake_get(url, params, headers, timeout):
        d = params["basDd"]
        calls.append(d)
        items = [
            {"ISU_CD": tk, "TDD_HGPRC": h, "TDD_CLSPRC": c, "CMPPREVDD_PRC": ch}
            for tk, h, c, ch in rows_by_date.get(d, [])
        ]
        return FakeResp({"OutBlock_1": items})
    return fake_get


def test_business_days_excludes_weekends():
    out = business_days(date(2026, 8, 14), date(2026, 8, 18))  # 금~화
    assert out == ["20260814", "20260817", "20260818"]


def test_backfill_loads_and_reports(tmp_path):
    db = _db(tmp_path)
    calls = []
    rows = {"20260818": [("005930", 110, 100, 0)], "20260819": [("005930", 120, 115, 15)]}
    res = backfill(db, "KEY", years=1, today=date(2026, 8, 19),
                   workers=2, _get=_maker(rows, calls))
    assert res["rows"] > 0
    assert db.last_loaded_date() == "20260819"


def test_backfill_skips_already_loaded_dates(tmp_path):
    db = _db(tmp_path)
    db.save_day("20260819", "KOSPI", [("005930", 1, 1, 0)])
    db.save_day("20260819", "KOSDAQ", [("035720", 1, 1, 0)])
    calls = []
    backfill(db, "KEY", years=1, today=date(2026, 8, 19), workers=2,
             _get=_maker({}, calls))
    assert "20260819" not in calls          # 이미 적재된 날짜는 요청하지 않는다


def test_sync_without_prior_data_requests_nothing(tmp_path):
    db = _db(tmp_path)
    calls = []
    res = sync(db, "KEY", today=date(2026, 8, 19), workers=2, _get=_maker({}, calls))
    assert res["requested"] == 0
    assert calls == []


def test_sync_fills_gap_since_last_loaded(tmp_path):
    db = _db(tmp_path)
    db.save_day("20260814", "KOSPI", [("005930", 100, 100, 0)])
    calls = []
    rows = {"20260818": [("005930", 110, 105, 5)], "20260819": [("005930", 120, 115, 10)]}
    sync(db, "KEY", today=date(2026, 8, 19), workers=2, _get=_maker(rows, calls))
    assert "20260817" in calls and "20260819" in calls
    assert "20260814" not in calls          # 이미 있는 날짜는 다시 받지 않는다
    assert db.last_loaded_date() == "20260819"


def test_rebuild_adjustments_persists_events(tmp_path):
    db = _db(tmp_path)
    db.save_day("20260105", "KOSPI", [("005930", 0, 2_650_000, 0)])
    db.save_day("20260106", "KOSPI", [("005930", 53_900, 51_900, -1_100)])
    n = rebuild_adjustments(db)
    assert n == 1
    evs = db.load_events("005930")
    assert len(evs) == 1 and evs[0].factor == 50.0
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_price_backfill.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.price_history.backfill'`

- [ ] **Step 3: 최소 구현** (`src/price_history/backfill.py`)

```python
"""일봉 적재 오케스트레이션 — 초기 백필과 일일 증분.

초기 백필은 약 5,400 요청(11년 x 2시장)으로 실측 16분이 걸린다. 이미
적재된 날짜를 건너뛰므로 중단돼도 다시 실행하면 이어받는다.
"""
from __future__ import annotations

from datetime import date, timedelta

from loguru import logger

from src.price_history.adjust import detect_adjustments
from src.price_history.fetcher import MARKET_ENDPOINTS, fetch_many


def business_days(start: date, end: date) -> list[str]:
    """start~end(양끝 포함)의 주말 제외 날짜. 공휴일은 빈 응답으로 걸러진다."""
    out: list[str] = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return out


def _load(db, api_key: str, days: list[str], workers: int, _get) -> dict:
    """이미 적재된 (market, d)는 건너뛰고 나머지를 받아 저장한다."""
    jobs: list[tuple[str, str]] = []
    skipped = 0
    for market in MARKET_ENDPOINTS:
        done = db.loaded_dates(market)
        for d in days:
            if d in done:
                skipped += 1
            else:
                jobs.append((market, d))

    rows_total = 0
    loaded_days = 0
    if jobs:
        for i, (market, d, rows) in enumerate(fetch_many(api_key, jobs, workers, _get), 1):
            if rows:
                rows_total += db.save_day(d, market, rows)
                loaded_days += 1
            if i % 200 == 0:
                logger.info(f"일봉 적재 {i}/{len(jobs)} 요청, {rows_total:,}행")

    return {
        "requested": len(jobs),
        "loaded_days": loaded_days,
        "rows": rows_total,
        "skipped": skipped,
    }


def backfill(
    db, api_key: str, years: int = 11, workers: int = 8,
    today: date | None = None, _get=None,
) -> dict:
    """years년치를 적재한다. 이미 있는 날짜는 건너뛴다(재개 가능)."""
    end = today or date.today()
    start = end - timedelta(days=int(365.25 * years))
    days = business_days(start, end)
    logger.info(f"일봉 백필 시작: {start} ~ {end} ({len(days)} 영업일 x {len(MARKET_ENDPOINTS)} 시장)")
    res = _load(db, api_key, days, workers, _get)
    db.set_meta("backfill_years", str(years))
    rebuilt = rebuild_adjustments(db)
    res["adjust_events"] = rebuilt
    logger.info(
        f"일봉 백필 완료: {res['rows']:,}행 적재, {res['skipped']}건 건너뜀, "
        f"수정 이벤트 {rebuilt}건"
    )
    return res


def sync(
    db, api_key: str, workers: int = 8,
    today: date | None = None, _get=None,
) -> dict:
    """마지막 적재일 다음날부터 오늘까지 채운다.

    저장소가 비어 있으면 아무것도 받지 않는다 — 16분짜리 백필을 자동으로
    시작하면 안 된다. 경고만 남기고 사용자가 'prices backfill'을 실행하게 한다.
    """
    end = today or date.today()
    last = db.last_loaded_date()
    if last is None:
        logger.warning(
            "일봉 저장소가 비어 있습니다. 'python -m src.cli prices backfill'을 "
            "먼저 실행하세요(약 16분). 돌파 신선도는 이번 실행에서 생략됩니다."
        )
        return {"requested": 0, "loaded_days": 0, "rows": 0, "skipped": 0}

    start = date(int(last[:4]), int(last[4:6]), int(last[6:8])) + timedelta(days=1)
    if start > end:
        return {"requested": 0, "loaded_days": 0, "rows": 0, "skipped": 0}

    days = business_days(start, end)
    res = _load(db, api_key, days, workers, _get)
    if res["rows"]:
        rebuild_adjustments(db)
    return res


def rebuild_adjustments(db, tickers: list[str] | None = None, since: str = "19900101") -> int:
    """티커별로 수정 이벤트를 다시 계산해 저장한다. 반환은 총 이벤트 수."""
    total = 0
    for ticker in (tickers if tickers is not None else db.tickers()):
        rows = db.load_rows(ticker, since=since)
        events = detect_adjustments(rows)
        db.save_events(ticker, events)
        total += len(events)
    return total
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_price_backfill.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: 실제 KRX로 소규모 적재 확인 (수동, 네트워크 필요)**

`.env`에 `KRX_API_KEY`가 있을 때만 실행한다. 없으면 건너뛰고 커밋 메시지에 남긴다.

```bash
.venv/bin/python -c "
from datetime import date
from src.config import Settings
from src.price_history.db import PriceDB
from src.price_history.backfill import backfill
db = PriceDB(path='/tmp/prices_smoke.db')
print(backfill(db, Settings().krx_api_key, years=0.1, today=date.today()))
"
```

Expected: `rows`가 수천 단위, `requested`가 수십 단위. 실패 없이 종료.

- [ ] **Step 6: 커밋**

```bash
git add src/price_history/backfill.py tests/test_price_backfill.py
git commit -m "feat(prices): 초기 백필과 일일 증분 적재 추가"
```

---

### Task 5: 조회 — `loader.py`

**Files:**
- Create: `src/price_history/loader.py`
- Test: `tests/test_price_loader.py`

**Interfaces:**
- Consumes: Task 1의 `adjusted_highs`, Task 2의 `PriceDB`, 기존 `src.breakout_recency.Bar`
- Produces: `def load_bars(db, ticker: str, as_of: date, years: int = 11) -> list[Bar] | None`
  - `as_of` 이후 날짜는 제외한다(과거 시점 재현용).
  - 고가가 0 이하인 봉(거래정지)은 제외한다.
  - 유효 봉이 2개 미만이면 `None`.

- [ ] **Step 1: 실패 테스트 작성** (`tests/test_price_loader.py`)

```python
from datetime import date

from src.price_history.db import PriceDB
from src.price_history.loader import load_bars


def _db(tmp_path):
    return PriceDB(path=str(tmp_path / "prices.db"))


def test_returns_none_when_ticker_absent(tmp_path):
    assert load_bars(_db(tmp_path), "005930", date(2026, 8, 19)) is None


def test_returns_none_with_fewer_than_two_valid_bars(tmp_path):
    db = _db(tmp_path)
    db.save_day("20260818", "KOSPI", [("005930", 100, 100, 0)])
    assert load_bars(db, "005930", date(2026, 8, 19)) is None


def test_returns_ascending_bars(tmp_path):
    db = _db(tmp_path)
    db.save_day("20260819", "KOSPI", [("005930", 120, 115, 5)])
    db.save_day("20260818", "KOSPI", [("005930", 110, 110, 0)])
    bars = load_bars(db, "005930", date(2026, 8, 19))
    assert [b.date for b in bars] == [date(2026, 8, 18), date(2026, 8, 19)]
    assert [b.high for b in bars] == [110.0, 120.0]


def test_excludes_dates_after_as_of(tmp_path):
    db = _db(tmp_path)
    for d, h in [("20260817", 100), ("20260818", 110), ("20260819", 120)]:
        db.save_day(d, "KOSPI", [("005930", h, h, 0)])
    bars = load_bars(db, "005930", date(2026, 8, 18))
    assert [b.date for b in bars] == [date(2026, 8, 17), date(2026, 8, 18)]


def test_drops_halted_days_with_zero_high(tmp_path):
    db = _db(tmp_path)
    db.save_day("20260817", "KOSPI", [("005930", 100, 100, 0)])
    db.save_day("20260818", "KOSPI", [("005930", 0, 100, 0)])   # 거래정지
    db.save_day("20260819", "KOSPI", [("005930", 120, 115, 5)])
    bars = load_bars(db, "005930", date(2026, 8, 19))
    assert [b.date for b in bars] == [date(2026, 8, 17), date(2026, 8, 19)]


def test_applies_stored_adjustment_events(tmp_path):
    db = _db(tmp_path)
    # 5:1 액면병합: 정지 전 396 -> 재개일 기준가 1,980
    db.save_day("20260817", "KOSPI", [("005930", 400, 396, 0)])
    db.save_day("20260818", "KOSPI", [("005930", 2065, 1720, -260)])
    from src.price_history.backfill import rebuild_adjustments
    rebuild_adjustments(db)

    bars = load_bars(db, "005930", date(2026, 8, 18))
    assert bars[0].high == 2000.0    # 400 x 5
    assert bars[1].high == 2065.0    # 이벤트 당일은 그대로
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_price_loader.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.price_history.loader'`

- [ ] **Step 3: 최소 구현** (`src/price_history/loader.py`)

```python
"""저장소에서 티커의 수정 일봉을 꺼낸다. 네트워크를 타지 않는다."""
from __future__ import annotations

from datetime import date, timedelta

from src.breakout_recency import Bar
from src.price_history.adjust import adjusted_highs


def load_bars(db, ticker: str, as_of: date, years: int = 11) -> list[Bar] | None:
    """as_of 기준 years년치 수정 일봉(날짜 오름차순).

    유효 봉이 2개 미만이면 None. 거래정지일(고가 0)은 제외한다.
    """
    since = (as_of - timedelta(days=int(365.25 * years))).strftime("%Y%m%d")
    rows = [r for r in db.load_rows(ticker, since=since) if r.d <= as_of]
    if len(rows) < 2:
        return None

    events = [e for e in db.load_events(ticker) if e.d <= as_of]
    bars = [
        Bar(date=d, high=high)
        for d, high in adjusted_highs(rows, events)
        if high > 0
    ]
    return bars if len(bars) >= 2 else None
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_price_loader.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/price_history/loader.py tests/test_price_loader.py
git commit -m "feat(prices): 저장소에서 수정 일봉을 읽는 loader 추가"
```

---

### Task 6: 신선도 계산을 달력 창으로 — `breakout_recency.py`

거래일 수 창(250)을 달력 일수 창(365)으로 바꾼다. 순진하게 구현하면 후보일마다 창을 다시 훑어 O(n²)이 되고, 2,700봉 종목에서 수 초가 걸린다. **단조 덱으로 롤링 최고가를 O(n)에 미리 구한다.** `prev_high_52w`는 그 배열의 마지막 원소로 공짜로 나온다.

**Files:**
- Modify: `src/breakout_recency.py`
- Test: `tests/test_breakout_recency.py` (전면 재작성)

**Interfaces:**
- Produces:
  - `def compute_recency(bars: list[Bar], window_days: int = 365) -> Recency | None` — `window` 파라미터가 `window_days`로 대체된다
  - `Recency` 필드는 불변

- [ ] **Step 1: 테스트 재작성** (`tests/test_breakout_recency.py` 전체 교체)

```python
from datetime import date, timedelta

from src.breakout_recency import Bar, compute_recency

END = date(2026, 8, 19)


def _daily(highs: list[float], end: date = END) -> list[Bar]:
    """마지막 원소가 end인 연속 일봉(달력 하루 간격)."""
    n = len(highs)
    return [Bar(date=end - timedelta(days=n - 1 - i), high=h) for i, h in enumerate(highs)]


def test_returns_none_when_fewer_than_two_bars():
    assert compute_recency([]) is None
    assert compute_recency([Bar(date=END, high=100.0)]) is None


def test_all_time_high_has_no_price_above_day():
    r = compute_recency(_daily([100.0 + i for i in range(400)]))
    assert r is not None
    assert r.days_since_price_above is None
    assert r.today_high == 499.0


def test_daily_streak_gives_one_day():
    r = compute_recency(_daily([100.0 + i for i in range(400)]))
    assert r.days_since_prev_new_high == 1


def test_prev_high_uses_calendar_window_not_bar_count():
    # 창(365일) 밖의 500은 제외되고, 창 안의 100만 본다
    highs = [500.0] + [100.0] * 400 + [150.0]
    r = compute_recency(_daily(highs), window_days=365)
    assert r.prev_high_52w == 100.0


def test_prev_high_is_zero_when_window_not_covered():
    r = compute_recency(_daily([100.0] * 100 + [150.0]), window_days=365)
    assert r.prev_high_52w == 0.0
    assert r.days_since_prev_new_high is None   # 워밍업 부족


def test_staircase_recovery_splits_the_two_metrics():
    # index 0: 옛 고점 300 / 1~400: 박스권 100 / 401: 150(그날 신고가) /
    # 402~500: 120 / 501: 오늘 250
    highs = [300.0] + [100.0] * 400 + [150.0] + [120.0] * 99 + [250.0]
    r = compute_recency(_daily(highs), window_days=365)
    assert r.days_since_price_above == 501          # index 0 까지
    assert r.days_since_prev_new_high == 100        # index 401 까지
    assert r.prev_high_52w == 150.0


def test_history_span_days_spans_first_to_last():
    r = compute_recency(_daily([100.0] * 11))
    assert r.history_span_days == 10


def test_gap_in_dates_does_not_break_window():
    # 휴장으로 날짜가 듬성해도 달력 기준으로 창을 잡는다
    bars = [Bar(date=END - timedelta(days=d), high=h) for d, h in
            [(400, 90.0), (370, 95.0), (200, 80.0), (100, 85.0), (0, 99.0)]]
    bars.sort(key=lambda b: b.date)
    r = compute_recency(bars, window_days=365)
    assert r is not None
    assert r.prev_high_52w == 85.0      # 365일 안: 80, 85 (95는 370일 전이라 제외)
    assert r.days_since_price_above is None
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_breakout_recency.py -q`
Expected: FAIL — `test_prev_high_uses_calendar_window_not_bar_count` 등이 기존 250봉 구현에서 실패

- [ ] **Step 3: 구현 교체** (`src/breakout_recency.py`)

`window: int = 250` 을 쓰던 부분을 아래로 교체한다. `Bar`/`Recency` 정의는 그대로 둔다.

```python
from collections import deque
from datetime import timedelta


def _prior_max_by_calendar(bars: list[Bar], window_days: int) -> list[float | None]:
    """각 봉 시점에서 '직전 window_days 구간(당일 제외)'의 최고 고가.

    창 전체를 덮는 데이터가 없으면 None. 단조 덱으로 O(n)에 구한다 —
    후보일마다 창을 다시 훑으면 O(n^2)이 되어 11년치에서 수 초가 걸린다.
    """
    out: list[float | None] = [None] * len(bars)
    if not bars:
        return out
    first = bars[0].date
    dq: deque[tuple] = deque()          # (date, high), high 내림차순
    for j, bar in enumerate(bars):
        win_start = bar.date - timedelta(days=window_days)
        while dq and dq[0][0] < win_start:
            dq.popleft()
        if first <= win_start and dq:
            out[j] = dq[0][1]
        # 당일은 자기 창에서 제외되므로 값을 읽은 뒤에 넣는다.
        while dq and dq[-1][1] <= bar.high:
            dq.pop()
        dq.append((bar.date, bar.high))
    return out


def compute_recency(bars: list[Bar], window_days: int = 365) -> Recency | None:
    """bars(날짜 오름차순, 마지막이 오늘)에서 A·B를 계산한다."""
    if len(bars) < 2:
        return None

    today = bars[-1]
    past = bars[:-1]

    # B: 오늘 고가 이상이었던 가장 최근 과거일. 창을 쓰지 않고 이력 전체를 본다.
    days_since_price_above: int | None = None
    for bar in reversed(past):
        if bar.high >= today.high:
            days_since_price_above = (today.date - bar.date).days
            break

    prior_max = _prior_max_by_calendar(bars, window_days)

    # A: 그날 자체가 52주 신고가였던 가장 최근 과거일.
    days_since_prev_new_high: int | None = None
    for j in range(len(past) - 1, -1, -1):
        pm = prior_max[j]
        if pm is not None and bars[j].high >= pm:
            days_since_prev_new_high = (today.date - bars[j].date).days
            break

    # 오늘 시점의 창 최고가가 곧 직전 52주 고점이다.
    prev_high_52w = prior_max[-1] if prior_max[-1] is not None else 0.0

    return Recency(
        days_since_prev_new_high=days_since_prev_new_high,
        days_since_price_above=days_since_price_above,
        history_span_days=(today.date - bars[0].date).days,
        prev_high_52w=prev_high_52w,
        today_high=today.high,
    )
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_breakout_recency.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/breakout_recency.py tests/test_breakout_recency.py
git commit -m "refactor(recency): 신선도 창을 250 거래일에서 365 달력일로 변경"
```

---

### Task 7: 소비 측 교체 — `recency_source.py`

**Files:**
- Modify: `src/recency_source.py` (대폭 축소)
- Modify: `tests/test_recency_source.py` (KRX 조회 테스트 삭제, 저장소 기반으로 재작성)

**Interfaces:**
- Consumes: Task 5의 `load_bars`
- Produces: `def enrich_highs(highs: list[StockHigh], as_of: date, db=None, window_days: int = 365) -> None` — **`client` 인자가 사라진다**

**삭제 대상** (존재 이유가 사라짐): `CHUNK_DAYS`, `fetch_bars`, `_to_bars`, 역방향 청크 루프, 빈 응답 재확인 로직, `KrxBlockedError` import.

- [ ] **Step 1: 테스트 재작성** (`tests/test_recency_source.py` 전체 교체)

```python
from datetime import date, timedelta

from src.models import StockHigh
from src.price_history.db import PriceDB
from src import recency_source


def _stock(ticker="005930", close=100.0):
    return StockHigh(
        ticker=ticker, name="테스트", market="KOSPI", sector="전기전자",
        close_price=close, high_52w=close, prev_high_52w=0.0,
        breakout_pct=0.0, volume=1, avg_volume_20d=0, change_pct=2.0,
    )


def _seeded_db(tmp_path, ticker="005930", days=400, high=100, last_high=110):
    """days일치 평탄한 이력 + 마지막 날 돌파."""
    db = PriceDB(path=str(tmp_path / "prices.db"))
    end = date(2026, 8, 19)
    for i in range(days):
        d = (end - timedelta(days=days - 1 - i)).strftime("%Y%m%d")
        h = last_high if i == days - 1 else high
        db.save_day(d, "KOSPI", [(ticker, h, h, 0)])
    return db


def test_enrich_fills_metrics_and_normalizes_breakout(tmp_path):
    db = _seeded_db(tmp_path)
    stock = _stock()
    recency_source.enrich_highs([stock], date(2026, 8, 19), db=db)

    assert stock.history_span_days == 399
    assert stock.days_since_price_above is None
    assert stock.days_since_prev_new_high == 1
    assert stock.prev_high_52w == 100.0
    assert stock.breakout_pct == 10.0
    assert stock.high_52w == 110.0


def test_enrich_leaves_stock_untouched_without_history(tmp_path):
    db = PriceDB(path=str(tmp_path / "prices.db"))
    stock = _stock()
    recency_source.enrich_highs([stock], date(2026, 8, 19), db=db)
    assert stock.history_span_days is None
    assert stock.breakout_pct == 0.0
    assert stock.change_pct == 2.0
    assert stock.high_52w == 100.0


def test_enrich_skips_stale_last_bar(tmp_path):
    # 저장소의 마지막 고가가 investing 종가보다 낮으면 그 종목은 건너뛴다
    db = _seeded_db(tmp_path, last_high=110)
    stock = _stock(close=200.0)
    recency_source.enrich_highs([stock], date(2026, 8, 19), db=db)
    assert stock.history_span_days is None


def test_enrich_isolates_per_stock_failure(tmp_path, monkeypatch):
    db = _seeded_db(tmp_path, ticker="000002")

    real = recency_source.load_bars

    def flaky(db_, ticker, as_of, **kw):
        if ticker == "000001":
            raise ValueError("boom")
        return real(db_, ticker, as_of, **kw)

    monkeypatch.setattr(recency_source, "load_bars", flaky)
    bad, ok = _stock("000001"), _stock("000002")
    recency_source.enrich_highs([bad, ok], date(2026, 8, 19), db=db)
    assert bad.history_span_days is None
    assert ok.history_span_days == 399


def test_fetch_bars_is_gone():
    """KRX 종목별 조회 경로는 삭제되었다."""
    assert not hasattr(recency_source, "fetch_bars")
    assert not hasattr(recency_source, "CHUNK_DAYS")
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_recency_source.py -q`
Expected: FAIL — `enrich_highs()` 시그니처 불일치 및 `fetch_bars` 잔존

- [ ] **Step 3: 구현 교체** (`src/recency_source.py` 전체를 아래로 대체)

```python
"""신고가 종목에 돌파 신선도를 채운다.

가격 이력은 로컬 저장소(data/prices.db)에서 읽는다. KRX에 종목별로
조회하던 경로는 제거됐다 — 그쪽은 2년 조회 상한이 있고 종목당 6콜이 들었다.
계산 자체는 src.breakout_recency의 순수 함수가 담당한다.
"""
from __future__ import annotations

from datetime import date

from loguru import logger

from src.breakout_recency import compute_recency
from src.models import StockHigh
from src.price_history.loader import load_bars


def enrich_highs(
    highs: list[StockHigh],
    as_of: date,
    db=None,
    window_days: int = 365,
) -> None:
    """highs 각 종목의 돌파 신선도를 계산해 제자리에서 채운다.

    이력을 못 읽은 종목은 지표를 None으로 남기고 기존 값을 보존한다.
    """
    if db is None:
        from src.price_history.db import PriceDB
        db = PriceDB()

    filled = 0
    for stock in highs:
        try:
            bars = load_bars(db, stock.ticker, as_of)
        except Exception as e:  # noqa: BLE001 — 개별 종목 실패는 나머지를 막지 않는다
            logger.warning(f"{stock.ticker} 신선도 계산 실패: {type(e).__name__}: {e}")
            continue

        if not bars:
            continue

        recency = compute_recency(bars, window_days=window_days)
        if recency is None:
            continue

        # 저장소의 마지막 봉이 오늘 것이 아니면 종가보다 낮은 고가가 나온다.
        # 실제 장중 고가는 같은 날 종가보다 낮을 수 없다.
        if recency.today_high < stock.close_price:
            logger.warning(
                f"{stock.name}({stock.ticker}) 최신 봉이 오늘 것이 아님 — "
                f"마지막 봉 {bars[-1].date} 고가 {recency.today_high:,.0f} < "
                f"종가 {stock.close_price:,.0f}, 신선도 계산 생략"
            )
            continue

        stock.days_since_prev_new_high = recency.days_since_prev_new_high
        stock.days_since_price_above = recency.days_since_price_above
        stock.history_span_days = recency.history_span_days
        stock.high_52w = recency.today_high
        stock.prev_high_52w = recency.prev_high_52w
        if recency.prev_high_52w > 0:
            stock.breakout_pct = round(
                (recency.today_high - recency.prev_high_52w) / recency.prev_high_52w * 100, 2
            )

        if recency.days_since_price_above is not None and recency.days_since_price_above < 365:
            logger.warning(
                f"{stock.name}({stock.ticker}) B={recency.days_since_price_above}일 — "
                f"52주 신고가와 불일치(액면병합 등 investing 미반영 가능성)"
            )
        filled += 1

    logger.info(f"돌파 신선도 산출: {filled}/{len(highs)}종목")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_recency_source.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/recency_source.py tests/test_recency_source.py
git commit -m "refactor(recency): 가격 이력 소스를 로컬 저장소로 교체하고 KRX 종목별 조회 제거"
```

---

### Task 8: CLI — `prices` 명령과 `run` 자동 동기화

**Files:**
- Modify: `src/cli.py`
- Test: `tests/test_cli.py` (추가)

**Interfaces:**
- Consumes: Task 4의 `backfill`/`sync`, Task 7의 `enrich_highs` 새 시그니처
- Produces: `prices` 하위 명령 앱 — `backfill` / `sync` / `status`

- [ ] **Step 1: 실패 테스트 작성** (`tests/test_cli.py` 끝에 추가)

```python
def test_cli_prices_subcommands_exist():
    """prices backfill / sync / status 가 노출된다."""
    from src.cli import app
    result = runner.invoke(app, ["prices", "--help"])
    assert result.exit_code == 0
    for sub in ("backfill", "sync", "status"):
        assert sub in result.output


def test_cli_prices_backfill_has_years_option():
    from src.cli import app
    result = runner.invoke(app, ["prices", "backfill", "--help"])
    assert result.exit_code == 0
    assert "--years" in result.output
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_cli.py -q`
Expected: FAIL — `prices` 명령이 없어 `--help` 종료 코드가 0이 아님

- [ ] **Step 3: `prices` 앱 추가** (`src/cli.py`)

파일 하단, 기존 명령들 뒤에 추가한다.

```python
prices_app = typer.Typer(help="일봉 이력 저장소 (data/prices.db)")
app.add_typer(prices_app, name="prices")


@prices_app.command("backfill")
def prices_backfill(
    years: int = typer.Option(11, "--years", "-y", help="적재할 과거 연수"),
    workers: int = typer.Option(8, "--workers", "-w", help="동시 요청 수"),
):
    """KRX Open API로 일봉 이력을 적재한다(11년 기준 약 16분). 재실행하면 이어받는다."""
    from src.price_history.backfill import backfill
    from src.price_history.db import PriceDB

    settings = Settings()
    if not settings.krx_api_key:
        console.print("[red]KRX_API_KEY가 없습니다. .env를 확인하세요.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold]일봉 백필 시작 ({years}년, 동시 {workers})[/bold]")
    res = backfill(PriceDB(), settings.krx_api_key, years=years, workers=workers)
    console.print(
        f"[green]완료: {res['rows']:,}행 적재, {res['skipped']}건 건너뜀, "
        f"수정 이벤트 {res['adjust_events']}건[/green]"
    )


@prices_app.command("sync")
def prices_sync():
    """마지막 적재일 이후를 채운다(평상시 2콜)."""
    from src.price_history.backfill import sync
    from src.price_history.db import PriceDB

    settings = Settings()
    res = sync(PriceDB(), settings.krx_api_key)
    console.print(f"[green]동기화: {res['rows']:,}행 추가 ({res['requested']}건 요청)[/green]")


@prices_app.command("status")
def prices_status():
    """저장소 현황을 출력한다."""
    from src.price_history.db import PriceDB

    db = PriceDB()
    last = db.last_loaded_date()
    if last is None:
        console.print("[yellow]저장소가 비어 있습니다. 'prices backfill'을 먼저 실행하세요.[/yellow]")
        return
    n_rows = db.con.execute("SELECT COUNT(*) FROM daily_px").fetchone()[0]
    n_days = db.con.execute("SELECT COUNT(DISTINCT d) FROM daily_px").fetchone()[0]
    n_tk = db.con.execute("SELECT COUNT(DISTINCT ticker) FROM daily_px").fetchone()[0]
    n_ev = db.con.execute("SELECT COUNT(*) FROM px_adjust").fetchone()[0]
    size_mb = os.path.getsize(db.path) / 1024 / 1024
    console.print(
        f"마지막 적재일 {last} | {n_rows:,}행 | 거래일 {n_days:,} | "
        f"종목 {n_tk:,} | 수정 이벤트 {n_ev:,} | {size_mb:.0f}MB"
    )
```

파일 상단 import에 `import os` 를 추가한다(없으면).

- [ ] **Step 4: `run`에서 자동 동기화 + `enrich_highs` 호출 변경** (`src/cli.py`)

`run`의 `else:` 분기에서 import 블록을 다음으로 바꾼다. `KrxBlockedError`는 더 이상 필요 없다.

```python
        from src.dart.cache import DartCache
        from src.collector import Collector
        from src.scanner import Scanner
        from src.investing_high import collect_investing_highs, InvestingFetchError, InvestingParseError
        from src.recency_source import enrich_highs
        from src.price_history.backfill import sync as price_sync
        from src.price_history.db import PriceDB
```

`collect_investing_highs` 호출 뒤 블록을 다음으로 교체한다.

```python
        console.print("[dim]1-3/5 일봉 저장소 동기화 중...[/dim]")
        price_db = PriceDB()
        sync_res = price_sync(price_db, settings.krx_api_key)
        if sync_res["rows"]:
            console.print(f"[dim]  {sync_res['rows']:,}행 추가[/dim]")

        console.print(f"[dim]1-3/5 돌파 신선도 계산 중... ({len(highs)}종목)[/dim]")
        enrich_highs(highs, scan_date, db=price_db)

        result = scanner.build_scan_result(scan_date, highs, len(highs))
        db.save_scan_result(result)
```

- [ ] **Step 5: 회귀 + 수동 확인**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS (기존 248 + 신규 태스크들의 테스트)

`.env`가 있으면 다음을 확인한다. 없으면 건너뛰고 커밋 메시지에 남긴다.

```bash
.venv/bin/python -m src.cli prices status
```

Expected: 저장소가 비어 있으면 백필 안내가, 있으면 현황 한 줄이 출력된다.

- [ ] **Step 6: 커밋**

```bash
git add src/cli.py tests/test_cli.py
git commit -m "feat(prices): prices 명령 추가하고 run에서 저장소 자동 동기화"
```

---

### Task 9: 리포트 — `⚠ 이력 불일치` 그룹

**Files:**
- Modify: `src/reporter.py`
- Modify: `src/scanner.py` (`build_scan_result`의 카운트)
- Modify: `src/cli.py` (AI 분석 대상에서 제외)
- Test: `tests/test_reporter.py`, `tests/test_scanner.py` (추가)

**Interfaces:**
- Consumes: Task 7이 채운 `days_since_price_above`
- Produces:
  - `src.breakout_recency.MISMATCH_MAX_DAYS: int = 365`
  - `src.breakout_recency.is_history_mismatch(days_since_price_above: int | None) -> bool`
  - `src.reporter.GROUP_MISMATCH: str` = `"⚠ 이력 불일치 · 52주 신고가 아님"`

**판정 규칙을 `breakout_recency`에 두는 이유**: `scanner.py`와 `cli.py`도 이 규칙이 필요한데, `reporter.py`에 두면 도메인이 표현 계층을 import하게 되어 의존 방향이 뒤집힌다. 원시 정수를 받게 해서 `breakout_recency`가 `StockHigh`를 모르는 순수 모듈로 남도록 유지한다.

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_reporter.py` 끝에 추가:

```python
def test_is_history_mismatch():
    from src.breakout_recency import is_history_mismatch
    assert is_history_mismatch(90) is True
    assert is_history_mismatch(364) is True
    assert is_history_mismatch(365) is False
    assert is_history_mismatch(None) is False


def test_mismatch_stock_goes_to_its_own_group_last():
    from src.reporter import Reporter, GROUP_MISMATCH

    highs = [_rstock("000001", "정상", a=1000, span=4000, b=2000),
             _rstock("000002", "불일치", a=10, span=4000, b=90)]
    text = Reporter(bot_token="", chat_id=0).format_report(_result(highs), [], [])

    assert f"[{GROUP_MISMATCH}]" in text
    listing = text.split("■ 전체 52주 신고가 목록")[1]
    assert listing.index("정상") < listing.index("불일치")
```

`tests/test_scanner.py` 끝에 추가:

```python
def test_build_scan_result_excludes_history_mismatch_from_count():
    """이력 불일치 종목은 신고가 카운트에서 빠진다."""
    from datetime import date
    from src.scanner import Scanner
    from src.models import StockHigh

    def _s(ticker, b):
        return StockHigh(
            ticker=ticker, name=ticker, market="KOSPI", sector="기타",
            close_price=100, high_52w=100, prev_high_52w=0.0, breakout_pct=0.0,
            volume=1, avg_volume_20d=0, change_pct=1.0,
            history_span_days=4000, days_since_price_above=b,
        )

    highs = [_s("000001", 2000), _s("000002", 90)]
    result = Scanner(collector=None).build_scan_result(date(2026, 8, 19), highs, 2)
    assert result.stats.new_high_count == 1
    assert len(result.highs) == 2          # 목록에는 남는다
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_reporter.py tests/test_scanner.py -q`
Expected: FAIL — `ImportError: cannot import name 'is_history_mismatch'`

- [ ] **Step 3: 판정 규칙과 리포터 구현**

`src/breakout_recency.py` 끝에 추가한다(이 모듈은 `StockHigh`를 모른다 — 원시 값을 받는다).

```python
MISMATCH_MAX_DAYS = 365


def is_history_mismatch(days_since_price_above: int | None) -> bool:
    """우리 이력이 52주 신고가를 반박하는가.

    B가 1년 미만이면 그 가격보다 높았던 날이 52주 안에 있다는 뜻이다.
    액면병합 등을 investing이 소급 반영하지 않아 생기는 거짓 신고가를 잡는다.
    """
    return days_since_price_above is not None and days_since_price_above < MISMATCH_MAX_DAYS
```

`src/reporter.py`의 그룹 상수 옆에 추가하고 `GROUP_ORDER` 맨 끝에 넣는다.

```python
from src.breakout_recency import is_history_mismatch

GROUP_MISMATCH = "⚠ 이력 불일치 · 52주 신고가 아님"
GROUP_ORDER = (GROUP_LONG, GROUP_MID, GROUP_STREAK, GROUP_NEW_LISTING, GROUP_UNKNOWN, GROUP_MISMATCH)
```

`_recency_group`의 맨 앞에 분기를 추가한다.

```python
def _recency_group(stock) -> str:
    if is_history_mismatch(stock.days_since_price_above):
        return GROUP_MISMATCH
    if stock.history_span_days is None:
        return GROUP_UNKNOWN
    ...
```

- [ ] **Step 4: 카운트·AI 제외**

`src/scanner.py`의 `build_scan_result`에서 `new_high_count`를 불일치 제외로 바꾼다.

```python
        from src.breakout_recency import is_history_mismatch
        genuine = [h for h in highs if not is_history_mismatch(h.days_since_price_above)]
```

`MarketStats(...)`의 `new_high_count=len(highs)` 를 `new_high_count=len(genuine)` 로 바꾼다. 시장별 카운트(`kospi_count` 등)도 `genuine` 기준으로 센다. `highs`와 `sector_breakdown`은 그대로 전체를 담는다.

`src/cli.py`의 AI 분석 대상 선정에서 제외한다.

```python
    from src.breakout_recency import is_history_mismatch
    done_tickers = set() if force else db.get_ai_analyzed_tickers(scan_date)
    remaining = [
        h for h in highs
        if h.ticker not in done_tickers and not is_history_mismatch(h.days_since_price_above)
    ]
```

- [ ] **Step 5: 전체 테스트**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS

- [ ] **Step 6: 커밋**

```bash
git add src/reporter.py src/scanner.py src/cli.py tests/test_reporter.py tests/test_scanner.py
git commit -m "feat(report): 이력 불일치 그룹 분리하고 카운트·AI 분석에서 제외"
```

---

### Task 10: README 갱신

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 주요 기능과 사용법 갱신**

주요 기능 목록에 한 줄 추가한다.

```markdown
- 로컬 일봉 이력 저장소(`data/prices.db`)로 수정주가 기준 돌파 신선도 산출 — 액면분할·병합 자동 보정
```

`### 개별 명령어` 블록에 다음을 추가한다.

```bash
# 일봉 이력 저장소 (최초 1회, 11년 기준 약 16분)
python -m src.cli prices backfill

# 저장소 현황
python -m src.cli prices status
```

그리고 그 아래에 한 문단을 넣는다.

```markdown
돌파 신선도 지표는 `data/prices.db`의 일봉 이력을 씁니다. 최초 1회 `prices backfill`이 필요하고, 이후에는 `run`이 실행 때마다 자동으로 동기화합니다(평상시 KRX 호출 2건). 이 파일은 언제든 재생성 가능한 캐시이므로 백업 대상이 아닙니다.
```

- [ ] **Step 2: 커밋**

```bash
git add README.md
git commit -m "docs: 일봉 이력 저장소 사용법 추가"
```

---

## 스펙 커버리지 확인

| 스펙 항목 | 구현 위치 |
|---|---|
| 당일 판별은 investing 유지 | 변경 없음 (Task 8이 `collect_investing_highs`를 그대로 둠) |
| Open API 일자별 벌크, 11년 | Task 3·4 |
| `data/prices.db` 분리 | Task 2 |
| 원주가 저장 + 이벤트 테이블 분리 | Task 1·2 |
| 기준가 역산 검출, 임계값 0.02 | Task 1 |
| 소급 보정(이벤트 이전만) | Task 1 |
| KRX 종목별 조회 경로 삭제 | Task 7 |
| 창을 365 달력일로 | Task 6 |
| B < 365 → `⚠ 이력 불일치` 그룹 | Task 9 (판정은 `breakout_recency`, 표시는 `reporter`) |
| 카운트·AI 분석 제외 | Task 9 |
| 백필 재개 가능 | Task 4 (`loaded_dates` 건너뛰기) |
| `sync()`가 빈 DB에서 자동 백필하지 않음 | Task 4 |
| 저장소 없을 때 열화 | Task 7 (`load_bars` → `None`) |
| 전 종목 저장(우선주 포함) | Task 3 (필터 없음) |
| 로그인 클라이언트는 섹터용으로 잔존 | 변경 없음 |

## 구현 중 확인할 것

- **임계값 0.02의 아래쪽**: Task 4의 수동 확인 후, 검출된 이벤트의 계수 분포에서 0.5~2% 구간에 실제 이벤트가 있는지 훑는다. 있으면 임계값을 낮춘다.
- **검출 이벤트 상위 50건 육안 확인**: 스펙의 리스크 항목. 백필 완료 후 계수가 큰 순으로 50건을 뽑아 액면분할·병합·무상증자로 설명되는지 본다. 설명 안 되는 건이 있으면 보고한다.
- **실제 용량**: 6컬럼 기준 11년 실측치를 기록한다(스펙의 780MB는 10컬럼 외삽 상한).
