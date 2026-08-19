# 52주 신고가 돌파 신선도 지표 구현 플랜

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 52주 신고가 종목마다 "직전 신고가 이후 며칠 지났는가(A)"와 "이 가격을 마지막으로 웃돈 게 며칠 전인가(B)"를 산출해 리포트·AI 프롬프트에 노출한다.

**Architecture:** 순수 계산 모듈 `src/breakout_recency.py`(일봉 리스트 → 지표)와 취득 어댑터 `src/recency_source.py`(KRX 11년 수정주가 조회 → 계산 → `StockHigh` 채움)를 분리한다. `src/cli.py`의 `run`이 investing 신고가 목록을 만든 직후 어댑터를 호출한다. 원시 시계열은 저장하지 않고 산출된 값만 `new_highs`에 적재한다.

**Tech Stack:** Python, pydantic(StockHigh), SQLAlchemy(SQLite), pandas(KRX 응답), typer(cli), pytest.

**Spec:** [docs/superpowers/specs/2026-08-19-52w-high-recency-design.md](../specs/2026-08-19-52w-high-recency-design.md)

## Global Constraints

- 파이썬 실행: `.venv/bin/python` 사용(`python` 미존재 환경). 워크트리에 `.venv`가 없으면 먼저 `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`.
- 52주 창 `window = 250 거래일`. 조회 깊이 `years = 11`(10년 + 52주 워밍업).
- KRX 조회는 **수정주가**(`adjusted=True`)로만. 다년치 비교에서 액면분할·증자 미반영 시 판정이 무너진다.
- 모든 지표는 **달력 일수** 기준(거래일 수 아님).
- 순수 계산부(`src/breakout_recency.py`)는 네트워크·DB·로깅 설정에 의존하지 않는다. import도 표준 라이브러리와 `dataclasses`만.
- `KrxBlockedError`는 어댑터에서 **즉시 전파**한다(차단 상태에서 추가 요청 금지). 개별 종목의 다른 실패는 해당 종목만 `None` 처리하고 계속 진행한다.
- 네트워크 없는 단위테스트만 작성한다. 실제 KRX 호출은 Task 4 Step 5·Task 6 Step 4의 수동 검증뿐이며, 자격증명이 없으면 건너뛰고 그 사실을 커밋 메시지에 남긴다.
- 기존 리포트 화면의 `+x.x%` 숫자와 정렬 순서는 **당일 등락률**을 계속 의미해야 한다(Task 3에서 `change_pct`로 옮긴다).
- 라벨 버킷 경계: A는 `5일` / `30일` / `365일`, B의 10년 판정은 `3650일`.

---

### Task 1: 순수 계산 모듈 — `compute_recency`

**Files:**
- Create: `src/breakout_recency.py`
- Test: `tests/test_breakout_recency.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Bar`: `date: datetime.date`, `high: float`
  - `@dataclass(frozen=True) class Recency`: `days_since_prev_new_high: int | None`, `days_since_price_above: int | None`, `history_span_days: int`, `prev_high_52w: float`, `today_high: float`
  - `def compute_recency(bars: list[Bar], window: int = 250) -> Recency | None`
    - `bars`는 날짜 오름차순, 마지막 원소가 오늘. `len(bars) < 2`이면 `None`.

- [ ] **Step 1: 실패 테스트 작성** (`tests/test_breakout_recency.py`)

```python
from datetime import date, timedelta

from src.breakout_recency import Bar, compute_recency


def _series(highs: list[float], end: date = date(2026, 8, 19)) -> list[Bar]:
    """마지막 원소가 end인 연속 일봉(날짜 오름차순)."""
    n = len(highs)
    return [Bar(date=end - timedelta(days=n - 1 - i), high=h) for i, h in enumerate(highs)]


def test_returns_none_when_fewer_than_two_bars():
    assert compute_recency([]) is None
    assert compute_recency([Bar(date=date(2026, 8, 19), high=100.0)]) is None


def test_all_time_high_has_no_price_above_day():
    # 단조 상승 → 오늘 고가를 웃돈 과거일이 없음
    bars = _series([100.0 + i for i in range(300)])
    r = compute_recency(bars, window=250)
    assert r is not None
    assert r.days_since_price_above is None
    assert r.today_high == 399.0


def test_daily_streak_gives_one_day_since_prev_new_high():
    # 매일 경신 → 직전 거래일도 그날의 52주 신고가였음
    bars = _series([100.0 + i for i in range(300)])
    r = compute_recency(bars, window=250)
    assert r is not None
    assert r.days_since_prev_new_high == 1


def test_staircase_recovery_splits_the_two_metrics():
    # index   0     : 옛 고점 300 (401일 전)
    # index 1~300   : 100 박스권
    # index 301     : 150 — 직전 250봉(51~300)이 모두 100이므로 그날이 52주 신고가
    # index 302~400 : 120 (150 아래)
    # index 401     : 오늘 250 — 옛 고점 300은 못 넘었지만 52주 신고가
    highs = [300.0] + [100.0] * 300 + [150.0] + [120.0] * 99 + [250.0]
    bars = _series(highs)
    r = compute_recency(bars, window=250)
    assert r is not None
    assert r.days_since_prev_new_high == 100   # A: index 301까지
    assert r.days_since_price_above == 401     # B: index 0까지
    assert r.prev_high_52w == 150.0


def test_price_above_day_is_the_most_recent_one():
    # 과거에 오늘 고가보다 높은 날이 있으면 그중 가장 최근 날짜까지의 일수
    highs = [300.0] + [100.0] * 400 + [250.0]
    bars = _series(highs)
    r = compute_recency(bars, window=250)
    assert r is not None
    assert r.days_since_price_above == 401  # index 0 → 오늘까지 401일


def test_prev_high_52w_is_zero_when_fewer_than_window_bars():
    bars = _series([100.0] * 100 + [150.0])
    r = compute_recency(bars, window=250)
    assert r is not None
    assert r.prev_high_52w == 0.0
    assert r.days_since_prev_new_high is None  # 워밍업 부족


def test_prev_high_52w_uses_the_last_window_bars():
    highs = [500.0] + [100.0] * 250 + [150.0]
    bars = _series(highs)
    r = compute_recency(bars, window=250)
    assert r is not None
    assert r.prev_high_52w == 100.0  # 직전 250봉만 봄 (500은 창 밖)


def test_history_span_days_spans_first_to_last_bar():
    bars = _series([100.0] * 11)
    r = compute_recency(bars, window=250)
    assert r is not None
    assert r.history_span_days == 10
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_breakout_recency.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.breakout_recency'`

- [ ] **Step 3: 최소 구현** (`src/breakout_recency.py`)

```python
"""52주 신고가의 '돌파 신선도' 계산 — 순수 함수.

네트워크·DB에 의존하지 않는다. 입력은 일봉 리스트, 출력은 지표 값뿐이다.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Bar:
    """하루치 일봉 — 이 계산에 필요한 것은 날짜와 고가뿐이다."""

    date: date
    high: float


@dataclass(frozen=True)
class Recency:
    days_since_prev_new_high: int | None   # A: 직전 52주 신고가일로부터의 일수
    days_since_price_above: int | None     # B: 오늘 고가 이상이었던 마지막 날로부터의 일수
    history_span_days: int                 # 확보된 이력 길이
    prev_high_52w: float                   # 직전 window 봉의 최고 고가 (0.0 = 봉 부족)
    today_high: float


def compute_recency(bars: list[Bar], window: int = 250) -> Recency | None:
    """bars(날짜 오름차순, 마지막이 오늘)에서 A·B를 계산한다."""
    if len(bars) < 2:
        return None

    today = bars[-1]
    past = bars[:-1]

    # B: 오늘 고가 이상이었던 가장 최근 과거일. rolling window가 필요 없으므로
    #    확보된 이력 전체를 본다.
    days_since_price_above: int | None = None
    for bar in reversed(past):
        if bar.high >= today.high:
            days_since_price_above = (today.date - bar.date).days
            break

    # A: 그날 자체가 52주 신고가였던 가장 최근 과거일.
    #    앞쪽 window개 봉은 rolling max 워밍업으로 소비된다.
    days_since_prev_new_high: int | None = None
    for j in range(len(past) - 1, window - 1, -1):
        prior_max = max(b.high for b in past[j - window:j])
        if past[j].high >= prior_max:
            days_since_prev_new_high = (today.date - past[j].date).days
            break

    prev_high_52w = max(b.high for b in past[-window:]) if len(past) >= window else 0.0

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
git commit -m "feat(recency): 돌파 신선도 순수 계산 모듈 추가"
```

---

### Task 2: 모델 필드 + DB 마이그레이션

**Files:**
- Modify: `src/models.py` (`StockHigh`)
- Modify: `src/db.py` (`NewHigh`, `Database.__init__`, `save_scan_result`, `get_scan_result_full`)
- Test: `tests/test_db.py` (테스트 추가)

**Interfaces:**
- Consumes: 없음
- Produces:
  - `StockHigh` 신규 필드 4개 — `days_since_prev_new_high: int | None = None`, `days_since_price_above: int | None = None`, `history_span_days: int | None = None`, `change_pct: float = 0.0`
  - `src.db._migrate_add_recency_columns(engine) -> None`

- [ ] **Step 1: 실패 테스트 작성** (`tests/test_db.py` 끝에 추가)

```python
def test_migration_helper_handles_missing_table(tmp_path):
    """new_highs 테이블이 없어도 마이그레이션은 조용히 반환한다."""
    from sqlalchemy import create_engine
    from src.db import _migrate_add_recency_columns

    engine = create_engine(f"sqlite:///{tmp_path}/empty.db")
    _migrate_add_recency_columns(engine)  # 예외가 나면 안 됨


def test_migration_adds_recency_columns_to_legacy_table(tmp_path):
    """구 스키마 테이블에 4개 컬럼이 추가되고, 두 번 실행해도 안전하다."""
    from sqlalchemy import create_engine, inspect, text
    from src.db import _migrate_add_recency_columns

    db_path = f"{tmp_path}/legacy.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE new_highs ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " scan_date DATE NOT NULL, ticker VARCHAR(10) NOT NULL,"
            " name VARCHAR(100) NOT NULL, market VARCHAR(10) NOT NULL,"
            " sector VARCHAR(50) NOT NULL, close_price FLOAT NOT NULL,"
            " high_52w FLOAT NOT NULL, prev_high_52w FLOAT NOT NULL,"
            " breakout_pct FLOAT NOT NULL, volume BIGINT NOT NULL,"
            " avg_volume_20d BIGINT NOT NULL)"
        ))

    _migrate_add_recency_columns(engine)
    _migrate_add_recency_columns(engine)  # 멱등

    cols = {c["name"] for c in inspect(engine).get_columns("new_highs")}
    assert {
        "days_since_prev_new_high", "days_since_price_above",
        "history_span_days", "change_pct",
    } <= cols


def test_scan_result_roundtrip_preserves_recency_fields(tmp_path):
    """저장 후 복원했을 때 신선도 필드가 그대로 살아온다."""
    from datetime import date
    from src.db import Database
    from src.models import ScanResult, StockHigh, MarketStats

    db = Database(url=f"sqlite:///{tmp_path}/rt.db")
    stock = StockHigh(
        ticker="005930", name="삼성전자", market="KOSPI", sector="전기전자",
        close_price=78500, high_52w=79000, prev_high_52w=77000,
        breakout_pct=2.6, volume=1000, avg_volume_20d=0,
        days_since_prev_new_high=1170, days_since_price_above=None,
        history_span_days=4017, change_pct=3.1,
    )
    result = ScanResult(
        scan_date=date(2026, 8, 19),
        stats=MarketStats(total_stocks=1, new_high_count=1, kospi_count=1),
        highs=[stock], sector_breakdown={"전기전자": [stock]},
    )
    db.save_scan_result(result)

    loaded = db.get_scan_result_full(date(2026, 8, 19))
    assert loaded is not None
    got = loaded.highs[0]
    assert got.days_since_prev_new_high == 1170
    assert got.days_since_price_above is None
    assert got.history_span_days == 4017
    assert got.change_pct == 3.1
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_db.py -q`
Expected: FAIL — `ImportError: cannot import name '_migrate_add_recency_columns' from 'src.db'`

- [ ] **Step 3: 모델 필드 추가** (`src/models.py`의 `StockHigh` 끝에)

```python
class StockHigh(BaseModel):
    """A single stock that hit a 52-week high."""

    ticker: str
    name: str
    market: str
    sector: str
    close_price: float
    high_52w: float
    prev_high_52w: float
    breakout_pct: float
    volume: int
    avg_volume_20d: int
    # -- 돌파 신선도 (이력 확보 실패 시 None) --
    days_since_prev_new_high: int | None = None   # A: 직전 신고가 이후 경과 일수
    days_since_price_above: int | None = None     # B: 오늘 고가를 마지막으로 웃돈 날 이후 경과 일수
    history_span_days: int | None = None          # 확보된 이력 길이
    change_pct: float = 0.0                       # 당일 등락률 (breakout_pct와 별개)
```

- [ ] **Step 4: DB 컬럼·마이그레이션·왕복 반영** (`src/db.py`)

`NewHigh` 클래스에 컬럼 4개를 추가한다:

```python
class NewHigh(Base):
    __tablename__ = "new_highs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_date = Column(Date, nullable=False, index=True)
    ticker = Column(String(10), nullable=False)
    name = Column(String(100), nullable=False)
    market = Column(String(10), nullable=False)
    sector = Column(String(50), nullable=False)
    close_price = Column(Float, nullable=False)
    high_52w = Column(Float, nullable=False)
    prev_high_52w = Column(Float, nullable=False)
    breakout_pct = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)
    avg_volume_20d = Column(BigInteger, nullable=False)
    days_since_prev_new_high = Column(Integer, nullable=True)
    days_since_price_above = Column(Integer, nullable=True)
    history_span_days = Column(Integer, nullable=True)
    change_pct = Column(Float, nullable=True)
```

파일 상단 import에 `text`를 추가한다 (`from sqlalchemy import (..., create_engine, delete, select, text)`).

`Base` 정의 아래에 마이그레이션 헬퍼를 추가한다:

```python
def _migrate_add_recency_columns(engine) -> None:
    """new_highs에 돌파 신선도 컬럼을 멱등 추가. 테이블이 없으면 조용히 반환."""
    from sqlalchemy import inspect

    insp = inspect(engine)
    try:
        existing = {col["name"] for col in insp.get_columns("new_highs")}
    except Exception:
        # 테이블이 아직 없음 — create_all이 컬럼까지 함께 만든다.
        return
    to_add = [
        ("days_since_prev_new_high", "INTEGER"),
        ("days_since_price_above", "INTEGER"),
        ("history_span_days", "INTEGER"),
        ("change_pct", "FLOAT"),
    ]
    with engine.begin() as conn:
        for name, sqltype in to_add:
            if name not in existing:
                conn.execute(text(f"ALTER TABLE new_highs ADD COLUMN {name} {sqltype}"))
```

`Database.__init__`에서 `create_all` 직후 호출한다:

```python
    def __init__(self, url: str = "sqlite:///data/scanner.db"):
        self.engine = create_engine(url)
        Base.metadata.create_all(self.engine)
        _migrate_add_recency_columns(self.engine)
```

`save_scan_result`의 `session.add(NewHigh(...))`에 4개 필드를 추가한다:

```python
                session.add(NewHigh(
                    scan_date=result.scan_date,
                    ticker=stock.ticker,
                    name=stock.name,
                    market=stock.market,
                    sector=stock.sector,
                    close_price=stock.close_price,
                    high_52w=stock.high_52w,
                    prev_high_52w=stock.prev_high_52w,
                    breakout_pct=stock.breakout_pct,
                    volume=stock.volume,
                    avg_volume_20d=stock.avg_volume_20d,
                    days_since_prev_new_high=stock.days_since_prev_new_high,
                    days_since_price_above=stock.days_since_price_above,
                    history_span_days=stock.history_span_days,
                    change_pct=stock.change_pct,
                ))
```

`get_scan_result_full`의 `StockHigh(...)` 복원에도 추가한다:

```python
            highs = [
                StockHigh(
                    ticker=r.ticker, name=r.name, market=r.market,
                    sector=r.sector, close_price=r.close_price,
                    high_52w=r.high_52w, prev_high_52w=r.prev_high_52w,
                    breakout_pct=r.breakout_pct, volume=r.volume,
                    avg_volume_20d=r.avg_volume_20d,
                    days_since_prev_new_high=r.days_since_prev_new_high,
                    days_since_price_above=r.days_since_price_above,
                    history_span_days=r.history_span_days,
                    change_pct=r.change_pct or 0.0,
                )
                for r in rows
            ]
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_db.py tests/test_models.py -q`
Expected: PASS

- [ ] **Step 6: 커밋**

```bash
git add src/models.py src/db.py tests/test_db.py
git commit -m "feat(recency): StockHigh 신선도 필드와 new_highs 마이그레이션 추가"
```

---

### Task 3: `change_pct` 분리 — 당일 등락률과 돌파율을 갈라놓기

이 작업은 **한 커밋 안에서 완결되어야 한다.** investing 쪽만 바꾸면 리포트가 `+0.0%`를 출력하는 중간 상태가 생긴다.

**Files:**
- Modify: `src/investing_high.py:143-162` (`build_highs`)
- Modify: `src/reporter.py:103` 및 `src/reporter.py:115-121` (표시·정렬)
- Test: `tests/test_investing_high.py`, `tests/test_reporter.py`

**Interfaces:**
- Consumes: Task 2의 `StockHigh.change_pct`
- Produces: `build_highs`가 `change_pct=row.change_pct`, `breakout_pct=0.0`으로 채움. 리포트는 `change_pct`로 표시·정렬.

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_investing_high.py` 끝에 추가:

```python
def test_build_highs_puts_daily_change_in_change_pct():
    """당일 등락률은 change_pct로, breakout_pct는 이력 확보 전까지 0.0."""
    from src.investing_high import InvestingHighRow, build_highs

    row = InvestingHighRow(
        name="삼성전자", ticker="005930",
        last_price=78500.0, change_pct=3.1, volume=1000,
    )
    highs = build_highs([(row, "005930", "KOSPI")], {}, {"005930": "전기전자"})

    assert highs[0].change_pct == 3.1
    assert highs[0].breakout_pct == 0.0
```

`tests/test_reporter.py` 끝에 추가:

```python
def test_report_shows_and_sorts_by_change_pct():
    """리포트의 +x.x%는 당일 등락률(change_pct)이고, 정렬도 그 기준이다."""
    from datetime import date
    from src.reporter import Reporter
    from src.models import ScanResult, StockHigh, MarketStats

    def _stock(ticker, name, change_pct):
        return StockHigh(
            ticker=ticker, name=name, market="KOSPI", sector="전기전자",
            close_price=1000, high_52w=1000, prev_high_52w=0.0,
            breakout_pct=0.0, volume=1, avg_volume_20d=0, change_pct=change_pct,
        )

    lows, highs_ = _stock("000001", "낮은종목", 1.0), _stock("000002", "높은종목", 9.0)
    result = ScanResult(
        scan_date=date(2026, 8, 19),
        stats=MarketStats(total_stocks=2, new_high_count=2, kospi_count=2),
        highs=[lows, highs_], sector_breakdown={"전기전자": [lows, highs_]},
    )

    text = Reporter(bot_token="", chat_id=0).format_report(result, [], [])

    assert "+9.0%" in text and "+1.0%" in text
    assert text.index("높은종목") < text.index("낮은종목")  # 등락률 내림차순
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_investing_high.py tests/test_reporter.py -q`
Expected: FAIL — `build_highs`가 `change_pct`를 채우지 않아 `assert highs[0].change_pct == 3.1`이 `0.0 == 3.1`로 실패

- [ ] **Step 3: `build_highs` 수정** (`src/investing_high.py`)

```python
def build_highs(
    matched: list[tuple[InvestingHighRow, str, str]],
    market_caps: dict[str, int],
    sector_map: dict[str, str],
) -> list[StockHigh]:
    highs: list[StockHigh] = []
    for row, ticker, market in matched:
        highs.append(StockHigh(
            ticker=ticker,
            name=row.name,
            market=market,
            sector=sector_map.get(ticker, "기타"),
            close_price=row.last_price,
            high_52w=row.last_price,
            prev_high_52w=0.0,
            # 돌파율은 이력을 확보한 뒤에야 알 수 있다 (recency_source.enrich_highs).
            breakout_pct=0.0,
            volume=row.volume,
            avg_volume_20d=0,
            change_pct=row.change_pct,
        ))
    return highs
```

- [ ] **Step 4: 리포트의 표시·정렬을 `change_pct`로 전환** (`src/reporter.py`)

AI 분석 섹션의 종목 줄(현재 `src/reporter.py:101-104`):

```python
                    lines.append(
                        f"▶ {link} | "
                        f"{stock.close_price:,.0f}원 | +{stock.change_pct:.1f}%"
                    )
```

전체 목록 섹션(현재 `src/reporter.py:114-121`):

```python
        lines.append("<b>■ 전체 52주 신고가 목록</b>")
        for stock in sorted(result.highs, key=lambda h: h.change_pct, reverse=True):
            link = _stock_link(stock.name, stock.ticker)
            lines.append(
                f"  {link} | {stock.close_price:,.0f}원 | "
                f"+{stock.change_pct:.1f}% | {escape(stock.sector)}"
            )
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_investing_high.py tests/test_reporter.py -q`
Expected: PASS

- [ ] **Step 6: 커밋**

```bash
git add src/investing_high.py src/reporter.py tests/test_investing_high.py tests/test_reporter.py
git commit -m "refactor(report): 당일 등락률을 change_pct로 분리해 돌파율과 구분"
```

---

### Task 4: 취득 어댑터 — `fetch_bars`

KRX BLD가 11년치를 한 번에 주는지는 확인되지 않았다(스펙의 리스크 항목). 수동 측정에 의존하는 대신, **역방향 청크 루프**로 두 경우를 모두 처리한다: 한 번에 다 오면 1콜로 끝나고, 잘리면 반환된 첫 거래일 직전까지 다시 요청한다. 빈 응답이 오면 그 지점이 상장 시점이므로 종료한다.

**Files:**
- Create: `src/recency_source.py`
- Test: `tests/test_recency_source.py`

**Interfaces:**
- Consumes: Task 1의 `Bar`
- Produces: `def fetch_bars(client, ticker: str, as_of: date, years: int = 11, max_calls: int = 4) -> list[Bar] | None`
  - `client.supports_history`가 False면 `None`
  - 반환 리스트는 날짜 오름차순, 중복 없음

- [ ] **Step 1: 실패 테스트 작성** (`tests/test_recency_source.py`)

```python
from datetime import date

import pandas as pd
import pytest

from src.recency_source import fetch_bars


class FakeClient:
    """get_market_ohlcv_by_date 호출을 기록하는 가짜 KRX 클라이언트."""

    def __init__(self, frames, supports_history=True):
        self._frames = list(frames)   # 호출 순서대로 반환할 DataFrame들
        self.supports_history = supports_history
        self.calls = []

    def get_market_ohlcv_by_date(self, fromdate, todate, ticker, adjusted=False):
        self.calls.append((fromdate, todate, ticker, adjusted))
        return self._frames.pop(0) if self._frames else pd.DataFrame()


def _frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    """[(YYYYMMDD, 고가)] → KRX 응답 모양(날짜 인덱스, '고가' 컬럼)."""
    idx = [d for d, _ in rows]
    return pd.DataFrame({"고가": [h for _, h in rows]}, index=idx)


def test_returns_none_when_client_has_no_history_support():
    client = FakeClient([], supports_history=False)
    assert fetch_bars(client, "005930", date(2026, 8, 19)) is None
    assert client.calls == []


def test_single_call_when_full_range_returned():
    frame = _frame([("20150820", 100.0), ("20260819", 200.0)])
    client = FakeClient([frame])

    bars = fetch_bars(client, "005930", date(2026, 8, 19), years=11)

    assert len(client.calls) == 1
    assert client.calls[0][3] is True          # adjusted=True 필수
    assert [b.date for b in bars] == [date(2015, 8, 20), date(2026, 8, 19)]


def test_sorts_ascending_regardless_of_response_order():
    # KRX는 최신순으로 주는 경우가 있다
    frame = _frame([("20260819", 200.0), ("20150820", 100.0)])
    client = FakeClient([frame])

    bars = fetch_bars(client, "005930", date(2026, 8, 19), years=11)

    assert [b.high for b in bars] == [100.0, 200.0]


def test_chunks_backwards_when_response_is_truncated():
    # 1차 응답이 최근 1년만 → 그 앞 구간을 다시 요청해 병합
    first = _frame([("20250820", 150.0), ("20260819", 200.0)])
    second = _frame([("20150820", 100.0), ("20250819", 140.0)])
    client = FakeClient([first, second])

    bars = fetch_bars(client, "005930", date(2026, 8, 19), years=11)

    assert len(client.calls) == 2
    assert client.calls[1][1] == "20250819"    # 1차 첫 거래일 직전까지
    assert [b.date for b in bars] == [
        date(2015, 8, 20), date(2025, 8, 19), date(2025, 8, 20), date(2026, 8, 19),
    ]


def test_stops_when_earlier_chunk_is_empty_newly_listed():
    # 상장 3년차 종목: 앞 구간을 물어봐도 빈 응답 → 그대로 종료
    first = _frame([("20230820", 100.0), ("20260819", 200.0)])
    client = FakeClient([first])   # 이후 호출은 빈 DataFrame

    bars = fetch_bars(client, "005930", date(2026, 8, 19), years=11)

    assert len(bars) == 2
    assert len(client.calls) == 2


def test_returns_none_on_empty_first_response():
    client = FakeClient([pd.DataFrame()])
    assert fetch_bars(client, "005930", date(2026, 8, 19)) is None


def test_returns_none_when_client_raises():
    class Boom:
        supports_history = True

        def get_market_ohlcv_by_date(self, *a, **k):
            raise ValueError("boom")

    assert fetch_bars(Boom(), "005930", date(2026, 8, 19)) is None


def test_propagates_krx_blocked_error():
    from src.krx_login_client import KrxBlockedError

    class Blocked:
        supports_history = True

        def get_market_ohlcv_by_date(self, *a, **k):
            raise KrxBlockedError("차단")

    with pytest.raises(KrxBlockedError):
        fetch_bars(Blocked(), "005930", date(2026, 8, 19))
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_recency_source.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.recency_source'`

- [ ] **Step 3: 최소 구현** (`src/recency_source.py`)

```python
"""KRX 일봉 이력 취득 어댑터 — 돌파 신선도 계산의 입력을 만든다.

계산 자체는 src.breakout_recency의 순수 함수가 담당한다. 이 모듈은
'어디서 어떻게 가져오는가'만 안다.
"""
from __future__ import annotations

from datetime import date, timedelta

from loguru import logger

from src.breakout_recency import Bar
from src.krx_login_client import KrxBlockedError


def _to_bars(df) -> list[Bar]:
    """KRX 응답 DataFrame(날짜 인덱스, '고가' 컬럼) → 날짜 오름차순 Bar 리스트."""
    if df is None or df.empty or "고가" not in df.columns:
        return []
    bars: list[Bar] = []
    for raw_date, row in df.iterrows():
        try:
            d = date(int(str(raw_date)[:4]), int(str(raw_date)[4:6]), int(str(raw_date)[6:8]))
            high = float(row["고가"])
        except (ValueError, TypeError):
            continue
        if high > 0:
            bars.append(Bar(date=d, high=high))
    bars.sort(key=lambda b: b.date)
    return bars


def fetch_bars(
    client,
    ticker: str,
    as_of: date,
    years: int = 11,
    max_calls: int = 4,
) -> list[Bar] | None:
    """as_of 기준 years년치 수정주가 일봉을 가져온다.

    한 번에 다 오면 1콜로 끝난다. 응답이 잘리면 반환된 첫 거래일 직전까지
    역방향으로 다시 요청한다. 빈 응답이 오면 그 지점을 상장 시점으로 보고 종료한다.
    supports_history가 False인 클라이언트에서는 None.
    """
    if not getattr(client, "supports_history", False):
        return None

    start = as_of - timedelta(days=int(365.25 * years))
    bars: list[Bar] = []
    cursor_end = as_of

    for _ in range(max_calls):
        if cursor_end < start:
            break
        try:
            df = client.get_market_ohlcv_by_date(
                start.strftime("%Y%m%d"), cursor_end.strftime("%Y%m%d"),
                ticker, adjusted=True,
            )
        except KrxBlockedError:
            raise
        except Exception as e:  # noqa: BLE001 — 개별 종목 실패는 스캔을 막지 않는다
            logger.warning(f"{ticker} 일봉 조회 실패: {type(e).__name__}: {e}")
            return bars or None

        chunk = _to_bars(df)
        if not chunk:
            break

        bars = chunk + bars
        # 요청 시작일 근처까지 왔으면 완료 (거래일 공백 감안해 7일 여유)
        if chunk[0].date <= start + timedelta(days=7):
            break
        cursor_end = chunk[0].date - timedelta(days=1)

    return bars or None
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_recency_source.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: 실제 KRX 응답으로 청크 동작 1회 확인 (수동, 네트워크 필요)**

`.env`에 `KRX_ID`/`KRX_PW`가 설정된 환경에서만 실행한다. 설정이 없으면 이 스텝을 건너뛰고 그 사실을 커밋 메시지에 남긴다.

```bash
.venv/bin/python -c "
from datetime import date
from src.config import Settings
from src.krx_client import create_krx_client
from src.recency_source import fetch_bars
s = Settings()
c = create_krx_client(krx_id=s.krx_id, krx_pw=s.krx_pw, krx_api_key=s.krx_api_key)
bars = fetch_bars(c, '005930', date.today())
print('bars:', len(bars), 'first:', bars[0].date, 'last:', bars[-1].date)
"
```

Expected: `first`가 약 11년 전(2015년경)이고 `bars`가 2,500개 이상. 1콜로 끝났는지 여러 콜이 필요했는지는 로그로 확인한다. 어느 쪽이든 코드는 정상이며, 콜 수만 기록해 둔다.

- [ ] **Step 6: 커밋**

```bash
git add src/recency_source.py tests/test_recency_source.py
git commit -m "feat(recency): KRX 11년 수정주가 일봉 취득 어댑터 추가"
```

---

### Task 5: `enrich_highs` — 지표를 `StockHigh`에 채우기

**Files:**
- Modify: `src/recency_source.py`
- Test: `tests/test_recency_source.py`

**Interfaces:**
- Consumes: Task 1의 `compute_recency`, Task 4의 `fetch_bars`, Task 2의 `StockHigh` 필드
- Produces: `def enrich_highs(client, highs: list[StockHigh], as_of: date, window: int = 250) -> None` — `highs`를 제자리에서 수정한다

- [ ] **Step 1: 실패 테스트 작성** (`tests/test_recency_source.py` 끝에 추가)

```python
def _stock(ticker="005930"):
    from src.models import StockHigh

    return StockHigh(
        ticker=ticker, name="테스트", market="KOSPI", sector="전기전자",
        close_price=100, high_52w=100, prev_high_52w=0.0,
        breakout_pct=0.0, volume=1, avg_volume_20d=0, change_pct=2.0,
    )


def test_enrich_highs_fills_metrics_and_normalizes_breakout(monkeypatch):
    """이력이 있으면 A·B·이력길이·직전고점·돌파율·오늘고가가 모두 채워진다."""
    from datetime import timedelta
    from src.breakout_recency import Bar
    from src import recency_source

    end = date(2026, 8, 19)
    # 300봉: 앞 299봉은 고가 100, 오늘 110 → 직전 250봉 최고 100, 돌파율 10%
    bars = [Bar(date=end - timedelta(days=299 - i), high=100.0) for i in range(299)]
    bars.append(Bar(date=end, high=110.0))
    monkeypatch.setattr(recency_source, "fetch_bars", lambda *a, **k: bars)

    stock = _stock()
    recency_source.enrich_highs(object(), [stock], end)

    assert stock.history_span_days == 299
    assert stock.days_since_price_above is None   # 110을 웃돈 과거일 없음
    assert stock.days_since_prev_new_high == 1    # 어제 봉도 그날의 52주 신고가
    assert stock.prev_high_52w == 100.0
    assert stock.breakout_pct == 10.0
    assert stock.high_52w == 110.0


def test_enrich_highs_leaves_stock_untouched_when_no_history(monkeypatch):
    """이력을 못 가져오면 지표는 None으로 남고 기존 값은 건드리지 않는다."""
    from src import recency_source

    monkeypatch.setattr(recency_source, "fetch_bars", lambda *a, **k: None)

    stock = _stock()
    recency_source.enrich_highs(object(), [stock], date(2026, 8, 19))

    assert stock.history_span_days is None
    assert stock.days_since_prev_new_high is None
    assert stock.breakout_pct == 0.0
    assert stock.change_pct == 2.0      # 당일 등락률은 보존
    assert stock.high_52w == 100        # 원래 값 보존


def test_enrich_highs_isolates_per_stock_failure(monkeypatch):
    """한 종목이 실패해도 나머지는 계속 처리된다."""
    from datetime import timedelta
    from src.breakout_recency import Bar
    from src import recency_source

    end = date(2026, 8, 19)
    good = [Bar(date=end - timedelta(days=1), high=100.0), Bar(date=end, high=110.0)]

    def fake_fetch(client, ticker, as_of, **k):
        if ticker == "000001":
            raise ValueError("boom")
        return good

    monkeypatch.setattr(recency_source, "fetch_bars", fake_fetch)

    bad, ok = _stock("000001"), _stock("000002")
    recency_source.enrich_highs(object(), [bad, ok], end)

    assert bad.history_span_days is None
    assert ok.history_span_days == 1


def test_enrich_highs_propagates_krx_blocked_error(monkeypatch):
    from src import recency_source
    from src.krx_login_client import KrxBlockedError

    def blocked(*a, **k):
        raise KrxBlockedError("차단")

    monkeypatch.setattr(recency_source, "fetch_bars", blocked)

    with pytest.raises(KrxBlockedError):
        recency_source.enrich_highs(object(), [_stock()], date(2026, 8, 19))
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_recency_source.py -q`
Expected: FAIL — `AttributeError: module 'src.recency_source' has no attribute 'enrich_highs'`

- [ ] **Step 3: 구현 추가** (`src/recency_source.py`)

파일 상단의 `from src.breakout_recency import Bar` 줄을 다음 두 줄로 교체한다:

```python
from src.breakout_recency import Bar, compute_recency
from src.models import StockHigh
```

파일 끝에 다음 함수를 추가한다:

```python
def enrich_highs(
    client,
    highs: list[StockHigh],
    as_of: date,
    window: int = 250,
) -> None:
    """highs 각 종목의 돌파 신선도를 계산해 제자리에서 채운다.

    이력을 못 가져온 종목은 지표를 None으로 남기고 기존 값을 보존한다.
    KrxBlockedError는 전파한다 — 차단 상태에서 추가 요청을 보내면 안 된다.
    """
    filled = 0
    for stock in highs:
        try:
            bars = fetch_bars(client, stock.ticker, as_of)
        except KrxBlockedError:
            logger.error(f"KRX 차단 감지 — 신선도 계산 중단 ({filled}/{len(highs)} 완료)")
            raise
        except Exception as e:  # noqa: BLE001 — 개별 종목 실패는 나머지를 막지 않는다
            logger.warning(f"{stock.ticker} 신선도 계산 실패: {type(e).__name__}: {e}")
            continue

        if not bars:
            continue

        recency = compute_recency(bars, window=window)
        if recency is None:
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

        # 52주 신고가라면 B는 최소 1년 이상이어야 한다. 아니면 investing 판정과
        # KRX 수정주가가 어긋난 것이므로 기록만 남기고 값은 그대로 쓴다.
        if recency.days_since_price_above is not None and recency.days_since_price_above < 365:
            logger.warning(
                f"{stock.name}({stock.ticker}) B={recency.days_since_price_above}일 — "
                f"52주 신고가와 불일치(수정주가 차이 가능성)"
            )
        filled += 1

    logger.info(f"돌파 신선도 산출: {filled}/{len(highs)}종목")
```

`fetch_bars`는 이미 개별 종목 예외를 흡수하지만, 위 `except Exception`은 테스트가 `fetch_bars` 자체를 교체하는 경우와 예상 밖의 예외까지 격리한다.

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_recency_source.py -q`
Expected: PASS (12 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/recency_source.py tests/test_recency_source.py
git commit -m "feat(recency): 신고가 종목에 돌파 신선도 지표 채우는 enrich_highs 추가"
```

---

### Task 6: 파이프라인 통합 — `cli.run`에서 호출

`run`은 무거운 의존성을 함수 안에서 import하고 네트워크·DB·OpenAI를 모두 거치므로, 이 저장소는 `tests/test_cli.py`에서 명령어 존재와 옵션만 검증한다(기존 5개 테스트 참고). 이 태스크도 그 관례를 따른다 — 배선의 검증은 전체 스위트 회귀 + Step 4의 수동 실행이다. 억지 단위테스트를 만들지 않는다.

**Files:**
- Modify: `src/cli.py:63-80` (`run`의 수집 분기)

**Interfaces:**
- Consumes: Task 5의 `enrich_highs`
- Produces: 없음 (배선만)

- [ ] **Step 1: import 추가** (`src/cli.py`의 `run` 안, 기존 지역 import 블록)

기존 블록:

```python
        from src.dart.cache import DartCache
        from src.collector import Collector
        from src.scanner import Scanner
        from src.investing_high import collect_investing_highs, InvestingFetchError, InvestingParseError
```

다음 두 줄을 뒤에 추가한다:

```python
        from src.recency_source import enrich_highs
        from src.krx_login_client import KrxBlockedError
```

- [ ] **Step 2: `collect_investing_highs` 뒤에 신선도 계산 삽입** (`src/cli.py`)

```python
        console.print("[dim]1-3/5 investing.com 52주 신고가 수집 중...[/dim]")
        try:
            highs, market_caps = collect_investing_highs(date_str, collector, corps)
        except (InvestingFetchError, InvestingParseError) as e:
            console.print(f"[red]investing 신고가 수집 실패: {e}[/red]")
            raise typer.Exit(code=1)

        console.print(f"[dim]1-3/5 돌파 신선도 계산 중... ({len(highs)}종목)[/dim]")
        try:
            enrich_highs(client, highs, scan_date)
        except KrxBlockedError:
            # 차단은 신선도 지표만 잃는다. 스캔·뉴스·AI·리포트는 그대로 진행한다.
            console.print("[yellow]KRX 차단으로 돌파 신선도 일부/전체 누락[/yellow]")

        result = scanner.build_scan_result(scan_date, highs, len(highs))
        db.save_scan_result(result)
```

- [ ] **Step 3: 전체 회귀 테스트**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS (기존 테스트 전부 + Task 1~5의 신규 테스트)

- [ ] **Step 4: 수동 실행 확인 (네트워크 + `.env` 필요)**

```bash
.venv/bin/python -m src.cli run --force
```

Expected: `1-3/5 돌파 신선도 계산 중... (N종목)` 출력 후 로그에 `돌파 신선도 산출: N/M종목`이 찍히고, 리포트 목록에 `🆕`/`🔁`/`🏔` 배지가 보인다(배지는 Task 7 완료 후). `.env`가 없으면 이 스텝을 건너뛰고 커밋 메시지에 남긴다.

- [ ] **Step 5: 커밋**

```bash
git add src/cli.py
git commit -m "feat(recency): run 파이프라인에 돌파 신선도 계산 연결"
```

---

### Task 7: 리포트 배지 + 그룹핑

**Files:**
- Modify: `src/reporter.py`
- Test: `tests/test_reporter.py`

**Interfaces:**
- Consumes: Task 2의 `StockHigh` 신선도 필드
- Produces: `_fmt_span`, `_recency_badge`, `_depth_badge`, `_recency_group`, `_stock_line` (모두 `src.reporter` 모듈 수준 함수)

- [ ] **Step 1: 실패 테스트 작성** (`tests/test_reporter.py` 끝에 추가)

```python
def _rstock(ticker="000001", name="종목", a=None, b=None, span=None, breakout=0.0):
    from src.models import StockHigh

    return StockHigh(
        ticker=ticker, name=name, market="KOSPI", sector="전기전자",
        close_price=1000, high_52w=1000, prev_high_52w=0.0,
        breakout_pct=breakout, volume=1, avg_volume_20d=0, change_pct=1.0,
        days_since_prev_new_high=a, days_since_price_above=b, history_span_days=span,
    )


def test_fmt_span_formats_years_months_days():
    from src.reporter import _fmt_span

    assert _fmt_span(1170) == "3년 2개월"
    assert _fmt_span(730) == "2년"
    assert _fmt_span(90) == "3개월"
    assert _fmt_span(12) == "12일"


def test_recency_badge_buckets():
    from src.reporter import _recency_badge

    assert _recency_badge(_rstock(a=1, span=4000)) == "🔁 신고가 행진"
    assert _recency_badge(_rstock(a=5, span=4000)) == "🔁 신고가 행진"
    assert _recency_badge(_rstock(a=6, span=4000)) == "🔁 6일 만"
    assert _recency_badge(_rstock(a=30, span=4000)) == "🔁 30일 만"
    assert _recency_badge(_rstock(a=90, span=4000)) == "🆕 3개월 만"
    assert _recency_badge(_rstock(a=1170, span=4000)) == "🆕 3년 2개월 만"


def test_recency_badge_when_no_prior_new_high_in_range():
    from src.reporter import _recency_badge

    badge = _recency_badge(_rstock(a=None, span=4000))
    assert "이상 만" in badge and "첫 돌파" in badge


def test_badges_are_omitted_without_history():
    from src.reporter import _recency_badge, _depth_badge

    stock = _rstock(a=None, b=None, span=None)
    assert _recency_badge(stock) is None
    assert _depth_badge(stock) is None


def test_depth_badge_buckets():
    from src.reporter import _depth_badge

    assert _depth_badge(_rstock(b=None, span=4000)) == "🏔 10년래 최고"
    assert _depth_badge(_rstock(b=None, span=1000)) == "🏔 상장 이후 최고"
    assert _depth_badge(_rstock(b=1170, span=4000)) == "🏔 3년 2개월 만의 최고가"


def test_recency_groups():
    from src.reporter import _recency_group

    assert _recency_group(_rstock(a=1000, span=4000)).startswith("장기 돌파")
    assert _recency_group(_rstock(a=None, span=4000)).startswith("장기 돌파")
    assert _recency_group(_rstock(a=100, span=4000)).startswith("중기 돌파")
    assert _recency_group(_rstock(a=10, span=4000)).startswith("신고가 행진")
    assert _recency_group(_rstock(span=None)) == "정보 없음"


def test_stock_line_shows_breakout_only_when_known():
    from src.reporter import _stock_line

    assert "돌파" not in _stock_line(_rstock(breakout=0.0))
    assert "↑1.4% 돌파" in _stock_line(_rstock(a=10, span=4000, breakout=1.4))


def _result(highs):
    from datetime import date
    from src.models import ScanResult, MarketStats

    return ScanResult(
        scan_date=date(2026, 8, 19),
        stats=MarketStats(total_stocks=len(highs), new_high_count=len(highs), kospi_count=len(highs)),
        highs=highs, sector_breakdown={"전기전자": highs},
    )


def test_report_groups_when_any_stock_has_metrics():
    from src.reporter import Reporter

    highs = [_rstock("000001", "장기", a=1000, span=4000), _rstock("000002", "미상", span=None)]
    text = Reporter(bot_token="", chat_id=0).format_report(_result(highs), [], [])

    assert "[장기 돌파 · 1년 이상 만]" in text
    assert "[정보 없음]" in text
    assert text.index("[장기 돌파 · 1년 이상 만]") < text.index("[정보 없음]")


def test_report_stays_flat_when_no_stock_has_metrics():
    from src.reporter import Reporter

    highs = [_rstock("000001", "가", span=None), _rstock("000002", "나", span=None)]
    text = Reporter(bot_token="", chat_id=0).format_report(_result(highs), [], [])

    assert "[정보 없음]" not in text
    assert "장기 돌파" not in text
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_reporter.py -q`
Expected: FAIL — `ImportError: cannot import name '_fmt_span' from 'src.reporter'`

- [ ] **Step 3: 헬퍼 구현** (`src/reporter.py`의 `_stock_link` 아래에 추가)

```python
# -- 돌파 신선도 라벨 -----------------------------------------------------

RECENCY_STREAK_DAYS = 5      # 이하 → 신고가 행진
RECENCY_SHORT_DAYS = 30      # 이하 → 단기 재돌파
RECENCY_LONG_DAYS = 365      # 초과 → 장기 돌파
DEPTH_DECADE_DAYS = 3650     # 이상 → 10년래 최고

GROUP_LONG = "장기 돌파 · 1년 이상 만"
GROUP_MID = "중기 돌파 · 1~12개월"
GROUP_STREAK = "신고가 행진 · 1개월 내 재돌파"
GROUP_UNKNOWN = "정보 없음"
GROUP_ORDER = (GROUP_LONG, GROUP_MID, GROUP_STREAK, GROUP_UNKNOWN)


def _fmt_span(days: int) -> str:
    """일수를 '3년 2개월' 같은 사람이 읽는 기간으로."""
    years, rest = divmod(days, 365)
    months = rest // 30
    if years and months:
        return f"{years}년 {months}개월"
    if years:
        return f"{years}년"
    if months:
        return f"{months}개월"
    return f"{days}일"


def _recency_badge(stock) -> str | None:
    """A(재돌파 간격) 배지. 이력이 없으면 None."""
    if stock.history_span_days is None:
        return None
    a = stock.days_since_prev_new_high
    if a is None:
        # 확보 구간 안에 직전 신고가가 없음 — 워밍업 52주를 뺀 값이 하한
        floor_days = max(stock.history_span_days - RECENCY_LONG_DAYS, 0)
        return f"🆕 {_fmt_span(floor_days)} 이상 만 (첫 돌파)"
    if a <= RECENCY_STREAK_DAYS:
        return "🔁 신고가 행진"
    if a <= RECENCY_SHORT_DAYS:
        return f"🔁 {a}일 만"
    return f"🆕 {_fmt_span(a)} 만"


def _depth_badge(stock) -> str | None:
    """B(갱신 깊이) 배지. 이력이 없으면 None."""
    span = stock.history_span_days
    if span is None:
        return None
    b = stock.days_since_price_above
    if b is None:
        return "🏔 10년래 최고" if span >= DEPTH_DECADE_DAYS else "🏔 상장 이후 최고"
    return f"🏔 {_fmt_span(b)} 만의 최고가"


def _recency_group(stock) -> str:
    """A 기준 그룹명."""
    if stock.history_span_days is None:
        return GROUP_UNKNOWN
    a = stock.days_since_prev_new_high
    if a is None or a > RECENCY_LONG_DAYS:
        return GROUP_LONG
    if a > RECENCY_SHORT_DAYS:
        return GROUP_MID
    return GROUP_STREAK


def _badges(stock) -> list[str]:
    return [b for b in (_recency_badge(stock), _depth_badge(stock)) if b]


def _stock_line(stock) -> str:
    """전체 목록의 한 줄."""
    parts = [
        f"  {_stock_link(stock.name, stock.ticker)}",
        f"{stock.close_price:,.0f}원",
        f"+{stock.change_pct:.1f}%",
    ]
    if stock.breakout_pct > 0:
        parts.append(f"↑{stock.breakout_pct:.1f}% 돌파")
    parts.extend(_badges(stock))
    parts.append(escape(stock.sector))
    return " | ".join(parts)
```

- [ ] **Step 4: `format_report`의 목록 섹션 교체** (`src/reporter.py`)

AI 분석 섹션의 종목 줄에도 배지를 붙인다:

```python
                    a = ai_map[stock.ticker]
                    link = _stock_link(stock.name, stock.ticker)
                    header = [
                        f"▶ {link}",
                        f"{stock.close_price:,.0f}원",
                        f"+{stock.change_pct:.1f}%",
                    ]
                    header.extend(_badges(stock))
                    lines.append(" | ".join(header))
```

전체 목록 섹션 전체를 다음으로 교체한다:

```python
        lines.append("<b>■ 전체 52주 신고가 목록</b>")
        ordered = sorted(result.highs, key=lambda h: h.change_pct, reverse=True)
        grouped: dict[str, list] = {g: [] for g in GROUP_ORDER}
        for stock in ordered:
            grouped[_recency_group(stock)].append(stock)

        has_metrics = any(h.history_span_days is not None for h in result.highs)
        for group in GROUP_ORDER:
            members = grouped[group]
            if not members:
                continue
            if has_metrics:
                lines.append(f"[{group}]")
            for stock in members:
                lines.append(_stock_line(stock))
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_reporter.py -q`
Expected: PASS

- [ ] **Step 6: 커밋**

```bash
git add src/reporter.py tests/test_reporter.py
git commit -m "feat(report): 돌파 신선도 배지와 신선도별 그룹핑 추가"
```

---

### Task 8: AI 프롬프트에 신선도 주입

**Files:**
- Modify: `src/ai_analyst.py:40-64` (`analyze_stock`)
- Test: `tests/test_ai_analyst.py`

**Interfaces:**
- Consumes: Task 2의 `StockHigh` 신선도 필드
- Produces: `def _recency_prompt_line(stock) -> str` — 지표가 없으면 빈 문자열

- [ ] **Step 1: 실패 테스트 작성** (`tests/test_ai_analyst.py` 끝에 추가)

```python
def _astock(a=None, b=None, span=None):
    from src.models import StockHigh

    return StockHigh(
        ticker="005930", name="삼성전자", market="KOSPI", sector="전기전자",
        close_price=78500, high_52w=79000, prev_high_52w=77000,
        breakout_pct=2.6, volume=1000, avg_volume_20d=0, change_pct=3.1,
        days_since_prev_new_high=a, days_since_price_above=b, history_span_days=span,
    )


def test_recency_prompt_line_is_empty_without_history():
    from src.ai_analyst import _recency_prompt_line

    assert _recency_prompt_line(_astock()) == ""


def test_recency_prompt_line_describes_both_axes():
    from src.ai_analyst import _recency_prompt_line

    line = _recency_prompt_line(_astock(a=1170, b=1500, span=4000))
    assert "돌파 신선도:" in line
    assert "1170일 전" in line
    assert "1500일 전" in line


def test_recency_prompt_line_marks_all_time_high():
    from src.ai_analyst import _recency_prompt_line

    line = _recency_prompt_line(_astock(a=1170, b=None, span=4000))
    assert "최고 수준" in line


def test_recency_prompt_line_marks_first_breakout():
    from src.ai_analyst import _recency_prompt_line

    line = _recency_prompt_line(_astock(a=None, b=None, span=4000))
    assert "첫 돌파" in line
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_ai_analyst.py -q`
Expected: FAIL — `ImportError: cannot import name '_recency_prompt_line' from 'src.ai_analyst'`

- [ ] **Step 3: 헬퍼 구현** (`src/ai_analyst.py`의 `_sanitize` 아래에 추가)

```python
def _recency_prompt_line(stock: StockHigh) -> str:
    """돌파 신선도를 프롬프트 한 줄로. 지표가 없으면 빈 문자열."""
    if stock.history_span_days is None:
        return ""

    if stock.days_since_prev_new_high is None:
        first = "조회 구간 내 직전 신고가 없음(첫 돌파)"
    else:
        first = f"직전 신고가 {stock.days_since_prev_new_high}일 전"

    if stock.days_since_price_above is None:
        second = "현재가는 확보된 이력 전체에서 최고 수준"
    else:
        second = f"현재가를 마지막으로 웃돈 시점은 {stock.days_since_price_above}일 전"

    return f"돌파 신선도: {first} / {second}\n"
```

- [ ] **Step 4: 프롬프트에 주입** (`src/ai_analyst.py`의 `analyze_stock`)

`prompt` f-string의 종목 정보 블록을 다음으로 교체한다. `breakout_pct`는 이력을 확보한 경우에만 의미가 있으므로 조건부로 만든다.

```python
        breakout_note = (
            f" (직전 52주 고점 대비 +{stock.breakout_pct:.1f}%)"
            if stock.breakout_pct > 0 else ""
        )

        prompt = f"""다음 종목이 52주 신고가를 기록했습니다. 관련 뉴스를 바탕으로 아래 형식에 맞춰 분석해주세요.

종목: {stock.name} ({stock.ticker})
시장: {stock.market} / 섹터: {stock.sector}
종가: {stock.close_price:,.0f}원 (당일 {stock.change_pct:+.1f}%)
52주 신고가: {stock.high_52w:,.0f}원{breakout_note}
거래량: {stock.volume:,}주
{_recency_prompt_line(stock)}
최근 뉴스:
{news_text}

아래 형식으로 한국어 분석을 작성해주세요:
[상승 원인] 이 종목이 52주 신고가를 기록한 핵심 원인을 2~3문장으로 구체적으로 설명 (실적, 수주, 정책, 수급 등 구체적 이유 포함)
[핵심 뉴스] 가장 관련도 높은 뉴스 1~2개를 한 줄씩 요약
[투자 포인트] 향후 주가에 영향을 줄 수 있는 핵심 변수 1~2개를 간단히 언급"""
```

- [ ] **Step 5: 전체 테스트 통과 확인**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS (전체 통과)

- [ ] **Step 6: 커밋**

```bash
git add src/ai_analyst.py tests/test_ai_analyst.py
git commit -m "feat(ai): 돌파 신선도를 AI 분석 프롬프트에 주입"
```

---

### Task 9: README 갱신

**Files:**
- Modify: `README.md` (주요 기능 섹션)

**Interfaces:**
- Consumes: 없음
- Produces: 없음

- [ ] **Step 1: 주요 기능 목록에 한 줄 추가** (`README.md`)

```markdown
- 돌파 신선도 판별: 직전 신고가 이후 경과 기간(재돌파 간격)과 몇 년 만의 최고가인지(갱신 깊이)를 함께 표기
```

- [ ] **Step 2: 커밋**

```bash
git add README.md
git commit -m "docs: 돌파 신선도 지표를 주요 기능에 추가"
```

---

## 스펙 커버리지 확인

| 스펙 항목 | 구현 위치 |
|---|---|
| A·B 정의와 계산 | Task 1 |
| 온디맨드 조회, 캐시 테이블 없음 | Task 4 |
| 순수 함수 분리 | Task 1(계산) / Task 4·5(취득) |
| 11년 조회 + 상장 시점 폴백 | Task 4 (역방향 청크 루프) |
| 수정주가 사용 | Task 4 (`adjusted=True`, 테스트로 검증) |
| 오늘 봉 포함 → `high_52w`·`prev_high_52w`·`breakout_pct` 정상화 | Task 5 |
| `supports_history=False` 열화 | Task 4(`None` 반환) / Task 7(배지 생략) |
| `change_pct` 분리 | Task 3 |
| 모델 필드 + DB 마이그레이션 | Task 2 |
| 라벨 버킷 | Task 7 |
| 리포트 그룹핑 | Task 7 |
| AI 프롬프트 주입 | Task 8 |
| B < 365일 경고 로그 | Task 5 |
| `KrxBlockedError` 즉시 전파 | Task 4·5(전파) / Task 6(cli에서 경고 후 진행) |
