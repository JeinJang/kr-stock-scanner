# AI HW / 빅테크 시총 비율 지표 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** AI HW 종목군 시총 합계 ÷ 빅테크 종목군 시총 합계 비율을 추적·시각화하고, 0.8 경고선 기준 상태를 텔레그램 채널로 전송하는 `aihw` CLI 명령 + 얇은 Claude 스킬을 만든다.

**Architecture:** 신규 모듈 `src/aihw/`에 fetcher(yfinance 수집) → db(SQLite 스냅샷) → compute(순수 계산) → report(plotly HTML/PNG + 캡션) → pipeline(오케스트레이션) 구조. CLI `aihw` 명령과 기존 `run` 파이프라인 마지막 단계에 통합. 계산 로직은 전부 순수 함수로 두고 네트워크는 얇은 레이어로 격리한다.

**Tech Stack:** Python 3.11+, yfinance, pandas, plotly + kaleido(PNG), SQLAlchemy(SQLite), typer, python-telegram-bot, jinja2, pytest

**Spec:** `docs/superpowers/specs/2026-08-23-aihw-bigtech-ratio-design.md`

## Global Constraints

- 시총은 **A 방식 근사**: 현재 상장주식수 × 과거 수정종가. 삼성전자·SK하이닉스는 일별 USD/KRW 환율(`KRW=X`)로 달러 환산.
- 부분 성공 불허: cap 종목(11개) 중 하나라도 수집 실패 시 명시적 예외로 중단.
- DB 규칙: `snapshot` 행은 `backfill`로 덮어쓰지 않는다. `snapshot`은 무엇이든 덮어쓴다.
- 텔레그램 캡션은 1,024자 이내. 비율 ≥ threshold(0.8)이면 첫 줄에 ⚠️.
- 기존 코드 컨벤션 준수: `from __future__ import annotations`, loguru logger, pydantic v2 모델, SQLAlchemy DeclarativeBase 패턴 (`src/market_data/db.py` 참고).
- 모든 커밋 메시지 끝에 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` 추가.
- 테스트 실행: 저장소 루트에서 `.venv/bin/pytest` (또는 활성화된 venv에서 `pytest`).

---

### Task 1: 의존성 및 설정 (config)

**Files:**
- Modify: `pyproject.toml` (dependencies에 2줄 추가)
- Modify: `src/config.py`
- Modify: `config.yaml`
- Test: `tests/test_config.py` (기존 파일에 테스트 추가)

**Interfaces:**
- Produces: `AihwSection` (pydantic) — 필드: `ai_hw_tickers: dict[str, str]`, `big_tech_tickers: dict[str, str]`, `benchmarks: list[str]`, `base_date: str`, `threshold: float`, `report_dir: str`, `auto_send: bool`. `ScannerConfig.aihw` 로 접근. `Settings.aihw_telegram_chat_id: int` (0이면 미설정).

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_config.py` 끝에 추가:

```python
class TestAihwSection:
    def test_default_aihw_config(self):
        config = ScannerConfig()
        assert config.aihw.threshold == 0.8
        assert config.aihw.base_date == "2026-01-10"
        assert config.aihw.auto_send is True
        assert "NVDA" in config.aihw.ai_hw_tickers
        assert "005930.KS" in config.aihw.ai_hw_tickers
        assert "MSFT" in config.aihw.big_tech_tickers
        assert config.aihw.benchmarks == ["SPY", "RSP"]
        assert config.aihw.report_dir == "reports"

    def test_aihw_config_from_yaml(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "aihw:\n  threshold: 0.75\n  auto_send: false\n",
            encoding="utf-8",
        )
        config = load_scanner_config(yaml_file)
        assert config.aihw.threshold == 0.75
        assert config.aihw.auto_send is False
        # 나머지는 기본값 유지
        assert "NVDA" in config.aihw.ai_hw_tickers
```

(파일 상단 import에 `ScannerConfig`, `load_scanner_config`가 이미 없으면 추가.)

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_config.py -v -k Aihw`
Expected: FAIL — `AttributeError: 'ScannerConfig' object has no attribute 'aihw'`

- [ ] **Step 3: 구현** — `src/config.py`의 `RelatedSection` 뒤에 추가:

```python
class AihwSection(BaseModel):
    ai_hw_tickers: dict[str, str] = {
        "NVDA": "엔비디아",
        "AVGO": "브로드컴",
        "005930.KS": "삼성전자",
        "000660.KS": "SK하이닉스",
        "MU": "마이크론",
        "SNDK": "샌디스크",
    }
    big_tech_tickers: dict[str, str] = {
        "AMZN": "아마존",
        "TSLA": "테슬라",
        "MSFT": "MS",
        "META": "메타",
        "GOOGL": "구글",
    }
    benchmarks: list[str] = ["SPY", "RSP"]
    base_date: str = "2026-01-10"
    threshold: float = 0.8
    report_dir: str = "reports"
    auto_send: bool = True
```

`ScannerConfig`에 `aihw: AihwSection = AihwSection()` 필드 추가.
`Settings`에 `aihw_telegram_chat_id: int = 0` 필드 추가 (`telegram_chat_id` 아래).

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_config.py -v`
Expected: 전체 PASS (기존 테스트 포함)

- [ ] **Step 5: config.yaml과 pyproject.toml 갱신**

`config.yaml` 끝에 추가 (기본값과 동일하지만 사용자가 종목을 바꾸는 자리임을 명시):

```yaml
aihw:
  ai_hw_tickers:
    NVDA: 엔비디아
    AVGO: 브로드컴
    005930.KS: 삼성전자
    000660.KS: SK하이닉스
    MU: 마이크론
    SNDK: 샌디스크
  big_tech_tickers:
    AMZN: 아마존
    TSLA: 테슬라
    MSFT: MS
    META: 메타
    GOOGL: 구글
  benchmarks: ["SPY", "RSP"]
  base_date: "2026-01-10"
  threshold: 0.8
  report_dir: "reports"
  auto_send: true
```

`pyproject.toml` `dependencies`에 추가:

```toml
    "yfinance>=0.2.40",
    "kaleido>=0.2.1",
```

설치: `pip install -e ".[dev]"`

- [ ] **Step 6: 커밋**

```bash
git add pyproject.toml src/config.py config.yaml tests/test_config.py
git commit -m "feat(aihw): 설정 섹션 및 yfinance/kaleido 의존성 추가"
```

---

### Task 2: 모델 + compute — 합산·비율·지수화

**Files:**
- Create: `src/aihw/__init__.py` (빈 파일)
- Create: `src/aihw/models.py`
- Create: `src/aihw/compute.py`
- Test: `tests/test_aihw_compute.py`

**Interfaces:**
- Produces:
  - `DailyCap(BaseModel)` — `date: date`, `ticker: str`, `close: float`, `shares: int | None`, `market_cap_usd: float | None`, `source: str` ("backfill" | "snapshot")
  - `AihwSeries(BaseModel)` — `dates: list[date]`, `ai_hw_total: list[float]`, `big_tech_total: list[float]`, `ratio: list[float]`, `indexed: dict[str, list[float]]` (키: "AI HW", "빅테크", 벤치마크 티커)
  - `build_series(caps: list[DailyCap], ai_hw: list[str], big_tech: list[str], benchmarks: list[str], base_date: date) -> AihwSeries`

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_aihw_compute.py`:

```python
from datetime import date

import pytest

from src.aihw.compute import build_series
from src.aihw.models import DailyCap


def _cap(d, ticker, cap_usd, close=100.0):
    return DailyCap(
        date=d, ticker=ticker, close=close, shares=1000,
        market_cap_usd=cap_usd, source="backfill",
    )


def _bench(d, ticker, close):
    return DailyCap(
        date=d, ticker=ticker, close=close, shares=None,
        market_cap_usd=None, source="backfill",
    )


D1, D2, D3 = date(2026, 1, 10), date(2026, 1, 11), date(2026, 1, 12)
AI_HW = ["NVDA", "MU"]
BIG_TECH = ["MSFT", "META"]


def _sample_caps():
    caps = []
    # D1: AI HW 합=300, 빅테크 합=500 → ratio 0.6
    caps += [_cap(D1, "NVDA", 200.0), _cap(D1, "MU", 100.0)]
    caps += [_cap(D1, "MSFT", 300.0), _cap(D1, "META", 200.0)]
    caps += [_bench(D1, "SPY", 100.0), _bench(D1, "RSP", 50.0)]
    # D2: AI HW 합=440, 빅테크 합=550 → ratio 0.8
    caps += [_cap(D2, "NVDA", 300.0), _cap(D2, "MU", 140.0)]
    caps += [_cap(D2, "MSFT", 330.0), _cap(D2, "META", 220.0)]
    caps += [_bench(D2, "SPY", 110.0), _bench(D2, "RSP", 51.0)]
    return caps


class TestBuildSeries:
    def test_group_totals_and_ratio(self):
        s = build_series(_sample_caps(), AI_HW, BIG_TECH, ["SPY", "RSP"], base_date=D1)
        assert s.dates == [D1, D2]
        assert s.ai_hw_total == [300.0, 440.0]
        assert s.big_tech_total == [500.0, 550.0]
        assert s.ratio == [pytest.approx(0.6), pytest.approx(0.8)]

    def test_indexed_to_base_date(self):
        s = build_series(_sample_caps(), AI_HW, BIG_TECH, ["SPY", "RSP"], base_date=D1)
        assert s.indexed["AI HW"] == [pytest.approx(100.0), pytest.approx(146.6667, abs=0.01)]
        assert s.indexed["빅테크"] == [pytest.approx(100.0), pytest.approx(110.0)]
        assert s.indexed["SPY"] == [pytest.approx(100.0), pytest.approx(110.0)]
        assert s.indexed["RSP"] == [pytest.approx(100.0), pytest.approx(102.0)]

    def test_base_date_on_holiday_uses_next_available(self):
        # base_date가 D1 이전(휴장)이면 첫 거래일(D1)을 기준으로 지수화
        s = build_series(
            _sample_caps(), AI_HW, BIG_TECH, ["SPY"], base_date=date(2026, 1, 9)
        )
        assert s.indexed["AI HW"][0] == pytest.approx(100.0)

    def test_date_with_missing_cap_ticker_is_dropped(self):
        caps = _sample_caps()
        # D3에는 NVDA가 빠짐 → D3 전체 제외
        caps += [_cap(D3, "MU", 150.0), _cap(D3, "MSFT", 340.0), _cap(D3, "META", 230.0)]
        s = build_series(caps, AI_HW, BIG_TECH, [], base_date=D1)
        assert s.dates == [D1, D2]

    def test_missing_benchmark_does_not_drop_date(self):
        caps = _sample_caps()
        # D2의 RSP 제거 → 날짜는 유지, RSP 지수만 해당일 생략 없이 이전값 유지 안 함
        caps = [c for c in caps if not (c.date == D2 and c.ticker == "RSP")]
        s = build_series(caps, AI_HW, BIG_TECH, ["SPY", "RSP"], base_date=D1)
        assert s.dates == [D1, D2]
        assert len(s.indexed["SPY"]) == 2
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_aihw_compute.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.aihw'`

- [ ] **Step 3: 구현**

`src/aihw/__init__.py`: 빈 파일 생성.

`src/aihw/models.py`:

```python
"""src/aihw/models.py

AI HW / 빅테크 시총 비율 지표의 데이터 모델.
"""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class DailyCap(BaseModel):
    """종목 1개의 일별 시총 스냅샷 (벤치마크·환율은 close만 유효)."""

    date: date
    ticker: str
    close: float  # 현지통화 종가
    shares: int | None = None
    market_cap_usd: float | None = None
    source: str = "backfill"  # "backfill" | "snapshot"


class AihwSeries(BaseModel):
    """그룹 합산·비율·지수화 시계열. 모든 리스트는 dates와 같은 길이."""

    dates: list[date]
    ai_hw_total: list[float]
    big_tech_total: list[float]
    ratio: list[float]
    indexed: dict[str, list[float]]  # "AI HW", "빅테크", 벤치마크 티커
```

`src/aihw/compute.py`:

```python
"""src/aihw/compute.py

순수 계산: 그룹 합산, 비율, 지수화. 네트워크·DB 접근 없음.
"""
from __future__ import annotations

from datetime import date

from src.aihw.models import AihwSeries, DailyCap


def _index_100(values: list[float], base_idx: int) -> list[float]:
    base = values[base_idx]
    return [v / base * 100.0 for v in values]


def build_series(
    caps: list[DailyCap],
    ai_hw: list[str],
    big_tech: list[str],
    benchmarks: list[str],
    base_date: date,
) -> AihwSeries:
    cap_tickers = set(ai_hw) | set(big_tech)
    by_date: dict[date, dict[str, DailyCap]] = {}
    for c in caps:
        by_date.setdefault(c.date, {})[c.ticker] = c

    # cap 종목이 전부 있는 날짜만 채택 (부분 데이터로 비율 왜곡 방지)
    dates = sorted(
        d for d, row in by_date.items()
        if cap_tickers <= {t for t, c in row.items() if c.market_cap_usd is not None}
    )
    if not dates:
        raise ValueError("cap 종목 전체가 존재하는 날짜가 없습니다")

    ai_hw_total = [sum(by_date[d][t].market_cap_usd for t in ai_hw) for d in dates]
    big_tech_total = [sum(by_date[d][t].market_cap_usd for t in big_tech) for d in dates]
    ratio = [a / b for a, b in zip(ai_hw_total, big_tech_total)]

    base_idx = next((i for i, d in enumerate(dates) if d >= base_date), 0)
    indexed: dict[str, list[float]] = {
        "AI HW": _index_100(ai_hw_total, base_idx),
        "빅테크": _index_100(big_tech_total, base_idx),
    }
    for bench in benchmarks:
        closes = [by_date[d][bench].close for d in dates if bench in by_date[d]]
        if len(closes) == len(dates):
            indexed[bench] = _index_100(closes, base_idx)

    return AihwSeries(
        dates=dates,
        ai_hw_total=ai_hw_total,
        big_tech_total=big_tech_total,
        ratio=ratio,
        indexed=indexed,
    )
```

참고: `test_missing_benchmark_does_not_drop_date`는 벤치마크 데이터가 일부 날짜에 없으면
해당 벤치마크 시리즈를 통째로 생략하는 동작을 검증한다 — 위 구현에서 `len(closes) == len(dates)`
조건이 그 역할을 한다. 이 경우 테스트의 `s.indexed["SPY"]`는 존재하고 RSP 키는 없어야 하므로
테스트 마지막 줄이 `assert len(s.indexed["SPY"]) == 2`이고 RSP는 검증하지 않는다.

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_aihw_compute.py -v`
Expected: 6개 전체 PASS

- [ ] **Step 5: 커밋**

```bash
git add src/aihw/ tests/test_aihw_compute.py
git commit -m "feat(aihw): 모델 및 그룹 합산·비율·지수화 계산 구현"
```

---

### Task 3: compute — 임계값 판정 + 요약 통계

**Files:**
- Modify: `src/aihw/models.py`
- Modify: `src/aihw/compute.py`
- Test: `tests/test_aihw_compute.py` (추가)

**Interfaces:**
- Produces:
  - `CompanySummary(BaseModel)` — `ticker: str`, `name: str`, `cap_usd: float`, `day_change_pct: float | None`
  - `GroupSummary(BaseModel)` — `name: str` ("AI HW" | "빅테크"), `total_usd: float`, `companies: list[CompanySummary]` (시총 내림차순)
  - `AihwSummary(BaseModel)` — `as_of: date`, `ratio: float`, `ratio_prev: float | None`, `change_pp: float | None`, `high_30d: float`, `low_30d: float`, `threshold: float`, `status: str | None`, `groups: list[GroupSummary]`
  - `threshold_status(ratio_today: float, ratio_prev: float | None, threshold: float) -> str | None` — "cross_up" | "cross_down" | "above" | None
  - `summarize(series: AihwSeries, caps: list[DailyCap], ai_hw: dict[str, str], big_tech: dict[str, str], threshold: float) -> AihwSummary`

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_aihw_compute.py`에 추가:

```python
from src.aihw.compute import summarize, threshold_status
from src.aihw.models import AihwSummary


class TestThresholdStatus:
    def test_cross_up(self):
        assert threshold_status(0.81, 0.79, 0.8) == "cross_up"

    def test_cross_down(self):
        assert threshold_status(0.79, 0.81, 0.8) == "cross_down"

    def test_above_no_cross(self):
        assert threshold_status(0.82, 0.81, 0.8) == "above"

    def test_below(self):
        assert threshold_status(0.75, 0.76, 0.8) is None

    def test_no_prev_above(self):
        assert threshold_status(0.85, None, 0.8) == "above"

    def test_no_prev_below(self):
        assert threshold_status(0.7, None, 0.8) is None


class TestSummarize:
    def test_summary_fields(self):
        caps = _sample_caps()
        series = build_series(caps, AI_HW, BIG_TECH, ["SPY", "RSP"], base_date=D1)
        summary = summarize(
            series, caps,
            ai_hw={"NVDA": "엔비디아", "MU": "마이크론"},
            big_tech={"MSFT": "MS", "META": "메타"},
            threshold=0.8,
        )
        assert summary.as_of == D2
        assert summary.ratio == pytest.approx(0.8)
        assert summary.ratio_prev == pytest.approx(0.6)
        assert summary.change_pp == pytest.approx(20.0)  # %p
        assert summary.high_30d == pytest.approx(0.8)
        assert summary.low_30d == pytest.approx(0.6)
        assert summary.status == "cross_up"

    def test_groups_sorted_by_cap_desc(self):
        caps = _sample_caps()
        series = build_series(caps, AI_HW, BIG_TECH, ["SPY"], base_date=D1)
        summary = summarize(
            series, caps,
            ai_hw={"NVDA": "엔비디아", "MU": "마이크론"},
            big_tech={"MSFT": "MS", "META": "메타"},
            threshold=0.8,
        )
        ai_group = summary.groups[0]
        assert ai_group.name == "AI HW"
        assert ai_group.total_usd == pytest.approx(440.0)
        assert [c.ticker for c in ai_group.companies] == ["NVDA", "MU"]
        # NVDA: D1 200 → D2 300 = +50%
        assert ai_group.companies[0].day_change_pct == pytest.approx(50.0)
        assert ai_group.companies[0].name == "엔비디아"
        big_group = summary.groups[1]
        assert big_group.name == "빅테크"
        assert [c.ticker for c in big_group.companies] == ["MSFT", "META"]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_aihw_compute.py -v -k "Threshold or Summarize"`
Expected: FAIL — `ImportError: cannot import name 'threshold_status'`

- [ ] **Step 3: 구현**

`src/aihw/models.py`에 추가:

```python
class CompanySummary(BaseModel):
    ticker: str
    name: str
    cap_usd: float
    day_change_pct: float | None = None


class GroupSummary(BaseModel):
    name: str  # "AI HW" | "빅테크"
    total_usd: float
    companies: list[CompanySummary]  # 시총 내림차순


class AihwSummary(BaseModel):
    as_of: date
    ratio: float
    ratio_prev: float | None
    change_pp: float | None  # 전일 대비 %p (ratio 차이 × 100)
    high_30d: float
    low_30d: float
    threshold: float
    status: str | None  # "cross_up" | "cross_down" | "above" | None
    groups: list[GroupSummary]
```

`src/aihw/compute.py`에 추가:

```python
def threshold_status(
    ratio_today: float, ratio_prev: float | None, threshold: float
) -> str | None:
    above = ratio_today >= threshold
    if ratio_prev is None:
        return "above" if above else None
    prev_above = ratio_prev >= threshold
    if above and not prev_above:
        return "cross_up"
    if not above and prev_above:
        return "cross_down"
    return "above" if above else None


def _group_summary(
    name: str,
    tickers: dict[str, str],
    today: dict[str, DailyCap],
    prev: dict[str, DailyCap] | None,
) -> "GroupSummary":
    companies = []
    for ticker, display_name in tickers.items():
        cap = today[ticker].market_cap_usd
        change = None
        if prev and ticker in prev and prev[ticker].market_cap_usd:
            change = (cap / prev[ticker].market_cap_usd - 1.0) * 100.0
        companies.append(CompanySummary(
            ticker=ticker, name=display_name, cap_usd=cap, day_change_pct=change,
        ))
    companies.sort(key=lambda c: c.cap_usd, reverse=True)
    return GroupSummary(
        name=name, total_usd=sum(c.cap_usd for c in companies), companies=companies,
    )


def summarize(
    series: AihwSeries,
    caps: list[DailyCap],
    ai_hw: dict[str, str],
    big_tech: dict[str, str],
    threshold: float,
) -> AihwSummary:
    as_of = series.dates[-1]
    prev_date = series.dates[-2] if len(series.dates) >= 2 else None

    by_date: dict[date, dict[str, DailyCap]] = {}
    for c in caps:
        by_date.setdefault(c.date, {})[c.ticker] = c
    today_row = by_date[as_of]
    prev_row = by_date.get(prev_date) if prev_date else None

    ratio = series.ratio[-1]
    ratio_prev = series.ratio[-2] if len(series.ratio) >= 2 else None
    last_30 = series.ratio[-30:]

    return AihwSummary(
        as_of=as_of,
        ratio=ratio,
        ratio_prev=ratio_prev,
        change_pp=(ratio - ratio_prev) * 100.0 if ratio_prev is not None else None,
        high_30d=max(last_30),
        low_30d=min(last_30),
        threshold=threshold,
        status=threshold_status(ratio, ratio_prev, threshold),
        groups=[
            _group_summary("AI HW", ai_hw, today_row, prev_row),
            _group_summary("빅테크", big_tech, today_row, prev_row),
        ],
    )
```

models import 라인을 `from src.aihw.models import AihwSeries, AihwSummary, CompanySummary, DailyCap, GroupSummary`로 갱신.

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_aihw_compute.py -v`
Expected: 전체 PASS

- [ ] **Step 5: 커밋**

```bash
git add src/aihw/models.py src/aihw/compute.py tests/test_aihw_compute.py
git commit -m "feat(aihw): 임계값 돌파 판정 및 요약 통계 구현"
```

---

### Task 4: DB — 일별 스냅샷 저장소

**Files:**
- Create: `src/aihw/db.py`
- Test: `tests/test_aihw_db.py`

**Interfaces:**
- Consumes: `DailyCap` (Task 2)
- Produces: `AihwDB` 클래스 —
  - `__init__(self, url: str = "sqlite:///data/aihw.db")`
  - `save_caps(self, rows: list[DailyCap]) -> int` (저장/갱신된 행 수 반환)
  - `load_caps(self, start: date, end: date) -> list[DailyCap]` (date, ticker 순 정렬)

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_aihw_db.py`:

```python
from datetime import date

from src.aihw.db import AihwDB
from src.aihw.models import DailyCap

D1, D2 = date(2026, 1, 10), date(2026, 1, 11)


def _row(d, ticker, cap, source):
    return DailyCap(
        date=d, ticker=ticker, close=100.0, shares=10,
        market_cap_usd=cap, source=source,
    )


def _make_db():
    return AihwDB(url="sqlite:///:memory:")


class TestAihwDB:
    def test_save_and_load(self):
        db = _make_db()
        n = db.save_caps([_row(D1, "NVDA", 100.0, "backfill"),
                          _row(D1, "SPY", None, "backfill")])
        assert n == 2
        rows = db.load_caps(D1, D1)
        assert len(rows) == 2
        assert rows[0].ticker == "NVDA"
        assert rows[0].market_cap_usd == 100.0

    def test_load_range_filters_dates(self):
        db = _make_db()
        db.save_caps([_row(D1, "NVDA", 100.0, "backfill"),
                      _row(D2, "NVDA", 110.0, "backfill")])
        rows = db.load_caps(D2, D2)
        assert len(rows) == 1
        assert rows[0].date == D2

    def test_backfill_does_not_overwrite_snapshot(self):
        db = _make_db()
        db.save_caps([_row(D1, "NVDA", 100.0, "snapshot")])
        db.save_caps([_row(D1, "NVDA", 999.0, "backfill")])
        rows = db.load_caps(D1, D1)
        assert rows[0].market_cap_usd == 100.0
        assert rows[0].source == "snapshot"

    def test_snapshot_overwrites_backfill(self):
        db = _make_db()
        db.save_caps([_row(D1, "NVDA", 100.0, "backfill")])
        db.save_caps([_row(D1, "NVDA", 105.0, "snapshot")])
        rows = db.load_caps(D1, D1)
        assert rows[0].market_cap_usd == 105.0
        assert rows[0].source == "snapshot"

    def test_backfill_overwrites_backfill(self):
        db = _make_db()
        db.save_caps([_row(D1, "NVDA", 100.0, "backfill")])
        db.save_caps([_row(D1, "NVDA", 101.0, "backfill")])
        rows = db.load_caps(D1, D1)
        assert rows[0].market_cap_usd == 101.0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_aihw_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.aihw.db'`

- [ ] **Step 3: 구현** — `src/aihw/db.py` (기존 `src/market_data/db.py` 패턴 준수):

```python
"""src/aihw/db.py

AI HW/빅테크 지표의 일별 시총 스냅샷 저장소 (data/aihw.db).
규칙: snapshot 행은 backfill로 덮어쓰지 않는다. snapshot은 무엇이든 덮어쓴다.
"""
from __future__ import annotations

from datetime import date

from sqlalchemy import BigInteger, Column, Date, Float, String, create_engine, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import DeclarativeBase, Session

from src.aihw.models import DailyCap


class AihwBase(DeclarativeBase):
    pass


class DailyCapRow(AihwBase):
    __tablename__ = "daily_caps"
    date = Column(Date, primary_key=True)
    ticker = Column(String(12), primary_key=True)
    close = Column(Float, nullable=False)
    shares = Column(BigInteger, nullable=True)
    market_cap_usd = Column(Float, nullable=True)
    source = Column(String(10), nullable=False)


class AihwDB:
    def __init__(self, url: str = "sqlite:///data/aihw.db"):
        self.engine = create_engine(url)
        AihwBase.metadata.create_all(self.engine)

    def save_caps(self, rows: list[DailyCap]) -> int:
        if not rows:
            return 0
        saved = 0
        with Session(self.engine) as session:
            for r in rows:
                stmt = sqlite_insert(DailyCapRow).values(
                    date=r.date, ticker=r.ticker, close=r.close,
                    shares=r.shares, market_cap_usd=r.market_cap_usd,
                    source=r.source,
                )
                set_ = {
                    "close": stmt.excluded.close,
                    "shares": stmt.excluded.shares,
                    "market_cap_usd": stmt.excluded.market_cap_usd,
                    "source": stmt.excluded.source,
                }
                if r.source == "backfill":
                    # backfill은 기존 snapshot을 건드리지 않는다
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["date", "ticker"], set_=set_,
                        where=(DailyCapRow.source != "snapshot"),
                    )
                else:
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["date", "ticker"], set_=set_,
                    )
                session.execute(stmt)
                saved += 1
            session.commit()
        return saved

    def load_caps(self, start: date, end: date) -> list[DailyCap]:
        with Session(self.engine) as session:
            rows = session.execute(
                select(DailyCapRow)
                .where(DailyCapRow.date >= start, DailyCapRow.date <= end)
                .order_by(DailyCapRow.date, DailyCapRow.ticker)
            ).scalars().all()
            return [
                DailyCap(
                    date=r.date, ticker=r.ticker, close=r.close,
                    shares=r.shares, market_cap_usd=r.market_cap_usd,
                    source=r.source,
                )
                for r in rows
            ]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_aihw_db.py -v`
Expected: 5개 전체 PASS

- [ ] **Step 5: 커밋**

```bash
git add src/aihw/db.py tests/test_aihw_db.py
git commit -m "feat(aihw): 일별 시총 SQLite 저장소 구현 (snapshot 우선 규칙)"
```

---

### Task 5: fetcher — yfinance 수집 + 환율 환산

**Files:**
- Create: `src/aihw/fetcher.py`
- Test: `tests/test_aihw_fetcher.py`

**Interfaces:**
- Consumes: `DailyCap` (Task 2)
- Produces:
  - `FetchError(Exception)` — 수집 실패 예외
  - `build_daily_caps(prices: pd.DataFrame, shares: dict[str, int], fx: pd.Series, cap_tickers: list[str], benchmark_tickers: list[str], snapshot_date: date | None) -> list[DailyCap]` — 순수 변환 함수. `prices`: index=날짜(DatetimeIndex), columns=티커, 값=수정종가. `fx`: index=날짜, 값=USD당 KRW. `.KS`로 끝나는 티커는 KRW→USD 환산. `snapshot_date`와 같은 날짜의 행은 source="snapshot", 나머지는 "backfill".
  - `fetch_all(cap_tickers: list[str], benchmark_tickers: list[str], start: date, end: date) -> list[DailyCap]` — 네트워크 포함 상위 함수. 내부에서 `_download_prices`, `_download_shares` 호출 후 `build_daily_caps` 위임.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_aihw_fetcher.py`:

```python
from datetime import date

import pandas as pd
import pytest

from src.aihw.fetcher import FetchError, build_daily_caps

IDX = pd.to_datetime(["2026-01-10", "2026-01-11", "2026-01-12"])


def _prices():
    return pd.DataFrame(
        {
            "NVDA": [100.0, 110.0, 120.0],
            "005930.KS": [70000.0, None, 72000.0],  # 1/11 한국 휴장
            "SPY": [500.0, 505.0, 510.0],
        },
        index=IDX,
    )


def _fx():
    # 1/11 환율 누락 → ffill로 1300 사용
    return pd.Series([1300.0, None, 1350.0], index=IDX)


SHARES = {"NVDA": 1000, "005930.KS": 5000}


class TestBuildDailyCaps:
    def test_usd_ticker_cap(self):
        caps = build_daily_caps(
            _prices(), SHARES, _fx(), ["NVDA", "005930.KS"], ["SPY"], None
        )
        nvda_d1 = next(c for c in caps if c.ticker == "NVDA" and c.date == date(2026, 1, 10))
        assert nvda_d1.market_cap_usd == pytest.approx(100.0 * 1000)
        assert nvda_d1.shares == 1000

    def test_krw_ticker_converted_with_ffilled_fx(self):
        caps = build_daily_caps(
            _prices(), SHARES, _fx(), ["NVDA", "005930.KS"], ["SPY"], None
        )
        # 1/11: 삼전 종가 ffill(70000), 환율 ffill(1300)
        s_d2 = next(c for c in caps if c.ticker == "005930.KS" and c.date == date(2026, 1, 11))
        assert s_d2.close == pytest.approx(70000.0)
        assert s_d2.market_cap_usd == pytest.approx(70000.0 * 5000 / 1300.0)

    def test_benchmark_rows_have_no_cap(self):
        caps = build_daily_caps(
            _prices(), SHARES, _fx(), ["NVDA", "005930.KS"], ["SPY"], None
        )
        spy = next(c for c in caps if c.ticker == "SPY" and c.date == date(2026, 1, 10))
        assert spy.market_cap_usd is None
        assert spy.shares is None
        assert spy.close == pytest.approx(500.0)

    def test_snapshot_date_marks_source(self):
        caps = build_daily_caps(
            _prices(), SHARES, _fx(), ["NVDA", "005930.KS"], ["SPY"],
            snapshot_date=date(2026, 1, 12),
        )
        assert all(
            c.source == ("snapshot" if c.date == date(2026, 1, 12) else "backfill")
            for c in caps
        )

    def test_missing_cap_ticker_column_raises(self):
        with pytest.raises(FetchError, match="MU"):
            build_daily_caps(_prices(), SHARES, _fx(), ["NVDA", "MU"], [], None)

    def test_missing_shares_raises(self):
        with pytest.raises(FetchError, match="005930.KS"):
            build_daily_caps(_prices(), {"NVDA": 1000}, _fx(), ["NVDA", "005930.KS"], [], None)

    def test_all_nan_cap_column_raises(self):
        prices = _prices()
        prices["NVDA"] = None
        with pytest.raises(FetchError, match="NVDA"):
            build_daily_caps(prices, SHARES, _fx(), ["NVDA"], [], None)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_aihw_fetcher.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.aihw.fetcher'`

- [ ] **Step 3: 구현** — `src/aihw/fetcher.py`:

```python
"""src/aihw/fetcher.py

yfinance로 종목 가격·상장주식수·환율을 수집해 일별 시총(USD)을 만든다.
A 방식 근사: 현재 상장주식수 × 과거 수정종가. .KS 종목은 KRW=X 환율로 달러 환산.
부분 성공 불허 — cap 종목 하나라도 실패하면 FetchError.
"""
from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
from loguru import logger

from src.aihw.models import DailyCap

FX_TICKER = "KRW=X"


class FetchError(Exception):
    """cap 종목 수집 실패 (부분 데이터로 비율을 계산하지 않기 위해 중단)."""


def build_daily_caps(
    prices: pd.DataFrame,
    shares: dict[str, int],
    fx: pd.Series,
    cap_tickers: list[str],
    benchmark_tickers: list[str],
    snapshot_date: date | None,
) -> list[DailyCap]:
    """가격/주식수/환율 → DailyCap 목록으로 변환하는 순수 함수."""
    for t in cap_tickers:
        if t not in prices.columns or prices[t].isna().all():
            raise FetchError(f"가격 데이터 없음: {t}")
        if t not in shares or not shares[t]:
            raise FetchError(f"상장주식수 없음: {t}")

    prices = prices.ffill()
    fx = fx.ffill()

    rows: list[DailyCap] = []
    for ts in prices.index:
        d = ts.date()
        source = "snapshot" if snapshot_date and d == snapshot_date else "backfill"
        for t in cap_tickers:
            close = prices.at[ts, t]
            if pd.isna(close):
                continue  # 시계열 시작부의 결측 (ffill 이전 구간)
            cap = float(close) * shares[t]
            if t.endswith(".KS"):
                rate = fx.get(ts)
                if pd.isna(rate):
                    continue
                cap = cap / float(rate)
            rows.append(DailyCap(
                date=d, ticker=t, close=float(close), shares=shares[t],
                market_cap_usd=cap, source=source,
            ))
        for t in benchmark_tickers:
            if t not in prices.columns:
                continue
            close = prices.at[ts, t]
            if pd.isna(close):
                continue
            rows.append(DailyCap(
                date=d, ticker=t, close=float(close), shares=None,
                market_cap_usd=None, source=source,
            ))
    return rows


def _download_prices(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """yf.download로 수정종가 DataFrame(index=날짜, columns=티커)을 받는다."""
    import yfinance as yf

    df = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if df is None or df.empty:
        raise FetchError("yfinance 가격 다운로드 결과가 비어 있음")
    close = df["Close"]
    if isinstance(close, pd.Series):  # 단일 티커
        close = close.to_frame(name=tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def _download_shares(tickers: list[str], retries: int = 3) -> dict[str, int]:
    """티커별 현재 상장주식수. 실패 시 재시도 후 FetchError."""
    import yfinance as yf

    shares: dict[str, int] = {}
    for t in tickers:
        n = None
        for attempt in range(retries):
            try:
                tk = yf.Ticker(t)
                n = tk.info.get("sharesOutstanding")
                if not n:
                    n = getattr(tk.fast_info, "shares", None)
                if n:
                    break
            except Exception as e:  # noqa: BLE001 — 재시도 후 FetchError로 변환
                logger.warning(f"{t} 주식수 조회 실패 (시도 {attempt + 1}): {e}")
            time.sleep(1.0)
        if not n:
            raise FetchError(f"상장주식수 조회 실패: {t}")
        shares[t] = int(n)
    return shares


def fetch_all(
    cap_tickers: list[str],
    benchmark_tickers: list[str],
    start: date,
    end: date,
) -> list[DailyCap]:
    """가격 + 주식수 + 환율 수집 → DailyCap 목록. 최신 거래일이 snapshot."""
    all_tickers = cap_tickers + benchmark_tickers + [FX_TICKER]
    prices = _download_prices(all_tickers, start, end)
    fx = prices[FX_TICKER]
    prices = prices.drop(columns=[FX_TICKER])
    shares = _download_shares(cap_tickers)
    snapshot_date = prices.index.max().date()
    logger.info(
        f"aihw 수집 완료: {len(prices)}일 × {len(prices.columns)}종목, "
        f"snapshot={snapshot_date}"
    )
    return build_daily_caps(
        prices, shares, fx, cap_tickers, benchmark_tickers, snapshot_date,
    )
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_aihw_fetcher.py -v`
Expected: 7개 전체 PASS

- [ ] **Step 5: yfinance 실동작 스모크 확인 (수동, 테스트 아님)**

Run: `python -c "
from datetime import date
from src.aihw.fetcher import fetch_all
caps = fetch_all(['NVDA', '005930.KS'], ['SPY'], date(2026, 8, 1), date(2026, 8, 22))
print(len(caps), caps[-1])
"`
Expected: 행 수와 마지막 DailyCap 출력 (NVDA 시총이 조 단위 USD인지, 삼전이 환산됐는지 눈으로 확인). 실패 시 yfinance API 변경 여부를 확인하고 `_download_prices`/`_download_shares`만 수정.

- [ ] **Step 6: 커밋**

```bash
git add src/aihw/fetcher.py tests/test_aihw_fetcher.py
git commit -m "feat(aihw): yfinance 시총 수집 및 원화 환산 구현"
```

---

### Task 6: report — 텔레그램 캡션 생성

**Files:**
- Create: `src/aihw/report.py`
- Test: `tests/test_aihw_report.py`

**Interfaces:**
- Consumes: `AihwSummary`, `GroupSummary`, `CompanySummary` (Task 3)
- Produces: `build_caption(summary: AihwSummary) -> str` — 1,024자 이내 보장

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_aihw_report.py`:

```python
from datetime import date

import pytest

from src.aihw.models import AihwSummary, CompanySummary, GroupSummary
from src.aihw.report import build_caption


def _summary(ratio=0.762, status=None):
    return AihwSummary(
        as_of=date(2026, 8, 22),
        ratio=ratio,
        ratio_prev=0.754,
        change_pp=0.8,
        high_30d=0.781,
        low_30d=0.74,
        threshold=0.8,
        status=status,
        groups=[
            GroupSummary(name="AI HW", total_usd=6.82e12, companies=[
                CompanySummary(ticker="NVDA", name="엔비디아", cap_usd=4.21e12, day_change_pct=1.2),
                CompanySummary(ticker="AVGO", name="브로드컴", cap_usd=1.35e12, day_change_pct=-0.5),
            ]),
            GroupSummary(name="빅테크", total_usd=8.95e12, companies=[
                CompanySummary(ticker="MSFT", name="MS", cap_usd=3.12e12, day_change_pct=0.4),
                CompanySummary(ticker="META", name="메타", cap_usd=1.52e12, day_change_pct=None),
            ]),
        ],
    )


class TestBuildCaption:
    def test_header_lines(self):
        caption = build_caption(_summary())
        lines = caption.split("\n")
        assert lines[0] == "📊 AI HW / 빅테크 비율: 76.2% (경고선 80%)"
        assert lines[1] == "전일 대비 +0.8%p · 30일 최고 78.1%"

    def test_group_and_company_lines(self):
        caption = build_caption(_summary())
        assert "[AI HW] $6.82T" in caption
        assert "· 엔비디아 $4.21T (+1.2%)" in caption
        assert "· 브로드컴 $1.35T (-0.5%)" in caption
        assert "[빅테크] $8.95T" in caption
        assert "· 메타 $1.52T (-)" in caption  # 전일 데이터 없음

    def test_warning_when_at_or_above_threshold(self):
        caption = build_caption(_summary(ratio=0.81, status="above"))
        assert caption.startswith("⚠️")

    def test_cross_up_marks_warning(self):
        caption = build_caption(_summary(ratio=0.80, status="cross_up"))
        assert caption.startswith("⚠️")
        assert "상향 돌파" in caption

    def test_under_1024_chars(self):
        # 실제 구성(11종목)보다 많은 20종목으로도 한도 이내인지 확인
        companies = [
            CompanySummary(ticker=f"T{i}", name=f"종목이름{i}", cap_usd=1.0e12, day_change_pct=1.23)
            for i in range(10)
        ]
        s = _summary()
        s.groups[0].companies = companies
        s.groups[1].companies = companies
        assert len(build_caption(s)) <= 1024
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_aihw_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.aihw.report'`

- [ ] **Step 3: 구현** — `src/aihw/report.py`:

```python
"""src/aihw/report.py

AI HW/빅테크 지표 산출물: 텔레그램 캡션, HTML 리포트, 공유용 PNG.
"""
from __future__ import annotations

from src.aihw.models import AihwSummary

CAPTION_LIMIT = 1024


def _fmt_t(cap_usd: float) -> str:
    return f"${cap_usd / 1e12:.2f}T"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "(-)"
    return f"({value:+.1f}%)"


def build_caption(summary: AihwSummary) -> str:
    warn = summary.status in ("above", "cross_up")
    head = "⚠️ " if warn else "📊 "
    lines = [
        f"{head}AI HW / 빅테크 비율: {summary.ratio * 100:.1f}% "
        f"(경고선 {summary.threshold * 100:.0f}%)"
    ]
    if summary.status == "cross_up":
        lines.append(f"🚨 경고선 {summary.threshold * 100:.0f}% 상향 돌파")
    elif summary.status == "cross_down":
        lines.append(f"경고선 {summary.threshold * 100:.0f}% 하향 이탈")

    parts = []
    if summary.change_pp is not None:
        parts.append(f"전일 대비 {summary.change_pp:+.1f}%p")
    parts.append(f"30일 최고 {summary.high_30d * 100:.1f}%")
    lines.append(" · ".join(parts))

    for group in summary.groups:
        lines.append("")
        lines.append(f"[{group.name}] {_fmt_t(group.total_usd)}")
        for c in group.companies:
            lines.append(f"· {c.name} {_fmt_t(c.cap_usd)} {_fmt_pct(c.day_change_pct)}")

    caption = "\n".join(lines)
    if len(caption) > CAPTION_LIMIT:
        caption = caption[: CAPTION_LIMIT - 1] + "…"
    return caption
```

주의: 테스트의 첫 줄 기대값은 `📊 ` 접두 포함 문자열이므로 `lines[0]` 비교가 성립하려면
head가 `"📊 "`(비경고)일 때 `"📊 AI HW / 빅테크 비율: 76.2% (경고선 80%)"`가 되어야 한다.

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_aihw_report.py -v`
Expected: 5개 전체 PASS

- [ ] **Step 5: 커밋**

```bash
git add src/aihw/report.py tests/test_aihw_report.py
git commit -m "feat(aihw): 텔레그램 캡션 생성 구현"
```

---

### Task 7: report — plotly 차트, HTML, PNG

**Files:**
- Modify: `src/aihw/report.py`
- Create: `src/aihw/templates/report.html`
- Test: `tests/test_aihw_report.py` (추가)

**Interfaces:**
- Consumes: `AihwSeries`, `AihwSummary` (Task 2·3)
- Produces:
  - `build_figures(series: AihwSeries, threshold: float) -> tuple[go.Figure, go.Figure]` — (비율 차트, 지수 차트)
  - `generate_html(series: AihwSeries, summary: AihwSummary, output_dir: str) -> str` — `reports/aihw-YYYY-MM-DD.html` 경로 반환
  - `generate_png(series: AihwSeries, summary: AihwSummary, output_dir: str) -> str` — `reports/aihw-YYYY-MM-DD.png` 경로 반환

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_aihw_report.py`에 추가:

```python
from src.aihw.models import AihwSeries
from src.aihw.report import build_figures, generate_html


def _series():
    return AihwSeries(
        dates=[date(2026, 1, 10), date(2026, 1, 11)],
        ai_hw_total=[6.0e12, 6.8e12],
        big_tech_total=[8.8e12, 8.9e12],
        ratio=[0.682, 0.764],
        indexed={
            "AI HW": [100.0, 113.3],
            "빅테크": [100.0, 101.1],
            "SPY": [100.0, 101.0],
            "RSP": [100.0, 100.4],
        },
    )


class TestBuildFigures:
    def test_ratio_figure_has_threshold_line(self):
        ratio_fig, index_fig = build_figures(_series(), threshold=0.8)
        # 경고선은 hline shape로 추가됨
        assert any(s.type == "line" for s in ratio_fig.layout.shapes)
        assert len(ratio_fig.data) == 1  # 비율 트레이스 1개

    def test_index_figure_has_all_series(self):
        _, index_fig = build_figures(_series(), threshold=0.8)
        names = {t.name for t in index_fig.data}
        assert names == {"AI HW", "빅테크", "SPY", "RSP"}


class TestGenerateHtml:
    def test_writes_file_with_table(self, tmp_path):
        path = generate_html(_series(), _summary(), output_dir=str(tmp_path))
        assert path.endswith("aihw-2026-08-22.html")
        html = open(path, encoding="utf-8").read()
        assert "엔비디아" in html
        assert "76.2%" in html
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_aihw_report.py -v -k "Figures or GenerateHtml"`
Expected: FAIL — `ImportError: cannot import name 'build_figures'`

- [ ] **Step 3: 구현** — `src/aihw/report.py`에 추가:

```python
import os
from pathlib import Path

import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader
from loguru import logger

from src.aihw.models import AihwSeries  # 상단 import에 병합

GROUP_COLORS = {"AI HW": "#f5a623", "빅테크": "#7b61c4", "SPY": "#4a90d9", "RSP": "#3d9970"}


def build_figures(series: AihwSeries, threshold: float) -> tuple[go.Figure, go.Figure]:
    ratio_fig = go.Figure()
    ratio_fig.add_trace(go.Scatter(
        x=series.dates, y=[r * 100 for r in series.ratio],
        mode="lines", name="AI HW / 빅테크",
        line=dict(color=GROUP_COLORS["AI HW"], width=2),
    ))
    ratio_fig.add_hline(
        y=threshold * 100, line_color="red", line_width=2,
        annotation_text=f"경고선 {threshold * 100:.0f}%",
    )
    ratio_fig.update_layout(
        title="AI HW 시총합 / 빅테크 시총합 비율 (%)",
        yaxis_title="%", template="plotly_white", height=420,
    )

    index_fig = go.Figure()
    for name, values in series.indexed.items():
        index_fig.add_trace(go.Scatter(
            x=series.dates, y=values, mode="lines", name=name,
            line=dict(color=GROUP_COLORS.get(name), width=2),
        ))
    base = series.dates[0].isoformat()
    index_fig.update_layout(
        title=f"시총 지수 비교 ({base} = 100)",
        template="plotly_white", height=420,
    )
    return ratio_fig, index_fig


def generate_html(series: AihwSeries, summary: AihwSummary, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"aihw-{summary.as_of.isoformat()}.html")
    ratio_fig, index_fig = build_figures(series, summary.threshold)

    env = Environment(loader=FileSystemLoader(str(Path(__file__).parent / "templates")))
    html = env.get_template("report.html").render(
        summary=summary,
        ratio_pct=f"{summary.ratio * 100:.1f}%",
        ratio_div=ratio_fig.to_html(full_html=False, include_plotlyjs="cdn"),
        index_div=index_fig.to_html(full_html=False, include_plotlyjs=False),
        fmt_t=_fmt_t,
        fmt_pct=_fmt_pct,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"HTML 리포트 저장: {path}")
    return path


def generate_png(series: AihwSeries, summary: AihwSummary, output_dir: str) -> str:
    from plotly.subplots import make_subplots

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"aihw-{summary.as_of.isoformat()}.png")
    ratio_fig, index_fig = build_figures(series, summary.threshold)

    combined = make_subplots(
        rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.12,
        subplot_titles=(
            "AI HW / 빅테크 시총 비율 (%)",
            f"시총 지수 비교 ({series.dates[0].isoformat()} = 100)",
        ),
    )
    for trace in ratio_fig.data:
        combined.add_trace(trace, row=1, col=1)
    combined.add_hline(y=summary.threshold * 100, line_color="red", line_width=2, row=1, col=1)
    for trace in index_fig.data:
        combined.add_trace(trace, row=2, col=1)
    combined.update_layout(
        template="plotly_white", height=900, width=1000,
        title=f"AI HW / 빅테크 고점 지표 — {summary.as_of.isoformat()}",
    )
    combined.write_image(path, scale=2)
    logger.info(f"PNG 저장: {path}")
    return path
```

`src/aihw/templates/report.html`:

```html
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>AI HW / 빅테크 비율 — {{ summary.as_of.isoformat() }}</title>
<style>
  body { font-family: -apple-system, "Apple SD Gothic Neo", sans-serif;
         max-width: 1080px; margin: 24px auto; padding: 0 16px; color: #222; }
  h1 { font-size: 22px; }
  .ratio { font-size: 34px; font-weight: 700;
           color: {{ '#c0392b' if summary.ratio >= summary.threshold else '#222' }}; }
  table { border-collapse: collapse; width: 100%; margin: 16px 0; }
  th, td { border-bottom: 1px solid #e0e0e0; padding: 8px 12px; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  .neg { color: #c0392b; } .pos { color: #1e7e34; }
  .group-row { background: #f6f6f6; font-weight: 700; }
</style>
</head>
<body>
<h1>AI HW / 빅테크 시총 비율 — {{ summary.as_of.isoformat() }}</h1>
<p class="ratio">{{ ratio_pct }} <span style="font-size:16px; color:#888;">
  (경고선 {{ '%.0f' % (summary.threshold * 100) }}%)</span></p>
{{ ratio_div }}
{{ index_div }}
<table>
<tr><th>종목</th><th>시총 (USD)</th><th>전일 대비</th></tr>
{% for group in summary.groups %}
<tr class="group-row"><td>[{{ group.name }}]</td><td>{{ fmt_t(group.total_usd) }}</td><td></td></tr>
{% for c in group.companies %}
<tr>
  <td>{{ c.name }} ({{ c.ticker }})</td>
  <td>{{ fmt_t(c.cap_usd) }}</td>
  <td class="{{ 'pos' if c.day_change_pct and c.day_change_pct > 0 else 'neg' if c.day_change_pct and c.day_change_pct < 0 else '' }}">
    {{ fmt_pct(c.day_change_pct) }}</td>
</tr>
{% endfor %}
{% endfor %}
</table>
</body>
</html>
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_aihw_report.py -v`
Expected: 전체 PASS

- [ ] **Step 5: PNG 스모크 확인 (수동)** — kaleido 렌더는 단위 테스트에서 제외:

Run: `python -c "
from tests.test_aihw_report import _series, _summary
from src.aihw.report import generate_png
print(generate_png(_series(), _summary(), 'reports'))
"`
Expected: `reports/aihw-2026-08-22.png` 생성. 열어서 차트 2개(비율+빨간선 / 지수 4선)가 보이는지 확인 후 스모크 파일 삭제: `rm reports/aihw-2026-08-22.png reports/aihw-2026-08-22.html 2>/dev/null; true`

- [ ] **Step 6: 커밋**

```bash
git add src/aihw/report.py src/aihw/templates/report.html tests/test_aihw_report.py
git commit -m "feat(aihw): plotly 차트, HTML 리포트, 공유용 PNG 생성 구현"
```

---

### Task 8: reporter.send_photo + pipeline

**Files:**
- Modify: `src/reporter.py` (`send` 메서드 아래에 `send_photo` 추가)
- Create: `src/aihw/pipeline.py`
- Test: `tests/test_aihw_pipeline.py`

**Interfaces:**
- Consumes: `fetch_all` (Task 5), `AihwDB` (Task 4), `build_series`/`summarize` (Task 2·3), `build_caption`/`generate_html`/`generate_png` (Task 6·7), `AihwSection` (Task 1)
- Produces:
  - `Reporter.send_photo(self, photo_path: str, caption: str) -> None` (async)
  - `AihwResult(BaseModel)` — `summary: AihwSummary`, `html_path: str`, `png_path: str`, `caption: str`
  - `run_aihw(config: AihwSection, db: AihwDB | None = None, fetch = fetch_all) -> AihwResult` — 수집→저장→로드→계산→리포트 생성. 텔레그램 전송은 CLI 쪽 책임 (pipeline은 부수효과를 파일 생성까지만).

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_aihw_pipeline.py`:

```python
from datetime import date

from src.aihw.db import AihwDB
from src.aihw.models import DailyCap
from src.aihw.pipeline import run_aihw
from src.config import AihwSection

D1, D2 = date(2026, 1, 12), date(2026, 1, 13)

CFG = AihwSection(
    ai_hw_tickers={"NVDA": "엔비디아"},
    big_tech_tickers={"MSFT": "MS"},
    benchmarks=["SPY"],
    base_date="2026-01-10",
    threshold=0.8,
)


def _fake_fetch(cap_tickers, benchmark_tickers, start, end):
    rows = []
    for d, nvda, msft, spy in [(D1, 3.0e12, 4.0e12, 500.0), (D2, 3.3e12, 4.0e12, 505.0)]:
        source = "snapshot" if d == D2 else "backfill"
        rows.append(DailyCap(date=d, ticker="NVDA", close=100.0, shares=10,
                             market_cap_usd=nvda, source=source))
        rows.append(DailyCap(date=d, ticker="MSFT", close=100.0, shares=10,
                             market_cap_usd=msft, source=source))
        rows.append(DailyCap(date=d, ticker="SPY", close=spy, shares=None,
                             market_cap_usd=None, source=source))
    return rows


class TestRunAihw:
    def test_full_pipeline(self, tmp_path):
        cfg = CFG.model_copy(update={"report_dir": str(tmp_path)})
        db = AihwDB(url="sqlite:///:memory:")
        result = run_aihw(cfg, db=db, fetch=_fake_fetch)
        assert result.summary.as_of == D2
        assert result.summary.ratio == 3.3e12 / 4.0e12
        assert result.html_path.endswith("aihw-2026-01-13.html")
        assert result.png_path.endswith("aihw-2026-01-13.png")
        assert "엔비디아" in result.caption
        # DB에 저장됐는지
        assert len(db.load_caps(D1, D2)) == 6

    def test_snapshot_persists_across_runs(self, tmp_path):
        cfg = CFG.model_copy(update={"report_dir": str(tmp_path)})
        db = AihwDB(url="sqlite:///:memory:")
        run_aihw(cfg, db=db, fetch=_fake_fetch)

        def _fetch_backfill_only(cap_tickers, benchmark_tickers, start, end):
            rows = _fake_fetch(cap_tickers, benchmark_tickers, start, end)
            # 두 번째 실행이 과거를 다른 값의 backfill로 다시 준다고 가정
            return [r.model_copy(update={"source": "backfill", "market_cap_usd":
                    (r.market_cap_usd or 0) * 2 or None}) for r in rows]

        run_aihw(cfg, db=db, fetch=_fetch_backfill_only)
        d2_nvda = [r for r in db.load_caps(D2, D2) if r.ticker == "NVDA"][0]
        # D2는 첫 실행에서 snapshot이었으므로 값이 유지된다
        assert d2_nvda.market_cap_usd == 3.3e12
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_aihw_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.aihw.pipeline'`

- [ ] **Step 3: 구현**

`src/aihw/pipeline.py`:

```python
"""src/aihw/pipeline.py

수집 → 저장 → 계산 → 리포트 생성 오케스트레이션.
텔레그램 전송은 CLI가 담당한다 (pipeline의 부수효과는 DB·파일까지).
"""
from __future__ import annotations

from datetime import date, datetime

from loguru import logger
from pydantic import BaseModel

from src.aihw.compute import build_series, summarize
from src.aihw.db import AihwDB
from src.aihw.fetcher import fetch_all
from src.aihw.models import AihwSummary
from src.aihw.report import build_caption, generate_html, generate_png
from src.config import AihwSection


class AihwResult(BaseModel):
    summary: AihwSummary
    html_path: str
    png_path: str
    caption: str


def run_aihw(
    config: AihwSection,
    db: AihwDB | None = None,
    fetch=fetch_all,
) -> AihwResult:
    if db is None:
        db = AihwDB()
    base_date = datetime.strptime(config.base_date, "%Y-%m-%d").date()
    cap_tickers = list(config.ai_hw_tickers) + list(config.big_tech_tickers)

    logger.info(f"aihw 수집 시작: {len(cap_tickers)}종목 + {config.benchmarks}")
    fetched = fetch(cap_tickers, config.benchmarks, base_date, date.today())
    db.save_caps(fetched)

    caps = db.load_caps(base_date, date.today())
    series = build_series(
        caps,
        ai_hw=list(config.ai_hw_tickers),
        big_tech=list(config.big_tech_tickers),
        benchmarks=config.benchmarks,
        base_date=base_date,
    )
    summary = summarize(
        series, caps, config.ai_hw_tickers, config.big_tech_tickers, config.threshold,
    )
    return AihwResult(
        summary=summary,
        html_path=generate_html(series, summary, config.report_dir),
        png_path=generate_png(series, summary, config.report_dir),
        caption=build_caption(summary),
    )
```

`src/reporter.py`의 `send` 메서드 아래에 추가:

```python
    async def send_photo(self, photo_path: str, caption: str) -> None:
        """사진 + 캡션 전송 (aihw 지표 공유용)."""
        bot = Bot(token=self.bot_token)
        with open(photo_path, "rb") as f:
            await bot.send_photo(chat_id=self.chat_id, photo=f, caption=caption)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_aihw_pipeline.py -v`
Expected: 2개 PASS. 주의 — `test_full_pipeline`은 kaleido PNG 렌더를 실제 수행한다. CI/로컬에서 kaleido 초기화 실패 시 `generate_png` 호출을 monkeypatch로 대체하도록 테스트를 조정하지 말고, 먼저 `pip install -U kaleido`로 해결을 시도한다.

- [ ] **Step 5: 커밋**

```bash
git add src/aihw/pipeline.py src/reporter.py tests/test_aihw_pipeline.py
git commit -m "feat(aihw): 파이프라인 오케스트레이션 및 텔레그램 사진 전송 추가"
```

---

### Task 9: CLI `aihw` 명령

**Files:**
- Modify: `src/cli.py` (`stats` 명령 뒤에 추가)
- Test: `tests/test_aihw_cli.py`

**Interfaces:**
- Consumes: `run_aihw`/`AihwResult` (Task 8), `Reporter.send_photo` (Task 8), `Settings.aihw_telegram_chat_id` (Task 1)
- Produces: `python -m src.cli aihw [--send] [--days N]` 명령

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_aihw_cli.py`:

```python
from datetime import date
from unittest.mock import patch

from typer.testing import CliRunner

from src.aihw.models import AihwSummary, CompanySummary, GroupSummary
from src.aihw.pipeline import AihwResult
from src.cli import app

runner = CliRunner()


def _result():
    summary = AihwSummary(
        as_of=date(2026, 8, 22), ratio=0.762, ratio_prev=0.754, change_pp=0.8,
        high_30d=0.781, low_30d=0.74, threshold=0.8, status=None,
        groups=[
            GroupSummary(name="AI HW", total_usd=6.82e12, companies=[
                CompanySummary(ticker="NVDA", name="엔비디아", cap_usd=4.21e12, day_change_pct=1.2),
            ]),
            GroupSummary(name="빅테크", total_usd=8.95e12, companies=[
                CompanySummary(ticker="MSFT", name="MS", cap_usd=3.12e12, day_change_pct=0.4),
            ]),
        ],
    )
    return AihwResult(
        summary=summary, html_path="reports/aihw-2026-08-22.html",
        png_path="reports/aihw-2026-08-22.png", caption="캡션",
    )


class TestAihwCommand:
    @patch("src.aihw.pipeline.run_aihw")
    def test_aihw_prints_summary(self, mock_run):
        mock_run.return_value = _result()
        result = runner.invoke(app, ["aihw"])
        assert result.exit_code == 0
        assert "76.2%" in result.output
        assert "aihw-2026-08-22.html" in result.output
        mock_run.assert_called_once()

    @patch("src.cli.asyncio.run")
    @patch("src.aihw.pipeline.run_aihw")
    def test_aihw_send_flag_sends_photo(self, mock_run, mock_asyncio, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
        mock_run.return_value = _result()
        result = runner.invoke(app, ["aihw", "--send"])
        assert result.exit_code == 0
        mock_asyncio.assert_called_once()  # send_photo 코루틴 실행
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_aihw_cli.py -v`
Expected: FAIL — exit code 2 (명령 없음)

- [ ] **Step 3: 구현** — `src/cli.py`의 `stats` 명령 뒤에 추가:

```python
@app.command()
def aihw(
    send: bool = typer.Option(False, "--send", help="PNG+캡션을 텔레그램 채널로 전송"),
    days: int = typer.Option(None, "--days", help="차트 기간 (기본: base_date부터)"),
):
    """AI HW / 빅테크 시총 비율 지표: 수집·저장·리포트 생성."""
    from src.aihw import pipeline as aihw_pipeline

    settings = Settings()
    config = load_scanner_config()
    aihw_config = config.aihw
    if days:
        from datetime import timedelta
        base = date.today() - timedelta(days=days)
        aihw_config = aihw_config.model_copy(update={"base_date": base.isoformat()})

    console.print("[bold]AI HW / 빅테크 시총 비율 수집 중...[/bold]")
    result = aihw_pipeline.run_aihw(aihw_config)
    s = result.summary

    warn = s.status in ("above", "cross_up")
    color = "red" if warn else "green"
    console.print(
        f"[bold {color}]비율: {s.ratio * 100:.1f}%[/bold {color}] "
        f"(경고선 {s.threshold * 100:.0f}%)"
    )
    if s.change_pp is not None:
        console.print(f"전일 대비 {s.change_pp:+.1f}%p · 30일 최고 {s.high_30d * 100:.1f}%")
    for group in s.groups:
        console.print(f"[bold][{group.name}][/bold] ${group.total_usd / 1e12:.2f}T")
    console.print(f"HTML: {result.html_path}")
    console.print(f"PNG:  {result.png_path}")

    if send:
        if not settings.telegram_bot_token:
            console.print("[red]TELEGRAM_BOT_TOKEN이 없어 전송을 건너뜁니다[/red]")
            raise typer.Exit(code=1)
        from src.reporter import Reporter

        chat_id = settings.aihw_telegram_chat_id or settings.telegram_chat_id
        reporter = Reporter(settings.telegram_bot_token, chat_id)
        asyncio.run(reporter.send_photo(result.png_path, result.caption))
        console.print("[green]텔레그램 전송 완료![/green]")
```

(파일 상단에 `from datetime import date`가 이미 있는지 확인 — `_get_date_str`에서 사용 중이면 재사용.)

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_aihw_cli.py -v`
Expected: 2개 PASS

- [ ] **Step 5: 실동작 확인 (수동)**

Run: `python -m src.cli aihw`
Expected: 실데이터 수집 → 터미널 요약 출력, `reports/aihw-<오늘>.html`·`.png` 생성, `data/aihw.db` 생성. HTML을 열어 차트·테이블 확인.

- [ ] **Step 6: 커밋**

```bash
git add src/cli.py tests/test_aihw_cli.py
git commit -m "feat(aihw): aihw CLI 명령 추가 (--send 텔레그램 전송)"
```

---

### Task 10: `run` 파이프라인 통합 (auto_send)

**Files:**
- Modify: `src/cli.py` (`run` 명령의 마지막, `console.print(f"[bold green]완료! ...")` 직전)
- Test: `tests/test_aihw_cli.py` (추가)

**Interfaces:**
- Consumes: `run_aihw` (Task 8), `AihwSection.auto_send` (Task 1)
- Produces: `run` 완료 시 aihw 전송 (config `aihw.auto_send: true`일 때). aihw 실패는 run 전체를 실패시키지 않는다.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_aihw_cli.py`에 추가:

```python
from src.cli import _run_aihw_step
from src.config import ScannerConfig, Settings


class TestRunAihwStep:
    @patch("src.cli.asyncio.run")
    @patch("src.aihw.pipeline.run_aihw")
    def test_sends_when_auto_send_enabled(self, mock_run, mock_asyncio):
        mock_run.return_value = _result()
        config = ScannerConfig()
        config.aihw.auto_send = True
        settings = Settings(telegram_bot_token="token", telegram_chat_id=123)
        _run_aihw_step(config, settings)
        mock_run.assert_called_once()
        mock_asyncio.assert_called_once()

    @patch("src.aihw.pipeline.run_aihw")
    def test_skips_when_disabled(self, mock_run):
        config = ScannerConfig()
        config.aihw.auto_send = False
        _run_aihw_step(config, Settings())
        mock_run.assert_not_called()

    @patch("src.aihw.pipeline.run_aihw", side_effect=RuntimeError("yfinance down"))
    def test_failure_does_not_raise(self, mock_run):
        config = ScannerConfig()
        config.aihw.auto_send = True
        settings = Settings(telegram_bot_token="token", telegram_chat_id=123)
        _run_aihw_step(config, settings)  # 예외가 전파되면 테스트 실패
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_aihw_cli.py -v -k RunAihwStep`
Expected: FAIL — `ImportError: cannot import name '_run_aihw_step'`

- [ ] **Step 3: 구현** — `src/cli.py`에 헬퍼 추가 (`_make_client` 근처):

```python
def _run_aihw_step(config, settings) -> None:
    """run 파이프라인 마지막 단계: aihw 지표 생성·전송. 실패해도 run을 막지 않는다."""
    if not config.aihw.auto_send:
        return
    try:
        from src.aihw import pipeline as aihw_pipeline

        console.print("[dim]aihw 지표 생성 중...[/dim]")
        result = aihw_pipeline.run_aihw(config.aihw)
        if settings.telegram_bot_token:
            from src.reporter import Reporter

            chat_id = settings.aihw_telegram_chat_id or settings.telegram_chat_id
            reporter = Reporter(settings.telegram_bot_token, chat_id)
            asyncio.run(reporter.send_photo(result.png_path, result.caption))
            console.print("[green]aihw 지표 전송 완료[/green]")
    except Exception as e:  # noqa: BLE001 — aihw 실패가 run 전체를 막으면 안 됨
        logger.warning(f"aihw 단계 실패 (run은 계속): {e}")
        console.print(f"[yellow]aihw 지표 실패: {e}[/yellow]")
```

`run` 명령 본문의 `console.print(f"[bold green]완료! ...")` 직전에 한 줄 추가:

```python
    _run_aihw_step(config, settings)
```

(`src/cli.py` 상단에 `from loguru import logger`가 없으면 추가.)

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_aihw_cli.py -v` 그리고 전체 회귀: `pytest`
Expected: 전체 PASS

- [ ] **Step 5: 커밋**

```bash
git add src/cli.py tests/test_aihw_cli.py
git commit -m "feat(aihw): run 파이프라인에 aihw 자동 전송 단계 통합"
```

---

### Task 11: Claude 스킬 + README

**Files:**
- Create: `/Users/jangjein/.claude/skills/aihw-ratio/SKILL.md`
- Modify: `README.md` (사용법 섹션에 aihw 명령 추가)

**Interfaces:**
- Consumes: `python -m src.cli aihw` (Task 9)

- [ ] **Step 1: 스킬 작성** — `/Users/jangjein/.claude/skills/aihw-ratio/SKILL.md`:

```markdown
---
name: aihw-ratio
description: AI HW/빅테크 시총 비율(고점 지표)을 조회한다. "AI HW 비율", "고점 지표", "빅테크 대비 비율", "80% 경고선" 등을 물으면 kr-stock-scanner CLI를 실행해 현재 비율과 추세를 답한다.
---

# AI HW / 빅테크 시총 비율 조회

AI 하드웨어 종목군(엔비디아·브로드컴·삼전·하닉·마이크론·샌디스크) 시총 합계를
빅테크 종목군(아마존·테슬라·MS·메타·구글) 시총 합계로 나눈 비율. 0.8(80%) 이상이면
고점 주의 구간으로 본다.

## 실행 방법

```bash
cd /Users/jangjein/Desktop/repositories/trading/kr-stock-scanner
.venv/bin/python -m src.cli aihw
```

- 텔레그램 채널 전송까지 원하면 `--send`를 붙인다 (사용자가 명시적으로 요청할 때만).
- 로직은 전부 CLI에 있다. 이 스킬은 실행과 해석만 한다.

## 결과 해석

- 출력의 "비율" 값이 현재 지표. 80% 경고선까지 남은 거리(%p)를 계산해 알려준다.
- "전일 대비"와 "30일 최고"로 추세(상승 중인지, 고점 근처인지)를 설명한다.
- 상태가 cross_up(상향 돌파)이면 명확히 경고한다.
- HTML 리포트 경로(`reports/aihw-YYYY-MM-DD.html`)를 안내한다.

## 주의

- 미국 장 마감 데이터 기준이므로 한국 시간 기준 하루 시차가 있다.
- 실행 실패(yfinance 오류 등) 시 에러를 그대로 요약해 전달하고, DB에 저장된
  마지막 데이터로 추정 답변을 하지 않는다.
```

- [ ] **Step 2: README 갱신** — `README.md` "개별 명령어" 섹션에 추가:

```markdown
# AI HW / 빅테크 시총 비율 지표 (고점 경고)
python -m src.cli aihw           # 수집 + HTML/PNG 리포트 + 터미널 요약
python -m src.cli aihw --send    # 텔레그램 채널로 PNG+요약 전송
```

그리고 주요 기능 목록에 한 줄 추가:

```markdown
- AI HW/빅테크 시총 비율 고점 지표: 매일 run 후 텔레그램 채널 전송, 0.8 경고선 감지
```

- [ ] **Step 3: 스킬 동작 확인 (수동)**

새 Claude Code 세션(또는 현재 세션)에서 "지금 AI HW 비율 어때?"라고 물었을 때 스킬이 CLI를 실행하고 해석하는지 확인. (자동화 불가 — 수동 확인.)

- [ ] **Step 4: 커밋**

```bash
git add README.md
git commit -m "docs(aihw): README에 aihw 명령 사용법 추가"
```

(스킬 파일은 홈 디렉토리라 저장소 커밋 대상이 아님.)

---

## Self-Review 결과

- 스펙 커버리지: fetcher(환율·ffill·부분실패)→T5, db(snapshot 규칙)→T4, compute→T2·3, report 3종→T6·7, CLI/--send/--days→T9, run 통합→T10, config→T1, 스킬→T11. 누락 없음.
- `--days`는 base_date를 임시로 당기는 방식으로 구현 (T9) — 스펙의 "조회·차트 기간" 요구 충족.
- 타입 일관성: `AihwSummary`/`AihwResult`/`DailyCap` 시그니처가 T2→T8→T9에서 동일하게 사용됨을 확인.
