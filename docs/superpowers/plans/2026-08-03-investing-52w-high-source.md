# investing.com 52주 신고가 소스 대체 구현 플랜

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 52주 신고가 탐지를 KRX 종목별 순회(~2,700 요청)에서 investing.com 신고가 목록(요청 1건)으로 대체해 KRX 차단을 제거한다.

**Architecture:** 새 모듈 `src/investing_high.py`가 curl_cffi로 investing.com 신고가 페이지를 받아 HTML 표를 파싱(+`__NEXT_DATA__`의 `total`로 커버리지 확인)하고, 거래량 필터·종목명→KRX 티커 매핑을 거쳐 `StockHigh` 목록을 만든다. `src/cli.py`의 `run`이 기존 KRX 신고가 계산 대신 이 모듈을 사용하고, 시총·섹터는 기존 KRX 벌크 호출로 보강한다.

**Tech Stack:** Python, curl_cffi(Cloudflare 우회), BeautifulSoup(bs4), pydantic(StockHigh), typer(cli), pytest.

## Global Constraints

- 취득은 `curl_cffi`로만. `impersonate` 폴백 순서 = `("chrome124", "safari17_0")`. 순수 `requests`는 403(금지).
- Cloudflare 챌린지("Just a moment"/"challenge-platform"가 있고 데이터 표가 없음)·403·구조변경 시 **명확한 예외로 중단**(무한 재시도·조용한 빈 결과 금지).
- 거래량이 0/없음인 종목은 제외.
- 종목명→티커·시장 매핑은 `dart_corp_info`(`DartCache.load_corp_info()` → `CorpInfo(corp_code,ticker,name,market)`) 기준. 미매칭은 로그·스킵.
- 네트워크 없는 단위테스트(픽스처 사용). 실제 호출은 수동 검증 1회.
- 파이썬 실행: `.venv/bin/python` 사용(`python` 미존재 환경).
- KRX 종목별 히스토리 순회만 제거. 시총(`collector.get_market_caps`)·섹터(`collector.get_sector_map`) 벌크 호출은 유지.

---

### Task 1: 모듈 뼈대 — 데이터 모델 + 거래량 파서/필터

**Files:**
- Create: `src/investing_high.py`
- Test: `tests/test_investing_high.py`

**Interfaces:**
- Produces:
  - `class InvestingHighRow` (dataclass): `name: str`, `last_price: float`, `change_pct: float`, `volume: int`
  - `_parse_volume(text: str) -> int` — "2.07M"→2070000, "617.58K"→617580, "1,234"→1234, ""/"-"/"N/A"→0
  - `filter_tradeable(rows: list[InvestingHighRow]) -> list[InvestingHighRow]` — volume>0만 남김
  - `class InvestingFetchError(RuntimeError)`, `class InvestingParseError(RuntimeError)`

- [ ] **Step 1: 실패 테스트 작성** (`tests/test_investing_high.py`)

```python
from src.investing_high import (
    InvestingHighRow, _parse_volume, filter_tradeable,
    InvestingFetchError, InvestingParseError,
)


def test_parse_volume_suffixes():
    assert _parse_volume("2.07M") == 2_070_000
    assert _parse_volume("617.58K") == 617_580
    assert _parse_volume("1,234") == 1234
    assert _parse_volume("131.15K") == 131_150
    for empty in ("", "-", "N/A", "  "):
        assert _parse_volume(empty) == 0


def test_filter_tradeable_drops_zero_volume():
    rows = [
        InvestingHighRow(name="A", last_price=100.0, change_pct=1.0, volume=5000),
        InvestingHighRow(name="B", last_price=200.0, change_pct=2.0, volume=0),
    ]
    out = filter_tradeable(rows)
    assert [r.name for r in out] == ["A"]
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_investing_high.py -q`
Expected: FAIL (ImportError: cannot import name ... from src.investing_high)

- [ ] **Step 3: 최소 구현** (`src/investing_high.py`)

```python
from __future__ import annotations

from dataclasses import dataclass


class InvestingFetchError(RuntimeError):
    """investing.com 취득 실패(Cloudflare 챌린지/403 등)."""


class InvestingParseError(RuntimeError):
    """investing.com 페이지 구조 파싱 실패."""


@dataclass
class InvestingHighRow:
    name: str
    last_price: float
    change_pct: float
    volume: int


def _parse_volume(text: str) -> int:
    """'2.07M'/'617.58K'/'1,234'/''/'-' → int (없으면 0)."""
    if not text:
        return 0
    t = text.strip().upper().replace(",", "")
    if t in ("", "-", "N/A"):
        return 0
    mult = 1
    if t.endswith("K"):
        mult, t = 1_000, t[:-1]
    elif t.endswith("M"):
        mult, t = 1_000_000, t[:-1]
    elif t.endswith("B"):
        mult, t = 1_000_000_000, t[:-1]
    try:
        return int(round(float(t) * mult))
    except ValueError:
        return 0


def filter_tradeable(rows: list[InvestingHighRow]) -> list[InvestingHighRow]:
    """거래량 없는(0) 종목 제외 — 정지·유동성 없는 종목 제거."""
    return [r for r in rows if r.volume > 0]
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_investing_high.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/investing_high.py tests/test_investing_high.py
git commit -m "feat(investing): 신고가 모듈 뼈대 — 모델·거래량 파서·필터"
```

---

### Task 2: HTML 표 파싱 + total 추출

**Files:**
- Modify: `src/investing_high.py`
- Test: `tests/test_investing_high.py`
- Create: `tests/fixtures/investing_52w_high.html`

**Interfaces:**
- Consumes: `InvestingHighRow`, `InvestingParseError` (Task 1)
- Produces: `parse_high_rows(html: str) -> tuple[list[InvestingHighRow], int | None]`
  - 가장 큰 `<table>`의 데이터행(td ≥ 7)에서 td[1]=종목명, td[2]=현재가, td[5]=변동%, td[6]=거래량 추출.
  - `total`은 `<script id="__NEXT_DATA__">` JSON의 `"total"` 값(정규식). 없으면 None.
  - 표가 없고 챌린지 문구가 있으면 `InvestingParseError`.

- [ ] **Step 1: 픽스처 생성** (`tests/fixtures/investing_52w_high.html`)

확인된 실제 구조(td[0]=플래그, td[1]=종목명(a), td[2]=현재가, td[3]=고가, td[4]=저가, td[5]=변동%, td[6]=거래량, td[7]=상승여력, td[8]=시간)를 축소 재현. `total`은 3.

```html
<html><head>
<script id="__NEXT_DATA__" type="application/json">{"props":{"pageProps":{"state":{"assetsCollectionStore":{"assetsCollection":{"_collection":[1,2,3]}}}}}},"pagination":{"total":3}}</script>
</head><body>
<table><tr><th></th><th>종목명</th><th>현재가</th><th>고가</th><th>저가</th><th>변동 %</th><th>거래량</th><th>상승 여력</th><th>시간</th></tr>
<tr><td></td><td><a href="/equities/icraft-co-ltd">아이크래프트</a></td><td>5,190</td><td>5,650</td><td>4,435</td><td>+12.34%</td><td>2.07M</td><td>aa.aa</td><td>15:29:59</td></tr>
<tr><td></td><td><a href="/equities/vect">벡트</a></td><td>3,680</td><td>3,800</td><td>3,135</td><td>+8.08%</td><td>617.58K</td><td>aa.aa</td><td>15:29:59</td></tr>
<tr><td></td><td><a href="/equities/novol">거래정지주</a></td><td>1,000</td><td>1,000</td><td>1,000</td><td>+0.00%</td><td>-</td><td>aa.aa</td><td>15:29:59</td></tr>
</table></body></html>
```

- [ ] **Step 2: 실패 테스트 작성** (기존 테스트 파일에 추가)

```python
from pathlib import Path
from src.investing_high import parse_high_rows

_FIXTURE = Path(__file__).parent / "fixtures" / "investing_52w_high.html"


def test_parse_high_rows_extracts_all_rows_and_total():
    html = _FIXTURE.read_text(encoding="utf-8")
    rows, total = parse_high_rows(html)
    assert total == 3
    assert [r.name for r in rows] == ["아이크래프트", "벡트", "거래정지주"]
    assert rows[0].last_price == 5190.0
    assert rows[0].change_pct == 12.34
    assert rows[0].volume == 2_070_000
    assert rows[2].volume == 0  # 거래량 '-'


def test_parse_high_rows_raises_on_challenge():
    from src.investing_high import InvestingParseError
    import pytest
    challenge = "<html><head><title>Just a moment...</title></head><body></body></html>"
    with pytest.raises(InvestingParseError):
        parse_high_rows(challenge)
```

- [ ] **Step 3: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_investing_high.py -q`
Expected: FAIL (cannot import name 'parse_high_rows')

- [ ] **Step 4: 구현** (`src/investing_high.py`에 추가)

```python
import re
from bs4 import BeautifulSoup


def _clean_num(text: str) -> float:
    t = (text or "").strip().replace(",", "").replace("%", "").replace("+", "")
    try:
        return float(t)
    except ValueError:
        return 0.0


def parse_high_rows(html: str) -> tuple[list[InvestingHighRow], int | None]:
    """investing 신고가 HTML → (행 목록, total). 표 없으면 InvestingParseError."""
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        raise InvestingParseError("데이터 표를 찾지 못함(Cloudflare 챌린지 또는 구조 변경)")
    table = max(tables, key=lambda t: len(t.find_all("tr")))

    rows: list[InvestingHighRow] = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 7:
            continue  # 헤더/빈 행
        name = tds[1].get_text(strip=True)
        if not name:
            continue
        rows.append(InvestingHighRow(
            name=name,
            last_price=_clean_num(tds[2].get_text(strip=True)),
            change_pct=_clean_num(tds[5].get_text(strip=True)),
            volume=_parse_volume(tds[6].get_text(strip=True)),
        ))
    if not rows:
        raise InvestingParseError("표는 있으나 데이터 행이 없음")

    total: int | None = None
    m = re.search(r'"total"\s*:\s*(\d+)', html)
    if m:
        total = int(m.group(1))
    return rows, total
```

- [ ] **Step 5: 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_investing_high.py -q`
Expected: PASS (4 passed)

- [ ] **Step 6: 커밋**

```bash
git add src/investing_high.py tests/test_investing_high.py tests/fixtures/investing_52w_high.html
git commit -m "feat(investing): HTML 표 파싱 + total 추출 (픽스처 테스트)"
```

---

### Task 3: 취득 계층 — curl_cffi 폴백 + fail-fast

**Files:**
- Modify: `src/investing_high.py`
- Test: `tests/test_investing_high.py`

**Interfaces:**
- Consumes: `parse_high_rows`, `InvestingFetchError` (Task 1·2)
- Produces:
  - `_fetch_html(url: str, targets: tuple[str, ...], _get=None) -> str` — `targets` 순차 시도. `_get`은 테스트 주입용(기본 curl_cffi). 모두 실패 시 `InvestingFetchError`.
  - `fetch_52w_high_rows(_get=None) -> tuple[list[InvestingHighRow], int | None]` — `_fetch_html` + `parse_high_rows`. `total`이 행 수보다 크면 경고 로그(커버리지 잘림, 조용한 누락 금지).
  - 상수 `URL = "https://kr.investing.com/equities/52-week-high"`, `IMPERSONATE_TARGETS = ("chrome124", "safari17_0")`

- [ ] **Step 1: 실패 테스트 작성**

```python
import pytest
from src.investing_high import _fetch_html, InvestingFetchError


class _Resp:
    def __init__(self, status, text):
        self.status_code = status
        self.text = text


def test_fetch_html_falls_back_to_next_target():
    calls = []
    def fake_get(url, impersonate, timeout, headers):
        calls.append(impersonate)
        if impersonate == "chrome124":
            return _Resp(403, "403")               # 첫 타깃 차단
        return _Resp(200, "<table><tr><td>x</td></tr></table>")  # 둘째 성공
    html = _fetch_html("http://x", ("chrome124", "safari17_0"), _get=fake_get)
    assert "<table>" in html
    assert calls == ["chrome124", "safari17_0"]


def test_fetch_html_raises_when_all_targets_blocked():
    def fake_get(url, impersonate, timeout, headers):
        return _Resp(403, "403")
    with pytest.raises(InvestingFetchError):
        _fetch_html("http://x", ("chrome124", "safari17_0"), _get=fake_get)
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_investing_high.py -q`
Expected: FAIL (cannot import name '_fetch_html')

- [ ] **Step 3: 구현** (`src/investing_high.py`에 추가)

```python
from loguru import logger

URL = "https://kr.investing.com/equities/52-week-high"
IMPERSONATE_TARGETS = ("chrome124", "safari17_0")
_HEADERS = {"Accept-Language": "ko-KR,ko;q=0.9"}


def _default_get(url, impersonate, timeout, headers):
    from curl_cffi import requests as cffi
    return cffi.get(url, impersonate=impersonate, timeout=timeout, headers=headers)


def _is_challenge(text: str) -> bool:
    return ("Just a moment" in text) and ("<table" not in text)


def _fetch_html(url: str, targets: tuple[str, ...], _get=None) -> str:
    get = _get or _default_get
    last = ""
    for target in targets:
        try:
            resp = get(url, impersonate=target, timeout=25, headers=_HEADERS)
        except Exception as e:  # noqa: BLE001 — 네트워크 예외는 다음 타깃으로
            last = f"{type(e).__name__}: {e}"
            continue
        text = resp.text
        if resp.status_code == 200 and "<table" in text and not _is_challenge(text):
            return text
        last = f"status={resp.status_code}, challenge={_is_challenge(text)}"
        logger.warning(f"investing 취득 실패(impersonate={target}): {last}")
    raise InvestingFetchError(f"모든 impersonate 타깃 실패: {last}")


def fetch_52w_high_rows(_get=None) -> tuple[list[InvestingHighRow], int | None]:
    html = _fetch_html(URL, IMPERSONATE_TARGETS, _get=_get)
    rows, total = parse_high_rows(html)
    if total is not None and total > len(rows):
        logger.warning(
            f"investing 신고가 커버리지 잘림: total={total} > 취득={len(rows)}. "
            f"페이지네이션 미구현으로 초기 배치만 사용합니다."
        )
    return rows, total
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_investing_high.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/investing_high.py tests/test_investing_high.py
git commit -m "feat(investing): curl_cffi 폴백 취득 + fail-fast + 커버리지 경고"
```

---

### Task 4: 종목명→KRX 매핑 + StockHigh 조립

**Files:**
- Modify: `src/investing_high.py`
- Test: `tests/test_investing_high.py`

**Interfaces:**
- Consumes: `InvestingHighRow` (Task 1), `src.models.StockHigh`
- Produces:
  - `resolve_to_krx(rows, name_to_ticker, name_to_market) -> tuple[list[tuple[InvestingHighRow, str, str]], list[str]]`
    - 반환: `(matched, unmatched_names)`. matched 원소 = `(row, ticker, market)`.
    - 매칭: `name_to_ticker[정규화된 이름]`. 정규화 = 공백/`(주)`·`㈜` 제거(`src.related.extractor.normalize_name` 재사용).
  - `build_highs(matched, market_caps, sector_map) -> list[StockHigh]`
    - `StockHigh(ticker, name, market, sector=sector_map.get(ticker,"기타"), close_price=row.last_price, high_52w=row.last_price, prev_high_52w=0.0, breakout_pct=row.change_pct, volume=row.volume, avg_volume_20d=0)`
    - 시총은 `StockHigh`에 필드가 없으므로 저장 안 함(기존 모델 유지). market_caps는 통계용으로 호출부에서 사용.

- [ ] **Step 1: 실패 테스트 작성**

```python
from src.investing_high import resolve_to_krx, build_highs
from src.models import StockHigh


def test_resolve_to_krx_maps_and_reports_unmatched():
    rows = [
        InvestingHighRow(name="아이크래프트", last_price=5190.0, change_pct=12.34, volume=2_070_000),
        InvestingHighRow(name="없는회사", last_price=1000.0, change_pct=1.0, volume=100),
    ]
    n2t = {"아이크래프트": "052460"}
    n2m = {"아이크래프트": "KOSDAQ"}
    matched, unmatched = resolve_to_krx(rows, n2t, n2m)
    assert [(m[1], m[2]) for m in matched] == [("052460", "KOSDAQ")]
    assert unmatched == ["없는회사"]


def test_build_highs_assembles_stockhigh():
    row = InvestingHighRow(name="아이크래프트", last_price=5190.0, change_pct=12.34, volume=2_070_000)
    matched = [(row, "052460", "KOSDAQ")]
    highs = build_highs(matched, market_caps={"052460": 123}, sector_map={"052460": "IT"})
    assert len(highs) == 1
    h = highs[0]
    assert isinstance(h, StockHigh)
    assert (h.ticker, h.market, h.sector) == ("052460", "KOSDAQ", "IT")
    assert h.close_price == 5190.0 and h.volume == 2_070_000
    assert h.breakout_pct == 12.34
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_investing_high.py -q`
Expected: FAIL (cannot import name 'resolve_to_krx')

- [ ] **Step 3: 구현** (`src/investing_high.py`에 추가)

```python
from src.models import StockHigh
from src.related.extractor import normalize_name


def resolve_to_krx(
    rows: list[InvestingHighRow],
    name_to_ticker: dict[str, str],
    name_to_market: dict[str, str],
) -> tuple[list[tuple[InvestingHighRow, str, str]], list[str]]:
    norm_to_ticker = {normalize_name(k): v for k, v in name_to_ticker.items()}
    norm_to_market = {normalize_name(k): v for k, v in name_to_market.items()}
    matched: list[tuple[InvestingHighRow, str, str]] = []
    unmatched: list[str] = []
    for row in rows:
        key = normalize_name(row.name)
        ticker = name_to_ticker.get(row.name) or norm_to_ticker.get(key)
        if ticker is None:
            unmatched.append(row.name)
            continue
        market = name_to_market.get(row.name) or norm_to_market.get(key) or ""
        matched.append((row, ticker, market))
    if unmatched:
        logger.warning(f"investing 미매칭 {len(unmatched)}종목: {unmatched[:20]}")
    return matched, unmatched


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
            breakout_pct=row.change_pct,
            volume=row.volume,
            avg_volume_20d=0,
        ))
    return highs
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_investing_high.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/investing_high.py tests/test_investing_high.py
git commit -m "feat(investing): 종목명→KRX 매핑 + StockHigh 조립"
```

---

### Task 5: cli.run 통합 + curl_cffi 의존성 명시

**Files:**
- Modify: `src/cli.py` (`run`의 Step 1-3 신고가 수집 블록)
- Modify: `pyproject.toml` (dependencies에 `curl_cffi` 추가)
- Test: `tests/test_investing_high.py` (조립 통합 함수 테스트)

**Interfaces:**
- Consumes: `fetch_52w_high_rows`, `filter_tradeable`, `resolve_to_krx`, `build_highs` (Task 1~4); `DartCache.load_corp_info()`; `Collector.get_market_caps`, `Collector.get_sector_map`.
- Produces: `src/investing_high.py`의 `collect_investing_highs(date_str, collector, corps, _get=None) -> tuple[list[StockHigh], dict[str,int]]` — 신고가 목록 + market_caps(통계용). cli는 이 함수만 호출.

- [ ] **Step 1: 통합 함수 실패 테스트 작성**

```python
def test_collect_investing_highs_end_to_end(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    import src.investing_high as inv

    html = (Path(__file__).parent / "fixtures" / "investing_52w_high.html").read_text(encoding="utf-8")

    def fake_get(url, impersonate, timeout, headers):
        return SimpleNamespace(status_code=200, text=html)

    corps = [
        SimpleNamespace(name="아이크래프트", ticker="052460", market="KOSDAQ"),
        SimpleNamespace(name="벡트", ticker="365900", market="KOSDAQ"),
        # '거래정지주'는 매핑 없음 + 거래량 0 → 이중으로 제외
    ]

    class FakeCollector:
        def get_market_caps(self, date_str, market="ALL"):
            return {"052460": 111, "365900": 222}
        def get_sector_map(self, date_str, market):
            return {"052460": "IT", "365900": "전기전자"}

    highs, caps = inv.collect_investing_highs("20260803", FakeCollector(), corps, _get=fake_get)
    names = sorted(h.name for h in highs)
    assert names == ["벡트", "아이크래프트"]          # 거래정지주(거래량0) 제외
    assert all(h.close_price > 0 for h in highs)
    assert caps["052460"] == 111
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_investing_high.py::test_collect_investing_highs_end_to_end -q`
Expected: FAIL (module 'src.investing_high' has no attribute 'collect_investing_highs')

- [ ] **Step 3: `collect_investing_highs` 구현** (`src/investing_high.py`에 추가)

```python
def collect_investing_highs(date_str, collector, corps, _get=None):
    """investing 신고가 → 거래량 필터 → KRX 매핑 → 시총/섹터 보강 → StockHigh 목록."""
    rows, _total = fetch_52w_high_rows(_get=_get)
    rows = filter_tradeable(rows)
    name_to_ticker = {c.name: c.ticker for c in corps}
    name_to_market = {c.name: c.market for c in corps}
    matched, _unmatched = resolve_to_krx(rows, name_to_ticker, name_to_market)
    market_caps = collector.get_market_caps(date_str)
    sector_map: dict[str, str] = {}
    for m in ("KOSPI", "KOSDAQ"):
        sector_map.update(collector.get_sector_map(date_str, m))
    highs = build_highs(matched, market_caps, sector_map)
    return highs, market_caps
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_investing_high.py::test_collect_investing_highs_end_to_end -q`
Expected: PASS

- [ ] **Step 5: cli.run 통합**

`src/cli.py`의 `run`에서, 캐시 없을 때(`else:` 블록)의 KRX 수집·스캔 부분을 investing 소스로 교체. 기존 `existing`(DB 재사용)·`--force`·Step4~5(뉴스/AI/리포트)는 그대로 둔다. `else:` 블록을 다음으로 대체:

```python
    else:
        from src.dart.cache import DartCache
        from src.collector import Collector
        from src.investing_high import collect_investing_highs, InvestingFetchError, InvestingParseError

        client = _make_client(settings)
        collector = Collector(client=client)
        corps = DartCache().load_corp_info(markets=["KOSPI", "KOSDAQ"])

        console.print("[dim]1-3/5 investing.com 52주 신고가 수집 중...[/dim]")
        try:
            highs, market_caps = collect_investing_highs(date_str, collector, corps)
        except (InvestingFetchError, InvestingParseError) as e:
            console.print(f"[red]investing 신고가 수집 실패: {e}[/red]")
            raise typer.Exit(code=1)

        result = scanner.build_scan_result(scan_date, highs, len(highs))
        db.save_scan_result(result)
```

(주의: `scanner`가 이 블록에서 쓰이므로, 기존처럼 `scanner = Scanner(collector=collector)`를 이 블록 상단에 유지한다. `Scanner.build_scan_result`는 `find_new_highs` 없이도 호출 가능 — highs를 직접 넘긴다.)

- [ ] **Step 6: pyproject.toml 의존성 추가**

`[project]`의 `dependencies` 배열에 `"curl_cffi>=0.7"` 한 줄 추가.

- [ ] **Step 7: 전체 테스트 + import 확인**

Run: `.venv/bin/python -m pytest -q && .venv/bin/python -c "import src.cli"`
Expected: 전체 PASS, import 에러 없음

- [ ] **Step 8: 커밋**

```bash
git add src/investing_high.py src/cli.py pyproject.toml tests/test_investing_high.py
git commit -m "feat(scanner): 52주 신고가 소스를 investing.com으로 교체 (cli 통합)"
```

---

### Task 6: 수동 검증 (실제 네트워크 1회)

**Files:** 없음(실행 검증). 산출물: `data/scanner.db`의 오늘자 scan_result.

- [ ] **Step 1: 실제 수집 검증**

Run: `.venv/bin/python -m src.cli run -f`
Expected 확인:
- investing에서 신고가 목록 수집(로그에 "investing.com 52주 신고가 수집").
- 미매칭/커버리지 경고 로그가 있으면 개수 확인(정상 동작 신호).
- 리포트에 신고가 종목이 출력되고, 거래량 0 종목이 없다.
- **KRX 종목별 히스토리 순회 로그·차단(ip-block)이 발생하지 않는다.**

- [ ] **Step 2: 결과 저장 확인**

Run: `sqlite3 data/scanner.db "SELECT COUNT(*) FROM new_highs WHERE scan_date=date('now');"`
Expected: 신고가 종목 수 > 0 (investing 목록 중 매핑·거래량 통과분).

- [ ] **Step 3: 결함 발견 시 수정 후 재실행**

파싱/매핑 이슈가 보이면 해당 모듈 수정·테스트 후 Step 1 재실행. 수정 시 커밋.

---

## Self-Review

**1. 스펙 커버리지:**
- curl_cffi + chrome124/safari17 폴백 → Task 3. ✅
- __NEXT_DATA__ total로 커버리지 확인 + 경고 → Task 2(파싱)·3(경고). ✅
- 거래량 필터 → Task 1(filter_tradeable) + Task 5 통합. ✅
- 종목명→dart_corp_info 매핑, 미매칭 로그·스킵 → Task 4. ✅
- 챌린지/403/구조변경 fail-fast → Task 2(ParseError)·3(FetchError)·5(cli에서 Exit). ✅
- KRX 최소 사용(시총·섹터 벌크 유지, 순회 제거) → Task 5. ✅
- curl_cffi 의존성 명시 → Task 5 Step 6. ✅
- 네트워크 없는 단위테스트 + 수동 검증 1회 → Task 1~5(픽스처·주입) + Task 6. ✅
- 의미 변화(신고가 정의 investing 기준) → Task 4에서 high_52w/prev_high_52w/breakout_pct 매핑으로 반영. ✅

**2. 플레이스홀더 스캔:** 모든 코드 스텝에 실제 코드 수록. "TBD/추후" 없음. 페이지네이션 미구현은 경고 로그로 안전 처리(스펙 범위 밖 명시). ✅

**3. 타입 일관성:** `InvestingHighRow`(name/last_price/change_pct/volume), `parse_high_rows→(rows,total)`, `resolve_to_krx→(matched:(row,ticker,market), unmatched)`, `build_highs→list[StockHigh]`, `collect_investing_highs→(highs,caps)` — Task 간 시그니처 일치. `StockHigh` 필드는 models.py와 일치. ✅
