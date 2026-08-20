# 52주 신고가 — 돌파 신선도(재돌파 간격·갱신 깊이) 지표 — 설계

- **작성일:** 2026-08-19
- **상태:** 설계 승인 대기(스펙 리뷰용)
- **목적:** 52주 신고가 종목에 대해 **"얼마 만의 돌파인가"** 를 두 축으로 산출·표기한다. 1년 이상 만의 장기 박스권 탈출과 며칠 만의 재돌파를 리포트에서 즉시 구분할 수 있게 한다.

## 배경 / 문제

현재 스캔 소스는 investing.com의 사전 계산된 신고가 목록이다(2026-08-03 설계). 그 결과 `build_highs`([src/investing_high.py:143](../../../src/investing_high.py))는 종목별 가격 이력을 전혀 들고 있지 않고, 다음과 같이 채운다.

- `high_52w = last_price` (종가)
- `prev_high_52w = 0.0` (미상)
- `breakout_pct = change_pct` — **직전 고점 대비 돌파폭이 아니라 당일 등락률**

즉 리포트에는 "오늘 신고가를 냈다"는 사실만 있고, **그 돌파가 얼마나 신선한지에 대한 정보가 전혀 없다.** 매일 신고가를 경신하며 오르는 종목과 3년 박스권을 처음 뚫은 종목이 같은 줄로 표시된다. 후자가 훨씬 유의미한 신호인데 구분이 안 된다.

또한 스캐너 자체 이력(`new_highs` 테이블)으로 이를 유추하는 것은 불가능하다. 스캐너 운영 시작 이전 구간이 비어 있고, investing.com 목록은 페이지네이션 미구현으로 잘릴 수 있어(`total > len(rows)` 경고 경로) 누락된 날이 생긴다.

## 두 지표의 정의

오늘 날짜를 `D`, 종목의 **오늘 고가**를 `H`, 일봉 시계열을 날짜 오름차순 `bars[(date, high)]`(오늘 포함, **수정주가 기준**), 52주 창을 `W = 250 거래일`이라 한다.

### A. 재돌파 간격 — `days_since_prev_new_high`

> 직전에 **그날 자체가 52주 신고가였던** 날로부터 며칠이 지났는가.

```
그날이 신고가인가(j):  bars[j].high >= max(bars[j-W : j].high)
A = max{ j | W <= j < len(bars)-1 이고 신고가인가(j) } 의 날짜로부터 D까지의 일수
    해당 j가 없으면 None
```

앞쪽 `W`개 봉은 rolling max 워밍업으로 소비되므로 A의 판정 구간은 **확보 이력 − 52주**다.

추세의 **지속 여부**를 본다. A가 작으면 진행 중인 모멘텀, 크면 오랜 침묵 뒤의 신선한 돌파다.

### B. 갱신 깊이 — `days_since_price_above`

> 이 가격을 마지막으로 웃돌았던 게 며칠 전인가.

```
B = max{ i | i < len(bars)-1 이고 bars[i].high >= H } 의 날짜로부터 D까지의 일수
    해당 i가 없으면 None  (= 확보 이력 전체에서 최고)
```

rolling window가 필요 없으므로 B의 판정 구간은 **확보 이력 전체**다. 52주 신고가라면 정의상 B는 최소 1년 이상이어야 한다. 1년 미만이 나오면 investing.com 판정과 KRX 수정주가 사이의 불일치이므로 **경고 로그를 남기고 값은 그대로 싣는다**(스캔은 중단하지 않는다).

장기 박스권 탈출의 **강도**를 본다.

### 두 지표가 상보적인 이유

| 상황 | A | B |
|---|---|---|
| 매일 경신하며 상승 중 (사상 최고 영역) | 1일 | `None` (사상 최고) |
| 3년 전 고점을 오늘 넘어섬, 직전 신고가는 1년 전 | 1년 | 3년 |
| 5년 박스권을 오늘 처음 탈출 | ≈5년 | ≈5년 |

두 번째 행이 A와 B가 갈리는 계단식 회복 케이스다. 한 축만으로는 표현되지 않는다.

## 확정된 설계 결정

1. **두 지표 모두 산출한다** (A만/B만이 아님).
2. **데이터 취득은 온디맨드.** 그날 신고가로 잡힌 종목에 대해서만 `get_market_ohlcv_by_date`를 호출한다. 원시 시계열을 저장하는 `daily_prices` 캐시 테이블은 **만들지 않는다**. 계산된 값만 `new_highs`에 저장한다.
3. **계산부는 순수 함수로 분리한다.** 입력은 `bars` 리스트, 출력은 지표 값. 네트워크·DB·설정에 의존하지 않는다. 나중에 캐시 테이블이 필요해지면 소스만 바꿔 끼운다.
4. **조회 깊이는 10년 + 52주 워밍업 = 11년.** 요청 시작일보다 첫 거래일이 늦게 나오면 그 지점이 상장 시점이므로, 확보된 만큼만 쓰고 라벨을 "상장 이후 최고"로 낮춘다.
5. **수정주가(`adjusted=True`)를 쓴다.** 다년치를 보는 지표이므로 액면분할·증자 미반영 시 판정이 무너진다.
6. **오늘 봉을 시계열에 포함시킨다.** 그러면 investing.com이 주는 종가(`last_price`) 대신 **오늘 일중 고가** 기준으로 일관되게 계산할 수 있고, 부수적으로 지금 비어 있는 `prev_high_52w`와 그 대비 돌파율을 제대로 채울 수 있다. 이에 따라 당일 등락률은 `change_pct` 필드로 분리한다(아래 「`breakout_pct` 의미 충돌 해소」 참조).
7. **`supports_history=False` 환경(`KrxOpenApiClient`)에서는 열화 동작한다.** 지표를 `None`으로 두고 리포트에서 해당 배지·그룹핑을 생략한다. 스캔 자체는 정상 동작한다.

## 모듈 경계

### 새 모듈 `src/breakout_recency.py` — 순수 계산

```
@dataclass(frozen=True)
class Bar:
    date: date
    high: float

@dataclass(frozen=True)
class Recency:
    days_since_prev_new_high: int | None   # A
    days_since_price_above: int | None     # B
    history_span_days: int                 # bars[0] ~ bars[-1]
    prev_high_52w: float                   # 직전 250봉 최고 고가 (0.0 = 봉 부족으로 산출 불가)
    today_high: float

def compute_recency(bars: list[Bar], window: int = 250) -> Recency | None
    # bars는 날짜 오름차순, 마지막 원소가 오늘.
    # len(bars) < 2 이면 None.
    # A는 워밍업 부족(len <= window)이면 None.
```

네트워크·DB·로깅 설정에 의존하지 않는다. 단위 테스트는 합성 시계열로 전부 커버된다.

### 새 모듈 `src/recency_source.py` — 취득 어댑터

```
def fetch_bars(client, ticker, as_of: date, years: int = 11) -> list[Bar] | None
    # client.supports_history가 False면 None (열화 경로)
    # get_market_ohlcv_by_date(start, as_of, ticker, adjusted=True) → Bar 리스트
    # 빈 응답·예외는 None으로 흡수하고 경고 로그

def enrich_highs(client, highs: list[StockHigh], as_of: date) -> None
    # highs를 순회하며 fetch_bars → compute_recency → StockHigh 필드 채움
    # KrxBlockedError는 즉시 전파(차단 시 추가 요청 금지)
    # 개별 종목 실패는 해당 종목만 None으로 두고 계속 진행
```

### 라벨링 `src/reporter.py` 내부 헬퍼

표시용 문자열 생성은 리포트 계층의 책임이다. 모델에는 숫자만 싣는다.

## 데이터 흐름

```
investing 52w-high 목록 → 티커/시장 매핑 → 시총·섹터 보강   [기존 경로]
  → [신규] 종목별 11년 수정주가 일봉 조회 (신고가 종목 수만큼, ~수십 건)
  → [신규] compute_recency (순수 함수)
  → StockHigh에 A·B·history_span_days·prev_high_52w·breakout_pct 채움
     (change_pct는 investing 단계에서 이미 채워져 있음)
  → ScanResult → DB 저장 → 뉴스/AI/리포트                  [기존 경로]
```

## 모델 변경 — `src/models.py`

`StockHigh`에 3개 필드 추가. 모두 기본값이 있어 기존 생성 코드는 깨지지 않는다.

```python
days_since_prev_new_high: int | None = None   # A
days_since_price_above: int | None = None     # B (None = 확보 이력 내 최고)
history_span_days: int | None = None          # 확보된 이력 길이
change_pct: float = 0.0                       # 당일 등락률 (investing 제공)
```

### `breakout_pct` 의미 충돌 해소

이 변경은 `breakout_pct`에 숨어 있던 의미 충돌을 드러낸다. 현재 이 필드에는 investing이 준 **당일 등락률**이 들어가는데, 이력을 확보하고 나면 같은 필드에 **직전 고점 대비 돌파율**을 넣을 수 있게 된다. 그대로 두면 같은 숫자가 경로에 따라 다른 뜻이 되고, 리포트는 그걸 구분하지 못한 채 `+{breakout_pct}%`로 출력한다.

따라서 **두 값을 필드로 분리한다.**

- `change_pct` — 당일 등락률. investing에서 항상 채워진다. **리포트의 기존 `+x.x%` 표시와 정렬은 이 필드로 옮긴다**(현재 동작·화면이 그대로 유지된다).
- `breakout_pct` — 직전 250봉 최고 고가 대비 돌파율. 이력을 확보한 경우에만 채워지고, 그렇지 않으면 `0.0`으로 남는다.
- `prev_high_52w` — 직전 250봉 최고 고가. 채워지지 않으면 기존과 같이 `0.0`.

리포트는 `breakout_pct > 0`일 때만 `↑1.4% 돌파`를 추가로 표시한다.

## DB 변경 — `src/db.py`

`new_highs` 테이블에 동일한 4개 컬럼(`days_since_prev_new_high`, `days_since_price_above`, `history_span_days`, `change_pct`)을 nullable로 추가한다. SQLAlchemy `create_all`은 기존 테이블에 컬럼을 추가하지 않으므로, 이 저장소의 기존 패턴([src/fundamentals/db.py:64](../../../src/fundamentals/db.py) `_migrate_add_enrichment_columns`)을 따라 수동 마이그레이션 헬퍼를 둔다.

```python
def _migrate_add_recency_columns(engine) -> None:
    # PRAGMA table_info(new_highs) → 없는 컬럼만 ALTER TABLE ADD COLUMN
    # 테이블 자체가 없으면 조용히 반환
```

`Database.__init__`에서 `create_all` 직후 호출한다. `save_scan_result` / `get_scan_result_full`도 새 필드를 왕복시킨다.

## 라벨 버킷

경계는 리포트 계층 상수로 둔다.

**A (재돌파 간격)** — 거래일이 아닌 **달력 일수** 기준:

| 조건 | 라벨 |
|---|---|
| `A <= 5` | `🔁 신고가 행진` |
| `5 < A <= 30` | `🔁 {A}일 만` |
| `30 < A <= 365` | `🆕 {n}개월 만` |
| `A > 365` | `🆕 {n}년 {m}개월 만` |
| `A is None` (워밍업 부족 또는 구간 내 첫 돌파) | `🆕 {확보 이력−1년} 이상 만 (첫 돌파)` |

**B (갱신 깊이)**:

| 조건 | 라벨 |
|---|---|
| `B is None` 이고 `history_span_days >= 10년` | `🏔 10년래 최고` |
| `B is None` 이고 `history_span_days < 10년` | `🏔 상장 이후 최고` |
| `B is not None` | `🏔 {n}년 {m}개월 만의 최고가` |

지표가 `None`이고 `history_span_days`도 `None`이면(열화 경로) **배지를 아예 생략한다.**

## 리포트 노출 — `src/reporter.py`

전체 목록을 A 기준으로 그룹핑하고, 각 줄에 배지를 붙인다.

```
■ 전체 52주 신고가 목록
[장기 돌파 · 1년 이상 만]
  삼성전자 | 89,000원 | +2.1% | ↑1.4% 돌파 | 🆕 3년 2개월 만 | 🏔 10년래 최고 | 반도체
[중기 돌파 · 1~12개월]
  ...
[신고가 행진 · 1개월 내 재돌파]
  ...
```

그룹 경계는 A 기준으로 `>365일 또는 None` / `30~365일` / `<=30일`이다.

지표가 없는 종목의 처리는 다음 한 가지 규칙으로 통일한다: **한 종목이라도 지표가 있으면 목록을 그룹핑하고, 지표가 없는 종목은 맨 아래 `[정보 없음]` 그룹에 모은다. 전 종목에 지표가 없으면(열화 경로) 그룹 헤더 없이 기존 평면 목록 그대로 출력한다.**

AI 분석 섹션의 종목 줄에도 동일한 배지를 붙인다.

## AI 프롬프트 노출 — `src/ai_analyst.py`

프롬프트에 한 줄을 주입한다.

```
돌파 신선도: 직전 신고가 3년 2개월 전 / 현재가는 10년 내 최고 수준
```

지표가 없으면 이 줄을 통째로 생략한다. 상승 원인 서술의 맥락이 달라진다(장기 턴어라운드 vs 진행 중인 모멘텀).

## 열화·실패 처리

| 상황 | 동작 |
|---|---|
| `supports_history=False` (API 키 전용 클라이언트) | 전 종목 지표 `None`, 배지·그룹핑 생략, 스캔 정상 진행 |
| 개별 종목 조회 실패(빈 응답·파싱 실패·예외) | 해당 종목만 `None`, 경고 로그, 나머지 계속 |
| `KrxBlockedError` | **즉시 전파해 중단.** 차단 상태에서 추가 요청 금지 |
| 이력이 2봉 미만 | 지표 `None` |
| 이력이 251봉 미만 | A만 `None`, B는 산출 |
| B가 365일 미만 | 경고 로그 후 값 그대로 사용 (investing 판정과 KRX 수정주가 불일치) |

## 리스크

**KRX 차단 재발.** 2026-08-03에 전 종목(~2,700) 순회로 IP/계정이 차단된 이력이 있고, 이 설계는 종목별 이력 조회를 **다시 도입**한다. 다만 대상이 그날의 신고가 목록(수십 건)으로 한정되고, 클라이언트에 0.2초 rate limit이 있어([src/krx_login_client.py:141](../../../src/krx_login_client.py)) 종목당 2콜(ISIN 조회 + 일봉) 기준 80종목이면 약 160콜·32초다. 차단 사고 당시의 1/17 수준이다. `KrxBlockedError` 즉시 전파로 폭주를 막는다.

**11년치 응답 크기.** KRX BLD `ohlcv_by_date`가 한 번에 반환하는 기간에 상한이 있는지 확인되지 않았다. 구현 첫 단계에서 단일 종목으로 11년 요청을 실제로 던져 **반환된 첫 거래일이 요청 시작일 근처인지 검증**한다. 잘린다면 연 단위로 분할 호출해 병합한다(종목당 콜 수가 늘어나므로 이 경우 조회 깊이 재검토).

**수정주가 정합성.** `adjStkPrc: 2`가 액면분할·증자를 어디까지 반영하는지는 KRX 구현에 달려 있다. B가 365일 미만으로 나오는 빈도를 경고 로그로 관측해 사후 판단한다.

## 테스트 전략

- **`compute_recency` 단위 테스트(합성 시계열)** — 이 기능의 핵심. 매일 경신 / 계단식 회복 / 장기 박스권 탈출 / 사상 최고 / 워밍업 부족 / 2봉 미만 / B가 1년 미만인 이상 케이스.
- **`fetch_bars` 어댑터** — 페이크 클라이언트로 `supports_history=False`, 빈 DataFrame, 예외, 정상 응답 4경로.
- **`enrich_highs`** — 개별 실패 격리, `KrxBlockedError` 전파.
- **마이그레이션 헬퍼** — 테이블 없음 / 컬럼 없음 / 이미 있음 3경로(기존 `test_fundamentals_db_enrichment.py` 패턴).
- **리포트 포매팅** — 라벨 버킷 경계값(5, 30, 365일), 지표 없는 종목의 배지 생략, 그룹핑(일부만 지표 있음 / 전 종목 지표 없음 2경로), `breakout_pct == 0.0`일 때 돌파율 표시 생략.
- **AI 프롬프트** — 지표 유무에 따른 줄 포함/생략.

## 범위 밖 (YAGNI)

- `daily_prices` 원시 시계열 캐시 테이블과 백필 CLI
- 과거 스캔 날짜에 대한 지표 소급 계산·백테스트
- 거래량·시총 등 다른 축의 신선도 지표
- investing.com 페이지네이션 구현 (별개 이슈)
