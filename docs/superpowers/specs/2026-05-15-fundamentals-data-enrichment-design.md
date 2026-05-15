# Fundamentals Data Enrichment — Design Doc

- **Date:** 2026-05-15
- **Status:** Draft — awaiting user review
- **Author:** Jane (jane@kodax.com)
- **Scope:** A + B + C only (시가총액·발행주식수 / 현금흐름표 / 배당). D (일별 시세) 는 별도 spec.

---

## 1. Context & Motivation

`/analyze-stock` 보고서가 가치 있으려면 종목의 **이익의 질**과 **자본 효율**, **주주환원**을 정량 비교할 수 있어야 한다. 현재 `fundamentals_metrics` 의 다음 슬롯들이 미완성이라 이를 못 한다.

- `ocf_to_ni_ratio`, `fcf_positive_years`, `interest_coverage`, `peg` — 전 종목 NULL.
- `cashflow_score` — 위 입력 부재로 전 종목 NULL. 결국 `total_score` 도 4영역 중 3영역만으로 산출 중.
- EPS / BPS / 배당수익률 / 배당성향 등은 컬럼 자체가 없음.

원인은 분명하다.

1. `dart_financials` 가 9개 계정만 수집 — 현금흐름표·배당 계정이 빠져 있음.
2. `compute_all(market_caps, eps_map, bps_map)` 슬롯은 있지만 호출자가 채우지 않음 — KRX 데이터 파이프라인이 없음.

본 작업은 위 두 빈자리를 채워 분석 보고서의 정량 근거를 두텁게 만든다.

---

## 2. Goals / Non-Goals

### Goals

- DART에서 **영업활동현금흐름 / 유형자산취득 / 배당총액** 계정 추가 수집.
- pykrx로 **연말 시가총액 / 발행주식수 / 종가** 7년치 백필 + 진행 연도 갱신.
- `fundamentals_metrics` 에 신규 컬럼 9개 추가 (EPS, BPS, PSR, OCF, FCF, CAPEX/Revenue, 배당수익률, 배당성향, 연속배당년수).
- 기존 NULL 슬롯 (`ocf_to_ni_ratio`, `fcf_positive_years`) 정상 산출.
- `cashflow_score` 정상 산출 (전 종목 NULL 해소).
- `/analyze-stock` 호출 시 해당 종목 1건만 시총·종가를 fresh 하게 갱신.

### Non-Goals (Out of Scope)

- D. 일별 시세 (변동성·베타·모멘텀).
- 이자비용 수집 / `interest_coverage` 정상화.
- 분기 보고서 데이터.
- 우선주 EPS·시총 별도 처리.
- 신규 카테고리 (Cash Cow / Dividend Payer 등).
- 스코어 임계값 (★ 컷오프) 재조정.
- DART API 사용량 모니터링.
- 외부 EPS 비교 검증 모듈.

위 항목들은 본 작업이 안정화된 후 별도 spec 에서 다룬다.

---

## 3. Architecture Overview

```
[ pykrx ]                            [ DART OpenAPI ]
   │                                       │
   ▼                                       ▼
  market_data/fetcher.py          dart/fetcher.py (확장: 신규 account 4개)
   │                                       │
   ▼                                       ▼
 corp_market_yearly             dart_financials
  (corp_code, year, as_of_date,   (account 신규: 영업활동현금흐름,
   market_cap, shares, close)      유형자산취득, 배당총액, 주당배당금)
   │                                       │
   └─────────────┬─────────────────────────┘
                 ▼
       fundamentals/calculator.py
       (시계열 피벗 + 파생지표 산출)
                 │
                 ▼
        fundamentals_metrics
        (신규 컬럼 9개 + 기존 NULL 슬롯 채움)
                 │
                 ▼
        fundamentals/scorer.py
        (cashflow_score 정상 산출)
                 │
                 ▼
        fundamentals_scores
```

### 원칙

1. **출처별 격리** — pykrx 실패가 DART 수집을 막지 않고, 반대도 마찬가지. 한쪽이 실패해도 다른 쪽 결과는 계속 사용 가능.
2. **하나의 진실 원천** — 시계열은 모두 `(corp_code, year)` 키. `dart_financials` 와 `corp_market_yearly` 둘만 raw. `fundamentals_metrics` 는 둘로부터 derive.
3. **점진적 백필** — 모든 fetcher 는 `(corp_code, year)` 단위 멱등. 중간 실패 후 재실행해도 안전.

---

## 4. Schema Changes

### 4-1. `dart_financials` 에 신규 `account` 4개

테이블 스키마 자체는 변경 없음. 다음 4개 account 값이 새로 들어온다.

| account | DART account_id 후보 | 단위 |
| --- | --- | --- |
| `영업활동현금흐름` | `ifrs-full_CashFlowsFromUsedInOperatingActivities` | 원 |
| `유형자산취득` | `ifrs-full_PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities` | 원 |
| `배당총액` | `ifrs-full_DividendsPaidClassifiedAsFinancingActivities` (재무활동 CF 의 음수 항) | 원 |
| `주당배당금(보통주)` | XBRL 표준계정에 없을 수 있음. fallback 필요. | 원 |

**주의:** `주당배당금` 은 XBRL 외 사업보고서 본문 표일 가능성. 미수집 시 `dps = 배당총액 / shares_outstanding` 로 fallback 계산. 본문 파싱은 별도 follow-up.

### 4-2. 신규 테이블 `corp_market_yearly`

```sql
CREATE TABLE corp_market_yearly (
  corp_code           VARCHAR(8)  NOT NULL,
  ticker              VARCHAR(10) NOT NULL,
  year                INTEGER     NOT NULL,
  as_of_date          DATE        NOT NULL,   -- 그 해의 기준 영업일
  market_cap          BIGINT,                  -- 원
  shares_outstanding  BIGINT,                  -- 주
  close_price         INTEGER,                 -- 원, 검증용
  PRIMARY KEY (corp_code, year)
);
CREATE INDEX ix_corp_market_yearly_ticker ON corp_market_yearly(ticker);
```

`as_of_date` 규칙:

| year 상태 | as_of_date |
| --- | --- |
| 종료된 연도 | 그 해 마지막 영업일 (예: 2024 → 2024-12-30) |
| 진행 중인 연도 | 백필 실행일 또는 가장 최근 영업일 (재실행 시 upsert) |

### 4-3. `fundamentals_metrics` 신규 컬럼 9개

```sql
ALTER TABLE fundamentals_metrics ADD COLUMN eps FLOAT;
ALTER TABLE fundamentals_metrics ADD COLUMN bps FLOAT;
ALTER TABLE fundamentals_metrics ADD COLUMN psr FLOAT;
ALTER TABLE fundamentals_metrics ADD COLUMN ocf FLOAT;                       -- 절대값(억원)
ALTER TABLE fundamentals_metrics ADD COLUMN fcf FLOAT;                       -- 절대값(억원)
ALTER TABLE fundamentals_metrics ADD COLUMN capex_to_revenue FLOAT;          -- %
ALTER TABLE fundamentals_metrics ADD COLUMN dividend_yield FLOAT;            -- %
ALTER TABLE fundamentals_metrics ADD COLUMN payout_ratio FLOAT;              -- %
ALTER TABLE fundamentals_metrics ADD COLUMN consecutive_dividend_years INTEGER;
```

모두 NULLABLE. 기존 쿼리·시리얼라이저 영향 없음.

### 4-4. 백필 기간 & 대상 universe

- **기간:** `dart_financials` 의 기존 수집 범위 (2019 ~ 가장 최근 완료 연도) 와 정확히 동일. 본 spec 시점 기준 약 7년치.
- **Universe:** 기존 `Pipeline` 기본값과 동일 — KOSPI + KOSDAQ 보통주. ETF·우선주는 본 작업 범위에서 자동 제외 (기존 universe 정의가 그대로 적용).

---

## 5. New Modules / Files

```
src/
├── market_data/                  ← 신규 패키지
│   ├── __init__.py
│   ├── models.py                  ← MarketYearly dataclass
│   ├── db.py                      ← corp_market_yearly CRUD
│   ├── fetcher.py                 ← pykrx 래퍼
│   └── pipeline.py                ← 백필/리프레시 오케스트레이션
│
├── dart/
│   └── fetcher.py                 ← 확장: ACCOUNT_WHITELIST 에 4개 추가
│
└── fundamentals/
    ├── calculator.py              ← 신규 derive 로직
    ├── scorer.py                  ← cashflow_score 정상화 (입력 채워지면 자동 동작 가정)
    ├── db.py                      ← ALTER + 신규 컬럼 직렬화
    ├── models.py                  ← FundamentalsMetrics 필드 추가
    └── pipeline.py                ← compute_all 시그니처 변경, market_data 통합 로드
```

### 5-1. `src/market_data/fetcher.py`

```python
def fetch_yearly_market_data(
    tickers: list[str],
    years: list[int],
) -> list[MarketYearly]:
    """각 (ticker, year) 의 기준 영업일 시총·주식수·종가를 pykrx로 수집.
    
    - 연도당 1회 호출 (전 종목 일괄): get_market_cap_by_ticker(date, market='ALL')
    - 각 연도 기준일: 종료 연도 = 12월 마지막 영업일 / 진행 연도 = 호출 시점 가장 최근 영업일
    - 실패한 (ticker, year) 는 결과에서 빠짐. 호출자가 누락 판단.
    """
```

### 5-2. `src/dart/fetcher.py` 확장

신규 메서드 없음. 기존 `fetch_financials()` 의 `ACCOUNT_WHITELIST` 에 4개 항목만 추가.

### 5-3. `src/market_data/pipeline.py`

```python
class MarketDataPipeline:
    def refresh(self, years: list[int], tickers: list[str]) -> RefreshReport:
        """연도별 (ticker, year) upsert. 멱등."""
```

---

## 6. Derived Metrics

### 표기 규칙

- **LY** = Latest Year (가장 최근 완료된 연간보고서 연도, 본 spec 시점 기준 2025).
- **`market_cap_now`** = `corp_market_yearly` 에서 진행 연도(예: 2026) 행의 `market_cap` — 즉 `as_of_date` 가 가장 최근 영업일인 시총. PE/PB/PSR/배당수익률 분자에 사용.
- **신규 9개 컬럼 vs 기존 5개 슬롯:** § 4-3 에서 ALTER 로 추가하는 컬럼은 9개. 아래 산식 표 14개 중 나머지 5개 (`pe`, `pb`, `peg`, `ocf_to_ni_ratio`, `fcf_positive_years`) 는 기존 컬럼이지만 현재 NULL 인 슬롯 — 이번에 처음 정상 산출됨.

| 지표 | 산식 | 의존 데이터 |
| --- | --- | --- |
| `eps` | `당기순이익_LY / shares_outstanding_LY` | dart + market |
| `bps` | `자본총계_LY / shares_outstanding_LY` | dart + market |
| `psr` | `market_cap_now / 매출액_LY` | dart + market |
| `pe` (정확도↑) | `market_cap_now / 당기순이익_LY` | dart + market |
| `pb` (정확도↑) | `market_cap_now / 자본총계_LY` | dart + market |
| `peg` | `pe / (op_income_cagr_3y × 100)` (양수일 때만, 음수면 NULL) | 기존 |
| `ocf` | `영업활동현금흐름_LY` (억원으로 환산) | dart |
| `fcf` | `영업활동현금흐름_LY - 유형자산취득_LY` (억원) | dart |
| `ocf_to_ni_ratio` | 최근 3년 평균(OCF/당기순이익) — 단년 노이즈 회피 | dart |
| `fcf_positive_years` | 최근 5년 중 `fcf > 0` 인 연도 수 | dart |
| `capex_to_revenue` | `유형자산취득_LY × 100 / 매출액_LY` | dart |
| `dividend_yield` | `배당총액_LY × 100 / market_cap_now` | dart + market |
| `payout_ratio` | `배당총액_LY × 100 / 당기순이익_LY` (순이익 양수일 때만) | dart |
| `consecutive_dividend_years` | 최근 5년 끝에서부터 `배당총액 > 0` 인 연속 연도 수 | dart |

### Null 전파 규칙

- 분자/분모 중 어느 하나라도 NULL → 결과 NULL.
- 0 나누기 → NULL (예외 던지지 않음).

### derivation_audit

`compute_metrics` 가 부수적으로 dict 반환:

```python
{
  "ticker": "353200",
  "eps": None, "eps_reason": "shares_outstanding NULL for year=2025",
  "fcf": None, "fcf_reason": "유형자산취득 missing",
  "dividend_yield": 0.0, "dividend_yield_reason": "ok",
}
```

DB에 저장 안 함. 로그·리포트에만 사용.

---

## 7. Scorer Changes

### 약속

1. **`scorer.py` 의 cashflow 계산 함수는 손대지 않는다** — 입력(OCF/NI, FCF양수연수)이 채워지면 동작하도록 이미 짜여 있을 것이라 가정. 실제 코드 확인 후 NULL 반환 버그가 있으면 그때 수정.
2. **Liquidity / Profitability / Growth 점수 산식·가중치는 변경하지 않는다** — 새 지표 9개는 점수에 들어가지 않고 표시·분류용으로만.

### 영향

- `cashflow_score` 가 채워지면서 모든 종목의 `total_score` 가 일제히 상향됨.
- 이는 의도된 변화. 다만 ★ 컷오프 재조정이 필요할 수 있음 → 백필 후 분포 확인하여 별도 PR로 처리.
- `categories` (Quality/Growth/GARP/Caution) 분류 규칙 손대지 않음.

---

## 8. Pipeline Integration

### 8-1. `Pipeline.refresh_data()` 확장

```python
async def refresh_data(self, force, years, markets=None, market_map=None):
    # 1. DART (기존, 신규 account 4개 자동 포함)
    await self._fetch_dart(force, years, markets, market_map)
    
    # 2. 신규: 시장 데이터
    if self._market_pipeline is not None:
        tickers = [c.ticker for c in self._cache.load_corp_info(markets=markets)]
        self._market_pipeline.refresh(years=years, tickers=tickers)
```

- `_market_pipeline` 은 생성자 주입(옵션 인자). 미주입 시 기존 동작 그대로.
- DART 실패는 market 단계로 전파되지 않도록 try/except 격리.

### 8-2. `Pipeline.compute_all()` 시그니처 변경

```python
def compute_all(
    self,
    market_yearly: dict[str, list[MarketYearly]],   # 시계열 dict
    markets: list[str] | None = None,
) -> tuple[list[FundamentalsMetrics], list[ScoreCard]]:
    ...
```

- **`market_caps`, `eps_map`, `bps_map` 인자 제거.**
  - `market_caps` → `market_yearly` 로 진화 (시계열 dict).
  - `eps_map`, `bps_map` → 내부 derive 되므로 외부 주입 슬롯 불필요.
- DB 의존은 CLI 에 둠 (pipeline 자체는 DB 강결합 아님 — 테스트에 fixture dict 주입 가능).

### 8-3. CLI 변화

`src/fundamentals/cli.py` `run` 안에서:

| 옵션 | 동작 |
| --- | --- |
| (기본) `run` | DART + pykrx 모두 갱신, 메트릭 재계산 |
| `--skip-market` | pykrx 건너뜀 |
| `--skip-dart` | DART 건너뜀 |
| `--retry-failed` | NULL 행만 재시도 |
| `--years 2019 2020 ...` | 특정 연도만 |

별도 신규 서브커맨드 없음. `python -m src.fundamentals.cli run` 한 번으로 시장 데이터까지 통합.

### 8-4. 기존 `fundamentals_metrics` 행 재계산

- 신규 컬럼은 NULL → 백필 후 `compute_all()` 재실행으로 새 `as_of_date` 행이 추가됨 (누적 패턴).
- `pe/pb` 값이 변할 수 있음 (이전엔 eps/bps_map 비어 있어 다른 fallback). 정확도 개선이지 회귀 아님.

---

## 9. `/analyze-stock` Integration

### 결합 옵션 (선택됨: 옵션 나 + 옵션 가)

**옵션 가 — 안내:**
- `/analyze-stock` 이 가장 최근 `fundamentals_metrics.as_of_date` 를 검사.
- 30일 이상 오래됐으면 보고서 상단에 경고 한 줄: `⚠ 데이터가 N일 됐습니다. python -m src.fundamentals.cli run 실행을 권장합니다.`
- 종목이 `fundamentals_metrics` 에 없으면 명확한 안내: `이 종목은 아직 펀더멘털 파이프라인을 통과하지 않았습니다. python -m src.fundamentals.cli run 을 먼저 실행하세요.`

**옵션 나 — 1건 fresh 갱신:**
- 분석 시작 전에 해당 ticker 1건만 pykrx 호출 → `corp_market_yearly` 의 진행 연도 행 upsert (시총·종가·주식수).
- DART 는 손대지 않음 (사업보고서는 연 1회 갱신).
- 분석 시간 +1~2초. 보고서의 "현재 PE" 가 항상 fresh.

### 자동 트리거 안 함

전체 universe refresh 는 분 단위 소요. `/analyze-stock` 마다 돌면 비현실적. 사용자가 명시적으로 `python -m src.fundamentals.cli run` 실행 (월 1회 권장).

### 보고서 템플릿 변경 (analyze-stock.md)

신규 컬럼 활용을 위해 `.claude/commands/analyze-stock.md` 도 함께 업데이트:

- 핵심 평가 / 강점 / 약점 섹션에 EPS·FCF·OCF/NI·배당수익률 인용 추가.
- 가치/성장/품질 포지셔닝 표에 `EPS`, `dividend_yield`, `payout_ratio` 행 추가.
- "이익의 질" 코멘트: `ocf_to_ni_ratio` 와 `fcf_positive_years` 가 같이 약하면 회계이익과 현금이익 괴리 가능성 시사.

---

## 10. Failure Handling & Observability

### 10-1. 격리 단위

| 단위 | 실패 시 |
| --- | --- |
| pykrx 호출 1건 (연도 1회) | 3회 재시도 (지수 백오프 1s/2s/4s). 그래도 실패 시 그 해 데이터 NULL — 다른 연도/티커 계속 |
| DART 호출 1건 | 기존 fetcher 재시도 정책 유지 |
| (ticker, year) 단일 행 산출 | 한 행 실패가 다른 행을 막지 않음. 누락은 NULL |
| `compute_metrics` 종목 단위 | 종목 1개 계산 실패가 전체 compute_all 을 막지 않음. 로그 ERROR, 결과 리스트에서 제외 |

### 10-2. RefreshReport (반환 객체)

```python
@dataclass
class RefreshReport:
    started_at: datetime
    finished_at: datetime
    market_data: SourceReport
    dart: SourceReport

@dataclass
class SourceReport:
    requested_years: list[int]
    requested_tickers_count: int
    successful_rows: int
    failed_items: list[FailedItem]   # 처음 N개만
    duration_seconds: float
```

CLI 종료 시 stdout 1회 요약:

```
시장 데이터: 2,654 / 2,658 종목 성공 (4 실패)
DART: 18,594 / 18,606 (corp_code×year) 성공 (12 실패)
펀더멘털 계산: 2,612 / 2,658 종목 성공 — 46 종목은 시장 데이터 누락
시간: 4분 12초
실패 상세: logs/refresh-2026-05-15.log
```

### 10-3. 로깅

- INFO: 단계 시작/완료, 진행률
- WARNING: 단일 (ticker, year) 누락
- ERROR: 모듈 전체 실패
- DEBUG: 응답 raw 덤프 (트러블슈팅 시)

로그 파일: `logs/refresh-YYYY-MM-DD.log`.

### 10-4. 일관성 보증

- 모든 fetcher/DB writer 멱등 — (corp_code, year) 또는 (corp_code, year, account) upsert.
- pykrx 1년치 = 1 트랜잭션 (수천 행 한 번에 commit).
- 부분 실패 후 재실행: `--retry-failed` 옵션은 NULL 행만 재시도.

---

## 11. Testing Strategy

### 11-1. 단위 테스트

`calculator.py` derive 함수를 fixture 로 검증. 타깃 종목: 대덕전자(353200), SK하이닉스(000660) — 분석 보고서가 이미 있어 손계산 검증 용이.

```python
def test_eps_from_net_income_and_shares()
def test_fcf_negative_when_capex_exceeds_ocf()
def test_consecutive_dividend_years_breaks_on_zero()
def test_null_propagation_when_denominator_zero()
def test_derivation_audit_reports_missing_inputs()
```

### 11-2. 통합 테스트

```python
def test_market_db_upsert_is_idempotent()
def test_dart_financials_new_accounts_dont_break_old_queries()
```

### 11-3. 회귀 테스트

```python
def test_existing_fundamentals_metrics_columns_unchanged_shape()
def test_score_computation_remains_within_tolerance()
# liquidity/profitability/growth_score 자체는 변하지 않아야 함
```

### 11-4. Smoke

pykrx, DART 실제 호출 1건씩. CI 분리(opt-in). 기본 test suite 는 캐싱된 fixture 사용.

### 11-5. 수동 검증 시나리오

| 시나리오 | 기대 |
| --- | --- |
| `/analyze-stock 353200` 재실행 | EPS/FCF/배당수익률/OCF/NI 신규 필드 정상 채워짐. PE/PB 정확도 개선 |
| `/analyze-stock 000660` 재실행 | `ocf_to_ni_ratio`, `cashflow_score` NULL → 정상값. 2023년 사이클 적자가 FCF 에도 잡힘 |
| 시장 vs 섹터 중앙값 쿼리 | 신규 컬럼 중앙값 산출 가능 |
| `total_score` 분포 점검 | 히스토그램으로 ★ 임계값 보정 필요성 판단 |

---

## 12. Open Questions

본 spec 작성 시점에 미결로 남긴 항목. 구현 도중 명시적으로 확정한다.

1. **DART account_id 정확한 식별자** — 위 표는 후보. 실제 fetch 시 응답 검증 후 본 spec 부록에 확정 기록.
2. **`주당배당금` 수집 가능성** — XBRL 외 본문 표일 가능성. 미수집 시 `dps = 배당총액 / shares_outstanding` fallback 사용.
3. **pykrx 응답 스키마 변경 리스크** — 컬럼명 변경 대응 어댑터 한 줄 두기.
4. **★ 컷오프 재조정 필요 여부** — 백필 후 `total_score` 분포 확인. 별도 PR.

---

## 13. Migration & Rollback

### 마이그레이션

1. `corp_market_yearly` 테이블 생성 (`CREATE TABLE IF NOT EXISTS`).
2. `fundamentals_metrics` 신규 9개 컬럼 `ALTER TABLE ... ADD COLUMN`. 모두 NULLABLE.
3. `python -m src.fundamentals.cli run` 1회 실행으로 백필.

### 롤백

- 신규 컬럼 NULL 만 있고 기존 컬럼 그대로 → 코드만 이전 버전으로 되돌리면 정상 동작.
- `corp_market_yearly` 테이블은 그대로 둬도 무방 (다른 코드가 안 봄).
- `dart_financials` 의 신규 account 행은 그대로 두면 기존 쿼리(`account IN ('매출액',...)`)에 영향 없음.

비파괴적 변경만 있어 롤백 안전.
