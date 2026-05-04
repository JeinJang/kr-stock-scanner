# Open DART 기반 펀더멘털 스크리너 설계

## 개요

Open DART API를 활용하여 KOSPI + KOSDAQ 전 상장사(~2,772개)의 재무 데이터를 수집하고, 4차원 점수와 카테고리 분류로 투자 후보군을 자동 발굴하는 스크리너.

기존 52주 신고가 스캐너 / forecast 모듈과 독립적으로 실행되며, 결과는 DB에 저장되어 후속 모듈(연관 기업 발굴 등)에서 재사용 가능하다.

## 프로젝트 구조

```
src/
├── cli.py                       # 기존 스캐너 (변경 없음)
├── forecast/                    # 기존 forecast 모듈
├── dart/                        # 신규 — Open DART 공유 데이터 레이어
│   ├── __init__.py
│   ├── client.py                # API 호출 (rate limit, 재시도)
│   ├── fetcher.py               # 종목 유니버스, 재무제표 수집
│   ├── cache.py                 # DB 기반 캐싱
│   └── models.py                # CorpInfo, FinancialStatement
└── fundamentals/                # 신규 — D 분석 레이어
    ├── __init__.py
    ├── cli.py                   # 엔트리포인트
    ├── scorer.py                # 4차원 점수 산정
    ├── classifier.py            # 카테고리 분류
    ├── report.py                # HTML 리포트
    ├── templates/report.html    # Jinja2 템플릿
    └── models.py                # FundamentalsMetrics, ScoreCard
```

## CLI 인터페이스

```bash
# 캐시된 데이터로 빠르게 스크리닝 + HTML 리포트
python -m src.fundamentals.cli run

# 데이터 갱신 후 스크리닝 (분기 보고서 공시 후)
python -m src.fundamentals.cli run --refresh

# 데이터 갱신만 (분석 없이)
python -m src.fundamentals.cli refresh

# 특정 종목 스코어카드 조회
python -m src.fundamentals.cli show 005930
```

- Typer 기반 (기존 스캐너와 동일 패턴)
- 실행 후 `reports/fundamentals-YYYY-MM-DD.html` 생성 및 브라우저 자동 오픈

## 데이터 수집 레이어 (`src/dart/`)

### `client.py` — Open DART API 호출

- 환경변수 `OPENDART_API_KEY` 사용
- 호출량 제한: 분당 1,000회, 일 20,000회 → `asyncio.Semaphore`로 자동 스로틀링
- 5xx 에러 시 지수 백오프 재시도 (최대 3회)
- 호출 단위: 다중회사 API(`fnlttMultiAcnt.json`)로 100개씩 배치

### `fetcher.py` — 데이터 수집 함수

```python
async def fetch_corp_universe() -> list[CorpInfo]:
    """KOSPI + KOSDAQ 전 상장사 마스터 수집."""

async def fetch_financials(
    corp_codes: list[str],
    years: list[int],
    report_codes: list[str],  # 사업보고서, 1/2/3분기
) -> list[FinancialStatement]:
    """N년치 재무제표 시계열 수집."""
```

### `cache.py` — DB 캐싱

기존 `data/scanner.db`에 테이블 추가:
- `dart_corp_info` — 종목 마스터
- `dart_financials` — 재무제표 시계열
- `dart_meta` — 마지막 갱신 시각

**갱신 로직:**
- 기본: DB에 데이터 있으면 재사용 (호출 0회)
- `--refresh` 플래그: 캐시 무시하고 강제 재수집
- 자동 갱신: 마지막 갱신이 `cache_ttl_days`(기본 30일) 이상 경과한 경우 자동으로 재수집 (사용자 안내 메시지 출력)

### `models.py`

```python
class CorpInfo(BaseModel):
    corp_code: str            # DART 고유 코드
    ticker: str
    name: str
    market: str               # "KOSPI" | "KOSDAQ"

class FinancialStatement(BaseModel):
    corp_code: str
    year: int
    quarter: int              # 0=사업보고서(연간), 1/2/3=분기
    account: str              # "매출액", "영업이익", "당기순이익", "자산총계" 등
    value: float
```

## 점수 산정 (`src/fundamentals/scorer.py`)

### 4개 차원 (각 0~25점)

**A. 유동성/안정성 (25점)**
- 유동비율: 1.5↑ 만점, 1.0↓ 0점
- 이자보상배율: 5↑ 만점, 1↓ 0점
- 부채비율: 100%↓ 만점, 200%↑ 0점

**B. 수익성 (25점)**
- ROE 3년 평균: 15%↑ 만점, 5%↓ 0점
- ROIC 3년 평균: 15%↑ 만점, 5%↓ 0점
- 영업이익률: 산업 중앙값 대비 상대 평가

**C. 성장성 (25점)**
- 매출 3년 CAGR: 15%↑ 만점, 0%↓ 0점
- 영업이익 3년 CAGR: 15%↑ 만점, 0%↓ 0점

**D. 현금흐름 품질 (25점)**
- OCF / 순이익 비율: 1.0~1.2 만점, 0.5↓ 감점
- FCF 흑자 연수 (3년 중)

**종합 점수 = 4개 합산 (최대 100점)**

### 등급 변환

| 종합 점수 | 등급 |
|----------|------|
| 90~100 | ★★★★★ |
| 75~89  | ★★★★☆ |
| 60~74  | ★★★☆☆ |
| 45~59  | ★★☆☆☆ |
| 0~44   | ★☆☆☆☆ |

### 데이터 부족 처리

특정 차원 산정 불가 시(예: 신생 상장사) 그 차원 점수는 `None`, 종합 점수는 가용 차원만으로 비례 환산. 리포트에 "데이터 부족" 명시.

## 카테고리 분류 (`src/fundamentals/classifier.py`)

종목을 5개 카테고리로 라벨링. 다중 라벨 가능.

**Quality (우량주)**
- 종합 점수 75점 이상
- ROE 3년 평균 15% 이상
- 부채비율 100% 미만

**Value (가치주)**
- P/E ≤ 시장(KOSPI/KOSDAQ 분리) 중앙값 × 0.7
- P/B ≤ 시장(KOSPI/KOSDAQ 분리) 중앙값 × 0.7
- 안정성 점수 18/25 이상 (가치 함정 회피)

**Growth (성장주)**
- 매출 3년 CAGR 20% 이상
- 영업이익 3년 CAGR 15% 이상 (적자 기업은 매출만 평가)

**GARP (합리적 가격의 성장주)**
- 매출 3년 CAGR 15% 이상
- PEG = (P/E) / (이익 성장률) ≤ 1.0

**Caution (주의)**
- 종합 점수 45점 미만, 또는
- OCF/순이익 비율 < 0.5 (회계 위험), 또는
- 이자보상배율 < 1 (재무 위기)

### P/E, P/B 데이터 출처
- DART에서 EPS, BPS, 발행주식수 수집
- KRX(`KrxClient`)에서 종가 조달
- P/E, P/B 자체 계산

## DB 스키마 (3계층)

### 1. 원본 재무 데이터 (`src/dart/cache.py` 관리)
- `dart_corp_info` — 종목 마스터
- `dart_financials` — 분기별 원본 계정 (매출액, 영업이익, 자산, 부채 등)

### 2. 파생 지표 (`src/fundamentals` 관리)
- `fundamentals_metrics`
  - 종목별 계산된 지표
  - ROE, ROIC, 영업이익률, 부채비율, 유동비율, 이자보상배율
  - 매출/영업이익 3년 CAGR
  - OCF/순이익 비율, FCF
  - P/E, P/B, PEG
  - 계산 기준일(`as_of_date`)

### 3. 점수 & 분류
- `fundamentals_scores`
  - 차원별 점수: liquidity_score, profitability_score, growth_score, cashflow_score
  - total_score, grade(★)
  - categories (JSON 배열, 다중 라벨)
  - computed_at

이 구조의 장점:
- **원본**: 신뢰 가능한 진실의 원천
- **파생 지표**: 다른 모듈에서 재사용 가능
- **점수**: 시점별 스코어 이력 추적 가능

## HTML 리포트 (`src/fundamentals/report.py`)

Plotly + Jinja2로 단일 HTML 파일 생성. `reports/fundamentals-YYYY-MM-DD.html`.

### 섹션 1 — 시장 개요
- 분석 종목 수, 카테고리별 분포 (도넛 차트)
- 시장 평균/중앙값 P/E, P/B (참고용)

### 섹션 2 — 통합 랭킹 (전체 ~2,772개)
- 종합 점수 내림차순 정렬, 컬럼 헤더 클릭으로 재정렬
- 컬럼: 순위 | 티커 | 종목명 | 시장 | 섹터 | 등급(★) | 종합점수 | 4차원 점수(소형 막대) | 카테고리 뱃지 | P/E | P/B | ROE
- 카테고리 뱃지 색상: Quality(파랑), Value(초록), Growth(빨강), GARP(보라), Caution(회색)
- 검색/필터 UI:
  - 종목명/티커 검색 박스
  - 시장 필터 (KOSPI/KOSDAQ/All)
  - 카테고리 다중 필터
  - 점수 범위 슬라이더
- 가상 스크롤로 화면에 보이는 행만 렌더링 (~30행)
- 행 클릭으로 상세 스코어카드 펼침

### 섹션 3 — 카테고리별 Top 10
- Quality / Value / Growth / GARP 4개 섹션, 각 Top 10 카드 그리드
- 카드: 종목명, 종합점수, 핵심 지표 3개 (카테고리별 다름)

### 섹션 4 — 4차원 점수 분포 산점도
- X축: 안정성 + 수익성 점수
- Y축: 성장성 점수
- 점 크기: 시가총액
- 색상: 카테고리

### 섹션 5 — 종목 상세 스코어카드 (섹션 2 행 펼침으로 통합)
- 4차원 레이더 차트
- 3년 시계열 미니 차트: 매출, 영업이익, ROE, OCF
- 산정에 사용된 원본 수치 테이블

### 출력
- `reports/fundamentals-YYYY-MM-DD.html`
- `webbrowser.open()`으로 자동 오픈

## 실행 파이프라인

`python -m src.fundamentals.cli run` 실행 시:

```
1. DB 캐시 확인
   ├── 없거나 --refresh: DART에서 데이터 수집 → DB 저장
   └── 있음: 캐시 로드

2. KRX에서 최신 종가 + 시가총액 수집 (P/E, P/B 계산용)

3. 파생 지표 계산 (ROE, ROIC, CAGR, P/E, P/B 등)
   └── fundamentals_metrics 테이블 저장

4. 4차원 점수 산정 + 카테고리 분류
   └── fundamentals_scores 테이블 저장

5. HTML 리포트 생성
   └── reports/fundamentals-YYYY-MM-DD.html

6. 브라우저 자동 오픈
```

## 의존성 추가

`pyproject.toml`:
```
# 기존 의존성으로 충분 (requests, sqlalchemy, pydantic, plotly, jinja2 등)
# Open DART는 단순 REST API라 별도 라이브러리 불필요
```

## 설정 추가

`.env`:
```
OPENDART_API_KEY=your_dart_api_key
```

`config.yaml`:
```yaml
fundamentals:
  years_lookback: 10           # 시계열 수집 연수
  cache_ttl_days: 30           # 캐시 만료 일수
  report_dir: "reports"
  market_filter: ["KOSPI", "KOSDAQ"]
```

## 향후 확장 (별도 스펙)

- **C: 연관 기업 발굴** — 사업보고서 텍스트 마이닝(NER) + 지식 그래프. `src/dart/`의 사업보고서 원본 수집 기능을 재사용.
- **신고가 스캐너 통합** — 52주 신고가 종목에 펀더멘털 등급 컬럼 추가하여 텔레그램 리포트에 반영.
