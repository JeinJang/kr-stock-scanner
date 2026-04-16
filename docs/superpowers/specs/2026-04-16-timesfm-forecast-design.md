# TimesFM 기반 주가 예측 기능 설계

## 개요

52주 신고가 스캔 결과를 기반으로, Google TimesFM(시계열 Foundation Model)을 활용하여 개별 종목 가격 예측 + 매크로 지수 예측을 수행하고, 인터랙티브 HTML 리포트를 생성하는 기능.

기존 `python -m src.cli run` (스캐너)과 독립적으로 실행되며, 스캔 결과를 DB에서 읽어 예측에 활용한다.

## 프로젝트 구조

기존 `kr-stock-scanner` 프로젝트에 `src/forecast/` 패키지를 추가한다.

```
src/
├── cli.py                        # 기존 스캐너 (변경 없음)
├── forecast/
│   ├── __init__.py
│   ├── cli.py                    # 예측 CLI 엔트리포인트
│   ├── macro_fetcher.py          # ECOS API + FRED API 매크로 데이터 수집
│   ├── stock_fetcher.py          # 개별 종목 과거 가격 수집 (KRX 클라이언트 재사용)
│   ├── predictor.py              # TimesFM 모델 로드 & 예측
│   ├── report.py                 # HTML 리포트 생성 (Plotly + Jinja2)
│   └── models.py                 # 예측 관련 Pydantic 모델
├── db.py                         # 공유 — 스캔 결과 & AI 분석 읽기
├── config.py                     # 공유 — ECOS/FRED API 키 추가
└── ...
```

## CLI 인터페이스

```bash
# 가장 최근 스캔 결과 기반으로 예측 실행
python -m src.forecast.cli run

# 특정 날짜 스캔 결과 기반
python -m src.forecast.cli run --date 20260416

# 예측 기간 변경 (기본 60 거래일, 약 3개월)
python -m src.forecast.cli run --horizon 40
```

- Typer 기반 CLI (기존 스캐너와 동일한 패턴)
- 실행 후 `reports/forecast-YYYY-MM-DD.html` 생성 및 브라우저 자동 오픈

## 데이터 수집

### 매크로 데이터 (`macro_fetcher.py`)

**한국은행 ECOS API** (국내 지표):
- KOSPI 지수
- KOSDAQ 지수
- 원/달러 환율
- 한국 기준금리

**FRED API** (글로벌 지표):
- S&P 500
- NASDAQ Composite
- US Federal Funds Rate (미국 기준금리)

각 지표별로 최근 250 거래일(약 1년) 데이터를 수집하여 TimesFM context로 사용.

API 키는 `.env`에 추가:
```
ECOS_API_KEY=...
FRED_API_KEY=...
```

### 종목 데이터 (`stock_fetcher.py`)

기존 `KrxLoginClient.get_market_ohlcv_by_date()`를 재사용하여 스캔 결과에 포함된 종목들의 과거 250일 종가 데이터를 수집.

## 예측 엔진 (`predictor.py`)

### TimesFM 모델

- 모델: `google/timesfm-2.5-200m-pytorch` (200M 파라미터)
- context: 최대 512 토큰 (과거 데이터)
- horizon: 20~60 거래일 (기본 60)
- quantile head 활성화 — 불확실성 추정 (10th~90th percentile)
- RevIN(Reverse Instance Normalization) 활성화 — 가격 정규화

### 디바이스 자동 감지

- Mac: CPU 자동 사용
- RTX 5070 데스크탑: CUDA 자동 활용
- PyTorch `torch.device` 자동 선택에 위임

### 예측 흐름

1. 매크로 지표 6개 배치 예측
2. 신고가 종목들 배치 예측 (한 번에 모두)
3. 각 예측마다 point forecast + quantile forecast 반환

### 출력 데이터 구조

```python
class ForecastResult(BaseModel):
    ticker: str              # 종목코드 또는 지표명 (예: "KOSPI", "005930")
    name: str                # 종목명 또는 지표명
    category: str            # "macro" | "stock"
    history: list[float]     # 과거 가격 (차트용)
    dates_history: list[str] # 과거 날짜
    forecast: list[float]    # point forecast
    dates_forecast: list[str]# 예측 날짜
    quantile_low: list[float]   # 10th percentile
    quantile_high: list[float]  # 90th percentile
    predicted_return: float  # 예측 수익률 (%)
    uncertainty: float       # quantile 폭 기반 불확실성 지표
```

## HTML 리포트 (`report.py`)

Plotly로 인터랙티브 차트를 생성하고, Jinja2 템플릿으로 단일 HTML 파일에 모든 것을 담는다. Plotly.js를 인라인으로 포함하여 오프라인에서도 동작.

### 리포트 구성 (우선순위 순서)

**섹션 1 — 매크로 대시보드**
- 6개 지표 예측 차트를 2x3 그리드로 배치
- 각 차트: 과거 가격(실선) + 예측(점선) + 신뢰구간(음영 band)
- 차트 아래 방향성 요약 테이블: 지표명 | 현재값 | 예측값(60일 후) | 변동률 | 방향(상승/하락/횡보)

**섹션 2 — 종목별 예측 차트**
- 종목당 개별 차트: 과거 가격 + 예측 + quantile band
- 차트에 52주 신고가 돌파 시점 마킹
- 종목이 많으면 접이식(collapsible) 처리

**섹션 3 — 종목 랭킹**
- 예측 수익률 기준 내림차순 테이블
- 컬럼: 순위 | 종목명 | 현재가 | 예측가(60일) | 예측 수익률 | 불확실성 | 섹터
- 상위 종목 하이라이트 (초록), 하락 예측 종목은 빨강

**섹션 4 — 리스크 지표**
- quantile 폭(90th - 10th) 기반으로 불확실성 점수 산출
- 수익률(Y축) vs 불확실성(X축) scatter plot — 좌상단(고수익+저불확실성)이 유망 종목
- 종목별 위험등급: 낮음/보통/높음

**섹션 5 — AI 분석 연동**
- 기존 스캐너의 GPT 분석 결과를 DB에서 가져와서 종목 차트 옆에 표시
- [상승 원인], [핵심 뉴스], [투자 포인트] 텍스트 포함

### 파일 출력

```
reports/forecast-YYYY-MM-DD.html
```

생성 후 `webbrowser.open()`으로 자동 오픈.

## 실행 파이프라인

`python -m src.forecast.cli run` 실행 시 전체 흐름:

```
1. DB에서 최근 스캔 결과 로드
   └── 신고가 종목 리스트 + AI 분석 결과

2. 데이터 수집 (asyncio 병렬)
   ├── macro_fetcher: ECOS API → KOSPI, KOSDAQ, 환율, 금리
   ├── macro_fetcher: FRED API → S&P500, NASDAQ, 미국금리
   └── stock_fetcher: KRX → 종목별 과거 250일 종가

3. TimesFM 예측 (배치)
   ├── 매크로 6개 지표 배치 예측
   └── 종목들 배치 예측

4. 후처리
   ├── 예측 수익률 계산
   ├── 불확실성 점수 산출
   └── 종목 랭킹 정렬

5. HTML 리포트 생성
   └── reports/forecast-YYYY-MM-DD.html

6. 브라우저 자동 오픈
```

- 스캔 결과가 DB에 없으면 에러: "먼저 `python -m src.cli run`을 실행하세요"

## 의존성 추가

`pyproject.toml`에 추가:

```
timesfm
torch
plotly
jinja2
fredapi
```

## 설정 추가

`.env`에 추가:
```
ECOS_API_KEY=...
FRED_API_KEY=...
```

`config.yaml`에 forecast 섹션 추가:
```yaml
forecast:
  horizon: 60
  model: "google/timesfm-2.5-200m-pytorch"
  report_dir: "reports"
```
