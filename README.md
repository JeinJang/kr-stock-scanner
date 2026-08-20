# kr-stock-scanner

한국 주식 시장 52주 신고가 종목 일일 스캐너. 장 마감 후 KOSPI/KOSDAQ/ETF 전 종목을 스캔하여 **장중 고가가 직전 52주 최고 고가를 돌파한 종목**을 찾고, 뉴스 기반 AI 분석 후 텔레그램으로 리포트를 전송합니다.

## 주요 기능

- KOSPI + KOSDAQ + ETF 전 종목 52주 신고가 감지
- 섹터별 분류 및 시장 통계
- 네이버 뉴스 수집 + OpenAI GPT 기반 상승 이유 분석
- 텔레그램 봇으로 일일 리포트 자동 전송
- SQLite에 일별 데이터 저장 (과거 조회 가능)
- 돌파 신선도 판별: 직전 신고가 이후 경과 기간(재돌파 간격)과 몇 년 만의 최고가인지(갱신 깊이)를 함께 표기

## 설치

```bash
cd kr-stock-scanner

# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 패키지 설치
pip install -e ".[dev]"
```

## API 키 설정

`.env.example`을 복사한 후 실제 API 키를 입력합니다.

```bash
cp .env.example .env
```

`.env` 파일을 열어 아래 5개 값을 입력합니다:

```
OPENAI_API_KEY=sk-...
TELEGRAM_BOT_TOKEN=7123456789:AAH1bGzR...
TELEGRAM_CHAT_ID=123456789
NAVER_CLIENT_ID=your_naver_client_id
NAVER_CLIENT_SECRET=your_naver_client_secret
```

### API 키 발급 방법

| API | 발급처 | 비용 |
|-----|--------|------|
| OpenAI | [platform.openai.com](https://platform.openai.com) | 종량제 (일 ~$0.01) |
| Telegram Bot | 텔레그램에서 [@BotFather](https://t.me/BotFather)에게 `/newbot` 전송 | 무료 |
| Telegram Chat ID | 봇에 메시지 전송 후 `https://api.telegram.org/bot<TOKEN>/getUpdates` 에서 확인 | - |
| Naver Search API | [developers.naver.com](https://developers.naver.com) 에서 애플리케이션 등록 | 무료 (일 25,000건) |

## 사용법

### 전체 파이프라인 실행

장 마감 후 (15:30 이후) 실행하면 데이터 수집 → 52주 신고가 스캔 → 뉴스 수집 → AI 분석 → 텔레그램 리포트 전송까지 자동으로 진행됩니다.

```bash
# 오늘 날짜로 실행
python -m src.cli run

# 특정 날짜 지정
python -m src.cli run --date 20260219
```

### 개별 명령어

```bash
# 데이터 수집만
python -m src.cli collect
python -m src.cli collect --date 20260219

# 과거 스캔 결과 조회
python -m src.cli history                    # 최근 추이
python -m src.cli history --date 20260219    # 특정 날짜 상세

# 통계
python -m src.cli stats              # 최근 30일 통계
python -m src.cli stats --days 60    # 최근 60일 통계
```

### 실행 예시

```
$ python -m src.cli run
52주 신고가 스캔 시작: 20260219
1/5 데이터 수집 중...
2/5 섹터 정보 수집 중...
3/5 52주 신고가 스캔 중...
4/5 뉴스 수집 및 AI 분석 중...
5/5 리포트 전송 중...
텔레그램 리포트 전송 완료!
완료! 32개 신고가 종목 발견
```

## 주가 예측 (Forecast)

52주 신고가 스캔 결과를 기반으로 TimesFM 모델을 이용한 주가/매크로 예측 및 HTML 리포트 생성.

### 설치

```bash
# 기본 패키지 설치 (스캐너만 사용)
pip install -e ".[dev]"

# forecast 모듈 추가 설치 (TimesFM + PyTorch)
pip install -e ".[forecast]"
```

### 추가 설정

`.env` 파일에 추가:
```
ECOS_API_KEY=your_ecos_api_key    # 한국은행 ECOS API
FRED_API_KEY=your_fred_api_key    # FRED API
```

| API | 발급처 | 비용 |
|-----|--------|------|
| ECOS | [ecos.bok.or.kr](https://ecos.bok.or.kr) → 개발자센터 → 인증키 신청 | 무료 |
| FRED | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) | 무료 |

### 사용법

```bash
# 스캐너 먼저 실행
python -m src.cli run

# 예측 실행 (가장 최근 스캔 결과 기반)
python -m src.forecast.cli run

# 특정 날짜 & 예측 기간 지정
python -m src.forecast.cli run --date 20260416 --horizon 40
```

리포트는 `reports/forecast-YYYY-MM-DD.html`에 생성되며 브라우저에서 자동으로 열립니다.

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

## 연관 기업 발굴 (Related)

DART 사업보고서를 GPT로 분석하여 공급망/고객사/경쟁사/계열사/자회사 관계를 추출하고, 인터랙티브 네트워크 그래프 HTML 리포트로 시각화합니다.

### 사전 조건

DART 데이터(corp_info)가 먼저 적재되어 있어야 합니다:

```bash
python -m src.fundamentals.cli refresh
```

### 사용법

```bash
# 단일 종목 (1-hop 기본)
python -m src.related.cli show 042700

# 2-hop 확장
python -m src.related.cli show 042700 --depth 2

# 강제 재추출 (사업보고서 미변경에도 GPT 재호출)
python -m src.related.cli show 042700 --refresh

# 명시적 티커 리스트 배치 추출
python -m src.related.cli batch --tickers 005930,000660,042700

# 저장된 관계 통계
python -m src.related.cli stats
```

리포트는 `reports/related-<ticker>-YYYY-MM-DD.html`에 생성되며 브라우저에서 자동으로 열립니다. 사업보고서 접수번호(`rcept_no`) 기준으로 캐싱되므로 새 보고서가 나올 때까지 GPT 재호출이 없습니다.

## 자동 실행 (cron)

매일 평일 16:00에 자동 실행하려면 crontab에 등록합니다:

```bash
crontab -e
```

아래 줄을 추가합니다 (경로를 실제 경로로 수정):

```
0 16 * * 1-5 cd /path/to/kr-stock-scanner && /path/to/kr-stock-scanner/.venv/bin/python -m src.cli run >> /tmp/kr-scanner.log 2>&1
```

## 설정 변경

`config.yaml`에서 스캔 설정을 변경할 수 있습니다:

```yaml
scanner:
  markets: ["KOSPI", "KOSDAQ", "ETF"]  # 대상 시장
  lookback_days: 250                    # 52주 = 약 250 거래일
  max_ai_analyze: 50                    # AI 분석 최대 종목 수

news:
  max_articles_per_stock: 5             # 종목당 뉴스 수집 수

ai:
  model: "gpt-5-nano"                   # OpenAI 모델
  max_tokens: 300                       # 응답 최대 토큰

telegram:
  enabled: true                         # false로 설정 시 콘솔 출력만
```

## 테스트

```bash
pytest tests/ -v
```

## 프로젝트 구조

```
kr-stock-scanner/
├── config.yaml           # 스캔 설정
├── .env                  # API 키 (gitignored)
├── data/
│   └── scanner.db        # SQLite DB (자동 생성)
├── src/
│   ├── cli.py            # 스캐너 CLI (run/collect/history/stats)
│   ├── config.py         # 설정 로더
│   ├── models.py         # 데이터 모델
│   ├── db.py             # SQLite ORM
│   ├── collector.py      # KRX 데이터 수집
│   ├── scanner.py        # 52주 신고가 감지
│   ├── news_fetcher.py   # 네이버 뉴스 수집
│   ├── ai_analyst.py     # OpenAI GPT 분석
│   ├── reporter.py       # 텔레그램 리포트
│   └── forecast/         # 주가 예측 모듈
│       ├── cli.py        # 예측 CLI (run/list-reports)
│       ├── macro_fetcher.py  # ECOS + FRED 매크로 데이터
│       ├── stock_fetcher.py  # KRX 종목 과거 가격
│       ├── predictor.py      # TimesFM 예측 엔진
│       ├── report.py         # HTML 리포트 생성
│       └── templates/        # Jinja2 템플릿
└── tests/                # 48개 테스트
```
