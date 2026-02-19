# kr-stock-scanner

한국 주식 시장 52주 신고가 종목 일일 스캐너. 장 마감 후 KOSPI/KOSDAQ/ETF 전 종목을 스캔하여 52주 신고가 종목을 찾고, 뉴스 기반 AI 분석 후 텔레그램으로 리포트를 전송합니다.

## 주요 기능

- KOSPI + KOSDAQ + ETF 전 종목 52주 신고가 감지
- 섹터별 분류 및 시장 통계
- 네이버 뉴스 수집 + OpenAI GPT 기반 상승 이유 분석
- 텔레그램 봇으로 일일 리포트 자동 전송
- SQLite에 일별 데이터 저장 (과거 조회 가능)

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
│   ├── cli.py            # CLI 진입점 (run/collect/history/stats)
│   ├── config.py         # 설정 로더
│   ├── models.py         # 데이터 모델
│   ├── db.py             # SQLite ORM
│   ├── collector.py      # KRX 데이터 수집 (pykrx)
│   ├── scanner.py        # 52주 신고가 감지
│   ├── news_fetcher.py   # 네이버 뉴스 수집
│   ├── ai_analyst.py     # OpenAI GPT 분석
│   └── reporter.py       # 텔레그램 리포트
└── tests/                # 26개 테스트
```
