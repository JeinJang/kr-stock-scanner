# AI HW / 빅테크 시총 비율 지표 설계

날짜: 2026-08-23
상태: 승인됨 (구현 계획 작성 대기)

## 목적

AI 하드웨어 종목 시총 합계와 빅테크 종목 시총 합계의 비율을 추적하여 시장 고점
신호를 감지한다. 비율이 0.8(80%) 이상이면 주의 구간으로 본다는 가설에 기반한다.

- AI HW 그룹: 엔비디아(NVDA), 브로드컴(AVGO), 삼성전자(005930.KS),
  SK하이닉스(000660.KS), 마이크론(MU), 샌디스크(SNDK)
- 빅테크 그룹: 아마존(AMZN), 테슬라(TSLA), 마이크로소프트(MSFT), 메타(META),
  구글(GOOGL)
- 벤치마크: SPY, RSP (시장 대비 과열 비교용)

## 결정 사항 (브레인스토밍 결과)

1. **배치**: 하이브리드. kr-stock-scanner에 CLI 명령으로 구현하고, 그 CLI를
   호출·해석하는 얇은 Claude 스킬을 별도 등록한다. 로직은 전부 CLI에 둔다.
2. **과거 시총 재구성(A 방식)**: 현재 상장주식수 × 과거 수정종가로 근사한다.
   자사주 매입 오차는 분자·분모 동일 방식이므로 비율에서는 거의 상쇄된다.
   이후 매일 실행 시 실제 시총 스냅샷을 저장해 정확한 데이터를 쌓는다.
3. **실행 방식**: 별도 명령(`aihw`)으로 제공하되, 매일 `run` 파이프라인 마지막에
   52주 신고가 리포트 전송 후 같은 텔레그램 채널로 PNG + 요약 캡션을 전송한다
   (config 토글로 제어).
4. **공유 산출물**: 본인용 인터랙티브 HTML 리포트 + 공유용 PNG 이미지 +
   텔레그램 캡션 텍스트. 캡션에는 종목별 시총과 전일 대비 변화율을 포함한다.

## 아키텍처

신규 모듈 `src/aihw/` — 기존 모듈 패턴(models/fetcher/db/compute/report/pipeline
분리)을 따른다.

### fetcher.py

- yfinance로 전 종목의 수정종가 이력과 현재 상장주식수를 조회한다.
- 일별 시총(USD) = 현재 상장주식수 × 해당일 수정종가.
- 삼성전자·SK하이닉스는 원화 가격이므로 일별 USD/KRW 환율(yfinance `KRW=X`)로
  달러 환산한다. 환율 누락일은 직전 영업일 값으로 전방 채움(ffill)한다.
- SPY·RSP는 종가만 수집한다 (지수화 비교용).
- 미국·한국 거래일 불일치: 날짜 축을 합집합으로 만들고 종목별 휴장일은 직전
  종가로 전방 채움한다.
- 네트워크 실패 시 종목 단위로 재시도하고, 끝내 실패한 종목은 명시적으로
  에러를 내고 중단한다 (일부 종목 누락 상태로 비율을 계산하면 왜곡되므로
  부분 성공을 허용하지 않는다).

### models.py + db.py

- SQLite `data/aihw.db`, SQLAlchemy 사용 (기존 market_data 패턴).
- `daily_caps` 테이블: date, ticker, close(현지통화), shares, market_cap_usd,
  source(`backfill` | `snapshot`), created_at. (date, ticker) 유니크.
- 백필: 기준일 이전 과거 구간을 A 방식 근사치로 1회 채운다. 매일 실행 시
  당일(미국 전일) 데이터는 `snapshot`으로 저장하며, 같은 (date, ticker)에
  snapshot이 이미 있으면 backfill로 덮어쓰지 않는다.
- 벤치마크(SPY, RSP)와 환율(KRW=X)도 같은 테이블에 저장한다 (shares,
  market_cap_usd는 NULL).

### compute.py (순수 계산 — 단위 테스트 대상)

- 그룹별 시총 합산 시계열.
- 비율 시계열: AI HW 합계 ÷ 빅테크 합계.
- 지수화: 기준일 = 100 (기본 2026-01-10, config으로 변경 가능).
- 임계값 판정: 당일 비율이 0.8 이상인지, 전일 대비 0.8을 상향/하향 돌파했는지.
- 요약 통계: 전일 대비 변화(%p), 최근 30일 최고/최저, 종목별 전일 대비 변화율.

### report.py

- plotly 차트 2개:
  1. 비율 추이 (절대값, y축 %, 0.8에 빨간 수평 경고선)
  2. 시총 지수 비교 (AI HW 합계, 빅테크 합계, SPY, RSP — 기준일=100)
- 산출물 3종:
  - `reports/aihw-YYYY-MM-DD.html` — 인터랙티브 HTML (차트 2개 + 종목별 시총
    테이블: 현재 시총, 전일 대비, 30일 대비)
  - `reports/aihw-YYYY-MM-DD.png` — 차트 2개를 세로로 합친 공유용 이미지
    (kaleido 렌더)
  - 텔레그램 캡션 텍스트 — 아래 형식, 1,024자 이내:

```
📊 AI HW / 빅테크 비율: 76.2% (경고선 80%)
전일 대비 +0.8%p · 30일 최고 78.1%

[AI HW] $6.82T
· 엔비디아    $4.21T (+1.2%)
· ...

[빅테크] $8.95T
· MS        $3.12T (+0.4%)
· ...
```

- 그룹 내 시총 내림차순 정렬, 괄호는 전일 대비 시총 변화율.
- 비율이 0.8 이상이면 캡션 첫 줄에 ⚠️ 경고 표시를 붙인다.

### pipeline.py

수집 → 저장 → 계산 → 리포트 생성 → (옵션) 텔레그램 전송 오케스트레이션.

## CLI

- `python -m src.cli aihw` — 수집·저장·리포트(HTML/PNG) 생성·터미널 요약(rich).
- `--send` — 추가로 PNG + 캡션을 텔레그램 채널로 전송.
- `--days N` — 조회·차트 기간 (기본: 기준일부터 오늘까지).
- `run` 파이프라인 통합: `config.yaml`의 `aihw.auto_send: true`이면 기존 52주
  신고가 리포트 전송 후 aihw 파이프라인을 실행해 같은 채널로 전송한다. aihw
  단계 실패는 로그만 남기고 run 전체를 실패시키지 않는다.

## 텔레그램

- 기존 `reporter.py`의 봇 인프라를 재사용하되, 사진 전송(`send_photo` +
  caption)이 없으므로 추가한다.
- 채널 chat_id: `AIHW_TELEGRAM_CHAT_ID` 환경변수가 있으면 사용, 없으면 기존
  `TELEGRAM_CHAT_ID`로 폴백.

## config.yaml 추가

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

## Claude 스킬

`~/.claude/skills/aihw-ratio/SKILL.md` — 사용자가 "AI HW 비율", "고점 지표",
"빅테크 대비" 등을 물으면:

1. kr-stock-scanner 디렉토리에서 `.venv` 활성화 후 `python -m src.cli aihw` 실행
2. 터미널 요약을 해석해 현재 비율, 0.8까지 거리, 추세를 답변
3. HTML 리포트 경로 안내

스킬에는 로직이 없다 — CLI 호출과 결과 해석만 담당한다.

## 의존성 추가

- `yfinance` — 미국·한국 종목 가격/주식수/환율 조회
- `kaleido` — plotly PNG 렌더링

## 테스트

- `compute.py` 전체: 합산, 비율, 지수화, 임계값 돌파 판정, 요약 통계 —
  단위 테스트 (TDD).
- `report.py` 캡션 생성: 형식·정렬·1,024자 제한·경고 표시 — 단위 테스트.
- `fetcher.py`: yfinance mock으로 환율 환산·전방 채움·실패 처리 테스트.
- `db.py`: 인메모리 SQLite로 upsert·snapshot 우선 규칙 테스트.
- 네트워크 실호출은 테스트에서 배제한다.

## 범위 제외 (YAGNI)

- 시총 상위 자동 선정 (종목은 config 수동 관리)
- 상장주식수 이력 기반 정밀 재구성 (B 방식)
- 매일 리포트에 비율 한 줄 포함 (C안 — 채택하지 않음, 캡션 전송으로 대체)
- 웹 대시보드
