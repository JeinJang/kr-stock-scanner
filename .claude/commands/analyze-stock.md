---
description: 종목 펀더멘털 + 연관기업 데이터를 종합한 정성적 분석 (파일 저장 포함)
argument-hint: <ticker>
---

당신은 한국 주식 시장 전문 애널리스트입니다. 아래 절차를 따라 `$ARGUMENTS` 종목에 대한 정성적 투자 분석을 작성하고, **반드시 `docs/analysis/` 디렉토리에 마크다운 파일로 저장**하세요.

## 사전 조건 (선행 커맨드)

이 분석은 아래 테이블에 데이터가 적재돼 있어야 합니다. 각 테이블을 채우는 커맨드와 권장 실행 순서는 다음과 같습니다.

| 읽는 테이블 | 용도 | 채우는 커맨드 |
| --- | --- | --- |
| `dart_corp_info`, `dart_financials` | 종목 기본정보 + 연도별 손익/BS 시계열 | `python -m src.fundamentals.cli refresh` |
| `fundamentals_metrics`, `fundamentals_scores`, `corp_market_yearly` | 지표·점수·연도별 시총 | `python -m src.fundamentals.cli run` (refresh 내용 포함, 5스텝 전체) |
| `related_edges` | 연관기업(공급망/고객/경쟁/계열/자회사) | `python -m src.related.cli show <ticker>` (종목별 on-demand) |
| `new_highs.sector` | 섹터 비교 (선택) | `python -m src.cli run` 또는 `collect` (52주 신고가 스캔) |

**권장 실행 순서**

```bash
# 1) 펀더멘털 전체 — metrics/scores/financials/시총을 한 번에 적재.
#    fundamentals_metrics.as_of_date 가 30일 이상 오래됐을 때만 실행 (아래 1-0 신선도 가드 참조).
python -m src.fundamentals.cli run

# 2) 분석 대상 종목의 연관기업 추출 — 종목 단위 on-demand. 신규 종목은 거의 항상 필요.
python -m src.related.cli show $ARGUMENTS

# 3) 분석 실행 (이 커맨드)
```

주의:
- `fundamentals.cli run` 이 `refresh` 의 일(DART 적재)을 포함하므로 둘 다 돌릴 필요는 없습니다.
- `related_edges` 는 전체 배치가 아니라 **종목별로** 채워지므로, 대상 종목에 데이터가 없으면 `related.cli show` 를 먼저 실행하세요.
- `new_highs.sector` 는 그 종목이 **52주 신고가에 걸린 날**에만 채워집니다. 없으면 섹터 비교는 생략하고 "섹터 정보 미수집"으로 표기합니다.

## 0. 종목 확인

`$ARGUMENTS` 가 비어있다면 사용자에게 분석할 종목 티커(6자리)를 물어보고 답변을 기다린 후 진행하세요.

## 1. 종목/펀더멘털/점수/연관 기업 데이터 수집

### 1-0. 데이터 신선도 확인 (자동 안내)

분석 시작 전, `fundamentals_metrics.as_of_date` 가 30일 이상 오래됐는지 확인합니다.

```bash
sqlite3 data/scanner.db "SELECT julianday('now') - julianday(MAX(as_of_date)) AS days_old FROM fundamentals_metrics WHERE ticker = '$ARGUMENTS';"
```

`days_old > 30` 또는 결과 없음이면 보고서 상단에 다음 경고를 출력:

> ⚠ **데이터 신선도 경고:** 이 종목의 펀더멘털 데이터가 `N`일 됐습니다 (또는 미존재). 정확도를 위해 다음 명령을 실행해 주세요:
> ```bash
> python -m src.fundamentals.cli run
> ```

Bash로 다음 SQLite 쿼리를 실행하여 컨텍스트를 모으세요. DB 경로: `data/scanner.db`.

```bash
sqlite3 data/scanner.db <<SQL
.headers on
.mode column

-- 종목 기본 정보 (시장)
SELECT ticker, name, market FROM dart_corp_info WHERE ticker = '$ARGUMENTS';

-- 섹터 (new_highs 테이블에 한해 보유; 없을 수도 있음)
SELECT DISTINCT sector FROM new_highs WHERE ticker = '$ARGUMENTS' LIMIT 1;

-- 펀더멘털 지표 (가장 최근 as_of_date)
SELECT
  ticker, as_of_date,
  current_ratio, debt_ratio, interest_coverage,
  roe, roic, operating_margin,
  revenue_cagr_3y, op_income_cagr_3y,
  ocf_to_ni_ratio, fcf_positive_years,
  pe, pb, peg
FROM fundamentals_metrics
WHERE ticker = '$ARGUMENTS'
ORDER BY as_of_date DESC LIMIT 1;

-- 종합 점수 + 카테고리
SELECT
  ticker, as_of_date,
  liquidity_score, profitability_score, growth_score, cashflow_score,
  total_score, grade, categories
FROM fundamentals_scores
WHERE ticker = '$ARGUMENTS'
ORDER BY as_of_date DESC LIMIT 1;

-- 연관 기업 관계 (양방향)
SELECT 'OUT' AS dir, relation, target_ticker, target_name, evidence
FROM related_edges WHERE source_ticker = '$ARGUMENTS'
UNION ALL
SELECT 'IN' AS dir, relation, source_ticker, '', evidence
FROM related_edges WHERE target_ticker = '$ARGUMENTS';
SQL
```

데이터가 없으면 사용자에게 명확히 안내하고 어떤 모듈을 먼저 실행해야 하는지 알려주세요:
- 펀더멘털 데이터 없음 → `python -m src.fundamentals.cli run`
- 연관 기업 데이터 없음 → `python -m src.related.cli show $ARGUMENTS`

### 1-1. 연도별 손익/BS 시계열 + 파생지표 (필수)

`fundamentals_metrics`는 단일 시점 요약이라 사이클 회복/하락 같은 **추세 변화**가 가려집니다. `dart_financials` 원천에서 연도별로 직접 뽑아 트렌드 표를 만드세요.

`dart_financials`는 같은 `(year, account)`에 **연결/별도 두 행**이 들어 있는 경우가 많고, 당기순이익은 지배/비지배 분리로 4행까지 들어옵니다. **`MAX(value)`로 통일**해 연결재무제표 기준에 가깝게 사용하세요.

단위 환산: `dart_financials.value`는 **원 단위**이므로 보고서 표시는 `/1e8` (억원)으로 변환.

```bash
sqlite3 data/scanner.db <<SQL
.headers on
.mode column

WITH t AS (SELECT corp_code FROM dart_corp_info WHERE ticker = '$ARGUMENTS'),
base AS (
  SELECT year, account, MAX(value) AS v
  FROM dart_financials
  WHERE corp_code = (SELECT corp_code FROM t) AND quarter = 0
    AND account IN ('매출액','영업이익','당기순이익','자산총계','자본총계','부채총계','유동자산','유동부채','이익잉여금')
  GROUP BY year, account
),
piv AS (
  SELECT
    year,
    MAX(CASE WHEN account = '매출액'     THEN v END) AS revenue,
    MAX(CASE WHEN account = '영업이익'   THEN v END) AS op_income,
    MAX(CASE WHEN account = '당기순이익' THEN v END) AS net_income,
    MAX(CASE WHEN account = '자산총계'   THEN v END) AS total_assets,
    MAX(CASE WHEN account = '자본총계'   THEN v END) AS total_equity,
    MAX(CASE WHEN account = '유동자산'   THEN v END) AS current_assets,
    MAX(CASE WHEN account = '유동부채'   THEN v END) AS current_liab,
    MAX(CASE WHEN account = '이익잉여금' THEN v END) AS retained
  FROM base GROUP BY year
)
SELECT
  year,
  ROUND(revenue   /1e8, 0)                            AS rev_억,
  ROUND(op_income /1e8, 0)                            AS op_억,
  ROUND(net_income/1e8, 0)                            AS ni_억,
  ROUND(op_income  * 100.0 / NULLIF(revenue, 0),    2) AS op_margin,
  ROUND(net_income * 100.0 / NULLIF(revenue, 0),    2) AS net_margin,
  ROUND(net_income * 100.0 / NULLIF(total_assets,0),2) AS roa,
  ROUND(total_equity * 100.0 / NULLIF(total_assets,0),2) AS equity_ratio,
  ROUND((current_assets - current_liab) / 1e8, 0)     AS wc_억,
  ROUND(retained / 1e8, 0)                            AS retained_억
FROM piv
ORDER BY year DESC
LIMIT 7;
SQL
```

위 결과로 추가 산출 / 해석할 것:

- **YoY 성장률**: 직전년 대비 매출/영업이익/순이익 변화율 (직전년 대비 % — 보고서 표에 컬럼으로 끼우거나, 인사이트 문장에 자연어로 포함).
- **이익 변동성**: 5년 영업이익(또는 순이익) 시계열의 변동계수(CV = 표준편차/평균) — 사이클 민감도 판단.
- **단년 vs CAGR 괴리**: `op_income_cagr_3y` 같은 단일 지표가 음수여도, 시계열을 보면 **최근 1-2년 회복** 중일 수 있음. 반드시 비교해 코멘트.
- **적자 연도 존재 여부**: 5년 내 순이익 음수 연도가 있으면 명시.
- **자기자본비율 추이**: 자본구조 안정성 변화.

## 2. 비교 컨텍스트 수집 (필수)

### 2-1. 시장(KOSPI/KOSDAQ) 중앙값

```bash
sqlite3 data/scanner.db <<SQL
.headers on
.mode column

WITH pool AS (
  SELECT m.* FROM fundamentals_metrics m
  JOIN dart_corp_info c ON c.ticker = m.ticker
  WHERE c.market = (SELECT market FROM dart_corp_info WHERE ticker = '$ARGUMENTS')
),
pe_s  AS (SELECT pe   v, ROW_NUMBER() OVER (ORDER BY pe)   rn, COUNT(*) OVER () cnt FROM pool WHERE pe   IS NOT NULL AND pe   > 0),
pb_s  AS (SELECT pb   v, ROW_NUMBER() OVER (ORDER BY pb)   rn, COUNT(*) OVER () cnt FROM pool WHERE pb   IS NOT NULL AND pb   > 0),
roe_s AS (SELECT roe  v, ROW_NUMBER() OVER (ORDER BY roe)  rn, COUNT(*) OVER () cnt FROM pool WHERE roe  IS NOT NULL),
dr_s  AS (SELECT debt_ratio       v, ROW_NUMBER() OVER (ORDER BY debt_ratio)       rn, COUNT(*) OVER () cnt FROM pool WHERE debt_ratio       IS NOT NULL),
om_s  AS (SELECT operating_margin v, ROW_NUMBER() OVER (ORDER BY operating_margin) rn, COUNT(*) OVER () cnt FROM pool WHERE operating_margin IS NOT NULL)
SELECT
  (SELECT market FROM dart_corp_info WHERE ticker = '$ARGUMENTS')                                          AS market,
  (SELECT COUNT(*) FROM pool)                                                                              AS n_companies,
  (SELECT ROUND(AVG(v), 2) FROM pe_s  WHERE rn IN ((cnt+1)/2, (cnt+2)/2))                                  AS median_pe,
  (SELECT ROUND(AVG(v), 2) FROM pb_s  WHERE rn IN ((cnt+1)/2, (cnt+2)/2))                                  AS median_pb,
  (SELECT ROUND(AVG(v), 2) FROM roe_s WHERE rn IN ((cnt+1)/2, (cnt+2)/2))                                  AS median_roe,
  (SELECT ROUND(AVG(v), 2) FROM dr_s  WHERE rn IN ((cnt+1)/2, (cnt+2)/2))                                  AS median_debt_ratio,
  (SELECT ROUND(AVG(v), 2) FROM om_s  WHERE rn IN ((cnt+1)/2, (cnt+2)/2))                                  AS median_op_margin;
SQL
```

### 2-2. 섹터 중앙값 (동일 섹터 동종업종 비교)

섹터 정보는 `new_highs.sector` 컬럼에 저장돼 있습니다. 종목이 그 테이블에 없으면 섹터 비교는 생략하고 보고서에 "섹터 정보 미수집"이라고 명시하세요.

```bash
sqlite3 data/scanner.db <<SQL
.headers on
.mode column

WITH target_sector AS (
  SELECT sector FROM new_highs WHERE ticker = '$ARGUMENTS' LIMIT 1
),
sector_tickers AS (
  SELECT DISTINCT ticker FROM new_highs
  WHERE sector = (SELECT sector FROM target_sector)
),
pool AS (
  SELECT m.* FROM fundamentals_metrics m
  JOIN sector_tickers s ON s.ticker = m.ticker
),
pe_s  AS (SELECT pe   v, ROW_NUMBER() OVER (ORDER BY pe)   rn, COUNT(*) OVER () cnt FROM pool WHERE pe   IS NOT NULL AND pe   > 0),
pb_s  AS (SELECT pb   v, ROW_NUMBER() OVER (ORDER BY pb)   rn, COUNT(*) OVER () cnt FROM pool WHERE pb   IS NOT NULL AND pb   > 0),
roe_s AS (SELECT roe  v, ROW_NUMBER() OVER (ORDER BY roe)  rn, COUNT(*) OVER () cnt FROM pool WHERE roe  IS NOT NULL),
dr_s  AS (SELECT debt_ratio       v, ROW_NUMBER() OVER (ORDER BY debt_ratio)       rn, COUNT(*) OVER () cnt FROM pool WHERE debt_ratio       IS NOT NULL),
om_s  AS (SELECT operating_margin v, ROW_NUMBER() OVER (ORDER BY operating_margin) rn, COUNT(*) OVER () cnt FROM pool WHERE operating_margin IS NOT NULL)
SELECT
  (SELECT sector FROM target_sector)                                                                       AS sector,
  (SELECT COUNT(*) FROM pool)                                                                              AS n_companies,
  (SELECT ROUND(AVG(v), 2) FROM pe_s  WHERE rn IN ((cnt+1)/2, (cnt+2)/2))                                  AS median_pe,
  (SELECT ROUND(AVG(v), 2) FROM pb_s  WHERE rn IN ((cnt+1)/2, (cnt+2)/2))                                  AS median_pb,
  (SELECT ROUND(AVG(v), 2) FROM roe_s WHERE rn IN ((cnt+1)/2, (cnt+2)/2))                                  AS median_roe,
  (SELECT ROUND(AVG(v), 2) FROM dr_s  WHERE rn IN ((cnt+1)/2, (cnt+2)/2))                                  AS median_debt_ratio,
  (SELECT ROUND(AVG(v), 2) FROM om_s  WHERE rn IN ((cnt+1)/2, (cnt+2)/2))                                  AS median_op_margin;
SQL
```

### 2-3. 섹터 내 본 종목의 분위(percentile) — 선택적

여유가 있으면 PE / ROE / 영업이익률 기준 섹터 내 순위(상위 N%)를 함께 산출해 강·약점 해석에 활용하세요.

### 2-4. 해당 종목 시총 1건 fresh 갱신 (자동)

DB 의 진행 연도 시총이 며칠 됐을 수 있으므로, 분석 시작 직전에 KRX 로 1건만 fresh:

```bash
python -c "
from src.market_data.fetcher import fetch_yearly_market_data
from src.market_data.db import MarketDB
from datetime import datetime
corp_code = '$(sqlite3 data/scanner.db \"SELECT corp_code FROM dart_corp_info WHERE ticker='\\''$ARGUMENTS'\\''\")' 
rows = fetch_yearly_market_data(
    tickers=['$ARGUMENTS'],
    years=[datetime.now().year],
    corp_code_map={'$ARGUMENTS': corp_code},
)
MarketDB().save_yearly(rows)
print(f'updated: {len(rows)} row')
"
```

DART 데이터는 갱신하지 않습니다 (연 1회 사업보고서 단위라 변동 없음).

## 3. 분석 작성 + **파일 저장 (필수)**

분석을 채팅에 출력하면서 **동일한 내용을 반드시 다음 경로의 마크다운 파일로 저장**하세요. 디렉토리가 없으면 만들고, 같은 날짜 파일이 이미 있으면 덮어쓰는 대신 사용자에게 확인을 받으세요.

- 경로: `docs/analysis/{TICKER}_{NAME}_{TODAY}.md`
  - `{TICKER}`: 6자리 티커
  - `{NAME}`: DB의 종목명 (공백·특수문자는 그대로 사용해도 무방)
  - `{TODAY}`: 보고서 작성일 `YYYY-MM-DD` (셸 `$(date +%Y-%m-%d)` 또는 시스템 컨텍스트의 `currentDate`)

저장 후 채팅 마지막에는 **파일 경로를 markdown 링크로** 알려주세요. 예: `[docs/analysis/353200_대덕전자_2026-05-14.md](docs/analysis/353200_대덕전자_2026-05-14.md)`

## 4. 보고서 구조

다음 구조로 마크다운 보고서를 작성하세요. 데이터에 근거해서 쓰되, 과도한 추측은 피하고 "데이터가 부족하다"고 솔직히 말할 곳에선 그렇게 하세요.

```markdown
# {종목명} ({티커}) 정성적 펀더멘털 분석

- **시장:** KOSPI / KOSDAQ
- **섹터:** {sector} (출처: new_highs)  ← 없으면 "섹터 정보 미수집"
- **등급:** {grade}
- **카테고리:** {categories}
- **데이터 기준일 (as_of_date):** YYYY-MM-DD
- **보고서 작성일:** YYYY-MM-DD

## 핵심 평가
종합 점수와 카테고리를 근거로 펀더멘털 위치 3-5문장 요약. 정량 수치 직접 인용. **최근 1-2년 이익 추세(회복/하락)가 단일 시점 지표에 가려지지 않았는지** 시계열 표와 대조해 한 문장 포함.

### 점수 요약
| 항목 | 점수 |
| --- | --- |
| Liquidity | ... |
| Profitability | ... |
| Growth | ... |
| Cashflow | ... |
| **Total** | ... |

## 연도별 손익 트렌드 & 파생지표 (5년)

| 연도 | 매출(억) | 영업이익(억) | 순이익(억) | 영업이익률 | 순이익률 | ROA | 자기자본비율 | 운전자본(억) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YYYY | ... | ... | ... | ... | ... | ... | ... | ... |

- 최근 YoY 변화율 한두 줄 (매출/영업이익/순이익).
- 5년 영업이익 변동계수(CV) 또는 적자 연도 유무.
- 단년 시계열과 `fundamentals_metrics`의 `*_cagr_3y` 사이 괴리가 있다면 명시.

## 강점
- 3개 이내. 각 항목마다 본 종목 수치 + **섹터·시장 중앙값 양쪽과 비교**.

## 약점 / 리스크
- 데이터 기반 약점/위험 신호. 회계·재무 위험 신호(OCF/순이익, 이자보상배율 등)도 명시.
- Caution 카테고리에 속한다면 그 이유.

## 가치 / 성장 / 품질 포지셔닝

| 지표 | 본 종목 | 섹터 중앙값 | 시장 중앙값 | 해석 |
| --- | --- | --- | --- | --- |
| PER | ... | ... | ... | ... |
| PBR | ... | ... | ... | ... |
| ROE | ... | ... | ... | ... |
| 부채비율 | ... | ... | ... | ... |
| 영업이익률 | ... | ... | ... | ... |
| EPS | ... | ... | ... | ... |
| 배당수익률 | ...% | ...% | ...% | 섹터·시장 대비 위치 |
| 배당성향 | ...% | — | — | 순이익의 N% 환원 |
| OCF/NI 비율 | ... | — | — | 1.0 근처면 이익의 질 양호 |
| FCF 양수 연수 | N/5 | — | — | 5/5 면 안정적 캐시 창출 |

- 섹터 vs 시장 비교가 **서로 다른 결론**을 내놓을 경우, 그 차이를 명시 (예: "섹터 대비는 평균이지만 시장 대비는 고평가").

## 이익의 질 코멘트

- `ocf_to_ni_ratio` 와 `fcf_positive_years` 가 같이 약하면 회계이익과 현금이익 괴리 가능성. 둘 다 강하면 발생주의 회계가 실제 현금흐름과 잘 매칭됨.
- `capex_to_revenue` 가 섹터 평균 대비 높으면 자본집약 단계(투자기), 낮으면 회수기.
- `consecutive_dividend_years` ≥ 5 면 배당 정책 안정성 시사.

## 연관 기업 컨텍스트
- `related_edges`에서 가져온 공급망/고객/경쟁사. 없으면 "데이터 없음".

## 투자 관점에서 주의할 점
- 펀더멘털만으로는 매수/매도 결정 불가능.
- 데이터 기준 시점(`as_of_date`) 한계.
- 거시/이벤트 별도 확인 필요.

## 부록: 원시 데이터
원시 SQLite 출력(펀더멘털 + 섹터/시장 중앙값) 코드블록 형태로 포함.
```

## 5. 중요 가이드라인

- **수치는 반드시 DB에서 가져온 것만** 사용. 외부 지식이나 추측 금지.
- **데이터가 부족한 항목**은 "데이터 없음"으로 표시.
- **섹터 비교는 시장 비교와 함께** 제시. 동일 섹터 비교가 종종 더 의미 있는 결론을 줍니다(예: 섹터 평균이 높은 PER 산업이면 시장 대비 고평가도 정상일 수 있음).
- **투자 추천 문구 금지** ("매수 추천", "사세요" 같은 표현 X). 분석가 입장에서 사실 + 해석만 제공.
- 본문 길이는 600-1,000자 내외. 표/부록은 별도 카운트.
- **파일 저장을 잊지 말 것.** 채팅 응답만 하고 파일을 만들지 않으면 작업 미완료.
