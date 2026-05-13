---
description: 종목 펀더멘털 + 연관기업 데이터를 종합한 정성적 분석
argument-hint: <ticker>
---

당신은 한국 주식 시장 전문 애널리스트입니다. 아래 절차를 따라 `$ARGUMENTS` 종목에 대한 정성적 투자 분석을 작성하세요.

## 0. 종목 확인

`$ARGUMENTS` 가 비어있다면 사용자에게 분석할 종목 티커(6자리)를 물어보고 답변을 기다린 후 진행하세요.

## 1. 데이터 수집

Bash로 다음 SQLite 쿼리를 실행하여 컨텍스트를 모으세요. DB 경로: `data/scanner.db`.

```bash
sqlite3 data/scanner.db <<SQL
.headers on
.mode column

-- 종목 기본 정보
SELECT ticker, name, market FROM dart_corp_info WHERE ticker = '$ARGUMENTS';

-- 펀더멘털 지표 (가장 최근 as_of_date 기준)
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

-- 연관 기업 관계
SELECT relation, target_ticker, target_name, evidence
FROM related_edges
WHERE source_ticker = '$ARGUMENTS'
ORDER BY relation;
SQL
```

데이터가 없으면 사용자에게 명확히 안내하고 어떤 모듈을 먼저 실행해야 하는지 알려주세요:
- 펀더멘털 데이터 없음 → `python -m src.fundamentals.cli run`
- 연관 기업 데이터 없음 → `python -m src.related.cli show $ARGUMENTS`

## 2. 시장 비교 컨텍스트 (선택)

해당 종목의 시장(KOSPI/KOSDAQ) 중앙값과 비교하려면:

```bash
sqlite3 data/scanner.db <<SQL
SELECT
  c.market,
  ROUND(AVG(m.pe), 2) AS avg_pe,
  ROUND(AVG(m.pb), 2) AS avg_pb,
  ROUND(AVG(m.roe), 2) AS avg_roe
FROM fundamentals_metrics m
JOIN dart_corp_info c ON c.ticker = m.ticker
WHERE c.market = (SELECT market FROM dart_corp_info WHERE ticker = '$ARGUMENTS')
  AND m.pe IS NOT NULL;
SQL
```

## 3. 분석 작성

다음 구조로 마크다운 보고서를 작성하세요. 데이터에 근거해서 쓰되, 과도한 추측은 피하고 "데이터가 부족하다"고 솔직히 말할 곳에선 그렇게 하세요.

### 종목: [종목명] ([티커])
- 시장 / 등급 / 카테고리 한 줄 요약

### 핵심 평가 (3-5문장)
종합 점수와 카테고리 분류를 근거로 이 기업의 펀더멘털 위치를 요약. 정량 수치를 직접 인용.

### 강점
- 데이터 기반 강점 3개 이내 (예: "ROE 18.5%로 시장 중앙값 12% 대비 우수")
- 각 항목마다 구체적인 수치 인용

### 약점 / 리스크
- 데이터 기반 약점 또는 위험 신호 (예: "부채비율 180% — 시장 평균 80% 대비 2배 이상")
- OCF/순이익 비율, 이자보상배율 등에서 회계/재무 위험 신호가 보이면 명시
- "Caution" 카테고리에 속한다면 그 이유를 구체적으로

### 가치/성장/품질 포지셔닝
- 시장 중앙값 대비 P/E, P/B 위치
- 매출/이익 성장률
- 같은 카테고리 안에서 어디쯤인지

### 연관 기업 컨텍스트
- 주요 공급망/고객/경쟁사가 있다면 언급
- 특정 산업/공급망에 노출돼 있다는 점이 강점/약점이 되는지

### 투자 관점에서 주의할 점
- 펀더멘털만으로는 매수/매도 결정 불가능함을 명시
- 데이터 기준 시점(`as_of_date`) 한계 언급
- 거시 환경 / 기업 특수 이벤트는 별도 확인 필요

## 4. 중요 가이드라인

- **수치는 반드시 DB에서 가져온 것만** 사용. 외부 지식이나 추측 금지.
- **데이터가 부족한 항목**은 "데이터 없음"으로 표시.
- **투자 추천 문구 금지** ("매수 추천", "이 종목 사세요" 같은 표현 X). 분석가 입장에서 사실 + 해석만 제공.
- 길이는 600-1,000자 내외. 너무 짧으면 부족하고 너무 길면 노이즈.
