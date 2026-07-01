# analyze-thesis 스킬 구현 플랜

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 특정 투자자의 미래 지향 정성 분석 관점(사이클·촉매·밸류체인)을 코드화한 프로젝트 스킬 `analyze-thesis`(종목/테마 2모드)를 만든다.

**Architecture:** 프로젝트 스킬로 구현한다. `SKILL.md`는 얇게(절차·모드 분기·품질 가드), 방법론은 `references/`의 세 마크다운(lens/stock-mode/theme-mode)으로 분리. 코드가 아닌 프롬프트 산출물이므로 태스크 단위는 "파일 작성 → 검증 → 커밋"이다. 정량 토대는 기존 `.claude/commands/analyze-stock.md`의 SQL을 재사용(중복 금지).

**Tech Stack:** Claude Code 프로젝트 스킬(마크다운 + YAML frontmatter). 런타임에 `sqlite3`(DB: `data/scanner.db`), `WebSearch`/`WebFetch`, 프로젝트 CLI(`python -m src.related.cli show`, `python -m src.fundamentals.cli run`).

## Global Constraints

- 스킬 위치: `.claude/skills/analyze-thesis/`. 호출: `/analyze-thesis <ticker>`(종목), `/analyze-thesis theme:<키워드>`(테마).
- DB 경로: `data/scanner.db`.
- 모든 렌즈 결론에 근거 태그 필수: `[DB]` / `[웹:출처]` / `[추정]`.
- 웹 인용은 URL + 날짜 기록.
- 방향성 견해 허용, 단 "매수/매도하세요" 단정 표현 금지. 리스크 섹션과 반드시 짝.
- 외부 수치와 DB 수치 충돌 시 양쪽 병기 + 출처 명시.
- 정량 SQL은 `.claude/commands/analyze-stock.md`의 섹션 1~2를 재사용(플랜/스킬에 SQL 재복제 금지, 참조로 처리).
- 결측 데이터 자동 복구: 가벼운 것(related `show <ticker>`, 단일 종목 시총 fresh)은 확인 없이 자동 실행 / 무거운 것(`fundamentals.cli run`, 전 종목)은 사용자 확인 후 실행.
- 문서 커밋은 `docs/`가 gitignore 되어 있으므로 `git add -f` 사용(기존 관례). 스킬 파일(`.claude/skills/...`)은 무시 대상 아님 — 일반 `git add`.
- 언어: 한국어 경어체(프로젝트 CLAUDE.md 규칙).

---

### Task 1: 스킬 뼈대 — `SKILL.md` 작성 및 디스커버리 확인

**Files:**
- Create: `.claude/skills/analyze-thesis/SKILL.md`

**Interfaces:**
- Produces: `/analyze-thesis` 슬래시 커맨드. `references/lens.md`, `references/stock-mode.md`, `references/theme-mode.md`를 참조로 지시(파일들은 Task 2~4에서 생성).
- Consumes: 없음(첫 태스크).

- [ ] **Step 1: `SKILL.md` 생성**

아래 내용을 그대로 작성한다.

````markdown
---
name: analyze-thesis
description: 특정 투자자의 미래 지향 정성 관점(사이클·촉매·밸류체인·피어 밸류에이션)으로 한국 주식을 분석. 종목 논거 모드와 테마 밸류체인 수혜주 발굴 모드를 제공하며, DB 정량 토대 + 웹 리서치를 결합하고 근거를 태깅해 보고서를 파일로 저장한다.
argument-hint: <ticker> | theme:<키워드>
---

당신은 사이클·촉매·밸류체인 중심의 미래 지향 정성 분석을 하는 한국 주식 애널리스트입니다. `analyze-stock`(과거 회계 정량)과 달리, 이 스킬은 **미래 실적 가시성과 투자 논거(thesis)**에 초점을 둡니다.

## 0. 모드 분기

인자 `$ARGUMENTS`를 보고 모드를 정합니다.

- `theme:` 로 시작 → **테마 모드**. `references/theme-mode.md`를 읽고 그 절차를 따릅니다.
- 6자리 숫자 티커 → **종목 모드**. `references/stock-mode.md`를 읽고 그 절차를 따릅니다.
- 비어있음 → 사용자에게 "종목 티커(6자리) 또는 `theme:<키워드>` 중 무엇을 분석할까요?"를 묻고 답을 기다립니다.

두 모드 모두 **먼저 `references/lens.md`를 읽어** 12개 분석 렌즈를 숙지한 뒤 시작합니다.

## 1. 공통 품질 가드 (반드시 준수)

- **근거 태깅:** 모든 렌즈 결론 문장 끝에 `[DB]` / `[웹:출처]` / `[추정]` 중 하나를 붙입니다. 근거를 못 찾은 렌즈는 "근거 부족"이라고 명시하고 추측으로 채우지 않습니다.
- **웹 인용:** 웹에서 가져온 사실은 출처 URL과 날짜를 기록합니다.
- **방향성 견해 허용, 단정 금지:** 사이클 위치·리레이팅 여력 등 방향성 견해는 냅니다. 그러나 "매수하세요/파세요" 같은 단정 표현은 쓰지 않으며, 견해는 반드시 리스크 섹션과 짝을 이룹니다.
- **수치 충돌:** 외부(웹) 수치와 DB 수치가 다르면 양쪽을 병기하고 출처를 밝힙니다.
- **DB 정량은 재사용:** 종목 모드의 정량 토대는 `.claude/commands/analyze-stock.md`의 "1. 데이터 수집"·"2. 비교 컨텍스트" SQL을 그대로 실행해 확보합니다(중복 작성 금지).

## 2. 결측 데이터 자동 복구

- **가벼운 것은 확인 없이 자동 실행:**
  - 연관기업 엣지 없음 → `python -m src.related.cli show <ticker>` (단일 종목).
  - 연도별 시총이 오래됨/없음 → `analyze-stock`의 "2-4. 시총 1건 fresh 갱신" 인라인 파이썬을 실행(단일 종목).
- **무거운 것은 사용자 확인 후 실행:**
  - 펀더멘털 데이터 없음 → `python -m src.fundamentals.cli run`은 KOSPI/KOSDAQ **전 종목** 파이프라인이라 수 분~수십 분·API 대량 호출입니다. "펀더멘털 데이터가 없습니다. 전 종목 파이프라인(`fundamentals.cli run`)을 지금 돌릴까요? 수 분 걸립니다." 라고 물어 **승인 시에만** 실행합니다. 거절 시 보고서 상단에 "정량 토대 결측" 경고를 달고 웹 렌즈 중심으로 부분 진행합니다.
- 자동 실행 후에도 데이터가 안 채워지면 원인을 보고서에 명시합니다.
- **신선도 경고:** `fundamentals_metrics.as_of_date`가 30일 이상 오래됐으면 보고서 상단에 경고합니다.

## 3. 출력 · 저장

- 분석을 채팅에 출력하면서 **동일 내용을 반드시 마크다운 파일로 저장**합니다.
  - 종목 모드: `docs/analysis/{TICKER}_{NAME}_thesis_{YYYY-MM-DD}.md`
  - 테마 모드: `docs/analysis/themes/{테마슬러그}_{YYYY-MM-DD}.md`
- 같은 날짜 파일이 이미 있으면 덮어쓰기 전에 사용자 확인을 받습니다.
- 저장 후 채팅 마지막에 파일 경로를 마크다운 링크로 안내합니다.
- 언어는 한국어 경어체.

세부 절차와 보고서 템플릿은 각 모드 파일(`references/stock-mode.md`, `references/theme-mode.md`)을 따릅니다.
````

- [ ] **Step 2: 스킬 디스커버리 확인**

Run: `ls .claude/skills/analyze-thesis/SKILL.md && head -4 .claude/skills/analyze-thesis/SKILL.md`
Expected: 파일이 존재하고 frontmatter에 `name: analyze-thesis`, `description:`, `argument-hint:`가 보인다.

- [ ] **Step 3: 커밋**

```bash
git add .claude/skills/analyze-thesis/SKILL.md
git commit -m "feat(analyze-thesis): 스킬 뼈대 SKILL.md 추가 (모드 분기·품질 가드·저장 규칙)"
```

---

### Task 2: 분석 렌즈 체크리스트 — `references/lens.md`

**Files:**
- Create: `.claude/skills/analyze-thesis/references/lens.md`

**Interfaces:**
- Consumes: 없음(독립 참조 문서).
- Produces: 두 모드가 공유하는 12개 렌즈 정의. stock-mode/theme-mode가 렌즈 번호로 참조.

- [ ] **Step 1: `references/lens.md` 생성**

아래 내용을 그대로 작성한다.

````markdown
# 분석 렌즈 체크리스트

이 투자자의 관점을 12개 렌즈로 코드화한 것입니다. 각 렌즈 결론에는 `[DB]` / `[웹:출처]` / `[추정]` 태그를 붙이고, 근거가 없으면 "근거 부족"으로 명시합니다.

| # | 렌즈 | 핵심 질문 | 소스 |
|---|---|---|---|
| 1 | P×Q 분해 | 매출 성장이 물량(Q)인가 단가(P)인가, 각각 지속되나 | DB + 웹 |
| 2 | 수주잔고·리드타임 | 잔고 규모, 매출인식 시차, 빈티지별 마진 개선 | 웹 |
| 3 | OPM 궤적·드라이버 | 마진 추이 + 변동 원인(원재료/믹스/고정비/판가) | DB + 웹 |
| 4 | 사이클 지속성 | 구조적 수요인가 순환적인가, 과거 사이클 대비, 피크아웃 신호 | 웹 + 판단 |
| 5 | 캐파 vs 수요 수급 | 과잉 증설 우려 검증 (자사·경쟁사·전방 캐파 대조) | 웹 |
| 6 | 촉매·모멘텀 | 신규 발주/수주/정책 이벤트, 타이밍 | 웹 |
| 7 | 정책·지정학 | 관세, 정부 정책, 규제 테일윈드 | 웹 |
| 8 | 매크로 민감도 | 환율·원자재·전방 가격의 손익 탄력 (가능하면 정량화) | DB + 웹 |
| 9 | 밸류에이션 vs 피어 | Forward PER(사이클 이익) + 글로벌 피어 멀티플, 리레이팅 여력 | DB + 웹 |
| 10 | 밸류체인 포지션 | 셀→소재→부품 중 리스크/리워드 최적 자리 (시총·수주가시성·경쟁강도) | DB + 웹 |
| 11 | 1차 소스 삼각검증 | 경쟁사·고객사·인접산업 컨콜로 교차확인 | 웹 |
| 12 | 리스크 명시 | 논거를 깨뜨릴 요인 별도 섹션 | 종합 |

## 적용 규칙

- **종목 모드:** 12개 렌즈를 전부 시도합니다. 웹 근거를 못 찾은 렌즈는 "근거 부족"으로 남깁니다.
- **테마 모드:** 4(사이클)·6(촉매)·7(정책)·10(밸류체인)을 중심으로 적용합니다.
- 각 렌즈 결론 문장 끝에 근거 태그를 붙입니다.

## 이 투자자의 판단 휴리스틱 (밸류체인 포지션 선정 시)

- 빅캡은 테마 탄력이 약하므로, 순수 노출도가 높은 중소형을 선호.
- 실제 대규모 수주 실적이 있는 벤더를 아직 수주가 없는 후보보다 우선.
- 해당 테마 매출 비중이 높을수록 "순수" 플레이로 간주.
- "성장하는 시장 + 정책이 밀어주는 점유율 이전"이 겹치는 지점을 최고의 자리로 봄.
- 사이클은 "구조적 수요 증가"인지 "단발성 이슈"인지로 지속성을 판단(과거 사이클과 대조).
````

- [ ] **Step 2: 완결성 확인**

Run: `grep -c "^| " .claude/skills/analyze-thesis/references/lens.md`
Expected: 표 행 14개 이상(헤더+구분선+12개 렌즈). 12개 렌즈가 모두 포함됐는지 육안 확인.

- [ ] **Step 3: 커밋**

```bash
git add .claude/skills/analyze-thesis/references/lens.md
git commit -m "feat(analyze-thesis): 12개 분석 렌즈 체크리스트 추가"
```

---

### Task 3: 종목 모드 절차·템플릿 — `references/stock-mode.md`

**Files:**
- Create: `.claude/skills/analyze-thesis/references/stock-mode.md`

**Interfaces:**
- Consumes: `references/lens.md`의 12개 렌즈, `.claude/commands/analyze-stock.md`의 정량 SQL.
- Produces: 종목 모드 실행 절차 + 보고서 템플릿.

- [ ] **Step 1: `references/stock-mode.md` 생성**

아래 내용을 그대로 작성한다.

````markdown
# 종목 논거 모드

`/analyze-thesis <ticker>` 로 호출됩니다. 한 종목을 12개 렌즈(`references/lens.md`)로 분석해 투자 논거를 작성하고 파일로 저장합니다.

## 절차

1. **티커 확인** — `$ARGUMENTS`가 6자리 티커인지 확인. 비어있으면 사용자에게 요청.
2. **정량 토대 확보 (렌즈 1·3·8·9·10의 DB 부분)** — `.claude/commands/analyze-stock.md`의 다음 SQL을 그대로 실행합니다(재작성 금지):
   - "1-1. 연도별 손익/BS 시계열 + 파생지표" → P×Q(렌즈 1)와 OPM 궤적(렌즈 3)의 정량 토대.
   - "2-1. 시장 중앙값", "2-2. 섹터 중앙값" → 밸류에이션 비교(렌즈 9).
   - "1. 데이터 수집"의 연관기업 엣지 쿼리 → 밸류체인 포지션(렌즈 10).
   - "1-0. 데이터 신선도 확인" → 30일 경과 시 상단 경고.
   - "2-4. 시총 1건 fresh" → 자동 실행(가벼움).
   - 데이터 결측 시 SKILL.md "2. 결측 데이터 자동 복구" 정책을 따릅니다.
3. **웹 리서치 (렌즈 2·4·5·6·7·9·11)** — `WebSearch`/`WebFetch`로 다음을 조사하고 각 발견에 URL·날짜를 기록:
   - 최근 IR/실적발표 자료, 수주잔고·수주공시, 가이던스(렌즈 2).
   - 산업 사이클 전망, 과거 사이클 사례(렌즈 4).
   - 자사/경쟁사/전방 캐파 증설(렌즈 5).
   - 임박 촉매·정책 뉴스(렌즈 6·7).
   - 글로벌 동종 피어의 PER/EV 멀티플(렌즈 9).
   - 경쟁사·고객사·인접산업 컨콜 코멘트로 교차검증(렌즈 11).
4. **12개 렌즈 적용** — 각 렌즈 결론에 `[DB]`/`[웹:출처]`/`[추정]` 태그. 근거 없으면 "근거 부족".
5. **논거 종합** — 방향성 견해(사이클 위치 + 핵심 드라이버) + 핵심 논거 3가지 + 이를 깨뜨릴 리스크. 단정 매수/매도 표현 금지.
6. **파일 저장** — `docs/analysis/{TICKER}_{NAME}_thesis_{YYYY-MM-DD}.md`. `{NAME}`은 DB 종목명, `{YYYY-MM-DD}`는 작성일. 같은 날짜 파일 존재 시 덮어쓰기 전 확인. 저장 후 경로를 마크다운 링크로 안내.

## 보고서 템플릿

```markdown
# {종목명} ({티커}) 투자 논거 분석

- **시장 / 섹터:** ...
- **작성일 / 데이터 기준일(as_of_date):** ...
- ⚠ (해당 시) 신선도 경고 / 정량 토대 결측 경고

## 한 줄 논거 (Thesis)
방향성 견해 1-2문장. 사이클 위치 + 핵심 드라이버.

## P×Q 성장 분해   [DB]+[웹]
매출 성장을 물량/단가로 분리, 각 지속성.

## 실적 궤적 & 마진 드라이버   [DB]
| 연도 | 매출(억) | 영업이익(억) | OPM | 순이익(억) |
| --- | --- | --- | --- | --- |
| ... | ... | ... | ... | ... |
마진 변동 원인(원재료/믹스/고정비/판가) 코멘트.

## 수주잔고 & 미래 실적 가시성   [웹]
잔고 규모, 매출인식 리드타임, 빈티지별 마진.

## 사이클 & 수급   [웹]+[추정]
구조적 vs 순환적, 과거 사이클 대비, 캐파-수요 밸런스, 피크아웃 판단.

## 촉매 & 정책   [웹]
임박 이벤트, 정책/지정학 테일윈드.

## 매크로 민감도   [DB]+[웹]
환율/원자재/전방 가격의 손익 탄력.

## 밸류에이션 vs 피어   [DB]+[웹]
Forward PER + 글로벌 피어 멀티플, 리레이팅 여력.

## 밸류체인 포지션   [DB]+[웹]
연관기업 맥락, 이 종목의 상대적 자리.

## 리스크 (논거를 깨뜨릴 요인)
별도 명시.

## 근거 등급 & 출처
렌즈별 [DB]/[웹]/[추정] 요약 + 웹 출처 URL·날짜 목록.
```
````

- [ ] **Step 2: 완결성 확인**

Run: `grep -c "^## " .claude/skills/analyze-thesis/references/stock-mode.md`
Expected: 섹션 헤더가 다수 존재하고, 템플릿에 "한 줄 논거"부터 "근거 등급 & 출처"까지 10개 보고서 섹션이 포함됐는지 육안 확인. "analyze-stock.md" 참조 문구가 존재.

- [ ] **Step 3: 커밋**

```bash
git add .claude/skills/analyze-thesis/references/stock-mode.md
git commit -m "feat(analyze-thesis): 종목 모드 절차·보고서 템플릿 추가"
```

---

### Task 4: 테마 모드 절차·템플릿 — `references/theme-mode.md`

**Files:**
- Create: `.claude/skills/analyze-thesis/references/theme-mode.md`

**Interfaces:**
- Consumes: `references/lens.md`(렌즈 4·6·7·10), 연관기업 DB, 웹 리서치.
- Produces: 테마 모드 실행 절차 + 보고서 템플릿. 우선 후보를 종목 모드로 연결.

- [ ] **Step 1: `references/theme-mode.md` 생성**

아래 내용을 그대로 작성한다.

````markdown
# 테마 밸류체인 모드

`/analyze-thesis theme:<키워드>` 로 호출됩니다. 하나의 메가트렌드를 밸류체인으로 분해해 한국 상장 수혜주 후보를 도출합니다. 개별 종목 심층이 아니라 **밸류체인 지도 + 노드별 후보 발굴**이 목적입니다. 렌즈 4·6·7·10(`references/lens.md`)을 중심으로 적용합니다.

## 절차

1. **테마 정의** — 인자에서 테마 키워드를 추출. 모호하면 1-2개 질문으로 범위를 좁힙니다(예: "데이터센터 전력"이면 발전/저장/송배전 중 어디까지?).
2. **메가트렌드 & 수혜 구조 (렌즈 4·7)** — `WebSearch`/`WebFetch`로 수요 성장 근거 + 정책/지정학 촉매(탈중국, 관세, 보조금)를 수집. "성장 + 점유율 이전"이 겹치는지 확인. 각 발견에 URL·날짜.
3. **밸류체인 매핑 (렌즈 10)** — 완제품/시스템 → 중간재/셀 → 부품/소재 단계로 분해. 각 단계의 한국 상장사 후보를 나열:
   - `data/scanner.db`의 `related_edges`에서 관련 기업 탐색.
   - 웹 리서치로 보강(공급망 기사, 증권사 밸류체인 리포트).
4. **노드별 스크리닝** — 각 후보에 대해 DB에서 빠르게 조회:
   ```bash
   sqlite3 data/scanner.db "SELECT c.name, c.market FROM dart_corp_info c WHERE c.ticker='<TICKER>';"
   sqlite3 data/scanner.db "SELECT ticker, as_of_date, pe, pb, roe, operating_margin FROM fundamentals_metrics WHERE ticker='<TICKER>' ORDER BY as_of_date DESC LIMIT 1;"
   ```
   시총·매출비중·수주가시성·경쟁강도를 정리하고 리스크/리워드를 태깅. 웹으로 수주 실적을 확인.
5. **최적 자리 도출** — `references/lens.md`의 판단 휴리스틱 적용(빅캡 탄력 약함, 수주 실적 있는 벤더 우선, 매출 비중 높을수록 순수 노출). 1-3개 우선 후보 + 선정 이유.
6. **파일 저장** — `docs/analysis/themes/{테마슬러그}_{YYYY-MM-DD}.md`. `{테마슬러그}`는 공백을 하이픈으로 치환한 키워드. 같은 날짜 파일 존재 시 덮어쓰기 전 확인. 저장 후 경로를 마크다운 링크로 안내.

## 보고서 템플릿

```markdown
# {테마} 밸류체인 분석

- **작성일:** ...

## 메가트렌드 & 수혜 구조   [웹]
수요 성장 근거 + 정책 촉매. "성장 + 점유율 이전" 성립 여부.

## 밸류체인 지도
| 단계 | 역할 | 한국 상장 후보 | 시총 | 순수도(테마 매출비중) |
| --- | --- | --- | --- | --- |
| 완제품/시스템 | ... | ... | ... | ... |
| 중간재/셀 | ... | ... | ... | ... |
| 부품/소재 | ... | ... | ... | ... |

## 노드별 리스크/리워드
각 후보: 강점·약점·수주가시성·경쟁강도. [DB]/[웹] 태그.

## 최적 포지션 (우선 후보)
1-3개 + 선정 이유(판단 휴리스틱 명시).
각 후보는 `/analyze-thesis <ticker>`(종목 모드)로 심화 가능.

## 리스크 & 출처
테마 전체 리스크 + 웹 출처 URL·날짜 목록.
```
````

- [ ] **Step 2: 완결성 확인**

Run: `grep -c "^## " .claude/skills/analyze-thesis/references/theme-mode.md`
Expected: 섹션 헤더 다수 존재. 절차 6단계와 템플릿 5개 섹션(메가트렌드/밸류체인 지도/노드별/최적 포지션/리스크)이 포함됐는지, 종목 모드 연결 문구가 있는지 육안 확인.

- [ ] **Step 3: 커밋**

```bash
git add .claude/skills/analyze-thesis/references/theme-mode.md
git commit -m "feat(analyze-thesis): 테마 밸류체인 모드 절차·보고서 템플릿 추가"
```

---

### Task 5: 수동 검증 (종목 모드 1회 실행)

**Files:**
- Test(수동): 실제 실행. 산출물 `docs/analysis/{TICKER}_..._thesis_{DATE}.md`.

**Interfaces:**
- Consumes: Task 1~4의 스킬 전체.
- Produces: 검증 완료 + 필요 시 수정.

- [ ] **Step 1: DB 보유 종목 1개 선정**

Run: `sqlite3 data/scanner.db "SELECT ticker, name FROM dart_corp_info LIMIT 5;"`
Expected: 티커 후보가 나온다. 하나를 골라 검증 대상으로 삼는다.

- [ ] **Step 2: 종목 모드 실행**

새 세션 또는 현재 세션에서 `/analyze-thesis <선정티커>`를 실행한다.
Expected 확인 항목:
- 정량 토대가 `analyze-stock` SQL로 채워진다(연도별 손익 표, 밸류에이션 비교).
- 웹 렌즈 결과에 출처 URL·날짜가 붙는다.
- 각 렌즈 결론에 `[DB]`/`[웹:출처]`/`[추정]` 태그가 있다. 근거 없는 렌즈는 "근거 부족".
- 방향성 견해가 있으나 "매수/매도" 단정 표현은 없고 리스크 섹션이 존재한다.
- `docs/analysis/{TICKER}_{NAME}_thesis_{DATE}.md` 파일이 저장되고 채팅에 경로 링크가 나온다.

- [ ] **Step 3: 결함 발견 시 수정**

체크 항목 중 누락/오류가 있으면 해당 `SKILL.md` 또는 `references/*.md`를 수정하고 재실행해 확인한다. 수정이 있었다면 커밋:

```bash
git add .claude/skills/analyze-thesis
git commit -m "fix(analyze-thesis): 수동 검증 반영 수정"
```

- [ ] **Step 4: 검증 산출물 커밋(선택)**

검증 보고서를 남기려면(docs는 gitignore이므로 -f):

```bash
git add -f docs/analysis/*_thesis_*.md
git commit -m "docs(analyze-thesis): 수동 검증 샘플 보고서"
```

---

## Self-Review

**1. Spec coverage:**
- 모드 2개(종목/테마) → Task 1(분기) + Task 3/4. ✅
- DB+웹 하이브리드, 근거 태깅 → Global Constraints + Task 2 규칙 + Task 3/4 절차. ✅
- 방향성 견해 + 리스크 짝, 단정 금지 → SKILL.md 품질 가드 + 템플릿 리스크 섹션. ✅
- 정량 토대 analyze-stock SQL 재사용 → Task 3 Step 1(참조). ✅
- 결측 자동 복구(가벼움 자동 / 무거움 확인) → SKILL.md 섹션 2. ✅
- 신선도 경고, 동일 날짜 파일 확인 → SKILL.md + 모드 절차. ✅
- 12개 렌즈 → Task 2. ✅
- 테스트(수동 1케이스 + frontmatter 노출) → Task 5 + Task 1 Step 2. ✅
- 범위 밖(analyze-stock 불변, 웹결과 DB미저장) → 준수(SQL 참조만, 저장은 보고서까지). ✅

**2. Placeholder scan:** 각 태스크에 실제 파일 전문을 수록. "TBD/추후" 없음. ✅

**3. Type consistency:** 파일 경로·모드 인자 형식(`<ticker>` / `theme:<키워드>`)·근거 태그(`[DB]`/`[웹:출처]`/`[추정]`)·저장 경로가 모든 태스크에서 일치. ✅
