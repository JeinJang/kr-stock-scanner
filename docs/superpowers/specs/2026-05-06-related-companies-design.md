# 연관 기업 발굴 설계 (Related Company Discovery)

## 개요

DART 사업보고서의 텍스트 섹션을 GPT로 분석하여, 한 종목의 공급망(supplier)/고객사(customer)/경쟁사(competitor)/계열사(affiliate)/자회사(subsidiary) 관계를 추출하고, 인터랙티브 네트워크 그래프 HTML 리포트로 시각화한다.

기존 펀더멘털 모듈과 같은 패턴(독립 CLI + DB 영구화 + HTML 리포트)으로 구현하되, GPT 호출 비용을 줄이기 위해 **on-demand 추출** 위주로 운영한다 (전체 일괄 배치 없음). 추출 결과는 사업보고서 접수번호(`rcept_no`)를 기준으로 캐싱되어 보고서가 갱신되기 전까지 재호출되지 않는다.

## 프로젝트 구조

```
src/related/                  # 신규 패키지
├── __init__.py
├── cli.py                    # 엔트리포인트 (show, batch, stats)
├── report_fetcher.py         # DART 사업보고서 본문 다운로드 + 섹션 파싱
├── extractor.py              # GPT로 기업명/관계 추출
├── db.py                     # SQLite 영구화 (related_edges, related_report_meta)
├── graph.py                  # NetworkX 그래프 빌드/multi-hop 탐색
├── report.py                 # HTML 리포트 (Plotly 네트워크 그래프)
├── templates/report.html
└── models.py                 # ReportSections, RelatedCompany 모델
```

기존 `src/dart/` 데이터 레이어를 재사용한다 (DartClient, dart_corp_info 테이블).

## CLI 인터페이스

```bash
# 단일 종목 조회 (1-hop 기본)
python -m src.related.cli show 042700

# 2-hop 확장
python -m src.related.cli show 042700 --depth 2

# 강제 재추출 (보고서 미변경에도 GPT 재호출)
python -m src.related.cli show 042700 --refresh

# 명시적 티커 리스트로 배치 추출 (전체 --all 옵션 없음)
python -m src.related.cli batch --tickers 005930,000660,042700

# DB에 저장된 관계 통계
python -m src.related.cli stats
```

실행 후 `reports/related-{ticker}-{YYYY-MM-DD}.html` 생성 및 브라우저 자동 오픈.

## 데이터 수집 (`report_fetcher.py`)

### DART 사업보고서 본문

1. **최신 접수번호 조회** — `/api/list.json?corp_code={code}&pblntf_detail_ty=A001&page_count=1&sort=date&sort_mth=desc`로 가장 최근 사업보고서(`reprt_code=11011`) 접수번호 획득.
2. **본문 다운로드** — `/api/document.xml?rcept_no={rcept_no}` → ZIP 응답을 메모리에서 압축 해제 → XML 추출.
3. **섹션 파싱** — XML 내 헤더 텍스트(`II. 사업의 내용`, `IX. 계열회사 등에 관한 사항`, `X. 대주주 등과의 거래내용`, 그리고 재무제표 주석 내 특수관계자 거래)를 기준으로 4개 섹션으로 분리.

### 텍스트 정제

- BeautifulSoup4로 HTML 태그 제거하고 텍스트만 추출
- 표는 행 단위로 텍스트화 (예: `거래처명 | 품목 | 거래금액`)
- 페이지 번호/목차/빈 줄 등 노이즈 정리

### 출력 모델

```python
class ReportSections(BaseModel):
    corp_code: str
    rcept_no: str
    business_content: str        # 사업의 내용
    affiliates: str              # 계열회사 등에 관한 사항
    related_party: str           # 대주주 등과의 거래내용
    related_party_notes: str     # 재무제표 주석 - 특수관계인 거래
```

## GPT 추출 (`extractor.py`)

### 모델

- `gpt-5-nano` (기존 펀더멘털/AI 분석 모듈과 동일)
- `response_format={"type": "json_object"}` 로 JSON 출력 강제

### 프롬프트

시스템 메시지로 5가지 관계 타입(Supplier/Customer/Competitor/Affiliate/Subsidiary)과 출력 스키마를 명시. 유저 메시지에 대상 기업명/티커와 4개 섹션 텍스트를 합쳐서 전달.

기대 JSON 응답:
```json
{
  "edges": [
    {
      "name": "SK하이닉스",
      "ticker": "000660",
      "relation": "Customer",
      "evidence": "당사 매출의 상당 부분은 SK하이닉스의 HBM 후공정 장비 수요에서 발생..."
    },
    ...
  ]
}
```

### 티커 매핑

GPT가 회사명만 반환(`ticker: null`)할 수 있으므로 `dart_corp_info.name → ticker` 매핑으로 보강:
- 정확 일치 우선
- 실패 시 부분 일치 (정리된 회사명 기준 — `(주)`, `주식회사` 제거 후 비교)
- 매칭 실패 시 `ticker=null` 로 저장 (비상장 또는 외국 기업)

### 토큰 제어

섹션이 매우 긴 경우(특히 사업의 내용)를 위해 섹션별 최대 토큰 제한 적용. 설정 가능 (`config.yaml`의 `related.max_tokens_per_section`, 기본 8000).

## DB 스키마 (`db.py`)

기존 `data/scanner.db`에 2개 테이블 추가.

### `related_report_meta`

캐시 무효화용 메타데이터.

| 컬럼 | 타입 | 설명 |
|------|------|------|
| ticker | String | primary key |
| rcept_no | String | 마지막 추출에 사용된 접수번호 |
| extracted_at | DateTime | 추출 시각 |

### `related_edges`

단방향 관계 엣지.

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | Integer | primary key, autoincrement |
| source_ticker | String | 보고서 작성 기업 (indexed) |
| target_ticker | String? | 연관 기업, 비상장이면 null (indexed) |
| target_name | String | 연관 기업명 (원문 그대로) |
| relation | String | Supplier/Customer/Competitor/Affiliate/Subsidiary |
| evidence | Text | GPT가 추출한 원문 인용 |
| extracted_at | DateTime | 추출 시각 |

### 캐시 무효화

```python
def needs_refresh(ticker: str, current_rcept_no: str) -> bool:
    meta = get_meta(ticker)
    if meta is None:
        return True
    return meta.rcept_no != current_rcept_no
```

`--refresh` 플래그 사용 시 위 검사를 건너뛰고 강제 재추출.

## 그래프 빌드 & 탐색 (`graph.py`)

### NetworkX DiGraph

방향성 있는 그래프 (Supplier ≠ Customer):

```python
def build_graph(edges: list[RelatedEdge]) -> nx.DiGraph:
    G = nx.DiGraph()
    for e in edges:
        target = e.target_ticker or f"_unlisted_{e.target_name}"
        G.add_edge(
            e.source_ticker, target,
            relation=e.relation, evidence=e.evidence,
        )
    return G
```

비상장 기업은 가상 노드(`_unlisted_{name}`)로 표현해 그래프에 포함시키되, 펀더멘털 메타데이터는 없음.

### Multi-hop 탐색

`expand(graph, root_ticker, depth)` 함수가 양방향(in + out edges) BFS로 depth-hop 이내 도달 가능한 모든 노드를 포함한 서브그래프를 반환.

### 메타데이터 보강

서브그래프 노드별로:
- 종목명: `dart_corp_info.name`
- 펀더멘털 등급/카테고리: `fundamentals_scores` 테이블에 데이터 있으면 합쳐서 표시

## HTML 리포트 (`report.py`)

`reports/related-{ticker}-{YYYY-MM-DD}.html` 생성.

### 섹션 1 — 헤더

대상 종목명/티커/시장 + 펀더멘털 등급(있으면) + depth + 총 연관 기업 수.

### 섹션 2 — 네트워크 그래프 (메인)

Plotly 인터랙티브 네트워크. NetworkX `spring_layout`으로 좌표 계산.

**노드:**
- 색상: 대상 종목(빨강) / 1-hop(파랑) / 2-hop(회색)
- 크기: 펀더멘털 종합 점수 (없으면 기본 크기)
- Hover: 종목명/티커/등급/카테고리

**엣지 (관계 타입별 색상):**
- Supplier 초록 / Customer 파랑 / Competitor 빨강 / Affiliate 보라 / Subsidiary 주황
- Hover: evidence 원문

### 섹션 3 — 관계 테이블

관계 타입별로 그룹핑한 테이블:

```
공급업체 (5)
  - 동진쎄미켐 (005290)  Quality ★★★★☆  "EMC 원재료 공급..."
  ...

고객사 (3)
  - SK하이닉스 (000660)  Quality ★★★★★  "HBM 후공정 장비..."
  ...
```

각 행: 종목명, 티커, 펀더멘털 등급(있으면), evidence.

### 섹션 4 — 메타정보

데이터 출처: 사업보고서 접수번호 + 접수일자, 마지막 갱신 시각.

## 실행 파이프라인

`python -m src.related.cli show 042700 --depth 1` 실행 시:

```
1. 입력 검증
   └── 042700이 dart_corp_info에 존재하는지 확인
       없으면 에러 안내

2. 보고서 접수번호 조회
   └── DART /api/list.json → 최신 사업보고서 rcept_no

3. 캐시 확인
   ├── needs_refresh(042700, current_rcept_no)?
   ├── False: 4번 스킵 → 5번으로
   └── True: 4번 진행

4. 추출 파이프라인
   ├── report_fetcher: 본문 다운로드 + 4개 섹션 파싱
   ├── extractor: GPT 호출 → JSON 응답
   ├── 티커 매핑 (회사명 → ticker)
   └── DB 저장: related_edges + related_report_meta

5. 그래프 빌드
   ├── 042700의 직접 엣지 로드
   ├── depth=2이면 1-hop 이웃의 엣지도 로드
   └── NetworkX DiGraph 구축

6. 메타데이터 보강
   ├── 종목명 (dart_corp_info)
   └── 펀더멘털 등급 (fundamentals_scores, 있으면)

7. HTML 리포트 생성
   └── reports/related-042700-YYYY-MM-DD.html

8. 브라우저 자동 오픈
```

## 의존성 및 설정

### `pyproject.toml`

추가:
```
"networkx>=3.0",
```
BeautifulSoup, OpenAI, Plotly, Jinja2는 기존 의존성 사용.

### `.env`

기존 `OPENDART_API_KEY`, `OPENAI_API_KEY` 사용. 신규 키 없음.

### `config.yaml`

신규 섹션:
```yaml
related:
  model: "gpt-5-nano"
  report_dir: "reports"
  max_tokens_per_section: 8000
```

## 비용 추정

- 평균 4개 섹션 합계 ~15,000 입력 토큰 + 출력 ~500 토큰
- gpt-5-nano 가격으로 1건당 약 $0.003
- 매일 신고가 종목 30~50개 × $0.003 = 약 $0.10/일 사용 시
- 사업보고서 접수번호 기반 캐싱으로 재실행 비용은 0

## 향후 확장 (별도 스펙)

- **신고가 스캐너 통합** — 일일 리포트에 "연관 종목" 섹션 자동 추가
- **테마 발굴** — 특정 키워드(예: HBM) 검색 → 해당 키워드를 evidence에 포함하는 모든 엣지 찾기
- **외국 기업 매핑** — 비상장 노드 중 외국 상장사(예: ASML, TSMC)는 별도 매핑 테이블로 통합
