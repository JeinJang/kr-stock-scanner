# 52주 신고가 소스를 investing.com으로 대체 — 설계

- **작성일:** 2026-08-03
- **상태:** 설계 승인 대기(스펙 리뷰용)
- **목적:** 52주 신고가 탐지를 **KRX 종목별 히스토리 순회(~2,700 요청)**에서 **investing.com의 사전 계산된 신고가 목록(요청 1건)**으로 대체해, KRX 과다요청에 따른 IP/계정 차단을 원천 제거한다.

## 배경 / 문제

현재 스캐너는 `collector.get_52w_high`가 로그인 KRX 클라이언트(`supports_history=True`)에서 **종목마다** `get_market_ohlcv_by_date`를 호출해 52주 최고가를 계산한다. 전 종목(~2,700) 순회가 짧은 시간에 몰려 KRX Data Marketplace가 IP/계정을 차단(`ip-block-page` HTML, 403)했다. (fail-fast 패치는 별도 커밋으로 이미 반영: 차단 감지 시 중단.)

investing.com은 한국 시장 **52주 신고가 목록을 이미 계산해 제공**한다(`kr.investing.com/equities/52-week-high`). 이를 소스로 쓰면 요청이 1건(+페이지네이션)으로 줄어 차단 문제가 사라진다.

## 실현성 스파이크 결과 (완료)

- `requests` 직접 GET → **403 Cloudflare**(불가).
- **`curl_cffi`(impersonate=`chrome124`) → HTTP 200**, 전체 페이지 수신. `chrome131`은 실패, `chrome124`·`safari17_0` 성공 → **impersonate 타깃 폴백 필요**.
- 페이지는 Next.js 앱. 데이터는 `<script id="__NEXT_DATA__">` JSON의
  `props.pageProps.state.assetsCollectionStore.assetsCollection._collection` 배열에 임베드되며, `total` 필드가 함께 온다.
  - 스파이크 시점: `_collection` 53개, `total` 53 → 그날 전량이 초기 페이로드에 포함.
  - 각 원소 필드: `Name`, `Last`, `changeOneDayPercent`, `avgVolume` 등(HTML 표보다 파싱 안정적).
- HTML `<table>`(SSR)로도 파싱 가능하나(종목명/현재가/고가/저가/변동%/거래량/시간), **`__NEXT_DATA__` JSON 파싱을 1차 경로로 채택**(구조가 명확하고 필드가 풍부).

## 확정된 설계 결정

1. **소스 교체**: investing.com 신고가 목록이 52주 신고가 탐지를 **대체**한다(KRX 종목별 순회 제거). investing의 신고가 "정의"를 그대로 채택한다(기존 "장중 고가가 직전 52주 최고 돌파" 규칙 대체 — 의미 변화 있음, 아래 명시).
2. **취득**: `curl_cffi`로 GET. `impersonate`는 `["chrome124", "safari17_0"]` 순차 폴백. Cloudflare 챌린지/403/구조 변경 시 **명확한 예외로 중단**(무한 재시도 금지).
3. **파싱**: `__NEXT_DATA__` → `_collection`에서 종목명·현재가·변동%·거래량 추출. `total > len(_collection)`이면 다음 페이지를 추가 로드해 병합.
4. **거래량 필터**: 거래량이 0/없음/파싱 불가인 종목은 제외(정지·유동성 없는 종목 제거).
5. **매핑**: 종목명 → `dart_corp_info.name` 정규화 매칭으로 6자리 티커·시장(KOSPI/KOSDAQ) 확정. 미매칭 종목은 로그 남기고 스킵(ETF·신규상장·해외 등).
6. **KRX 최소 사용 유지**: 매칭된 신고가 종목의 **시가총액·섹터**는 기존 KRX 벌크 호출(`get_market_caps`, 섹터맵)로 보강. 이들은 시장당 1건 벌크라 차단 위험이 없다. **종목별 히스토리 순회만 제거**된다.

## 구조 (모듈 경계)

새 모듈 **`src/investing_high.py`** — 단일 책임: "오늘의 52주 신고가 목록(티커·시장·가격·거래량) 반환".

```
fetch_52w_high_rows() -> list[InvestingHighRow]
    # curl_cffi fetch + __NEXT_DATA__ 파싱 + 페이지네이션 병합 + 거래량 필터
    # 실패(챌린지/구조변경) 시 InvestingFetchError 발생

resolve_to_krx(rows, name_to_ticker, name_to_market)
    -> (matched: list[High], unmatched: list[str])
    # 종목명 → dart_corp_info 매핑, 미매칭 분리
```

`InvestingHighRow`: `name, last_price, change_pct, volume`.
`High`(스캐너 소비 형태): `ticker, name, market, price, volume(+ 시총·섹터는 KRX 보강 후 채움)`.

**스캐너 통합**: `src/cli.py`의 `run`에서 52주 신고가 소스를 `investing_high` 결과로 교체. 이후 시총/섹터 보강 → 기존 `build_scan_result`/뉴스/AI/리포트 파이프라인은 그대로.

## 데이터 흐름

```
investing 52w-high 페이지 (curl_cffi, 1~n 요청)
  → __NEXT_DATA__ JSON 파싱 → 거래량 필터
  → 종목명→티커·시장 매핑(dart_corp_info)  [미매칭 스킵·로그]
  → KRX 벌크로 시총·섹터 보강 (시장당 1건)
  → ScanResult 구성 → 뉴스/AI/리포트 (기존 경로)
```

## 에러 처리

- **Cloudflare 챌린지/403**: impersonate 폴백 모두 실패 시 `InvestingFetchError` 발생 → run 중단(명확 메시지). 무한 재시도 금지.
- **구조 변경**(`__NEXT_DATA__` 없음 / `_collection` 경로 부재): `InvestingParseError`로 중단(조용한 빈 결과 금지).
- **페이지네이션 미해결**: `total > 취득 수`인데 다음 페이지를 못 가져오면, **커버리지가 잘렸음을 경고 로그**로 남긴다(조용한 누락 금지).
- **미매칭 종목**: 개수와 이름을 로그로 남긴다(무단 누락 아님).

## 의존성

- `curl_cffi`(이미 venv에 존재, `pyproject.toml`에 명시 추가). `bs4`는 기존 의존성.

## 테스트

- **네트워크 없는 단위테스트**: 저장한 `__NEXT_DATA__` JSON 픽스처로
  ① 파싱(행 수·필드), ② 거래량 0/없음 필터, ③ 종목명→티커 매핑(정상·미매칭·정규화 케이스), ④ `total>len` 시 페이지 병합 로직(모킹), ⑤ 챌린지/구조변경 시 예외.
- **impersonate 폴백** 로직 단위테스트(첫 타깃 403 → 다음 타깃 성공 모킹).
- 실제 네트워크 호출은 테스트에서 제외(수동 검증 1회로 대체).

## 범위 밖 (YAGNI)

- 과거 신고가 백필(당일 목록만).
- investing의 "상승 여력"(유료) 등 부가 컬럼.
- KRX 신고가 bld 방식(대안으로 검토했으나 이번 스펙은 investing 채택).
- 페이지네이션 엔드포인트의 완전 리버스엔지니어링 — 구현 시 신고가 많은 날에 실제 `total>len`을 만나 파라미터를 확정하고, 그전까지는 초기 배치 + 경고 로그로 안전 처리.

## 의미 변화 (주의)

- 신고가 "정의"가 기존 스캐너 자체 계산(장중 고가 돌파)에서 **investing.com 기준**으로 바뀐다. 산출 종목 집합이 완전히 동일하지 않을 수 있으며, 이는 의도된 변경이다. 리포트에 소스가 investing임을 표기한다.
