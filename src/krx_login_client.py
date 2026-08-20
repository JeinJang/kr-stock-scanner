# src/krx_login_client.py
"""KRX client using session login authentication.

Logs in via data.krx.co.kr and uses getJsonData.cmd to fetch market data.
Supports per-ticker historical queries and sector classifications.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests
from loguru import logger

_LOGIN_PAGE = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd"
_LOGIN_JSP = "https://data.krx.co.kr/contents/MDC/COMS/client/view/login.jsp?site=mdc"
_LOGIN_URL = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001D1.cmd"
_DATA_URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
_MAIN_PAGE = "https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201"

_BLD = {
    "ohlcv_by_ticker": "dbms/MDC/STAT/standard/MDCSTAT01501",
    "ohlcv_by_date": "dbms/MDC/STAT/standard/MDCSTAT01701",
    "etf_by_ticker": "dbms/MDC/STAT/standard/MDCSTAT04301",
    "sector": "dbms/MDC/STAT/standard/MDCSTAT03901",
    "finder": "dbms/comm/finder/finder_stkisu",
}

_MARKET_CODES = {"KOSPI": "STK", "KOSDAQ": "KSQ", "KONEX": "KNX"}

# 전 시장 통합 조회(mktId=ALL)의 MKT_NM -> 저장소 시장 구분. 실측(20260819)
# KOSPI 942·KOSDAQ 1,771·KOSDAQ GLOBAL 50·KONEX 109 — 942는 저장소 KOSPI
# 개수와 정확히 일치, 1,771+50=1,821은 저장소 KOSDAQ 개수와 정확히 일치해
# KOSDAQ GLOBAL을 KOSDAQ에 합치는 매핑이 맞음을 확인했다. KONEX·그 외는 버림.
_MKT_NM_TO_MARKET = {"KOSPI": "KOSPI", "KOSDAQ": "KOSDAQ", "KOSDAQ GLOBAL": "KOSDAQ"}

# KRX가 IP/계정을 차단하면 JSON 대신 이 문구가 담긴 HTML 에러페이지(200)를 반환.
_BLOCK_MARKERS = ("ip-block-page", "에러페이지 - 한국거래소")


def _is_block_page(text: str) -> bool:
    """KRX Data Marketplace 차단/에러 HTML 페이지인지 판별."""
    if not text:
        return False
    return any(marker in text for marker in _BLOCK_MARKERS)


class KrxBlockedError(RuntimeError):
    """KRX가 IP 또는 계정을 차단했을 때 발생. 추가 요청을 즉시 중단하기 위함."""


class KrxLoginClient:
    """KRX client with session login auth. Supports per-ticker history."""

    supports_history = True

    def __init__(self, krx_id: str, krx_pw: str):
        self._krx_id = krx_id
        self._krx_pw = krx_pw
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": _MAIN_PAGE,
        })
        self._logged_in = False
        self._blocked = False
        self._session_initialized = False
        self._last_req_time = 0.0
        self._name_cache: dict[str, str] | None = None
        self._etf_name_cache: dict[str, str] | None = None
        self._isin_cache: dict[str, str] = {}

    # -- Session & auth ----------------------------------------------------

    def _init_session(self) -> None:
        """Visit login page + JSP to get session cookies (JSESSIONID etc.)."""
        if self._session_initialized:
            return
        try:
            self._session.get(_LOGIN_PAGE, timeout=15)
            self._session.get(
                _LOGIN_JSP,
                headers={"Referer": _LOGIN_PAGE},
                timeout=15,
            )
            self._session_initialized = True
        except requests.RequestException as e:
            logger.warning(f"KRX 세션 초기화 실패: {e}")

    def _login(self) -> None:
        if self._logged_in:
            return
        self._init_session()
        payload = {
            "mbrNm": "",
            "telNo": "",
            "di": "",
            "certType": "",
            "mbrId": self._krx_id,
            "pw": self._krx_pw,
        }
        try:
            resp = self._session.post(
                _LOGIN_URL,
                data=payload,
                headers={"Referer": _LOGIN_PAGE},
                timeout=15,
            )
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"KRX 로그인 요청 실패: {e}")
            return

        error_code = data.get("_error_code", "")

        # CD011 = 중복 로그인 — skipDup=Y 로 재시도
        if error_code == "CD011":
            logger.debug("KRX 중복 로그인 감지 — skipDup=Y 재시도")
            payload["skipDup"] = "Y"
            try:
                resp = self._session.post(
                    _LOGIN_URL,
                    data=payload,
                    headers={"Referer": _LOGIN_PAGE},
                    timeout=15,
                )
                error_code = resp.json().get("_error_code", "")
            except (requests.RequestException, ValueError) as e:
                logger.warning(f"KRX 중복 로그인 재시도 실패: {e}")
                return

        if error_code == "CD001":
            self._logged_in = True
            logger.info("KRX 로그인 성공")
        else:
            logger.warning(
                f"KRX 로그인 실패 (error_code={error_code}). "
                f"KRX_ID/KRX_PW를 확인하거나 data.krx.co.kr에서 직접 로그인을 시도하세요."
            )

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_req_time
        if elapsed < 0.2:
            time.sleep(0.2 - elapsed)
        self._last_req_time = time.time()
        

    def _guard_block(self, body: str) -> None:
        """차단 페이지면 blocked 플래그를 세우고 즉시 중단(추가 요청 방지)."""
        if _is_block_page(body):
            self._blocked = True
            logger.error(
                "KRX 차단 페이지 감지(IP 또는 계정 차단). 이후 요청을 모두 중단합니다. "
                "재시도는 차단을 연장시킬 수 있으니, 수 시간 뒤 또는 data.krx.co.kr "
                "계정/IP 상태 확인 후 진행하세요."
            )
            raise KrxBlockedError("KRX Data Marketplace 차단 페이지 수신")

    def _post(self, bld: str, params: dict) -> list[dict]:
        """POST to getJsonData.cmd. Tries without login first, retries with login on LOGOUT."""
        if self._blocked:
            # 이미 차단 감지 — 네트워크 요청 없이 즉시 중단
            raise KrxBlockedError("KRX 차단 상태 — 요청 중단")
        self._init_session()
        self._rate_limit()
        data = {"bld": bld, **params}

        resp = self._session.post(_DATA_URL, data=data, timeout=60)
        body = resp.text.strip()
        self._guard_block(body)

        if body == "LOGOUT" or resp.status_code == 400:
            if not self._logged_in:
                logger.debug("KRX LOGOUT 응답 — 로그인 시도")
                self._login()
                if self._logged_in:
                    self._rate_limit()
                    resp = self._session.post(_DATA_URL, data=data, timeout=60)
                    body = resp.text.strip()
                    self._guard_block(body)

            if body == "LOGOUT" or resp.status_code == 400:
                logger.warning(
                    f"KRX 데이터 요청 실패 (LOGOUT). BLD={bld}. "
                    f"data.krx.co.kr에서 계정 상태를 확인하세요."
                )
                return []

        if resp.status_code != 200:
            logger.warning(f"KRX 응답 오류: HTTP {resp.status_code}")
            return []

        try:
            result = resp.json()
        except ValueError:
            logger.warning(f"KRX JSON 파싱 실패: {body[:100]}")
            return []

        return result.get("OutBlock_1", result.get("block1", result.get("output", [])))

    def _resolve_isin(self, ticker: str) -> str:
        if ticker in self._isin_cache:
            return self._isin_cache[ticker]
        rows = self._post(_BLD["finder"], {"mktsel": "ALL", "searchText": ticker})
        for row in rows:
            short = row.get("short_code", "")
            isin = row.get("full_code", "")
            if short and isin:
                self._isin_cache[short] = isin
                if short == ticker:
                    return isin
        logger.warning(f"ISIN 변환 실패: {ticker}")
        return ticker

    @staticmethod
    def _to_numeric(df: pd.DataFrame, int_cols: list[str], float_cols: list[str]) -> pd.DataFrame:
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                ).fillna(0).astype(np.int64)
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                ).fillna(0.0).astype(np.float64)
        return df

    # -- Public API --------------------------------------------------------

    def get_market_ohlcv_by_ticker(self, date: str, market: str = "KOSPI") -> pd.DataFrame:
        mkt_id = _MARKET_CODES.get(market)
        if mkt_id is None:
            return pd.DataFrame()
        rows = self._post(_BLD["ohlcv_by_ticker"], {"mktId": mkt_id, "trdDd": date})
        if not rows:
            return pd.DataFrame()

        for row in rows:
            short = row.get("ISU_SRT_CD", "")
            isin = row.get("ISU_CD", "")
            if short and isin:
                self._isin_cache[short] = isin

        df = pd.DataFrame(rows)
        col_map = {
            "ISU_SRT_CD": "티커", "TDD_OPNPRC": "시가", "TDD_HGPRC": "고가",
            "TDD_LWPRC": "저가", "TDD_CLSPRC": "종가", "ACC_TRDVOL": "거래량",
            "ACC_TRDVAL": "거래대금", "FLUC_RT": "등락률", "MKTCAP": "시가총액",
            "LIST_SHRS": "상장주식수",
        }
        available = {k: v for k, v in col_map.items() if k in df.columns}
        if not available:
            return pd.DataFrame()
        df = df[list(available.keys())].rename(columns=available)
        if "티커" not in df.columns:
            return pd.DataFrame()
        df = df.set_index("티커")
        df = self._to_numeric(
            df,
            [c for c in ["시가", "고가", "저가", "종가", "거래량", "거래대금", "시가총액", "상장주식수"] if c in df.columns],
            [c for c in ["등락률"] if c in df.columns],
        )
        return df

    def get_all_market_ohlcv(self, date: str) -> pd.DataFrame:
        """전 시장 통합 당일 시세(mktId=ALL) — 한 번의 호출로 KOSPI+KOSDAQ 전 종목.

        오픈 API가 아직 당일 데이터를 주지 않을 때(장 마감 후 최대 2시간
        16분 실측) 당일 하루치를 채우는 용도. 인덱스는 6자리 티커.
        """
        rows = self._post(_BLD["ohlcv_by_ticker"], {"mktId": "ALL", "trdDd": date})
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        col_map = {
            "ISU_SRT_CD": "티커", "MKT_NM": "시장", "TDD_HGPRC": "고가",
            "TDD_CLSPRC": "종가", "CMPPREVDD_PRC": "전일대비",
        }
        available = {k: v for k, v in col_map.items() if k in df.columns}
        if "ISU_SRT_CD" not in available or "MKT_NM" not in available:
            return pd.DataFrame()
        df = df[list(available.keys())].rename(columns=available)

        df["시장"] = df["시장"].map(_MKT_NM_TO_MARKET)
        df = df[df["시장"].notna()]
        if df.empty:
            return pd.DataFrame()

        df = df.set_index("티커")
        df = self._to_numeric(
            df,
            [c for c in ["고가", "종가", "전일대비"] if c in df.columns],
            [],
        )
        return df

    def get_etf_ohlcv_by_ticker(self, date: str) -> pd.DataFrame:
        rows = self._post(_BLD["etf_by_ticker"], {"trdDd": date})
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        col_map = {
            "ISU_SRT_CD": "티커", "NAV": "NAV", "TDD_OPNPRC": "시가",
            "TDD_HGPRC": "고가", "TDD_LWPRC": "저가", "TDD_CLSPRC": "종가",
            "ACC_TRDVOL": "거래량", "ACC_TRDVAL": "거래대금", "OBJ_STKPRC_IDX": "기초지수",
        }
        available = {k: v for k, v in col_map.items() if k in df.columns}
        if not available:
            return pd.DataFrame()
        df = df[list(available.keys())].rename(columns=available)
        if "티커" not in df.columns:
            return pd.DataFrame()
        df = df.set_index("티커")
        df = self._to_numeric(
            df,
            [c for c in ["시가", "고가", "저가", "종가", "거래량", "거래대금"] if c in df.columns],
            [c for c in ["NAV", "기초지수"] if c in df.columns],
        )
        return df

    def get_market_ohlcv_by_date(
        self, fromdate: str, todate: str, ticker: str, adjusted: bool = False,
    ) -> pd.DataFrame:
        isin = self._resolve_isin(ticker)
        rows = self._post(_BLD["ohlcv_by_date"], {
            "isuCd": isin, "strtDd": fromdate, "endDd": todate,
            "adjStkPrc": 2 if adjusted else 1,
        })
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        col_map = {
            "TRD_DD": "날짜", "TDD_OPNPRC": "시가", "TDD_HGPRC": "고가",
            "TDD_LWPRC": "저가", "TDD_CLSPRC": "종가", "ACC_TRDVOL": "거래량",
            "ACC_TRDVAL": "거래대금", "FLUC_RT": "등락률",
        }
        available = {k: v for k, v in col_map.items() if k in df.columns}
        if not available:
            return pd.DataFrame()
        df = df[list(available.keys())].rename(columns=available)
        if "날짜" in df.columns:
            df["날짜"] = df["날짜"].str.replace("/", "", regex=False)
            df = df.set_index("날짜")
        df = self._to_numeric(
            df,
            [c for c in ["시가", "고가", "저가", "종가", "거래량", "거래대금"] if c in df.columns],
            [c for c in ["등락률"] if c in df.columns],
        )
        return df

    def get_market_sector_classifications(self, date: str, market: str) -> pd.DataFrame:
        mkt_id = _MARKET_CODES.get(market)
        if mkt_id is None:
            return pd.DataFrame()
        rows = self._post(_BLD["sector"], {"mktId": mkt_id, "trdDd": date})
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        col_map = {"ISU_SRT_CD": "티커", "ISU_ABBRV": "종목명", "IDX_IND_NM": "업종명"}
        available = {k: v for k, v in col_map.items() if k in df.columns}
        if not available:
            return pd.DataFrame()
        return df[list(available.keys())].rename(columns=available)

    def get_market_cap_by_ticker(self, date: str, market: str = "KOSPI") -> pd.DataFrame:
        df = self.get_market_ohlcv_by_ticker(date, market=market)
        if df.empty:
            return df
        keep = [c for c in ["종가", "시가총액", "거래량", "거래대금", "상장주식수"] if c in df.columns]
        return df[keep].sort_values("시가총액", ascending=False)

    def get_market_ticker_name(self, ticker: str, date: str = "") -> str:
        if self._name_cache is None and date:
            self._build_name_cache(date)
        return (self._name_cache or {}).get(ticker, "")

    def get_etf_ticker_name(self, ticker: str, date: str = "") -> str:
        if self._etf_name_cache is None and date:
            self._build_etf_name_cache(date)
        return (self._etf_name_cache or {}).get(ticker, "")

    def _build_name_cache(self, date: str) -> None:
        self._name_cache = {}
        for market in ["KOSPI", "KOSDAQ"]:
            mkt_id = _MARKET_CODES.get(market)
            if mkt_id is None:
                continue
            rows = self._post(_BLD["ohlcv_by_ticker"], {"mktId": mkt_id, "trdDd": date})
            for row in rows:
                short = row.get("ISU_SRT_CD", "")
                name = row.get("ISU_ABBRV", "")
                if short and name:
                    self._name_cache[short] = name

    def _build_etf_name_cache(self, date: str) -> None:
        self._etf_name_cache = {}
        rows = self._post(_BLD["etf_by_ticker"], {"trdDd": date})
        for row in rows:
            short = row.get("ISU_SRT_CD", "")
            name = row.get("ISU_ABBRV", "")
            if short and name:
                self._etf_name_cache[short] = name
