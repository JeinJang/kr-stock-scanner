# src/krx_openapi_client.py
"""KRX Open API client using API key authentication.

Uses https://data-dbg.krx.co.kr/svc/apis/ endpoints with AUTH_KEY.
Does NOT support per-ticker historical queries or sector classifications.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests
from loguru import logger

_BASE_URL = "https://data-dbg.krx.co.kr/svc/apis"

_MARKET_ENDPOINTS = {
    "KOSPI": ("sto", "stk_bydd_trd"),
    "KOSDAQ": ("sto", "ksq_bydd_trd"),
}
_BASE_INFO_ENDPOINTS = {
    "KOSPI": ("sto", "stk_isu_base_info"),
    "KOSDAQ": ("sto", "ksq_isu_base_info"),
}


class KrxOpenApiClient:
    """KRX Open API client with API key auth and rate limiting."""

    supports_history = False

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._session = requests.Session()
        self._last_req_time = 0.0
        self._name_cache: dict[str, str] | None = None
        self._etf_name_cache: dict[str, str] | None = None

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_req_time
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)
        self._last_req_time = time.time()

    def _get(self, category: str, endpoint: str, bas_dd: str) -> list[dict]:
        self._rate_limit()
        url = f"{_BASE_URL}/{category}/{endpoint}"
        resp = self._session.get(
            url,
            params={"basDd": bas_dd},
            headers={"AUTH_KEY": self._api_key},
            timeout=30,
        )
        if resp.status_code == 401:
            raise RuntimeError(
                f"KRX API 인증 실패 (401). API 키를 확인하고, "
                f"openapi.krx.co.kr에서 '{endpoint}' 서비스 이용 신청이 "
                f"완료되었는지 확인하세요."
            )
        resp.raise_for_status()
        return resp.json().get("OutBlock_1", [])

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

    # -- Name caches -------------------------------------------------------

    def _ensure_name_cache(self, date: str) -> None:
        if self._name_cache is not None:
            return
        self._name_cache = {}
        for market, (cat, ep) in _BASE_INFO_ENDPOINTS.items():
            rows = self._get(cat, ep, date)
            for row in rows:
                short = row.get("ISU_SRT_CD", "")
                name = row.get("ISU_ABBRV", "")
                if short and name:
                    self._name_cache[short] = name
        if not self._name_cache:
            for market, (cat, ep) in _MARKET_ENDPOINTS.items():
                rows = self._get(cat, ep, date)
                for row in rows:
                    short = row.get("ISU_CD", "")
                    name = row.get("ISU_NM", "")
                    if short and name:
                        self._name_cache[short] = name

    def _ensure_etf_name_cache(self, date: str) -> None:
        if self._etf_name_cache is not None:
            return
        self._etf_name_cache = {}
        rows = self._get("etp", "etf_bydd_trd", date)
        for row in rows:
            short = row.get("ISU_CD", "")
            name = row.get("ISU_NM", "")
            if short and name:
                self._etf_name_cache[short] = name

    # -- Public API --------------------------------------------------------

    def get_market_ohlcv_by_ticker(self, date: str, market: str = "KOSPI") -> pd.DataFrame:
        if market not in _MARKET_ENDPOINTS:
            return pd.DataFrame()
        cat, ep = _MARKET_ENDPOINTS[market]
        rows = self._get(cat, ep, date)
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        col_map = {
            "ISU_CD": "티커",
            "TDD_OPNPRC": "시가",
            "TDD_HGPRC": "고가",
            "TDD_LWPRC": "저가",
            "TDD_CLSPRC": "종가",
            "ACC_TRDVOL": "거래량",
            "ACC_TRDVAL": "거래대금",
            "FLUC_RT": "등락률",
            "MKTCAP": "시가총액",
            "LIST_SHRS": "상장주식수",
        }
        available = {k: v for k, v in col_map.items() if k in df.columns}
        df = df[list(available.keys())].rename(columns=available)
        df = df.set_index("티커")
        df = self._to_numeric(
            df,
            [c for c in ["시가", "고가", "저가", "종가", "거래량", "거래대금", "시가총액", "상장주식수"] if c in df.columns],
            [c for c in ["등락률"] if c in df.columns],
        )
        return df

    def get_etf_ohlcv_by_ticker(self, date: str) -> pd.DataFrame:
        rows = self._get("etp", "etf_bydd_trd", date)
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        col_map = {
            "ISU_CD": "티커",
            "NAV": "NAV",
            "TDD_OPNPRC": "시가",
            "TDD_HGPRC": "고가",
            "TDD_LWPRC": "저가",
            "TDD_CLSPRC": "종가",
            "ACC_TRDVOL": "거래량",
            "ACC_TRDVAL": "거래대금",
            "OBJ_STKPRC_IDX": "기초지수",
        }
        available = {k: v for k, v in col_map.items() if k in df.columns}
        df = df[list(available.keys())].rename(columns=available)
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
        return pd.DataFrame()

    def get_market_sector_classifications(self, date: str, market: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_all_market_ohlcv(self, date: str) -> pd.DataFrame:
        """당일 전 시장 통합 조회는 로그인 클라이언트 전용 — 이 클라이언트는 미지원."""
        return pd.DataFrame()

    def get_market_cap_by_ticker(self, date: str, market: str = "KOSPI") -> pd.DataFrame:
        df = self.get_market_ohlcv_by_ticker(date, market=market)
        if df.empty:
            return df
        keep = [c for c in ["종가", "시가총액", "거래량", "거래대금", "상장주식수"] if c in df.columns]
        return df[keep].sort_values("시가총액", ascending=False)

    def get_market_ticker_name(self, ticker: str, date: str = "") -> str:
        if self._name_cache is None and date:
            self._ensure_name_cache(date)
        return (self._name_cache or {}).get(ticker, "")

    def get_etf_ticker_name(self, ticker: str, date: str = "") -> str:
        if self._etf_name_cache is None and date:
            self._ensure_etf_name_cache(date)
        return (self._etf_name_cache or {}).get(ticker, "")
