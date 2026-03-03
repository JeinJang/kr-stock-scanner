# src/krx_client.py
"""KRX client Protocol + factory.

Defines the KrxClient protocol and a factory function that returns either
KrxLoginClient (session-based) or KrxOpenApiClient (API key-based) depending
on available credentials.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class KrxClient(Protocol):
    """Protocol for KRX data clients."""

    @property
    def supports_history(self) -> bool:
        """Whether per-ticker historical OHLCV queries are supported."""
        ...

    def get_market_ohlcv_by_ticker(self, date: str, market: str = "KOSPI") -> pd.DataFrame: ...

    def get_etf_ohlcv_by_ticker(self, date: str) -> pd.DataFrame: ...

    def get_market_ohlcv_by_date(
        self, fromdate: str, todate: str, ticker: str, adjusted: bool = False,
    ) -> pd.DataFrame: ...

    def get_market_sector_classifications(self, date: str, market: str) -> pd.DataFrame: ...

    def get_market_cap_by_ticker(self, date: str, market: str = "KOSPI") -> pd.DataFrame: ...

    def get_market_ticker_name(self, ticker: str, date: str = "") -> str: ...

    def get_etf_ticker_name(self, ticker: str, date: str = "") -> str: ...


def create_krx_client(*, krx_id: str = "", krx_pw: str = "", krx_api_key: str = "") -> KrxClient:
    """Create a KRX client based on available credentials.

    - krx_id + krx_pw  -> KrxLoginClient  (session login, supports per-ticker history)
    - krx_api_key      -> KrxOpenApiClient (API key auth, batch only)
    - Both provided    -> KrxLoginClient preferred
    """
    if krx_id and krx_pw:
        from src.krx_login_client import KrxLoginClient
        return KrxLoginClient(krx_id=krx_id, krx_pw=krx_pw)

    if krx_api_key:
        from src.krx_openapi_client import KrxOpenApiClient
        return KrxOpenApiClient(api_key=krx_api_key)

    raise ValueError(
        "KRX 인증 정보가 없습니다. .env에 KRX_ID+KRX_PW 또는 KRX_API_KEY를 설정하세요."
    )
