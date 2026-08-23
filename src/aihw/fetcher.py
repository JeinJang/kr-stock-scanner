"""src/aihw/fetcher.py

yfinance로 종목 가격·상장주식수·환율을 수집해 일별 시총(USD)을 만든다.
A 방식 근사: 현재 상장주식수 × 과거 수정종가. .KS 종목은 KRW=X 환율로 달러 환산.
부분 성공 불허 — cap 종목 하나라도 실패하면 FetchError.
"""
from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
from loguru import logger

from src.aihw.models import DailyCap

FX_TICKER = "KRW=X"


class FetchError(Exception):
    """cap 종목 수집 실패 (부분 데이터로 비율을 계산하지 않기 위해 중단)."""


def build_daily_caps(
    prices: pd.DataFrame,
    shares: dict[str, int],
    fx: pd.Series,
    cap_tickers: list[str],
    benchmark_tickers: list[str],
    snapshot_date: date | None,
) -> list[DailyCap]:
    """가격/주식수/환율 → DailyCap 목록으로 변환하는 순수 함수."""
    for t in cap_tickers:
        if t not in prices.columns or prices[t].isna().all():
            raise FetchError(f"가격 데이터 없음: {t}")
        if t not in shares or not shares[t]:
            raise FetchError(f"상장주식수 없음: {t}")

    prices = prices.ffill()
    fx = fx.ffill()

    rows: list[DailyCap] = []
    for ts in prices.index:
        d = ts.date()
        source = "snapshot" if snapshot_date and d == snapshot_date else "backfill"
        for t in cap_tickers:
            close = prices.at[ts, t]
            if pd.isna(close):
                continue  # 시계열 시작부의 결측 (ffill 이전 구간)
            cap = float(close) * shares[t]
            if t.endswith(".KS"):
                rate = fx.get(ts)
                if pd.isna(rate):
                    continue
                cap = cap / float(rate)
            rows.append(DailyCap(
                date=d, ticker=t, close=float(close), shares=shares[t],
                market_cap_usd=cap, source=source,
            ))
        for t in benchmark_tickers:
            if t not in prices.columns:
                continue
            close = prices.at[ts, t]
            if pd.isna(close):
                continue
            rows.append(DailyCap(
                date=d, ticker=t, close=float(close), shares=None,
                market_cap_usd=None, source=source,
            ))
    return rows


def _download_prices(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """yf.download로 수정종가 DataFrame(index=날짜, columns=티커)을 받는다."""
    import yfinance as yf

    df = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if df is None or df.empty:
        raise FetchError("yfinance 가격 다운로드 결과가 비어 있음")
    close = df["Close"]
    if isinstance(close, pd.Series):  # 단일 티커
        close = close.to_frame(name=tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def _resolve_shares(info: dict, fast_shares: int | None) -> int | None:
    """상장주식수 우선순위: impliedSharesOutstanding > sharesOutstanding > fast_info.shares.

    듀얼클래스 종목(GOOGL, META 등)은 `sharesOutstanding`이 일부 클래스만
    집계해 실제보다 과소평가되는 경우가 있어, 전체 클래스를 합산한
    `impliedSharesOutstanding`을 우선한다.
    """
    n = info.get("impliedSharesOutstanding") or info.get("sharesOutstanding")
    if not n:
        n = fast_shares
    return int(n) if n else None


def _download_shares(tickers: list[str], retries: int = 3) -> dict[str, int]:
    """티커별 현재 상장주식수. 실패 시 재시도 후 FetchError."""
    import yfinance as yf

    shares: dict[str, int] = {}
    for t in tickers:
        n = None
        for attempt in range(retries):
            try:
                tk = yf.Ticker(t)
                fast_shares = getattr(tk.fast_info, "shares", None)
                n = _resolve_shares(tk.info, fast_shares)
                if n:
                    break
            except Exception as e:  # noqa: BLE001 — 재시도 후 FetchError로 변환
                logger.warning(f"{t} 주식수 조회 실패 (시도 {attempt + 1}): {e}")
            time.sleep(1.0)
        if not n:
            raise FetchError(f"상장주식수 조회 실패: {t}")
        shares[t] = int(n)
    return shares


def fetch_all(
    cap_tickers: list[str],
    benchmark_tickers: list[str],
    start: date,
    end: date,
) -> list[DailyCap]:
    """가격 + 주식수 + 환율 수집 → DailyCap 목록. 최신 거래일이 snapshot."""
    all_tickers = cap_tickers + benchmark_tickers + [FX_TICKER]
    prices = _download_prices(all_tickers, start, end)
    fx = prices[FX_TICKER]
    prices = prices.drop(columns=[FX_TICKER])
    shares = _download_shares(cap_tickers)
    snapshot_date = prices.index.max().date()
    logger.info(
        f"aihw 수집 완료: {len(prices)}일 × {len(prices.columns)}종목, "
        f"snapshot={snapshot_date}"
    )
    return build_daily_caps(
        prices, shares, fx, cap_tickers, benchmark_tickers, snapshot_date,
    )
