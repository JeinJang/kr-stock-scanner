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

    # FX(KRW=X) 전용 유령 거래일 제거: cap 종목이 전부 결측인 행은 주말/휴장일에
    # KRW=X만 거래되며 생긴 가짜 행이다. ffill 이전에 제거해야 그 행이
    # "완전한 거래일"로 둔갑해 snapshot으로 동결되는 것을 막는다.
    ghost_mask = prices[cap_tickers].isna().all(axis=1)
    if ghost_mask.any():
        prices = prices[~ghost_mask]

    # snapshot_date(마지막 완전 거래일) 이후의 트레일링 행 제거: 16:00 KST 실행 시
    # 한국 종가만 있고 미국 종가는 ffill로 채워진 하이브리드 행이 남는데, 이 행이
    # 리포트의 최신 날짜가 되면 미국 종목 전일 대비가 0%로 표시된다.
    # (중간 휴장일의 ffill은 유지 — 문제는 아무도 정정하지 않는 꼬리쪽 행뿐이다.)
    if snapshot_date is not None:
        prices = prices[[ts.date() <= snapshot_date for ts in prices.index]]

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


def _last_complete_date(prices: pd.DataFrame, cap_tickers: list[str]) -> date | None:
    """모든 cap_tickers가 원본(ffill 이전) 관측치를 가진 마지막 날짜.

    주말 FX 전용 유령일이나, 16:00 KST 실행 시 일부 해외 종목이 아직
    당일 종가를 갖지 못해 생기는 한국-오늘/미국-어제 하이브리드 행은
    "완전한 거래일"이 아니므로 제외한다.
    """
    mask = prices[cap_tickers].notna().all(axis=1)
    complete_dates = prices.index[mask]
    if len(complete_dates) == 0:
        return None
    return complete_dates.max().date()


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
                info = tk.info
                fast_shares = None
                if not (info.get("impliedSharesOutstanding") or info.get("sharesOutstanding")):
                    # info에 상장주식수가 없을 때만 fast_info를 추가 조회 (불필요한 API 호출 회피)
                    try:
                        fast_shares = getattr(tk.fast_info, "shares", None)
                    except Exception as e:  # noqa: BLE001 — fast_info 실패해도 info 결과로 재시도 가능
                        logger.warning(f"{t} fast_info 조회 실패: {e}")
                n = _resolve_shares(info, fast_shares)
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
    snapshot_date = _last_complete_date(prices, cap_tickers)
    logger.info(
        f"aihw 수집 완료: {len(prices)}일 × {len(prices.columns)}종목, "
        f"snapshot={snapshot_date}"
    )
    return build_daily_caps(
        prices, shares, fx, cap_tickers, benchmark_tickers, snapshot_date,
    )
