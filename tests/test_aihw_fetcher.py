import sys
import types
from datetime import date

import pandas as pd
import pytest

from src.aihw.fetcher import (
    FetchError,
    _download_shares,
    _last_complete_date,
    _resolve_shares,
    build_daily_caps,
)

IDX = pd.to_datetime(["2026-01-10", "2026-01-11", "2026-01-12"])


def _prices():
    return pd.DataFrame(
        {
            "NVDA": [100.0, 110.0, 120.0],
            "005930.KS": [70000.0, None, 72000.0],  # 1/11 한국 휴장
            "SPY": [500.0, 505.0, 510.0],
        },
        index=IDX,
    )


def _fx():
    # 1/11 환율 누락 → ffill로 1300 사용
    return pd.Series([1300.0, None, 1350.0], index=IDX)


SHARES = {"NVDA": 1000, "005930.KS": 5000}


class TestBuildDailyCaps:
    def test_usd_ticker_cap(self):
        caps = build_daily_caps(
            _prices(), SHARES, _fx(), ["NVDA", "005930.KS"], ["SPY"], None
        )
        nvda_d1 = next(c for c in caps if c.ticker == "NVDA" and c.date == date(2026, 1, 10))
        assert nvda_d1.market_cap_usd == pytest.approx(100.0 * 1000)
        assert nvda_d1.shares == 1000

    def test_krw_ticker_converted_with_ffilled_fx(self):
        caps = build_daily_caps(
            _prices(), SHARES, _fx(), ["NVDA", "005930.KS"], ["SPY"], None
        )
        # 1/11: 삼전 종가 ffill(70000), 환율 ffill(1300)
        s_d2 = next(c for c in caps if c.ticker == "005930.KS" and c.date == date(2026, 1, 11))
        assert s_d2.close == pytest.approx(70000.0)
        assert s_d2.market_cap_usd == pytest.approx(70000.0 * 5000 / 1300.0)

    def test_benchmark_rows_have_no_cap(self):
        caps = build_daily_caps(
            _prices(), SHARES, _fx(), ["NVDA", "005930.KS"], ["SPY"], None
        )
        spy = next(c for c in caps if c.ticker == "SPY" and c.date == date(2026, 1, 10))
        assert spy.market_cap_usd is None
        assert spy.shares is None
        assert spy.close == pytest.approx(500.0)

    def test_snapshot_date_marks_source(self):
        caps = build_daily_caps(
            _prices(), SHARES, _fx(), ["NVDA", "005930.KS"], ["SPY"],
            snapshot_date=date(2026, 1, 12),
        )
        assert all(
            c.source == ("snapshot" if c.date == date(2026, 1, 12) else "backfill")
            for c in caps
        )

    def test_trailing_rows_keep_only_raw_observations(self):
        # 16:00 KST 실행: 마지막 날(1/12)에 한국 종가는 있지만 미국(NVDA)은 아직 없음.
        # snapshot_date(마지막 완전 거래일) 이후 구간은 실제 관측치만 배출하고,
        # ffill 복사본(미국 종가)은 배출하지 않는다 — 복사본이 배출되면 미국 종목
        # 전일 대비가 0%로 표시된다.
        prices = _prices()
        prices.loc[IDX[2], "NVDA"] = None  # 미국 미마감
        caps = build_daily_caps(
            prices, SHARES, _fx(), ["NVDA", "005930.KS"], ["SPY"],
            snapshot_date=date(2026, 1, 11),
        )
        d3 = [c for c in caps if c.date == date(2026, 1, 12)]
        tickers_d3 = {c.ticker for c in d3}
        assert "NVDA" not in tickers_d3  # ffill 복사본 미배출
        assert "005930.KS" in tickers_d3  # 실제 관측치는 배출
        assert "SPY" in tickers_d3
        ks = next(c for c in d3 if c.ticker == "005930.KS")
        assert ks.close == pytest.approx(72000.0)
        assert ks.source == "backfill"
        # 완전 구간(1/11 이하)은 기존처럼 ffill 유지
        assert any(c.ticker == "NVDA" and c.date == date(2026, 1, 11) for c in caps)

    def test_missing_cap_ticker_column_raises(self):
        with pytest.raises(FetchError, match="MU"):
            build_daily_caps(_prices(), SHARES, _fx(), ["NVDA", "MU"], [], None)

    def test_missing_shares_raises(self):
        with pytest.raises(FetchError, match="005930.KS"):
            build_daily_caps(_prices(), {"NVDA": 1000}, _fx(), ["NVDA", "005930.KS"], [], None)

    def test_all_nan_cap_column_raises(self):
        prices = _prices()
        prices["NVDA"] = None
        with pytest.raises(FetchError, match="NVDA"):
            build_daily_caps(prices, SHARES, _fx(), ["NVDA"], [], None)

    def test_leading_nan_rows_are_omitted(self):
        prices = _prices()
        prices.loc[IDX[0], "005930.KS"] = None  # 시작일 결측 (ffill 불가)
        fx = _fx()
        fx.loc[IDX[0]] = None
        caps = build_daily_caps(prices, SHARES, fx, ["NVDA", "005930.KS"], ["SPY"], None)
        # 시작일의 삼전 행은 조용히 생략되고, 에러 없이 나머지는 생성된다
        assert not any(c.ticker == "005930.KS" and c.date == date(2026, 1, 10) for c in caps)
        assert any(c.ticker == "005930.KS" and c.date == date(2026, 1, 12) for c in caps)
        assert any(c.ticker == "NVDA" and c.date == date(2026, 1, 10) for c in caps)

    def test_fx_only_ghost_day_is_removed_entirely(self):
        # 1/11: 주말/휴장으로 cap 종목(NVDA, 005930.KS)이 전부 결측 —
        # KRW=X만 거래되는 유령 거래일. ffill로 "가짜 완전일"이 되면 안 된다.
        prices = pd.DataFrame(
            {
                "NVDA": [100.0, None, 120.0],
                "005930.KS": [70000.0, None, 72000.0],
                "SPY": [500.0, None, 510.0],
            },
            index=IDX,
        )
        fx = pd.Series([1300.0, 1310.0, 1350.0], index=IDX)  # FX는 매일 거래
        caps = build_daily_caps(
            prices, SHARES, fx, ["NVDA", "005930.KS"], ["SPY"], None
        )
        assert not any(c.date == date(2026, 1, 11) for c in caps)
        # ffill이 유령일을 건너뛰고 직전 실거래일 값을 정상적으로 이어받는다
        nvda_d3 = next(c for c in caps if c.ticker == "NVDA" and c.date == date(2026, 1, 12))
        assert nvda_d3.close == pytest.approx(120.0)


class TestLastCompleteDate:
    def test_returns_last_date_all_cap_tickers_have_raw_observation(self):
        idx = pd.to_datetime(["2026-01-10", "2026-01-11", "2026-01-12"])
        prices = pd.DataFrame(
            {
                "NVDA": [100.0, 110.0, 120.0],
                "005930.KS": [70000.0, 71000.0, None],  # 마지막날 결측(하이브리드 행)
            },
            index=idx,
        )
        assert _last_complete_date(prices, ["NVDA", "005930.KS"]) == date(2026, 1, 11)

    def test_all_complete_returns_max_date(self):
        idx = pd.to_datetime(["2026-01-10", "2026-01-11"])
        prices = pd.DataFrame({"NVDA": [100.0, 110.0]}, index=idx)
        assert _last_complete_date(prices, ["NVDA"]) == date(2026, 1, 11)

    def test_no_complete_date_returns_none(self):
        idx = pd.to_datetime(["2026-01-10"])
        prices = pd.DataFrame({"NVDA": [None]}, index=idx)
        assert _last_complete_date(prices, ["NVDA"]) is None


class TestResolveShares:
    def test_prefers_implied_over_shares_outstanding(self):
        # GOOGL류 듀얼클래스: sharesOutstanding은 일부 클래스만 집계 → implied 우선
        info = {"impliedSharesOutstanding": 12_229_934_831, "sharesOutstanding": 5_867_155_790}
        assert _resolve_shares(info, fast_shares=999) == 12_229_934_831

    def test_falls_back_to_shares_outstanding_when_implied_missing(self):
        info = {"sharesOutstanding": 7_430_000_000}
        assert _resolve_shares(info, fast_shares=999) == 7_430_000_000

    def test_falls_back_to_fast_info_when_info_empty(self):
        assert _resolve_shares({}, fast_shares=24_221_000_000) == 24_221_000_000

    def test_returns_none_when_nothing_available(self):
        assert _resolve_shares({}, fast_shares=None) is None


class TestDownloadShares:
    """fast_info는 info에 상장주식수가 없을 때만 조회해야 한다 (I3)."""

    def test_skips_fast_info_when_info_has_shares(self, monkeypatch):
        class FakeTicker:
            def __init__(self, ticker):
                self.info = {"impliedSharesOutstanding": 12_000_000_000}

            @property
            def fast_info(self):
                raise AssertionError("info에 상장주식수가 있으면 fast_info를 조회하면 안 됨")

        monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))
        shares = _download_shares(["NVDA"])
        assert shares["NVDA"] == 12_000_000_000

    def test_queries_fast_info_when_info_lacks_shares(self, monkeypatch):
        class FakeFastInfo:
            shares = 24_221_000_000

        class FakeTicker:
            def __init__(self, ticker):
                self.info = {}
                self.fast_info = FakeFastInfo()

        monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))
        shares = _download_shares(["MU"])
        assert shares["MU"] == 24_221_000_000
