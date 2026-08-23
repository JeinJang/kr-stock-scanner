from datetime import date

import pandas as pd
import pytest

from src.aihw.fetcher import FetchError, build_daily_caps

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
