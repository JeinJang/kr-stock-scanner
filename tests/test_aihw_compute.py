from datetime import date

import pytest

from src.aihw.compute import build_series
from src.aihw.models import DailyCap


def _cap(d, ticker, cap_usd, close=100.0):
    return DailyCap(
        date=d, ticker=ticker, close=close, shares=1000,
        market_cap_usd=cap_usd, source="backfill",
    )


def _bench(d, ticker, close):
    return DailyCap(
        date=d, ticker=ticker, close=close, shares=None,
        market_cap_usd=None, source="backfill",
    )


D1, D2, D3 = date(2026, 1, 10), date(2026, 1, 11), date(2026, 1, 12)
AI_HW = ["NVDA", "MU"]
BIG_TECH = ["MSFT", "META"]


def _sample_caps():
    caps = []
    # D1: AI HW 합=300, 빅테크 합=500 → ratio 0.6
    caps += [_cap(D1, "NVDA", 200.0), _cap(D1, "MU", 100.0)]
    caps += [_cap(D1, "MSFT", 300.0), _cap(D1, "META", 200.0)]
    caps += [_bench(D1, "SPY", 100.0), _bench(D1, "RSP", 50.0)]
    # D2: AI HW 합=440, 빅테크 합=550 → ratio 0.8
    caps += [_cap(D2, "NVDA", 300.0), _cap(D2, "MU", 140.0)]
    caps += [_cap(D2, "MSFT", 330.0), _cap(D2, "META", 220.0)]
    caps += [_bench(D2, "SPY", 110.0), _bench(D2, "RSP", 51.0)]
    return caps


class TestBuildSeries:
    def test_group_totals_and_ratio(self):
        s = build_series(_sample_caps(), AI_HW, BIG_TECH, ["SPY", "RSP"], base_date=D1)
        assert s.dates == [D1, D2]
        assert s.ai_hw_total == [300.0, 440.0]
        assert s.big_tech_total == [500.0, 550.0]
        assert s.ratio == [pytest.approx(0.6), pytest.approx(0.8)]

    def test_indexed_to_base_date(self):
        s = build_series(_sample_caps(), AI_HW, BIG_TECH, ["SPY", "RSP"], base_date=D1)
        assert s.indexed["AI HW"] == [pytest.approx(100.0), pytest.approx(146.6667, abs=0.01)]
        assert s.indexed["빅테크"] == [pytest.approx(100.0), pytest.approx(110.0)]
        assert s.indexed["SPY"] == [pytest.approx(100.0), pytest.approx(110.0)]
        assert s.indexed["RSP"] == [pytest.approx(100.0), pytest.approx(102.0)]

    def test_base_date_on_holiday_uses_next_available(self):
        # base_date가 D1 이전(휴장)이면 첫 거래일(D1)을 기준으로 지수화
        s = build_series(
            _sample_caps(), AI_HW, BIG_TECH, ["SPY"], base_date=date(2026, 1, 9)
        )
        assert s.indexed["AI HW"][0] == pytest.approx(100.0)

    def test_date_with_missing_cap_ticker_is_dropped(self):
        caps = _sample_caps()
        # D3에는 NVDA가 빠짐 → D3 전체 제외
        caps += [_cap(D3, "MU", 150.0), _cap(D3, "MSFT", 340.0), _cap(D3, "META", 230.0)]
        s = build_series(caps, AI_HW, BIG_TECH, [], base_date=D1)
        assert s.dates == [D1, D2]

    def test_missing_benchmark_does_not_drop_date(self):
        caps = _sample_caps()
        # D2의 RSP 제거 → 날짜는 유지, RSP 지수만 해당일 생략 없이 이전값 유지 안 함
        caps = [c for c in caps if not (c.date == D2 and c.ticker == "RSP")]
        s = build_series(caps, AI_HW, BIG_TECH, ["SPY", "RSP"], base_date=D1)
        assert s.dates == [D1, D2]
        assert len(s.indexed["SPY"]) == 2
        assert "RSP" not in s.indexed
