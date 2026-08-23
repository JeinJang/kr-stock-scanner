from datetime import date

import pytest

from src.aihw.compute import build_series, summarize, threshold_status
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

    def test_company_caps_grouped_by_group_name(self):
        s = build_series(_sample_caps(), AI_HW, BIG_TECH, ["SPY"], base_date=D1)
        assert s.company_caps["AI HW"]["NVDA"] == [200.0, 300.0]
        assert s.company_caps["AI HW"]["MU"] == [100.0, 140.0]
        assert s.company_caps["빅테크"]["MSFT"] == [300.0, 330.0]
        assert s.company_caps["빅테크"]["META"] == [200.0, 220.0]
        # 벤치마크는 포함되지 않는다
        assert "SPY" not in s.company_caps["AI HW"]
        assert "SPY" not in s.company_caps["빅테크"]

    def test_missing_benchmark_does_not_drop_date(self):
        caps = _sample_caps()
        # D2의 RSP 제거 → 날짜는 유지, RSP 지수만 해당일 생략 없이 이전값 유지 안 함
        caps = [c for c in caps if not (c.date == D2 and c.ticker == "RSP")]
        s = build_series(caps, AI_HW, BIG_TECH, ["SPY", "RSP"], base_date=D1)
        assert s.dates == [D1, D2]
        assert len(s.indexed["SPY"]) == 2
        assert "RSP" not in s.indexed


class TestThresholdStatus:
    def test_cross_up(self):
        assert threshold_status(0.81, 0.79, 0.8) == "cross_up"

    def test_cross_down(self):
        assert threshold_status(0.79, 0.81, 0.8) == "cross_down"

    def test_above_no_cross(self):
        assert threshold_status(0.82, 0.81, 0.8) == "above"

    def test_below(self):
        assert threshold_status(0.75, 0.76, 0.8) is None

    def test_no_prev_above(self):
        assert threshold_status(0.85, None, 0.8) == "above"

    def test_no_prev_below(self):
        assert threshold_status(0.7, None, 0.8) is None


class TestSummarize:
    def test_summary_fields(self):
        caps = _sample_caps()
        series = build_series(caps, AI_HW, BIG_TECH, ["SPY", "RSP"], base_date=D1)
        summary = summarize(
            series, caps,
            ai_hw={"NVDA": "엔비디아", "MU": "마이크론"},
            big_tech={"MSFT": "MS", "META": "메타"},
            threshold=0.8,
        )
        assert summary.as_of == D2
        assert summary.ratio == pytest.approx(0.8)
        assert summary.ratio_prev == pytest.approx(0.6)
        assert summary.change_pp == pytest.approx(20.0)  # %p
        assert summary.high_30d == pytest.approx(0.8)
        assert summary.low_30d == pytest.approx(0.6)
        assert summary.status == "cross_up"

    def test_groups_sorted_by_cap_desc(self):
        caps = _sample_caps()
        series = build_series(caps, AI_HW, BIG_TECH, ["SPY"], base_date=D1)
        summary = summarize(
            series, caps,
            ai_hw={"NVDA": "엔비디아", "MU": "마이크론"},
            big_tech={"MSFT": "MS", "META": "메타"},
            threshold=0.8,
        )
        ai_group = summary.groups[0]
        assert ai_group.name == "AI HW"
        assert ai_group.total_usd == pytest.approx(440.0)
        assert [c.ticker for c in ai_group.companies] == ["NVDA", "MU"]
        # NVDA: D1 200 → D2 300 = +50%
        assert ai_group.companies[0].day_change_pct == pytest.approx(50.0)
        assert ai_group.companies[0].name == "엔비디아"
        big_group = summary.groups[1]
        assert big_group.name == "빅테크"
        assert [c.ticker for c in big_group.companies] == ["MSFT", "META"]
