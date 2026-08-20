from datetime import date

from src.price_history.adjust import (
    PxRow, AdjustEvent, detect_adjustments, adjusted_highs,
)


def _row(day, high, close, chg):
    return PxRow(d=date(2026, 1, day), high=high, close=close, chg=chg)


def test_no_event_when_base_matches_prev_close():
    rows = [_row(5, 110, 100, 0), _row(6, 115, 105, 5), _row(7, 108, 102, -3)]
    assert detect_adjustments(rows) == []


def test_detects_split_factor_fifty():
    # 액면분할: 정지 전 종가 2,650,000 -> 재개일 기준가 53,000
    rows = [
        _row(5, 0, 2_650_000, 0),
        _row(6, 53_900, 51_900, -1_100),   # 기준가 53,000, 계수 50
    ]
    evs = detect_adjustments(rows)
    assert len(evs) == 1
    assert evs[0].d == date(2026, 1, 6)
    assert evs[0].factor == 50.0


def test_detects_reverse_split_factor_below_one():
    # 액면병합 5:1 — 정지 전 396 -> 재개일 기준가 1,980
    rows = [_row(5, 0, 396, 0), _row(6, 2_065, 1_720, -260)]
    evs = detect_adjustments(rows)
    assert len(evs) == 1
    assert round(evs[0].factor, 4) == 0.2


def test_ignores_moves_below_threshold():
    # 기준가와 전일종가가 1% 어긋나면 이벤트가 아니다
    rows = [_row(5, 110, 1_000, 0), _row(6, 1_020, 1_010, 20)]  # 기준가 990
    assert detect_adjustments(rows) == []


def test_skips_rows_with_nonpositive_prices():
    rows = [_row(5, 0, 0, 0), _row(6, 100, 100, 0)]
    assert detect_adjustments(rows) == []


def test_adjusted_highs_scales_only_dates_before_event():
    rows = [_row(5, 400, 396, 0), _row(6, 2_065, 1_720, -260), _row(7, 1_870, 1_691, -29)]
    evs = detect_adjustments(rows)
    out = dict(adjusted_highs(rows, evs))
    assert out[date(2026, 1, 5)] == 2_000.0   # 400 x 5
    assert out[date(2026, 1, 6)] == 2_065.0   # 이벤트 당일은 그대로
    assert out[date(2026, 1, 7)] == 1_870.0


def test_adjusted_highs_accumulates_multiple_events():
    # 1/6에 계수 2, 1/8에 계수 5. 1/5 가격은 두 계수가 모두 걸려 1/10이 된다.
    rows = [
        _row(5, 110, 100, 0),
        _row(6, 60, 55, 5),      # 기준가 50, 전일종가 100 -> 계수 2
        _row(7, 65, 60, 5),      # 기준가 55 = 전일종가, 이벤트 아님
        _row(8, 18, 15, 3),      # 기준가 12, 전일종가 60 -> 계수 5
    ]
    evs = detect_adjustments(rows)
    assert [e.factor for e in evs] == [2.0, 5.0]

    out = dict(adjusted_highs(rows, evs))
    assert out[date(2026, 1, 8)] == 18.0    # 마지막 봉은 무보정
    assert out[date(2026, 1, 7)] == 13.0    # 65 x 1/5
    assert out[date(2026, 1, 6)] == 12.0    # 60 x 1/5 (이벤트 당일은 그 계수를 안 먹는다)
    assert out[date(2026, 1, 5)] == 11.0    # 110 x 1/5 x 1/2


def test_adjusted_highs_without_events_is_identity():
    rows = [_row(5, 110, 100, 0), _row(6, 115, 105, 5)]
    out = dict(adjusted_highs(rows, []))
    assert out == {date(2026, 1, 5): 110.0, date(2026, 1, 6): 115.0}
