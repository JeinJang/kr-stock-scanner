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
    # 기준가와 전일종가가 0.3%만 어긋나면 이벤트가 아니다
    rows = [_row(5, 110, 1_000, 0), _row(6, 1_005, 1_000, 3)]  # 기준가 997, 편차 0.301%
    assert detect_adjustments(rows) == []


def test_detects_stock_dividend_just_above_threshold():
    """주식배당 수준(약 1~2%)도 검출한다 — 실측상 이 구간에 실제 이벤트가 산다."""
    rows = [_row(5, 2_100, 2_060, 0), _row(6, 2_070, 2_040, 20)]  # 기준가 2,020, 전일종가 2,060 -> 계수 1.0198
    evs = detect_adjustments(rows)
    assert len(evs) == 1
    assert round(evs[0].factor, 4) == 1.0198


def test_ignores_rounding_noise_below_half_percent():
    """0.5% 미만은 호가 반올림 잡음이므로 이벤트가 아니다."""
    rows = [_row(5, 15_600, 15_525, 0), _row(6, 15_560, 15_535, 5)]  # 기준가 15,530, 편차 0.032%
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


def test_ignores_halt_resume_auction_base_near_one():
    """장기 정지 해제일의 경매 기준가는 이벤트가 아니다 — 실측 009410 20241031."""
    # 실측은 153일 정지. 정지 1행만으로는(HALT_MIN_ROWS 미달) 정지로 보지
    # 않으므로, 최소 조건을 만족하도록 정지일을 2행으로 재현한다.
    # 정지 직전 종가 4,700 -> 재개일 기준가 4,781 (계수 0.983)
    rows = [
        _row(4, 0, 4_700, 0),
        _row(5, 0, 4_700, 0),
        _row(6, 4_800, 4_620, -161),
    ]
    assert detect_adjustments(rows) == []


def test_detects_split_even_when_preceded_by_halt():
    """정지 다음 날이라도 계수가 1에서 멀면(액면분할) 그대로 검출한다."""
    rows = [_row(5, 0, 2_650_000, 0), _row(6, 53_900, 51_900, -1_100)]
    assert [e.factor for e in detect_adjustments(rows)] == [50.0]


def test_near_one_factor_without_preceding_halt_is_still_an_event():
    """정지가 없었다면 1 근처 계수도 실제 이벤트(주식배당 등)로 남긴다."""
    rows = [_row(5, 2_100, 2_060, 0), _row(6, 2_070, 2_040, 20)]  # 계수 1.0198
    assert len(detect_adjustments(rows)) == 1


def test_halt_resume_boundary_is_ten_percent():
    """계수가 정확히 ±10% 밖이면 정지 다음 날이라도 이벤트로 남긴다."""
    # 기준가 1,000, 전일종가 1,100 -> 계수 1.1 (HALT_RESUME_MAX 밖)
    rows = [_row(5, 0, 1_100, 0), _row(6, 1_050, 1_020, 20)]
    assert [round(e.factor, 4) for e in detect_adjustments(rows)] == [1.1]


def test_single_no_trade_day_is_not_a_halt_event_still_detected():
    """무거래 하루(고가 0)뿐이면 정지로 보지 않는다 — 실측 001067 20230313."""
    # 기준가 2,020, 전일종가 2,060 -> 계수 1.0198 (HALT_RESUME_MAX 이내지만
    # 정지 1행뿐이라 HALT_MIN_ROWS 미달 -> 그대로 이벤트로 남는다).
    rows = [_row(5, 0, 2_060, 0), _row(6, 2_070, 2_040, 20)]
    evs = detect_adjustments(rows)
    assert len(evs) == 1
    assert round(evs[0].factor, 4) == 1.0198


def test_two_no_trade_days_are_a_halt_event_not_detected():
    """같은 계수라도 정지 2행 이상 이어지면 경매 잡음으로 버린다."""
    # 정지 1일차, 정지 2일차(둘 다 고가 0, 종가 유지) 이후 재개.
    # 기준가 2,020, 전일종가 2,060 -> 계수 1.0198이지만 HALT_MIN_ROWS 충족.
    rows = [
        _row(4, 0, 2_060, 0),
        _row(5, 0, 2_060, 0),
        _row(6, 2_070, 2_040, 20),
    ]
    assert detect_adjustments(rows) == []
