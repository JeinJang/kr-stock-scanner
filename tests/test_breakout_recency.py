from datetime import date, timedelta

from src.breakout_recency import Bar, compute_recency

END = date(2026, 8, 19)


def _daily(highs: list[float], end: date = END) -> list[Bar]:
    """마지막 원소가 end인 연속 일봉(달력 하루 간격)."""
    n = len(highs)
    return [Bar(date=end - timedelta(days=n - 1 - i), high=h) for i, h in enumerate(highs)]


def test_returns_none_when_fewer_than_two_bars():
    assert compute_recency([]) is None
    assert compute_recency([Bar(date=END, high=100.0)]) is None


def test_all_time_high_has_no_price_above_day():
    r = compute_recency(_daily([100.0 + i for i in range(400)]))
    assert r is not None
    assert r.days_since_price_above is None
    assert r.today_high == 499.0


def test_daily_streak_gives_one_day():
    r = compute_recency(_daily([100.0 + i for i in range(400)]))
    assert r.days_since_prev_new_high == 1


def test_prev_high_uses_calendar_window_not_bar_count():
    # 창(365일) 밖의 500은 제외되고, 창 안의 100만 본다
    highs = [500.0] + [100.0] * 400 + [150.0]
    r = compute_recency(_daily(highs), window_days=365)
    assert r.prev_high_52w == 100.0


def test_prev_high_is_zero_when_window_not_covered():
    r = compute_recency(_daily([100.0] * 100 + [150.0]), window_days=365)
    assert r.prev_high_52w == 0.0
    assert r.days_since_prev_new_high is None   # 워밍업 부족


def test_staircase_recovery_splits_the_two_metrics():
    # index 0: 옛 고점 300 / 1~400: 박스권 100 / 401: 150(그날 신고가) /
    # 402~500: 120 / 501: 오늘 250
    highs = [300.0] + [100.0] * 400 + [150.0] + [120.0] * 99 + [250.0]
    r = compute_recency(_daily(highs), window_days=365)
    assert r.days_since_price_above == 501          # index 0 까지
    assert r.days_since_prev_new_high == 100        # index 401 까지
    assert r.prev_high_52w == 150.0


def test_history_span_days_spans_first_to_last():
    r = compute_recency(_daily([100.0] * 11))
    assert r.history_span_days == 10


def test_gap_in_dates_does_not_break_window():
    # 휴장으로 날짜가 듬성해도 달력 기준으로 창을 잡는다
    bars = [Bar(date=END - timedelta(days=d), high=h) for d, h in
            [(400, 90.0), (370, 95.0), (200, 80.0), (100, 85.0), (0, 99.0)]]
    bars.sort(key=lambda b: b.date)
    r = compute_recency(bars, window_days=365)
    assert r is not None
    assert r.prev_high_52w == 85.0      # 365일 안: 80, 85 (95는 370일 전이라 제외)
    assert r.days_since_price_above is None
