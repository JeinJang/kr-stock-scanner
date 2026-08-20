from datetime import date, timedelta

from src.breakout_recency import Bar, compute_recency


def _series(highs: list[float], end: date = date(2026, 8, 19)) -> list[Bar]:
    """마지막 원소가 end인 연속 일봉(날짜 오름차순)."""
    n = len(highs)
    return [Bar(date=end - timedelta(days=n - 1 - i), high=h) for i, h in enumerate(highs)]


def test_returns_none_when_fewer_than_two_bars():
    assert compute_recency([]) is None
    assert compute_recency([Bar(date=date(2026, 8, 19), high=100.0)]) is None


def test_all_time_high_has_no_price_above_day():
    # 단조 상승 → 오늘 고가를 웃돈 과거일이 없음
    bars = _series([100.0 + i for i in range(300)])
    r = compute_recency(bars, window=250)
    assert r is not None
    assert r.days_since_price_above is None
    assert r.today_high == 399.0


def test_daily_streak_gives_one_day_since_prev_new_high():
    # 매일 경신 → 직전 거래일도 그날의 52주 신고가였음
    bars = _series([100.0 + i for i in range(300)])
    r = compute_recency(bars, window=250)
    assert r is not None
    assert r.days_since_prev_new_high == 1


def test_staircase_recovery_splits_the_two_metrics():
    # index   0     : 옛 고점 300 (401일 전)
    # index 1~300   : 100 박스권
    # index 301     : 150 — 직전 250봉(51~300)이 모두 100이므로 그날이 52주 신고가
    # index 302~400 : 120 (150 아래)
    # index 401     : 오늘 250 — 옛 고점 300은 못 넘었지만 52주 신고가
    highs = [300.0] + [100.0] * 300 + [150.0] + [120.0] * 99 + [250.0]
    bars = _series(highs)
    r = compute_recency(bars, window=250)
    assert r is not None
    assert r.days_since_prev_new_high == 100   # A: index 301까지
    assert r.days_since_price_above == 401     # B: index 0까지
    assert r.prev_high_52w == 150.0


def test_price_above_day_is_the_most_recent_one():
    # 과거에 오늘 고가보다 높은 날이 있으면 그중 가장 최근 날짜까지의 일수
    highs = [300.0] + [100.0] * 400 + [250.0]
    bars = _series(highs)
    r = compute_recency(bars, window=250)
    assert r is not None
    assert r.days_since_price_above == 401  # index 0 → 오늘까지 401일


def test_prev_high_52w_is_zero_when_fewer_than_window_bars():
    bars = _series([100.0] * 100 + [150.0])
    r = compute_recency(bars, window=250)
    assert r is not None
    assert r.prev_high_52w == 0.0
    assert r.days_since_prev_new_high is None  # 워밍업 부족


def test_prev_high_52w_uses_the_last_window_bars():
    highs = [500.0] + [100.0] * 250 + [150.0]
    bars = _series(highs)
    r = compute_recency(bars, window=250)
    assert r is not None
    assert r.prev_high_52w == 100.0  # 직전 250봉만 봄 (500은 창 밖)


def test_history_span_days_spans_first_to_last_bar():
    bars = _series([100.0] * 11)
    r = compute_recency(bars, window=250)
    assert r is not None
    assert r.history_span_days == 10
