from datetime import date

import pandas as pd
import pytest

from src.recency_source import fetch_bars


class FakeClient:
    """get_market_ohlcv_by_date 호출을 기록하는 가짜 KRX 클라이언트."""

    def __init__(self, frames, supports_history=True):
        self._frames = list(frames)   # 호출 순서대로 반환할 DataFrame들
        self.supports_history = supports_history
        self.calls = []

    def get_market_ohlcv_by_date(self, fromdate, todate, ticker, adjusted=False):
        self.calls.append((fromdate, todate, ticker, adjusted))
        return self._frames.pop(0) if self._frames else pd.DataFrame()


def _frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    """[(YYYYMMDD, 고가)] → KRX 응답 모양(날짜 인덱스, '고가' 컬럼)."""
    idx = [d for d, _ in rows]
    return pd.DataFrame({"고가": [h for _, h in rows]}, index=idx)


def test_returns_none_when_client_has_no_history_support():
    client = FakeClient([], supports_history=False)
    assert fetch_bars(client, "005930", date(2026, 8, 19)) is None
    assert client.calls == []


def test_single_call_when_full_range_returned():
    frame = _frame([("20150820", 100.0), ("20260819", 200.0)])
    client = FakeClient([frame])

    bars = fetch_bars(client, "005930", date(2026, 8, 19), years=11)

    assert len(client.calls) == 1
    assert client.calls[0][3] is True          # adjusted=True 필수
    assert [b.date for b in bars] == [date(2015, 8, 20), date(2026, 8, 19)]


def test_sorts_ascending_regardless_of_response_order():
    # KRX는 최신순으로 주는 경우가 있다
    frame = _frame([("20260819", 200.0), ("20150820", 100.0)])
    client = FakeClient([frame])

    bars = fetch_bars(client, "005930", date(2026, 8, 19), years=11)

    assert [b.high for b in bars] == [100.0, 200.0]


def test_chunks_backwards_when_response_is_truncated():
    # 1차 응답이 최근 1년만 → 그 앞 구간을 다시 요청해 병합
    first = _frame([("20250820", 150.0), ("20260819", 200.0)])
    second = _frame([("20150820", 100.0), ("20250819", 140.0)])
    client = FakeClient([first, second])

    bars = fetch_bars(client, "005930", date(2026, 8, 19), years=11)

    assert len(client.calls) == 2
    assert client.calls[1][1] == "20250819"    # 1차 첫 거래일 직전까지
    assert [b.date for b in bars] == [
        date(2015, 8, 20), date(2025, 8, 19), date(2025, 8, 20), date(2026, 8, 19),
    ]


def test_stops_when_earlier_chunk_is_empty_newly_listed():
    # 상장 3년차 종목: 앞 구간을 물어봐도 빈 응답 → 그대로 종료
    first = _frame([("20230820", 100.0), ("20260819", 200.0)])
    client = FakeClient([first])   # 이후 호출은 빈 DataFrame

    bars = fetch_bars(client, "005930", date(2026, 8, 19), years=11)

    assert len(bars) == 2
    assert len(client.calls) == 2


def test_returns_none_on_empty_first_response():
    client = FakeClient([pd.DataFrame()])
    assert fetch_bars(client, "005930", date(2026, 8, 19)) is None


def test_returns_none_when_client_raises():
    class Boom:
        supports_history = True

        def get_market_ohlcv_by_date(self, *a, **k):
            raise ValueError("boom")

    assert fetch_bars(Boom(), "005930", date(2026, 8, 19)) is None


def test_propagates_krx_blocked_error():
    from src.krx_login_client import KrxBlockedError

    class Blocked:
        supports_history = True

        def get_market_ohlcv_by_date(self, *a, **k):
            raise KrxBlockedError("차단")

    with pytest.raises(KrxBlockedError):
        fetch_bars(Blocked(), "005930", date(2026, 8, 19))


def test_returns_none_when_max_calls_exhausted_without_reaching_start():
    # 매 응답이 계속 짧게 잘려 나와 start까지 도달하지 못하면(호출 상한 소진)
    # 이력이 실제로 완결됐는지 알 수 없으므로 부분 리스트 대신 None.
    frames = [
        _frame([("20260101", 100.0)]),
        _frame([("20250101", 100.0)]),
        _frame([("20240101", 100.0)]),
        _frame([("20230101", 100.0)]),
    ]
    client = FakeClient(frames)

    bars = fetch_bars(client, "005930", date(2026, 8, 19))

    assert bars is None
    assert len(client.calls) == 4    # max_calls 기본값만큼만 호출


def test_returns_none_when_later_chunk_fails_after_partial_success():
    # 1차 응답은 왔지만 start까지 못 미쳤는데 2차 호출이 실패하면,
    # 1차만으로는 이력 완결을 보장할 수 없으므로 부분 리스트를 넘기지 않는다.
    first = _frame([("20250820", 150.0), ("20260819", 200.0)])

    class PartialThenBoom:
        supports_history = True

        def __init__(self):
            self.calls = 0

        def get_market_ohlcv_by_date(self, *a, **k):
            self.calls += 1
            if self.calls == 1:
                return first
            raise ValueError("boom")

    client = PartialThenBoom()
    assert fetch_bars(client, "005930", date(2026, 8, 19)) is None
