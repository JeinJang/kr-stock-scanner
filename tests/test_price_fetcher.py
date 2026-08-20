import pytest

from src.price_history.fetcher import (
    KrxApiError, fetch_day, fetch_many, parse_rows,
)


class FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload


def _payload(*items):
    return {"OutBlock_1": [
        {"ISU_CD": tk, "TDD_HGPRC": h, "TDD_CLSPRC": c, "CMPPREVDD_PRC": ch}
        for tk, h, c, ch in items
    ]}


def test_parse_rows_strips_commas_and_casts():
    rows = parse_rows(_payload(("005930", "53,900", "51,900", "-1,100")))
    assert rows == [("005930", 53900, 51900, -1100)]


def test_parse_rows_skips_rows_without_ticker():
    rows = parse_rows({"OutBlock_1": [{"TDD_HGPRC": "1", "TDD_CLSPRC": "1"}]})
    assert rows == []


def test_parse_rows_on_empty_payload():
    assert parse_rows({}) == []


def test_fetch_day_passes_auth_key_and_date():
    seen = {}

    def fake_get(url, params, headers, timeout):
        seen.update(url=url, params=params, headers=headers)
        return FakeResp(200, _payload(("005930", "1", "1", "0")))

    rows = fetch_day("KEY123", "KOSPI", "20260819", _get=fake_get)
    assert rows == [("005930", 1, 1, 0)]
    assert seen["url"].endswith("/stk_bydd_trd")
    assert seen["params"] == {"basDd": "20260819"}
    assert seen["headers"]["AUTH_KEY"] == "KEY123"


def test_fetch_day_raises_on_401():
    def fake_get(url, params, headers, timeout):
        return FakeResp(401)

    with pytest.raises(KrxApiError):
        fetch_day("BAD", "KOSPI", "20260819", _get=fake_get)


def test_fetch_day_returns_empty_on_holiday():
    def fake_get(url, params, headers, timeout):
        return FakeResp(200, {"OutBlock_1": []})

    assert fetch_day("KEY", "KOSPI", "20260101", _get=fake_get) == []


def test_fetch_many_covers_every_job():
    def fake_get(url, params, headers, timeout):
        return FakeResp(200, _payload(("005930", "1", "1", "0")))

    jobs = [("KOSPI", "20260818"), ("KOSDAQ", "20260818"), ("KOSPI", "20260819")]
    out = list(fetch_many("KEY", jobs, workers=2, _get=fake_get))
    assert len(out) == 3
    assert {(m, d) for m, d, _ in out} == set(jobs)


def test_fetch_many_propagates_auth_error():
    def fake_get(url, params, headers, timeout):
        return FakeResp(401)

    with pytest.raises(KrxApiError):
        list(fetch_many("BAD", [("KOSPI", "20260819")], workers=2, _get=fake_get))
