from pathlib import Path
import pytest
from src.investing_high import (
    InvestingHighRow, _parse_volume, filter_tradeable,
    InvestingFetchError, InvestingParseError, parse_high_rows, _fetch_html,
    resolve_to_krx, build_highs,
)
from src.models import StockHigh

_FIXTURE = Path(__file__).parent / "fixtures" / "investing_52w_high.html"


def test_parse_volume_suffixes():
    assert _parse_volume("2.07M") == 2_070_000
    assert _parse_volume("617.58K") == 617_580
    assert _parse_volume("1,234") == 1234
    assert _parse_volume("131.15K") == 131_150
    for empty in ("", "-", "N/A", "  "):
        assert _parse_volume(empty) == 0


def test_filter_tradeable_drops_zero_volume():
    rows = [
        InvestingHighRow(name="A", last_price=100.0, change_pct=1.0, volume=5000),
        InvestingHighRow(name="B", last_price=200.0, change_pct=2.0, volume=0),
    ]
    out = filter_tradeable(rows)
    assert [r.name for r in out] == ["A"]


def test_parse_high_rows_extracts_all_rows_and_total():
    html = _FIXTURE.read_text(encoding="utf-8")
    rows, total = parse_high_rows(html)
    assert total == 3
    assert [r.name for r in rows] == ["아이크래프트", "벡트", "거래정지주"]
    assert rows[0].last_price == 5190.0
    assert rows[0].change_pct == 12.34
    assert rows[0].volume == 2_070_000
    assert rows[2].volume == 0  # 거래량 '-'


def test_parse_high_rows_raises_on_challenge():
    challenge = "<html><head><title>Just a moment...</title></head><body></body></html>"
    with pytest.raises(InvestingParseError):
        parse_high_rows(challenge)


class _Resp:
    def __init__(self, status, text):
        self.status_code = status
        self.text = text


def test_fetch_html_falls_back_to_next_target():
    calls = []
    def fake_get(url, impersonate, timeout, headers):
        calls.append(impersonate)
        if impersonate == "chrome124":
            return _Resp(403, "403")               # 첫 타깃 차단
        return _Resp(200, "<table><tr><td>x</td></tr></table>")  # 둘째 성공
    html = _fetch_html("http://x", ("chrome124", "safari17_0"), _get=fake_get)
    assert "<table>" in html
    assert calls == ["chrome124", "safari17_0"]


def test_fetch_html_raises_when_all_targets_blocked():
    def fake_get(url, impersonate, timeout, headers):
        return _Resp(403, "403")
    with pytest.raises(InvestingFetchError):
        _fetch_html("http://x", ("chrome124", "safari17_0"), _get=fake_get)


def test_resolve_to_krx_maps_and_reports_unmatched():
    rows = [
        InvestingHighRow(name="아이크래프트", last_price=5190.0, change_pct=12.34, volume=2_070_000),
        InvestingHighRow(name="없는회사", last_price=1000.0, change_pct=1.0, volume=100),
    ]
    n2t = {"아이크래프트": "052460"}
    n2m = {"아이크래프트": "KOSDAQ"}
    matched, unmatched = resolve_to_krx(rows, n2t, n2m)
    assert [(m[1], m[2]) for m in matched] == [("052460", "KOSDAQ")]
    assert unmatched == ["없는회사"]


def test_build_highs_assembles_stockhigh():
    row = InvestingHighRow(name="아이크래프트", last_price=5190.0, change_pct=12.34, volume=2_070_000)
    matched = [(row, "052460", "KOSDAQ")]
    highs = build_highs(matched, market_caps={"052460": 123}, sector_map={"052460": "IT"})
    assert len(highs) == 1
    h = highs[0]
    assert isinstance(h, StockHigh)
    assert (h.ticker, h.market, h.sector) == ("052460", "KOSDAQ", "IT")
    assert h.close_price == 5190.0 and h.volume == 2_070_000
    assert h.breakout_pct == 12.34


def test_collect_investing_highs_end_to_end(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    import src.investing_high as inv

    html = (Path(__file__).parent / "fixtures" / "investing_52w_high.html").read_text(encoding="utf-8")

    def fake_get(url, impersonate, timeout, headers):
        return SimpleNamespace(status_code=200, text=html)

    corps = [
        SimpleNamespace(name="아이크래프트", ticker="052460", market="KOSDAQ"),
        SimpleNamespace(name="벡트", ticker="365900", market="KOSDAQ"),
        # '거래정지주'는 매핑 없음 + 거래량 0 → 이중으로 제외
    ]

    class FakeCollector:
        def get_market_caps(self, date_str, market="ALL"):
            return {"052460": 111, "365900": 222}
        def get_sector_map(self, date_str, market):
            return {"052460": "IT", "365900": "전기전자"}

    highs, caps = inv.collect_investing_highs("20260803", FakeCollector(), corps, _get=fake_get)
    names = sorted(h.name for h in highs)
    assert names == ["벡트", "아이크래프트"]          # 거래정지주(거래량0) 제외
    assert all(h.close_price > 0 for h in highs)
    assert caps["052460"] == 111
