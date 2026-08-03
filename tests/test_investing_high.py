from pathlib import Path
import pytest
from src.investing_high import (
    InvestingHighRow, filter_tradeable,
    InvestingFetchError, InvestingParseError, parse_high_rows, _fetch_html,
    resolve_to_krx, build_highs,
)
from src.models import StockHigh

_FIXTURE = Path(__file__).parent / "fixtures" / "investing_52w_high.html"


def test_filter_tradeable_drops_zero_volume():
    rows = [
        InvestingHighRow(name="A", ticker="000001", last_price=100.0, change_pct=1.0, volume=5000),
        InvestingHighRow(name="B", ticker="000002", last_price=200.0, change_pct=2.0, volume=0),
    ]
    out = filter_tradeable(rows)
    assert [r.name for r in out] == ["A"]


def test_parse_high_rows_extracts_next_data_collection_and_total():
    html = _FIXTURE.read_text(encoding="utf-8")
    rows, total = parse_high_rows(html)
    # total은 _collection 길이가 아니라 JSON의 "total" 실제 신호(정규식)에서 옴.
    # 이 픽스처는 total=4, _collection도 4건이라 커버리지 경고는 뜨지 않아야 정상.
    assert total == 4
    assert [r.name for r in rows] == ["아이크래프트", "벡트", "거래정지주", "Foreign Co"]
    assert rows[0].ticker == "052460"
    assert rows[0].last_price == 5190.0
    assert rows[0].change_pct == 12.34
    assert rows[0].volume == 2_070_000
    assert rows[2].volume == 0  # 거래정지: volumeOneDay=0


def test_parse_high_rows_total_reflects_json_signal_not_collection_length():
    """total은 "total":N 정규식 신호를 그대로 반영해야 한다 — _collection 개수로
    치환되면 신고가가 배치보다 많은 날의 커버리지 경고(total > len(rows))가
    영원히 발동하지 않게 되어 조용한 누락을 놓친다."""
    html = (
        '<script id="__NEXT_DATA__" type="application/json">'
        '{"props":{"pageProps":{"state":{"assetsCollectionStore":{"assetsCollection":'
        '{"_collection":[{"name":"아이크래프트","symbol":"052460","last":5190.0,'
        '"changeOneDayPercent":12.34,"volumeOneDay":2070000}]},'
        '"pagination":{"total":9}}}}}}'
        '</script>'
    )
    rows, total = parse_high_rows(html)
    assert total == 9  # _collection은 1건뿐이지만 total은 JSON 신호 그대로 9
    assert len(rows) == 1


def test_fetch_52w_high_rows_warns_when_total_exceeds_fetched_rows():
    """total(9) > len(rows)(1)일 때 커버리지 경고 로그가 실제로 발동하는지 확인."""
    html = (
        '<table></table>'
        '<script id="__NEXT_DATA__" type="application/json">'
        '{"props":{"pageProps":{"state":{"assetsCollectionStore":{"assetsCollection":'
        '{"_collection":[{"name":"아이크래프트","symbol":"052460","last":5190.0,'
        '"changeOneDayPercent":12.34,"volumeOneDay":2070000}]},'
        '"pagination":{"total":9}}}}}}'
        '</script>'
    )

    def fake_get(url, impersonate, timeout, headers):
        return _Resp(200, html)

    from src import investing_high as inv
    rows, total = inv.fetch_52w_high_rows(_get=fake_get)
    assert total == 9
    assert len(rows) == 1  # 실제로 total > len(rows) 조건이 성립함을 재확인


def test_parse_high_rows_raises_when_next_data_script_missing():
    challenge = "<html><head><title>Just a moment...</title></head><body></body></html>"
    with pytest.raises(InvestingParseError):
        parse_high_rows(challenge)


def test_parse_high_rows_raises_on_invalid_json():
    html = '<script id="__NEXT_DATA__" type="application/json">{not valid json</script>'
    with pytest.raises(InvestingParseError):
        parse_high_rows(html)


def test_parse_high_rows_raises_when_collection_path_absent():
    html = '<script id="__NEXT_DATA__" type="application/json">{"props":{"pageProps":{}}}</script>'
    with pytest.raises(InvestingParseError):
        parse_high_rows(html)


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


def test_resolve_to_krx_maps_by_ticker_and_reports_unmatched():
    rows = [
        InvestingHighRow(name="아이크래프트", ticker="052460", last_price=5190.0, change_pct=12.34, volume=2_070_000),
        InvestingHighRow(name="Foreign Co", ticker="US1234567890", last_price=42.5, change_pct=3.21, volume=150_000),
    ]
    ticker_to_market = {"052460": "KOSDAQ"}
    matched, unmatched = resolve_to_krx(rows, ticker_to_market)
    assert [(m[1], m[2]) for m in matched] == [("052460", "KOSDAQ")]
    assert unmatched == ["Foreign Co"]


def test_build_highs_assembles_stockhigh():
    row = InvestingHighRow(name="아이크래프트", ticker="052460", last_price=5190.0, change_pct=12.34, volume=2_070_000)
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
        # '거래정지주'(volumeOneDay=0)와 'Foreign Co'(유니버스 밖 ticker)는 매핑되어도/안 되어도 제외 대상
    ]

    class FakeCollector:
        def get_market_caps(self, date_str, market="ALL"):
            return {"052460": 111, "365900": 222}
        def get_sector_map(self, date_str, market):
            return {"052460": "IT", "365900": "전기전자"}

    highs, caps = inv.collect_investing_highs("20260803", FakeCollector(), corps, _get=fake_get)
    names = sorted(h.name for h in highs)
    assert names == ["벡트", "아이크래프트"]          # 거래정지주(거래량0), Foreign Co(미매칭) 제외
    assert all(h.close_price > 0 for h in highs)
    assert caps["052460"] == 111
