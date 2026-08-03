from pathlib import Path
import pytest
from src.investing_high import (
    InvestingHighRow, _parse_volume, filter_tradeable,
    InvestingFetchError, InvestingParseError, parse_high_rows,
)

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
