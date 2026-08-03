from src.investing_high import (
    InvestingHighRow, _parse_volume, filter_tradeable,
    InvestingFetchError, InvestingParseError,
)


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
