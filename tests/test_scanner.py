# tests/test_scanner.py
from datetime import date
from unittest.mock import MagicMock


def test_find_new_highs():
    """Scanner should identify stocks at 52-week highs."""
    from src.scanner import Scanner

    daily_data = {
        "005930": {
            "market": "KOSPI", "open": 70000, "high": 79000,
            "low": 69000, "close": 78500, "volume": 15000000, "change_pct": 2.0,
        },
        "000660": {
            "market": "KOSPI", "open": 200000, "high": 210000,
            "low": 199000, "close": 205000, "volume": 5000000, "change_pct": 1.5,
        },
        "035420": {
            "market": "KOSDAQ", "open": 210000, "high": 215000,
            "low": 208000, "close": 212000, "volume": 3000000, "change_pct": -0.5,
        },
    }

    sector_map = {
        "005930": "전기전자",
        "000660": "전기전자",
        "035420": "서비스업",
    }

    name_map = {
        "005930": "삼성전자",
        "000660": "SK하이닉스",
        "035420": "NAVER",
    }

    mock_collector = MagicMock()
    # 005930: high=79000, 52w high=79000 -> matches today, NEW HIGH (prev was 77000)
    # 000660: high=210000, 52w high=215000 -> doesn't match, NOT new high
    # 035420: high=215000, 52w high=215000 -> matches today, NEW HIGH (prev was 214000)
    mock_collector.get_52w_high.side_effect = lambda t, d, **kw: {
        "005930": 79000,
        "000660": 215000,
        "035420": 215000,
    }[t]

    scanner = Scanner(collector=mock_collector)
    highs = scanner.find_new_highs(
        daily_data=daily_data,
        date_str="20260219",
        sector_map=sector_map,
        name_map=name_map,
        prev_highs={"005930": 77000, "000660": 215000, "035420": 214000},
    )

    tickers = [h.ticker for h in highs]
    assert "005930" in tickers
    assert "035420" in tickers
    assert "000660" not in tickers


def test_build_scan_result():
    """Scanner should produce a complete ScanResult with stats."""
    from src.scanner import Scanner
    from src.models import StockHigh

    mock_collector = MagicMock()
    scanner = Scanner(collector=mock_collector)

    highs = [
        StockHigh(
            ticker="005930", name="삼성전자", market="KOSPI", sector="전기전자",
            close_price=78500, high_52w=79000, prev_high_52w=77000,
            breakout_pct=2.60, volume=15000000, avg_volume_20d=12000000,
        ),
        StockHigh(
            ticker="035420", name="NAVER", market="KOSDAQ", sector="서비스업",
            close_price=212000, high_52w=215000, prev_high_52w=214000,
            breakout_pct=0.47, volume=3000000, avg_volume_20d=2500000,
        ),
    ]

    result = scanner.build_scan_result(
        scan_date=date(2026, 2, 19),
        highs=highs,
        total_stocks=2500,
    )

    assert result.stats.new_high_count == 2
    assert result.stats.kospi_count == 1
    assert result.stats.kosdaq_count == 1
    assert "전기전자" in result.sector_breakdown
    assert len(result.sector_breakdown["전기전자"]) == 1
