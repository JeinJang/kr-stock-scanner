# tests/test_integration.py
"""Integration test: verifies the full pipeline works end-to-end with mocks."""
import pytest
from datetime import date
from unittest.mock import MagicMock


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
    monkeypatch.setenv("NAVER_CLIENT_ID", "test-naver-id")
    monkeypatch.setenv("NAVER_CLIENT_SECRET", "test-naver-secret")


def test_full_pipeline_with_mocks(mock_env, tmp_path):
    """Full pipeline should collect, scan, analyze, and produce a report."""
    from src.collector import Collector
    from src.scanner import Scanner
    from src.reporter import Reporter
    from src.db import Database

    # Setup
    db = Database(f"sqlite:///{tmp_path}/test.db")

    # Mock collector
    daily_data = {
        "005930": {
            "market": "KOSPI", "open": 70000, "high": 79000,
            "low": 69000, "close": 78500, "volume": 15000000, "change_pct": 2.0,
        },
    }
    sector_map = {"005930": "전기전자"}
    name_map = {"005930": "삼성전자"}
    market_caps = {"005930": 450000000000000}

    mock_collector = MagicMock(spec=Collector)
    mock_collector.collect_daily.return_value = daily_data
    mock_collector.get_sector_map.return_value = sector_map
    mock_collector.get_market_caps.return_value = market_caps
    mock_collector.get_52w_high.return_value = 79000

    # Scan
    scanner = Scanner(collector=mock_collector)
    highs = scanner.find_new_highs(
        daily_data=daily_data, date_str="20260219",
        sector_map=sector_map, name_map=name_map,
        prev_highs={"005930": 77000},
    )
    result = scanner.build_scan_result(date(2026, 2, 19), highs, 1)

    assert result.stats.new_high_count == 1
    assert result.highs[0].ticker == "005930"

    # Save to DB
    db.save_scan_result(result)
    loaded = db.get_scan_result(date(2026, 2, 19))
    assert len(loaded) == 1

    # Format report (without actually sending)
    reporter = Reporter(bot_token="test", chat_id=123)
    text = reporter.format_report(result, [], [])
    assert "삼성전자" in text
    assert "52주 신고가 리포트" in text
