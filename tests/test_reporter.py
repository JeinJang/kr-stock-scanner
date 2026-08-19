# tests/test_reporter.py
import pytest
from datetime import date
from unittest.mock import AsyncMock, patch


def test_format_report():
    """Reporter should format a ScanResult into a readable Telegram message."""
    from src.reporter import Reporter
    from src.models import ScanResult, StockHigh, MarketStats, AIAnalysisResult

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

    result = ScanResult(
        scan_date=date(2026, 2, 19),
        stats=MarketStats(
            total_stocks=2500, new_high_count=2,
            kospi_count=1, kosdaq_count=1, etf_count=0,
        ),
        highs=highs,
        sector_breakdown={
            "전기전자": [highs[0]],
            "서비스업": [highs[1]],
        },
    )

    ai_analyses = [
        AIAnalysisResult(
            ticker="005930",
            news_summary="HBM4 수주 확대",
            ai_analysis="HBM4 수주 확대에 따른 실적 개선 기대감.",
        ),
    ]
    trend = [{"date": date(2026, 2, i), "count": c} for i, c in
             [(17, 18), (18, 24), (19, 32)]]

    reporter = Reporter(bot_token="test", chat_id=123)
    text = reporter.format_report(result, ai_analyses, trend)

    assert "52주 신고가 리포트" in text
    assert "2026-02-19" in text
    assert "삼성전자" in text
    assert "NAVER" in text
    assert "전기전자" in text
    assert "HBM4" in text


def test_split_message():
    """Should split long messages respecting 4096 char limit."""
    from src.reporter import split_message

    short = "Hello"
    assert split_message(short) == ["Hello"]

    long_text = "line\n" * 2000
    chunks = split_message(long_text, max_length=4096)
    assert all(len(c) <= 4096 for c in chunks)
    assert len(chunks) > 1


@pytest.mark.asyncio
async def test_send_report():
    """Reporter should send message via Telegram Bot."""
    from src.reporter import Reporter

    with patch("src.reporter.Bot") as mock_bot_cls:
        mock_bot = AsyncMock()
        mock_bot_cls.return_value = mock_bot

        reporter = Reporter(bot_token="test-token", chat_id=123456)
        await reporter.send("Test message")

        mock_bot.send_message.assert_called_once()
        call_kwargs = mock_bot.send_message.call_args[1]
        assert call_kwargs["chat_id"] == 123456
        assert call_kwargs["text"] == "Test message"


def test_report_shows_and_sorts_by_change_pct():
    """리포트의 +x.x%는 당일 등락률(change_pct)이고, 정렬도 그 기준이다."""
    from datetime import date
    from src.reporter import Reporter
    from src.models import ScanResult, StockHigh, MarketStats

    def _stock(ticker, name, change_pct):
        return StockHigh(
            ticker=ticker, name=name, market="KOSPI", sector="전기전자",
            close_price=1000, high_52w=1000, prev_high_52w=0.0,
            breakout_pct=0.0, volume=1, avg_volume_20d=0, change_pct=change_pct,
        )

    lows, highs_ = _stock("000001", "낮은종목", 1.0), _stock("000002", "높은종목", 9.0)
    result = ScanResult(
        scan_date=date(2026, 8, 19),
        stats=MarketStats(total_stocks=2, new_high_count=2, kospi_count=2),
        highs=[lows, highs_], sector_breakdown={"전기전자": [lows, highs_]},
    )

    text = Reporter(bot_token="", chat_id=0).format_report(result, [], [])

    assert "+9.0%" in text and "+1.0%" in text
    # 섹터별 TOP 섹션에도 종목명이 나오므로, 전체 목록 구간만 잘라서 순서를 본다
    listing = text.split("■ 전체 52주 신고가 목록")[1]
    assert listing.index("높은종목") < listing.index("낮은종목")  # 등락률 내림차순
