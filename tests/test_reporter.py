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


def _rstock(ticker="000001", name="종목", a=None, b=None, span=None, breakout=0.0):
    from src.models import StockHigh

    return StockHigh(
        ticker=ticker, name=name, market="KOSPI", sector="전기전자",
        close_price=1000, high_52w=1000, prev_high_52w=0.0,
        breakout_pct=breakout, volume=1, avg_volume_20d=0, change_pct=1.0,
        days_since_prev_new_high=a, days_since_price_above=b, history_span_days=span,
    )


def test_fmt_span_formats_years_months_days():
    from src.reporter import _fmt_span

    assert _fmt_span(1170) == "3년 2개월"
    assert _fmt_span(730) == "2년"
    assert _fmt_span(90) == "3개월"
    assert _fmt_span(12) == "12일"


def test_fmt_span_clamps_month_rollover_at_boundaries():
    """rest // 30 이 12가 되는 구간(365일 직전)에서도 '12개월'이 나오면 안 된다."""
    from src.reporter import _fmt_span

    assert _fmt_span(364) == "11개월"
    assert _fmt_span(365) == "1년"
    assert _fmt_span(725) == "1년 11개월"
    assert _fmt_span(1090) == "2년 11개월"


def test_recency_badge_buckets():
    from src.reporter import _recency_badge

    assert _recency_badge(_rstock(a=1, span=4000)) == "🔁 신고가 행진"
    assert _recency_badge(_rstock(a=5, span=4000)) == "🔁 신고가 행진"
    assert _recency_badge(_rstock(a=6, span=4000)) == "🔁 6일 만"
    assert _recency_badge(_rstock(a=30, span=4000)) == "🔁 30일 만"
    assert _recency_badge(_rstock(a=90, span=4000)) == "🆕 3개월 만"
    assert _recency_badge(_rstock(a=1170, span=4000)) == "🆕 3년 2개월 만"


def test_recency_badge_when_no_prior_new_high_in_range():
    from src.reporter import _recency_badge

    badge = _recency_badge(_rstock(a=None, span=4000))
    assert "이상 만" in badge and "첫 돌파" in badge


def test_badges_are_omitted_without_history():
    from src.reporter import _recency_badge, _depth_badge

    stock = _rstock(a=None, b=None, span=None)
    assert _recency_badge(stock) is None
    assert _depth_badge(stock) is None


def test_depth_badge_buckets():
    from src.reporter import _depth_badge

    assert _depth_badge(_rstock(b=None, span=4000)) == "🏔 10년래 최고"
    assert _depth_badge(_rstock(b=None, span=1000)) == "🏔 상장 이후 최고"
    assert _depth_badge(_rstock(b=1170, span=4000)) == "🏔 3년 2개월 만의 최고가"


def test_depth_badge_decade_boundary():
    from src.reporter import _depth_badge

    assert _depth_badge(_rstock(b=None, span=3650)) == "🏔 10년래 최고"
    assert _depth_badge(_rstock(b=None, span=3649)) == "🏔 상장 이후 최고"


def test_recency_groups():
    from src.reporter import _recency_group

    assert _recency_group(_rstock(a=1000, span=4000)).startswith("장기 돌파")
    assert _recency_group(_rstock(a=None, span=4000)).startswith("장기 돌파")
    assert _recency_group(_rstock(a=100, span=4000)).startswith("중기 돌파")
    assert _recency_group(_rstock(a=10, span=4000)).startswith("신고가 행진")
    assert _recency_group(_rstock(span=None)) == "정보 없음"


def test_recency_group_boundary_at_365_days():
    """A == 365 는 장기 돌파(1년 이상 만), A == 364 는 중기 돌파여야 배지 문구와 그룹명이 맞는다."""
    from src.reporter import _recency_group

    assert _recency_group(_rstock(a=365, span=4000)).startswith("장기 돌파")
    assert _recency_group(_rstock(a=364, span=4000)).startswith("중기 돌파")


def test_stock_line_shows_breakout_only_when_known():
    from src.reporter import _stock_line

    assert "돌파" not in _stock_line(_rstock(breakout=0.0))
    assert "↑1.4% 돌파" in _stock_line(_rstock(a=10, span=4000, breakout=1.4))


def _result(highs):
    from datetime import date
    from src.models import ScanResult, MarketStats

    return ScanResult(
        scan_date=date(2026, 8, 19),
        stats=MarketStats(total_stocks=len(highs), new_high_count=len(highs), kospi_count=len(highs)),
        highs=highs, sector_breakdown={"전기전자": highs},
    )


def test_report_groups_when_any_stock_has_metrics():
    from src.reporter import Reporter

    highs = [_rstock("000001", "장기", a=1000, span=4000), _rstock("000002", "미상", span=None)]
    text = Reporter(bot_token="", chat_id=0).format_report(_result(highs), [], [])

    assert "[장기 돌파 · 1년 이상 만]" in text
    assert "[정보 없음]" in text
    assert text.index("[장기 돌파 · 1년 이상 만]") < text.index("[정보 없음]")


def test_report_stays_flat_when_no_stock_has_metrics():
    from src.reporter import Reporter

    highs = [_rstock("000001", "가", span=None), _rstock("000002", "나", span=None)]
    text = Reporter(bot_token="", chat_id=0).format_report(_result(highs), [], [])

    assert "[정보 없음]" not in text
    assert "장기 돌파" not in text


def test_recency_badge_is_omitted_for_recently_listed_stock():
    """상장 1년 미만(이력 < 워밍업 52주)이면 하한을 말할 수 없으므로 A 배지는 없다."""
    from src.reporter import _recency_badge, _depth_badge

    stock = _rstock(a=None, b=None, span=99)
    assert _recency_badge(stock) is None
    # 깊이 배지는 참이므로 그대로 남는다
    assert _depth_badge(stock) == "🏔 상장 이후 최고"


def test_recency_badge_floor_boundary_at_one_year():
    """하한이 0 이하면 생략, 1일이라도 남으면 표기한다."""
    from src.reporter import _recency_badge

    assert _recency_badge(_rstock(a=None, span=365)) is None
    assert _recency_badge(_rstock(a=None, span=366)) == "🆕 1일 이상 만 (첫 돌파)"


def test_recently_listed_stock_goes_to_new_listing_group():
    from src.reporter import _recency_group, GROUP_NEW_LISTING

    assert _recency_group(_rstock(a=None, span=99)) == GROUP_NEW_LISTING
    # 이력이 1년 이상이면 종전대로 장기 돌파
    assert _recency_group(_rstock(a=None, span=4000)).startswith("장기 돌파")
    # A를 아는 종목은 이력이 짧아도 A 기준 그룹을 따른다
    assert _recency_group(_rstock(a=10, span=99)).startswith("신고가 행진")


def test_report_shows_new_listing_group_without_empty_floor_badge():
    from src.reporter import Reporter, GROUP_NEW_LISTING

    highs = [_rstock("000001", "신규상장", a=None, b=None, span=99)]
    text = Reporter(bot_token="", chat_id=0).format_report(_result(highs), [], [])

    assert f"[{GROUP_NEW_LISTING}]" in text
    assert "0일 이상 만" not in text
    assert "[장기 돌파 · 1년 이상 만]" not in text
    assert "🏔 상장 이후 최고" in text
