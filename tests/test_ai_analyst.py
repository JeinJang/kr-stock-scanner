import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_analyze_stock():
    """AIAnalyst should call OpenAI API and return analysis."""
    from src.ai_analyst import AIAnalyst
    from src.models import StockHigh, NewsArticle

    stock = StockHigh(
        ticker="005930", name="삼성전자", market="KOSPI", sector="전기전자",
        close_price=78500, high_52w=79000, prev_high_52w=77000,
        breakout_pct=2.60, volume=15000000, avg_volume_20d=12000000,
    )
    news = [
        NewsArticle(title="삼성전자 HBM4 수주 확대", link="https://test.com",
                     description="HBM4 양산 준비 박차"),
    ]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(
        content="HBM4 수주 확대에 따른 실적 개선 기대감이 52주 신고가를 견인했습니다."
    ))]

    with patch("src.ai_analyst.openai.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_cls.return_value = mock_client

        analyst = AIAnalyst(api_key="test-key", model="gpt-5-nano")
        result = await analyst.analyze_stock(stock, news)

    assert result.ticker == "005930"
    assert "HBM4" in result.ai_analysis
    assert "HBM4" in result.news_summary


@pytest.mark.asyncio
async def test_analyze_stocks_respects_max_limit():
    """Should only analyze up to max_analyze stocks by market cap."""
    from src.ai_analyst import AIAnalyst
    from src.models import StockHigh

    stocks = [
        StockHigh(
            ticker=f"00{i}000", name=f"Stock{i}", market="KOSPI", sector="테스트",
            close_price=10000 * i, high_52w=10000 * i, prev_high_52w=9000 * i,
            breakout_pct=11.1, volume=1000000, avg_volume_20d=800000,
        )
        for i in range(1, 6)
    ]
    news_map = {s.name: [] for s in stocks}
    market_caps = {s.ticker: s.close_price * 1000000 for s in stocks}

    with patch("src.ai_analyst.openai.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="분석 결과"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_cls.return_value = mock_client

        analyst = AIAnalyst(api_key="test-key", model="gpt-5-nano")
        results = await analyst.analyze_stocks(
            stocks, news_map, market_caps, max_analyze=3
        )

    assert len(results) == 3


def _astock(a=None, b=None, span=None):
    from src.models import StockHigh

    return StockHigh(
        ticker="005930", name="삼성전자", market="KOSPI", sector="전기전자",
        close_price=78500, high_52w=79000, prev_high_52w=77000,
        breakout_pct=2.6, volume=1000, avg_volume_20d=0, change_pct=3.1,
        days_since_prev_new_high=a, days_since_price_above=b, history_span_days=span,
    )


def test_recency_prompt_line_is_empty_without_history():
    from src.ai_analyst import _recency_prompt_line

    assert _recency_prompt_line(_astock()) == ""


def test_recency_prompt_line_describes_both_axes():
    from src.ai_analyst import _recency_prompt_line

    line = _recency_prompt_line(_astock(a=1170, b=1500, span=4000))
    assert "돌파 신선도:" in line
    assert "1170일 전" in line
    assert "1500일 전" in line


def test_recency_prompt_line_marks_all_time_high():
    from src.ai_analyst import _recency_prompt_line

    line = _recency_prompt_line(_astock(a=1170, b=None, span=4000))
    assert "최고 수준" in line


def test_recency_prompt_line_marks_first_breakout():
    from src.ai_analyst import _recency_prompt_line

    line = _recency_prompt_line(_astock(a=None, b=None, span=4000))
    assert "첫 돌파" in line
