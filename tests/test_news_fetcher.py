# tests/test_news_fetcher.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx


@pytest.mark.asyncio
async def test_fetch_naver_api_news():
    """NewsFetcher should parse Naver Search API response correctly."""
    from src.news_fetcher import NewsFetcher

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "items": [
            {
                "title": "<b>삼성전자</b> HBM4 수주 확대",
                "originallink": "https://example.com/1",
                "link": "https://n.news.naver.com/1",
                "description": "<b>삼성전자</b>가 HBM4 양산 준비에 박차를 가하고 있다.",
                "pubDate": "Wed, 19 Feb 2026 10:00:00 +0900",
            },
            {
                "title": "반도체 업황 개선 신호",
                "originallink": "https://example.com/2",
                "link": "https://n.news.naver.com/2",
                "description": "글로벌 반도체 시장이 회복세를 보이고 있다.",
                "pubDate": "Wed, 19 Feb 2026 09:00:00 +0900",
            },
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("src.news_fetcher.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        fetcher = NewsFetcher(
            naver_client_id="test-id",
            naver_client_secret="test-secret",
        )
        articles = await fetcher.fetch_news("삼성전자", max_articles=5)

    assert len(articles) == 2
    assert "삼성전자" in articles[0].title
    assert "<b>" not in articles[0].title
    assert articles[0].link == "https://n.news.naver.com/1"


@pytest.mark.asyncio
async def test_fetch_news_for_stocks():
    """Should fetch news for multiple stocks."""
    from src.news_fetcher import NewsFetcher
    from src.models import NewsArticle

    fetcher = NewsFetcher(naver_client_id="test", naver_client_secret="test")

    async def mock_fetch(query, max_articles=5):
        return [NewsArticle(title=f"{query} 뉴스", link="https://test.com", description="test")]

    fetcher.fetch_news = mock_fetch

    results = await fetcher.fetch_news_for_stocks(["삼성전자", "SK하이닉스"])

    assert "삼성전자" in results
    assert "SK하이닉스" in results
    assert len(results["삼성전자"]) == 1
