import asyncio
import re

import httpx
from loguru import logger

from src.models import NewsArticle


def _strip_html(text: str) -> str:
    """Remove HTML tags from Naver API response text."""
    return re.sub(r"<[^>]+>", "", text).strip()


class NewsFetcher:
    """Fetches stock news from Naver Search API."""

    def __init__(self, naver_client_id: str, naver_client_secret: str):
        self.client_id = naver_client_id
        self.client_secret = naver_client_secret

    async def fetch_news(self, query: str, max_articles: int = 5) -> list[NewsArticle]:
        """Fetch news articles for a stock name from Naver Search API."""
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
        }
        params = {
            "query": query,
            "display": max_articles,
            "sort": "date",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()

        data = response.json()
        articles = []
        for item in data.get("items", []):
            articles.append(NewsArticle(
                title=_strip_html(item["title"]),
                link=item.get("link", item.get("originallink", "")),
                description=_strip_html(item.get("description", "")),
                source="",
                pub_date=item.get("pubDate", ""),
            ))

        logger.debug(f"Fetched {len(articles)} articles for '{query}'")
        return articles

    async def fetch_news_for_stocks(
        self, stock_names: list[str], max_articles: int = 5, delay: float = 0.2
    ) -> dict[str, list[NewsArticle]]:
        """Fetch news for multiple stocks with rate limiting."""
        results = {}
        for name in stock_names:
            try:
                articles = await self.fetch_news(name, max_articles)
                results[name] = articles
            except Exception as e:
                logger.warning(f"Failed to fetch news for '{name}': {e}")
                results[name] = []
            await asyncio.sleep(delay)
        return results
