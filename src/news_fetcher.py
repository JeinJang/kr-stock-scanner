import asyncio
import html
import re

import httpx
from loguru import logger

from src.models import NewsArticle


def _clean_text(text: str) -> str:
    """Remove HTML tags, decode entities, and strip characters that break JSON."""
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    # Remove control chars + surrogates + other problematic unicode
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    return text.strip()


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
                title=_clean_text(item["title"]),
                link=item.get("link", item.get("originallink", "")),
                description=_clean_text(item.get("description", "")),
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
