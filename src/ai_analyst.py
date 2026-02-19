import asyncio

import openai
from loguru import logger

from src.models import StockHigh, NewsArticle, AIAnalysisResult


class AIAnalyst:
    """Analyzes stock rise reasons using OpenAI GPT."""

    def __init__(self, api_key: str, model: str = "gpt-5-nano", max_tokens: int = 300):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    async def analyze_stock(
        self, stock: StockHigh, news: list[NewsArticle]
    ) -> AIAnalysisResult:
        """Analyze why a stock hit its 52-week high using news context."""
        news_text = "\n".join(
            f"- {a.title}: {a.description}" for a in news
        ) if news else "관련 뉴스 없음"

        prompt = f"""다음 종목이 52주 신고가를 기록했습니다. 관련 뉴스를 바탕으로 상승 이유를 1-2문장으로 분석해주세요.

종목: {stock.name} ({stock.ticker})
시장: {stock.market} / 섹터: {stock.sector}
종가: {stock.close_price:,.0f}원
52주 신고가: {stock.high_52w:,.0f}원 (전고점 대비 +{stock.breakout_pct:.1f}%)
거래량: {stock.volume:,}주

최근 뉴스:
{news_text}

분석 (1-2문장, 한국어):"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.3,
        )

        analysis = response.choices[0].message.content.strip()

        return AIAnalysisResult(
            ticker=stock.ticker,
            news_summary=news_text[:500],
            ai_analysis=analysis,
        )

    async def analyze_stocks(
        self,
        stocks: list[StockHigh],
        news_map: dict[str, list[NewsArticle]],
        market_caps: dict[str, int],
        max_analyze: int = 50,
    ) -> list[AIAnalysisResult]:
        """Analyze multiple stocks, limited by max_analyze (sorted by market cap)."""
        sorted_stocks = sorted(
            stocks,
            key=lambda s: market_caps.get(s.ticker, 0),
            reverse=True,
        )[:max_analyze]

        results = []
        for stock in sorted_stocks:
            try:
                news = news_map.get(stock.name, [])
                result = await self.analyze_stock(stock, news)
                results.append(result)
            except Exception as e:
                logger.warning(f"AI analysis failed for {stock.ticker}: {e}")
            await asyncio.sleep(0.5)

        logger.info(f"Completed AI analysis for {len(results)}/{len(stocks)} stocks")
        return results
