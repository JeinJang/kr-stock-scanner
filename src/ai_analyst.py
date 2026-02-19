import asyncio

import openai
from loguru import logger

from src.models import StockHigh, NewsArticle, AIAnalysisResult


class AIAnalyst:
    """Analyzes stock rise reasons using OpenAI GPT."""

    def __init__(self, api_key: str, model: str = "gpt-5-nano", max_tokens: int = 600):
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

        prompt = f"""다음 종목이 52주 신고가를 기록했습니다. 관련 뉴스를 바탕으로 아래 형식에 맞춰 분석해주세요.

종목: {stock.name} ({stock.ticker})
시장: {stock.market} / 섹터: {stock.sector}
종가: {stock.close_price:,.0f}원
52주 신고가: {stock.high_52w:,.0f}원 (전고점 대비 +{stock.breakout_pct:.1f}%)
거래량: {stock.volume:,}주

최근 뉴스:
{news_text}

아래 형식으로 한국어 분석을 작성해주세요:
[상승 원인] 이 종목이 52주 신고가를 기록한 핵심 원인을 2~3문장으로 구체적으로 설명 (실적, 수주, 정책, 수급 등 구체적 이유 포함)
[핵심 뉴스] 가장 관련도 높은 뉴스 1~2개를 한 줄씩 요약
[투자 포인트] 향후 주가에 영향을 줄 수 있는 핵심 변수 1~2개를 간단히 언급"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=self.max_tokens,
        )

        content = response.choices[0].message.content
        analysis = content.strip() if content else "분석 결과를 생성하지 못했습니다."

        news_links = [a.link for a in news if a.link]

        return AIAnalysisResult(
            ticker=stock.ticker,
            news_summary=news_text[:500],
            ai_analysis=analysis,
            news_links=news_links[:3],
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
