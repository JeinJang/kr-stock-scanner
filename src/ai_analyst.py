import asyncio
import re

import openai
from loguru import logger

from src.models import StockHigh, NewsArticle, AIAnalysisResult


def _strip_preamble(text: str) -> str:
    """Remove any preamble the model adds before the first ``[...]`` section.

    가끔 모델이 "다음은 제시해주신 정보를 바탕으로 한 분석입니다." 같은 서두를
    지정된 형식 앞에 붙이는데, 첫 ``[`` 섹션 헤더 이전 텍스트를 잘라냅니다.
    """
    idx = text.find("[")
    if idx > 0:
        return text[idx:].strip()
    return text


def _sanitize(text: str) -> str:
    """Remove characters that can break JSON serialization."""
    # Remove surrogates and invalid Unicode by round-tripping through UTF-8
    text = text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")
    # Remove control characters (keep \n and \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Remove lone surrogates that survived encoding
    text = re.sub(r"[\ud800-\udfff]", "", text)
    return text


def _recency_prompt_line(stock: StockHigh) -> str:
    """돌파 신선도를 프롬프트 한 줄로. 지표가 없으면 빈 문자열."""
    if stock.history_span_days is None:
        return ""

    if stock.days_since_prev_new_high is None:
        first = "조회 구간 내 직전 신고가 없음(첫 돌파)"
    else:
        first = f"직전 신고가 {stock.days_since_prev_new_high}일 전"

    if stock.days_since_price_above is None:
        second = "현재가는 확보된 이력 전체에서 최고 수준"
    else:
        second = f"현재가를 마지막으로 웃돈 시점은 {stock.days_since_price_above}일 전"

    return f"돌파 신선도: {first} / {second}\n"


class AIAnalyst:
    """Analyzes stock rise reasons using OpenAI GPT."""

    def __init__(self, api_key: str, model: str = "gpt-5-nano"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def analyze_stock(
        self, stock: StockHigh, news: list[NewsArticle]
    ) -> AIAnalysisResult:
        """Analyze why a stock hit its 52-week high using news context."""
        news_text = "\n".join(
            f"- {_sanitize(a.title)}: {_sanitize(a.description)}" for a in news
        ) if news else "관련 뉴스 없음"

        breakout_note = (
            f" (직전 52주 고점 대비 +{stock.breakout_pct:.1f}%)"
            if stock.breakout_pct > 0 else ""
        )

        prompt = f"""다음 종목이 52주 신고가를 기록했습니다. 관련 뉴스를 바탕으로 아래 형식에 맞춰 분석해주세요.

종목: {stock.name} ({stock.ticker})
시장: {stock.market} / 섹터: {stock.sector}
종가: {stock.close_price:,.0f}원 (당일 {stock.change_pct:+.1f}%)
52주 신고가: {stock.high_52w:,.0f}원{breakout_note}
거래량: {stock.volume:,}주
{_recency_prompt_line(stock)}
최근 뉴스:
{news_text}

아래 형식으로 한국어 분석을 작성해주세요:
[상승 원인] 이 종목이 52주 신고가를 기록한 핵심 원인을 2~3문장으로 구체적으로 설명 (실적, 수주, 정책, 수급 등 구체적 이유 포함)
[핵심 뉴스] 가장 관련도 높은 뉴스 1~2개를 한 줄씩 요약
[투자 포인트] 향후 주가에 영향을 줄 수 있는 핵심 변수 1~2개를 간단히 언급"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": _sanitize(prompt)}],
        )

        content = response.choices[0].message.content
        analysis = _strip_preamble(content.strip()) if content else "분석 결과를 생성하지 못했습니다."

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
        fail_count = 0
        last_error = None
        for stock in sorted_stocks:
            try:
                news = news_map.get(stock.name, [])
                result = await self.analyze_stock(stock, news)
                results.append(result)
            except Exception as e:
                fail_count += 1
                last_error = e
                logger.error(f"AI analysis failed for {stock.name} ({stock.ticker}): {type(e).__name__}: {e}")
                # Stop early if all attempts are failing with the same error
                if fail_count >= 3 and len(results) == 0:
                    logger.error(f"AI analysis aborted: {fail_count} consecutive failures. Last error: {e}")
                    break
            await asyncio.sleep(0.5)

        logger.info(f"Completed AI analysis: {len(results)} success / {fail_count} failed / {len(sorted_stocks)} total")
        return results
