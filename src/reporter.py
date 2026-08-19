# src/reporter.py
import asyncio
from html import escape

from loguru import logger
from telegram import Bot
from telegram.constants import ParseMode

from src.models import ScanResult, AIAnalysisResult

NAVER_STOCK_URL = "https://finance.naver.com/item/main.naver?code={ticker}"


def split_message(text: str, max_length: int = 4096) -> list[str]:
    """Split a long message into chunks that fit Telegram's limit."""
    if len(text) <= max_length:
        return [text]

    chunks = []
    current = ""

    for line in text.split("\n"):
        if len(current) + len(line) + 1 > max_length:
            if current:
                chunks.append(current)
                current = ""
            while len(line) > max_length:
                chunks.append(line[:max_length])
                line = line[max_length:]
            current = line
        else:
            current = f"{current}\n{line}" if current else line

    if current:
        chunks.append(current)
    return chunks


def _stock_link(name: str, ticker: str) -> str:
    """Create an HTML link to Naver Finance for a stock."""
    url = NAVER_STOCK_URL.format(ticker=ticker)
    return f'<a href="{url}">{escape(name)} ({ticker})</a>'


class Reporter:
    """Formats and sends reports via Telegram."""

    def __init__(self, bot_token: str, chat_id: int):
        self.bot_token = bot_token
        self.chat_id = chat_id

    def format_report(
        self,
        result: ScanResult,
        ai_analyses: list[AIAnalysisResult],
        trend: list[dict],
    ) -> str:
        """Format a ScanResult into a Telegram HTML message string."""
        s = result.stats
        lines = []

        lines.append(f"📊 <b>52주 신고가 리포트 ({result.scan_date})</b>")
        lines.append("<i>장중 고가 기준 (오늘 고가 ≥ 직전 250 거래일 최고 고가)</i>")
        lines.append("")

        lines.append("<b>■ 시장 요약</b>")
        lines.append(
            f"• 신고가 종목: {s.new_high_count}개 "
            f"(KOSPI {s.kospi_count} / KOSDAQ {s.kosdaq_count} / ETF {s.etf_count})"
        )
        if len(trend) >= 2:
            prev = trend[-2]["count"]
            diff = s.new_high_count - prev
            sign = "+" if diff >= 0 else ""
            lines.append(f"• 전일 대비: {sign}{diff}개")
        if trend:
            trend_str = "→".join(str(t["count"]) for t in trend)
            lines.append(f"• 최근 추이: {trend_str}")
        lines.append("")

        sorted_sectors = sorted(
            result.sector_breakdown.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )
        lines.append("<b>■ 섹터별 TOP</b>")
        for i, (sector, stocks) in enumerate(sorted_sectors[:5], 1):
            names = ", ".join(escape(st.name) for st in stocks[:3])
            suffix = "..." if len(stocks) > 3 else ""
            lines.append(f"{i}. {escape(sector)} ({len(stocks)}종목): {names}{suffix}")
        lines.append("")

        ai_map = {a.ticker: a for a in ai_analyses}
        if ai_analyses:
            lines.append("<b>■ 주요 종목 AI 분석</b>")
            lines.append("")
            for stock in result.highs:
                if stock.ticker in ai_map:
                    a = ai_map[stock.ticker]
                    link = _stock_link(stock.name, stock.ticker)
                    lines.append(
                        f"▶ {link} | "
                        f"{stock.close_price:,.0f}원 | +{stock.change_pct:.1f}%"
                    )
                    lines.append(escape(a.ai_analysis))
                    if a.news_links:
                        lines.append("관련 기사:")
                        for news_link in a.news_links:
                            lines.append(f"  🔗 {news_link}")
                    lines.append("─" * 30)
                    lines.append("")
            lines.append("")

        lines.append("<b>■ 전체 52주 신고가 목록</b>")
        for stock in sorted(result.highs, key=lambda h: h.change_pct, reverse=True):
            link = _stock_link(stock.name, stock.ticker)
            lines.append(
                f"  {link} | {stock.close_price:,.0f}원 | "
                f"+{stock.change_pct:.1f}% | {escape(stock.sector)}"
            )

        return "\n".join(lines)

    async def send(self, text: str) -> None:
        """Send a message via Telegram Bot, splitting if needed."""
        bot = Bot(token=self.bot_token)
        for chunk in split_message(text):
            await bot.send_message(
                chat_id=self.chat_id,
                text=chunk,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
            await asyncio.sleep(0.5)

    async def send_report(
        self,
        result: ScanResult,
        ai_analyses: list[AIAnalysisResult],
        trend: list[dict],
    ) -> None:
        """Format and send a complete scan report."""
        text = self.format_report(result, ai_analyses, trend)
        await self.send(text)
        logger.info(f"Report sent to Telegram chat {self.chat_id}")
