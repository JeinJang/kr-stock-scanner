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


# -- 돌파 신선도 라벨 -----------------------------------------------------

RECENCY_STREAK_DAYS = 5      # 이하 → 신고가 행진
RECENCY_SHORT_DAYS = 30      # 이하 → 단기 재돌파
RECENCY_LONG_DAYS = 365      # 초과 → 장기 돌파
DEPTH_DECADE_DAYS = 3650     # 이상 → 10년래 최고

GROUP_LONG = "장기 돌파 · 1년 이상 만"
GROUP_MID = "중기 돌파 · 1~12개월"
GROUP_STREAK = "신고가 행진 · 1개월 내 재돌파"
GROUP_UNKNOWN = "정보 없음"
GROUP_ORDER = (GROUP_LONG, GROUP_MID, GROUP_STREAK, GROUP_UNKNOWN)


def _fmt_span(days: int) -> str:
    """일수를 '3년 2개월' 같은 사람이 읽는 기간으로."""
    years, rest = divmod(days, 365)
    months = rest // 30
    if years and months:
        return f"{years}년 {months}개월"
    if years:
        return f"{years}년"
    if months:
        return f"{months}개월"
    return f"{days}일"


def _recency_badge(stock) -> str | None:
    """A(재돌파 간격) 배지. 이력이 없으면 None."""
    if stock.history_span_days is None:
        return None
    a = stock.days_since_prev_new_high
    if a is None:
        # 확보 구간 안에 직전 신고가가 없음 — 워밍업 52주를 뺀 값이 하한
        floor_days = max(stock.history_span_days - RECENCY_LONG_DAYS, 0)
        return f"🆕 {_fmt_span(floor_days)} 이상 만 (첫 돌파)"
    if a <= RECENCY_STREAK_DAYS:
        return "🔁 신고가 행진"
    if a <= RECENCY_SHORT_DAYS:
        return f"🔁 {a}일 만"
    return f"🆕 {_fmt_span(a)} 만"


def _depth_badge(stock) -> str | None:
    """B(갱신 깊이) 배지. 이력이 없으면 None."""
    span = stock.history_span_days
    if span is None:
        return None
    b = stock.days_since_price_above
    if b is None:
        return "🏔 10년래 최고" if span >= DEPTH_DECADE_DAYS else "🏔 상장 이후 최고"
    return f"🏔 {_fmt_span(b)} 만의 최고가"


def _recency_group(stock) -> str:
    """A 기준 그룹명."""
    if stock.history_span_days is None:
        return GROUP_UNKNOWN
    a = stock.days_since_prev_new_high
    if a is None or a > RECENCY_LONG_DAYS:
        return GROUP_LONG
    if a > RECENCY_SHORT_DAYS:
        return GROUP_MID
    return GROUP_STREAK


def _badges(stock) -> list[str]:
    return [b for b in (_recency_badge(stock), _depth_badge(stock)) if b]


def _stock_line(stock) -> str:
    """전체 목록의 한 줄."""
    parts = [
        f"  {_stock_link(stock.name, stock.ticker)}",
        f"{stock.close_price:,.0f}원",
        f"+{stock.change_pct:.1f}%",
    ]
    if stock.breakout_pct > 0:
        parts.append(f"↑{stock.breakout_pct:.1f}% 돌파")
    parts.extend(_badges(stock))
    parts.append(escape(stock.sector))
    return " | ".join(parts)


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
                    header = [
                        f"▶ {link}",
                        f"{stock.close_price:,.0f}원",
                        f"+{stock.change_pct:.1f}%",
                    ]
                    header.extend(_badges(stock))
                    lines.append(" | ".join(header))
                    lines.append(escape(a.ai_analysis))
                    if a.news_links:
                        lines.append("관련 기사:")
                        for news_link in a.news_links:
                            lines.append(f"  🔗 {news_link}")
                    lines.append("─" * 30)
                    lines.append("")
            lines.append("")

        lines.append("<b>■ 전체 52주 신고가 목록</b>")
        ordered = sorted(result.highs, key=lambda h: h.change_pct, reverse=True)
        grouped: dict[str, list] = {g: [] for g in GROUP_ORDER}
        for stock in ordered:
            grouped[_recency_group(stock)].append(stock)

        has_metrics = any(h.history_span_days is not None for h in result.highs)
        for group in GROUP_ORDER:
            members = grouped[group]
            if not members:
                continue
            if has_metrics:
                lines.append(f"[{group}]")
            for stock in members:
                lines.append(_stock_line(stock))

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
