from datetime import date

from pydantic import BaseModel


class StockHigh(BaseModel):
    """A single stock that hit a 52-week high."""

    ticker: str
    name: str
    market: str
    sector: str
    close_price: float
    high_52w: float
    prev_high_52w: float
    breakout_pct: float
    volume: int
    avg_volume_20d: int
    # -- 돌파 신선도 (이력 확보 실패 시 None) --
    days_since_prev_new_high: int | None = None   # A: 직전 신고가 이후 경과 일수
    days_since_price_above: int | None = None     # B: 오늘 고가를 마지막으로 웃돈 날 이후 경과 일수
    history_span_days: int | None = None          # 확보된 이력 길이
    change_pct: float = 0.0                       # 당일 등락률 (breakout_pct와 별개)


class MarketStats(BaseModel):
    """Aggregate market statistics for a scan."""

    total_stocks: int
    new_high_count: int
    kospi_count: int = 0
    kosdaq_count: int = 0
    etf_count: int = 0


class ScanResult(BaseModel):
    """Complete result of a daily 52-week high scan."""

    scan_date: date
    stats: MarketStats
    highs: list[StockHigh]
    sector_breakdown: dict[str, list[StockHigh]]


class NewsArticle(BaseModel):
    """A single news article."""

    title: str
    link: str
    description: str = ""
    source: str = ""
    pub_date: str = ""


class AIAnalysisResult(BaseModel):
    """AI analysis result for a single stock."""

    ticker: str
    news_summary: str
    ai_analysis: str
    news_links: list[str] = []
