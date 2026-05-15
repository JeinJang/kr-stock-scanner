from datetime import date
from pydantic import BaseModel


class MarketYearly(BaseModel):
    """Yearly market data snapshot for a single ticker.

    For completed years: as_of_date = last trading day of that year.
    For the in-progress year: as_of_date = backfill execution date (latest business day).
    """

    corp_code: str
    ticker: str
    year: int
    as_of_date: date
    market_cap: int | None = None        # 원
    shares_outstanding: int | None = None  # 주
    close_price: int | None = None        # 원 (verification)
