from __future__ import annotations
from datetime import date
from sqlalchemy import (
    Column, Integer, BigInteger, String, Date,
    create_engine, select,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import DeclarativeBase, Session

from src.market_data.models import MarketYearly


class MarketBase(DeclarativeBase):
    pass


class MarketYearlyRow(MarketBase):
    __tablename__ = "corp_market_yearly"
    corp_code = Column(String(8), primary_key=True)
    year = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    as_of_date = Column(Date, nullable=False)
    market_cap = Column(BigInteger, nullable=True)
    shares_outstanding = Column(BigInteger, nullable=True)
    close_price = Column(Integer, nullable=True)


class MarketDB:
    """Persistence for yearly market data (corp_market_yearly)."""

    def __init__(self, url: str = "sqlite:///data/scanner.db"):
        self.engine = create_engine(url)
        MarketBase.metadata.create_all(self.engine)

    def save_yearly(self, rows: list[MarketYearly]) -> None:
        if not rows:
            return
        with Session(self.engine) as session:
            for r in rows:
                stmt = sqlite_insert(MarketYearlyRow).values(
                    corp_code=r.corp_code, year=r.year, ticker=r.ticker,
                    as_of_date=r.as_of_date,
                    market_cap=r.market_cap,
                    shares_outstanding=r.shares_outstanding,
                    close_price=r.close_price,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=["corp_code", "year"],
                    set_={
                        "ticker": stmt.excluded.ticker,
                        "as_of_date": stmt.excluded.as_of_date,
                        "market_cap": stmt.excluded.market_cap,
                        "shares_outstanding": stmt.excluded.shares_outstanding,
                        "close_price": stmt.excluded.close_price,
                    },
                )
                session.execute(stmt)
            session.commit()

    def load_for_corp(self, corp_code: str) -> list[MarketYearly]:
        with Session(self.engine) as session:
            rows = session.execute(
                select(MarketYearlyRow).where(MarketYearlyRow.corp_code == corp_code)
            ).scalars().all()
            return [_row_to_model(r) for r in rows]

    def load_all(self) -> dict[str, list[MarketYearly]]:
        with Session(self.engine) as session:
            rows = session.execute(select(MarketYearlyRow)).scalars().all()
            out: dict[str, list[MarketYearly]] = {}
            for r in rows:
                out.setdefault(r.corp_code, []).append(_row_to_model(r))
            return out


def _row_to_model(r: MarketYearlyRow) -> MarketYearly:
    return MarketYearly(
        corp_code=r.corp_code, ticker=r.ticker,
        year=r.year, as_of_date=r.as_of_date,
        market_cap=r.market_cap,
        shares_outstanding=r.shares_outstanding,
        close_price=r.close_price,
    )
