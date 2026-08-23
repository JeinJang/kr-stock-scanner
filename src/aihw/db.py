"""src/aihw/db.py

AI HW/빅테크 지표의 일별 시총 스냅샷 저장소 (data/aihw.db).
규칙: snapshot 행은 backfill로 덮어쓰지 않는다. snapshot은 무엇이든 덮어쓴다.
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import BigInteger, Column, Date, DateTime, Float, String, create_engine, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import DeclarativeBase, Session

from src.aihw.models import DailyCap


class AihwBase(DeclarativeBase):
    pass


class DailyCapRow(AihwBase):
    __tablename__ = "daily_caps"
    date = Column(Date, primary_key=True)
    ticker = Column(String(12), primary_key=True)
    close = Column(Float, nullable=False)
    shares = Column(BigInteger, nullable=True)
    market_cap_usd = Column(Float, nullable=True)
    source = Column(String(10), nullable=False)
    created_at = Column(DateTime, default=datetime.now)


class AihwDB:
    def __init__(self, url: str = "sqlite:///data/aihw.db"):
        self.engine = create_engine(url)
        AihwBase.metadata.create_all(self.engine)

    def save_caps(self, rows: list[DailyCap]) -> int:
        if not rows:
            return 0
        saved = 0
        with Session(self.engine) as session:
            for r in rows:
                stmt = sqlite_insert(DailyCapRow).values(
                    date=r.date, ticker=r.ticker, close=r.close,
                    shares=r.shares, market_cap_usd=r.market_cap_usd,
                    source=r.source,
                )
                set_ = {
                    "close": stmt.excluded.close,
                    "shares": stmt.excluded.shares,
                    "market_cap_usd": stmt.excluded.market_cap_usd,
                    "source": stmt.excluded.source,
                }
                if r.source == "backfill":
                    # backfill은 기존 snapshot을 건드리지 않는다
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["date", "ticker"], set_=set_,
                        where=(DailyCapRow.source != "snapshot"),
                    )
                else:
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["date", "ticker"], set_=set_,
                    )
                result = session.execute(stmt)
                saved += result.rowcount
            session.commit()
        return saved

    def load_caps(self, start: date, end: date) -> list[DailyCap]:
        with Session(self.engine) as session:
            rows = session.execute(
                select(DailyCapRow)
                .where(DailyCapRow.date >= start, DailyCapRow.date <= end)
                .order_by(DailyCapRow.date, DailyCapRow.ticker)
            ).scalars().all()
            return [
                DailyCap(
                    date=r.date, ticker=r.ticker, close=r.close,
                    shares=r.shares, market_cap_usd=r.market_cap_usd,
                    source=r.source,
                )
                for r in rows
            ]
