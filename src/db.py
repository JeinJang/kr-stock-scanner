# src/db.py
from datetime import date, datetime

from sqlalchemy import (
    Column, Integer, String, Float, BigInteger, Date, DateTime, Text,
    create_engine, select,
)
from sqlalchemy.orm import DeclarativeBase, Session

from src.models import ScanResult, AIAnalysisResult


class Base(DeclarativeBase):
    pass


class DailyScan(Base):
    __tablename__ = "daily_scans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_date = Column(Date, nullable=False)
    total_stocks = Column(Integer, nullable=False)
    new_high_count = Column(Integer, nullable=False)
    market_type = Column(String(10), nullable=False)


class NewHigh(Base):
    __tablename__ = "new_highs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_date = Column(Date, nullable=False, index=True)
    ticker = Column(String(10), nullable=False)
    name = Column(String(100), nullable=False)
    market = Column(String(10), nullable=False)
    sector = Column(String(50), nullable=False)
    close_price = Column(Float, nullable=False)
    high_52w = Column(Float, nullable=False)
    prev_high_52w = Column(Float, nullable=False)
    breakout_pct = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)
    avg_volume_20d = Column(BigInteger, nullable=False)


class AIAnalysis(Base):
    __tablename__ = "ai_analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False)
    scan_date = Column(Date, nullable=False, index=True)
    news_summary = Column(Text, nullable=False)
    ai_analysis = Column(Text, nullable=False)
    news_links = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.now)


class Database:
    def __init__(self, url: str = "sqlite:///data/scanner.db"):
        self.engine = create_engine(url)
        Base.metadata.create_all(self.engine)

    def save_scan_result(self, result: ScanResult) -> None:
        with Session(self.engine) as session:
            session.add(DailyScan(
                scan_date=result.scan_date,
                total_stocks=result.stats.total_stocks,
                new_high_count=result.stats.new_high_count,
                market_type="ALL",
            ))
            for stock in result.highs:
                session.add(NewHigh(
                    scan_date=result.scan_date,
                    ticker=stock.ticker,
                    name=stock.name,
                    market=stock.market,
                    sector=stock.sector,
                    close_price=stock.close_price,
                    high_52w=stock.high_52w,
                    prev_high_52w=stock.prev_high_52w,
                    breakout_pct=stock.breakout_pct,
                    volume=stock.volume,
                    avg_volume_20d=stock.avg_volume_20d,
                ))
            session.commit()

    def get_scan_result(self, scan_date: date) -> list[dict]:
        with Session(self.engine) as session:
            rows = session.execute(
                select(NewHigh).where(NewHigh.scan_date == scan_date)
            ).scalars().all()
            return [
                {
                    "ticker": r.ticker, "name": r.name, "market": r.market,
                    "sector": r.sector, "close_price": r.close_price,
                    "high_52w": r.high_52w, "breakout_pct": r.breakout_pct,
                }
                for r in rows
            ]

    def get_high_count_history(self, days: int = 5) -> list[dict]:
        with Session(self.engine) as session:
            rows = session.execute(
                select(DailyScan)
                .order_by(DailyScan.scan_date.desc())
                .limit(days)
            ).scalars().all()
            return [
                {"date": r.scan_date, "count": r.new_high_count}
                for r in reversed(rows)
            ]

    def save_ai_analysis(self, scan_date: date, analysis: AIAnalysisResult) -> None:
        with Session(self.engine) as session:
            session.add(AIAnalysis(
                ticker=analysis.ticker,
                scan_date=scan_date,
                news_summary=analysis.news_summary,
                ai_analysis=analysis.ai_analysis,
                news_links="\n".join(analysis.news_links),
            ))
            session.commit()

    def get_ai_analysis(self, scan_date: date, ticker: str) -> dict | None:
        with Session(self.engine) as session:
            row = session.execute(
                select(AIAnalysis).where(
                    AIAnalysis.scan_date == scan_date,
                    AIAnalysis.ticker == ticker,
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            return {
                "ticker": row.ticker,
                "news_summary": row.news_summary,
                "ai_analysis": row.ai_analysis,
            }
