# src/db.py
from datetime import date, datetime

from sqlalchemy import (
    Column, Integer, String, Float, BigInteger, Date, DateTime, Text,
    create_engine, delete, select, text,
)
from sqlalchemy.orm import DeclarativeBase, Session

from src.models import ScanResult, AIAnalysisResult


class Base(DeclarativeBase):
    pass


def _migrate_add_recency_columns(engine) -> None:
    """new_highs에 돌파 신선도 컬럼을 멱등 추가하고 레거시 행을 보정한다.

    스키마뿐 아니라 데이터도 마이그레이션한다. 이 브랜치 이전 코드는
    breakout_pct에 '당일 등락률'을 저장했고 change_pct는 존재하지 않았다.
    지금의 breakout_pct는 '직전 52주 고점 대비 돌파율'이라 의미가 다르므로,
    레거시 행을 그대로 읽으면 +5.2% 상승이 "5.2% 돌파"로 둔갑한다.

    따라서 change_pct가 NULL인 행에 한해 값을 change_pct로 옮기고
    breakout_pct는 0으로 되돌린다. 신규 코드는 change_pct를 항상 NOT NULL로
    쓰므로 WHERE 절은 레거시 행에만 걸리고, 매 기동마다 실행해도 안전하다
    (한 번 보정된 행은 change_pct가 채워져 다시 걸리지 않는다).

    테이블이 없으면 조용히 반환한다.
    """
    from sqlalchemy import inspect

    insp = inspect(engine)
    try:
        existing = {col["name"] for col in insp.get_columns("new_highs")}
    except Exception:
        # 테이블이 아직 없음 — create_all이 컬럼까지 함께 만든다.
        return
    to_add = [
        ("days_since_prev_new_high", "INTEGER"),
        ("days_since_price_above", "INTEGER"),
        ("history_span_days", "INTEGER"),
        ("change_pct", "FLOAT"),
    ]
    with engine.begin() as conn:
        for name, sqltype in to_add:
            if name not in existing:
                conn.execute(text(f"ALTER TABLE new_highs ADD COLUMN {name} {sqltype}"))
        # 레거시 행 보정: 당일 등락률이 breakout_pct에 들어가 있던 시절의 데이터
        conn.execute(text(
            "UPDATE new_highs SET change_pct = breakout_pct, breakout_pct = 0 "
            "WHERE change_pct IS NULL"
        ))


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
    days_since_prev_new_high = Column(Integer, nullable=True)
    days_since_price_above = Column(Integer, nullable=True)
    history_span_days = Column(Integer, nullable=True)
    change_pct = Column(Float, nullable=True)


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
        _migrate_add_recency_columns(self.engine)

    def save_scan_result(self, result: ScanResult) -> None:
        with Session(self.engine) as session:
            # Delete existing data for the same date
            session.execute(
                delete(DailyScan).where(DailyScan.scan_date == result.scan_date)
            )
            session.execute(
                delete(NewHigh).where(NewHigh.scan_date == result.scan_date)
            )
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
                    days_since_prev_new_high=stock.days_since_prev_new_high,
                    days_since_price_above=stock.days_since_price_above,
                    history_span_days=stock.history_span_days,
                    change_pct=stock.change_pct,
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
            # Delete existing analysis for the same date + ticker
            session.execute(
                delete(AIAnalysis).where(
                    AIAnalysis.scan_date == scan_date,
                    AIAnalysis.ticker == analysis.ticker,
                )
            )
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

    def get_ai_analyzed_tickers(self, scan_date: date) -> set[str]:
        """해당 날짜에 이미 AI 분석 완료된 티커 set 반환."""
        with Session(self.engine) as session:
            rows = session.execute(
                select(AIAnalysis.ticker).where(AIAnalysis.scan_date == scan_date)
            ).scalars().all()
            return set(rows)

    def get_all_ai_analyses(self, scan_date: date) -> list[AIAnalysisResult]:
        """해당 날짜의 모든 AI 분석 결과를 AIAnalysisResult로 반환."""
        with Session(self.engine) as session:
            rows = session.execute(
                select(AIAnalysis).where(AIAnalysis.scan_date == scan_date)
            ).scalars().all()
            return [
                AIAnalysisResult(
                    ticker=r.ticker,
                    news_summary=r.news_summary,
                    ai_analysis=r.ai_analysis,
                    news_links=r.news_links.split("\n") if r.news_links else [],
                )
                for r in rows
            ]

    def get_scan_result_full(self, scan_date: date) -> ScanResult | None:
        """DB에서 ScanResult 객체를 복원. 없으면 None."""
        with Session(self.engine) as session:
            scan = session.execute(
                select(DailyScan).where(DailyScan.scan_date == scan_date)
            ).scalar_one_or_none()
            if scan is None:
                return None

            rows = session.execute(
                select(NewHigh).where(NewHigh.scan_date == scan_date)
            ).scalars().all()

            from src.models import StockHigh, MarketStats
            highs = [
                StockHigh(
                    ticker=r.ticker, name=r.name, market=r.market,
                    sector=r.sector, close_price=r.close_price,
                    high_52w=r.high_52w, prev_high_52w=r.prev_high_52w,
                    breakout_pct=r.breakout_pct, volume=r.volume,
                    avg_volume_20d=r.avg_volume_20d,
                    days_since_prev_new_high=r.days_since_prev_new_high,
                    days_since_price_above=r.days_since_price_above,
                    history_span_days=r.history_span_days,
                    change_pct=r.change_pct or 0.0,
                )
                for r in rows
            ]

            sector_breakdown: dict[str, list] = {}
            for h in highs:
                sector_breakdown.setdefault(h.sector, []).append(h)

            kospi = sum(1 for h in highs if h.market == "KOSPI")
            kosdaq = sum(1 for h in highs if h.market == "KOSDAQ")
            etf = sum(1 for h in highs if h.market == "ETF")

            return ScanResult(
                scan_date=scan_date,
                stats=MarketStats(
                    total_stocks=scan.total_stocks,
                    new_high_count=scan.new_high_count,
                    kospi_count=kospi,
                    kosdaq_count=kosdaq,
                    etf_count=etf,
                ),
                highs=highs,
                sector_breakdown=sector_breakdown,
            )

    def delete_ai_analyses(self, scan_date: date) -> None:
        """해당 날짜의 AI 분석 결과 전체 삭제."""
        with Session(self.engine) as session:
            session.execute(
                delete(AIAnalysis).where(AIAnalysis.scan_date == scan_date)
            )
            session.commit()
