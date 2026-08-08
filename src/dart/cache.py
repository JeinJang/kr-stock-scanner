from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, DateTime,
    create_engine, delete, select, inspect, text,
)
from sqlalchemy.orm import DeclarativeBase, Session

from src.dart.models import CorpInfo, FinancialStatement


class DartBase(DeclarativeBase):
    pass


class DartCorpInfoRow(DartBase):
    __tablename__ = "dart_corp_info"
    corp_code = Column(String(8), primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    market = Column(String(10), nullable=False)


class DartFinancialRow(DartBase):
    __tablename__ = "dart_financials"
    id = Column(Integer, primary_key=True, autoincrement=True)
    corp_code = Column(String(8), nullable=False, index=True)
    year = Column(Integer, nullable=False)
    quarter = Column(Integer, nullable=False)
    account = Column(String(50), nullable=False)
    value = Column(Float, nullable=False)
    fs_div = Column(String(3), nullable=True, index=True)  # "CFS" | "OFS" | None


class DartMetaRow(DartBase):
    __tablename__ = "dart_meta"
    key = Column(String(50), primary_key=True)
    value = Column(String(100), nullable=False)


class DartCache:
    """SQLite-based cache for Open DART data."""

    def __init__(self, url: str = "sqlite:///data/scanner.db"):
        self.engine = create_engine(url)
        DartBase.metadata.create_all(self.engine)
        self._migrate()

    def _migrate(self) -> None:
        """기존 dart_financials 테이블에 fs_div 컬럼을 더합니다(없을 때만)."""
        insp = inspect(self.engine)
        if "dart_financials" not in insp.get_table_names():
            return
        cols = {c["name"] for c in insp.get_columns("dart_financials")}
        if "fs_div" not in cols:
            with self.engine.begin() as conn:
                conn.execute(text("ALTER TABLE dart_financials ADD COLUMN fs_div VARCHAR(3)"))

    def save_corp_info(self, corps: list[CorpInfo]) -> None:
        with Session(self.engine) as session:
            session.execute(delete(DartCorpInfoRow))
            for c in corps:
                session.add(DartCorpInfoRow(
                    corp_code=c.corp_code, ticker=c.ticker,
                    name=c.name, market=c.market,
                ))
            session.commit()

    def load_corp_info(self, markets: list[str] | None = None) -> list[CorpInfo]:
        with Session(self.engine) as session:
            stmt = select(DartCorpInfoRow)
            if markets:
                stmt = stmt.where(DartCorpInfoRow.market.in_(markets))
            rows = session.execute(stmt).scalars().all()
            return [
                CorpInfo(corp_code=r.corp_code, ticker=r.ticker, name=r.name, market=r.market)
                for r in rows
            ]

    def save_financials(self, statements: list[FinancialStatement]) -> None:
        with Session(self.engine) as session:
            corp_codes = {s.corp_code for s in statements}
            for cc in corp_codes:
                session.execute(
                    delete(DartFinancialRow).where(DartFinancialRow.corp_code == cc)
                )
            for s in statements:
                session.add(DartFinancialRow(
                    corp_code=s.corp_code, year=s.year, quarter=s.quarter,
                    account=s.account, value=s.value, fs_div=s.fs_div,
                ))
            session.commit()

    def load_financials(
        self, corp_codes: list[str] | None = None,
    ) -> list[FinancialStatement]:
        with Session(self.engine) as session:
            stmt = select(DartFinancialRow)
            if corp_codes:
                stmt = stmt.where(DartFinancialRow.corp_code.in_(corp_codes))
            rows = session.execute(stmt).scalars().all()
            return [
                FinancialStatement(
                    corp_code=r.corp_code, year=r.year, quarter=r.quarter,
                    account=r.account, value=r.value, fs_div=r.fs_div,
                )
                for r in rows
            ]

    def last_updated(self) -> datetime | None:
        with Session(self.engine) as session:
            row = session.execute(
                select(DartMetaRow).where(DartMetaRow.key == "last_updated")
            ).scalar_one_or_none()
            if row is None:
                return None
            return datetime.fromisoformat(row.value)

    def set_last_updated(self, ts: datetime) -> None:
        with Session(self.engine) as session:
            session.execute(delete(DartMetaRow).where(DartMetaRow.key == "last_updated"))
            session.add(DartMetaRow(key="last_updated", value=ts.isoformat()))
            session.commit()
