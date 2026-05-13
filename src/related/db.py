from __future__ import annotations

from collections import Counter
from datetime import datetime
from dataclasses import dataclass

from sqlalchemy import (
    Column, Integer, String, Text, DateTime,
    create_engine, delete, select,
)
from sqlalchemy.orm import DeclarativeBase, Session

from src.related.models import ExtractedEdge


class RelatedBase(DeclarativeBase):
    pass


class RelatedReportMetaRow(RelatedBase):
    __tablename__ = "related_report_meta"
    ticker = Column(String(10), primary_key=True)
    rcept_no = Column(String(20), nullable=False)
    extracted_at = Column(DateTime, nullable=False)


class RelatedEdgeRow(RelatedBase):
    __tablename__ = "related_edges"
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_ticker = Column(String(10), nullable=False, index=True)
    target_ticker = Column(String(10), nullable=True, index=True)
    target_name = Column(String(100), nullable=False)
    relation = Column(String(20), nullable=False)
    evidence = Column(Text, nullable=False)
    extracted_at = Column(DateTime, nullable=False)


@dataclass
class StoredEdge:
    source_ticker: str
    target_ticker: str | None
    target_name: str
    relation: str
    evidence: str
    extracted_at: datetime


@dataclass
class ReportMeta:
    ticker: str
    rcept_no: str
    extracted_at: datetime


class RelatedDB:
    """SQLite persistence for related-company edges and report meta."""

    def __init__(self, url: str = "sqlite:///data/scanner.db"):
        self.engine = create_engine(url)
        RelatedBase.metadata.create_all(self.engine)

    def get_meta(self, ticker: str) -> ReportMeta | None:
        with Session(self.engine) as s:
            row = s.execute(
                select(RelatedReportMetaRow).where(RelatedReportMetaRow.ticker == ticker)
            ).scalar_one_or_none()
            if row is None:
                return None
            return ReportMeta(ticker=row.ticker, rcept_no=row.rcept_no, extracted_at=row.extracted_at)

    def set_meta(self, ticker: str, rcept_no: str, extracted_at: datetime) -> None:
        with Session(self.engine) as s:
            s.execute(delete(RelatedReportMetaRow).where(RelatedReportMetaRow.ticker == ticker))
            s.add(RelatedReportMetaRow(ticker=ticker, rcept_no=rcept_no, extracted_at=extracted_at))
            s.commit()

    def needs_refresh(self, ticker: str, current_rcept_no: str) -> bool:
        meta = self.get_meta(ticker)
        if meta is None:
            return True
        return meta.rcept_no != current_rcept_no

    def save_edges(
        self, source_ticker: str, edges: list[ExtractedEdge], extracted_at: datetime,
    ) -> None:
        with Session(self.engine) as s:
            s.execute(delete(RelatedEdgeRow).where(RelatedEdgeRow.source_ticker == source_ticker))
            for e in edges:
                s.add(RelatedEdgeRow(
                    source_ticker=source_ticker,
                    target_ticker=e.ticker,
                    target_name=e.name,
                    relation=e.relation,
                    evidence=e.evidence,
                    extracted_at=extracted_at,
                ))
            s.commit()

    def load_edges(self, source_tickers: list[str] | None = None) -> list[StoredEdge]:
        with Session(self.engine) as s:
            stmt = select(RelatedEdgeRow)
            if source_tickers is not None:
                stmt = stmt.where(RelatedEdgeRow.source_ticker.in_(source_tickers))
            rows = s.execute(stmt).scalars().all()
            return [
                StoredEdge(
                    source_ticker=r.source_ticker,
                    target_ticker=r.target_ticker,
                    target_name=r.target_name,
                    relation=r.relation,
                    evidence=r.evidence,
                    extracted_at=r.extracted_at,
                )
                for r in rows
            ]

    def stats(self) -> dict:
        with Session(self.engine) as s:
            rows = s.execute(select(RelatedEdgeRow)).scalars().all()
            counter: Counter = Counter(r.relation for r in rows)
            sources = len({r.source_ticker for r in rows})
            return {
                "sources": sources,
                "edges": len(rows),
                "by_relation": dict(counter),
            }
