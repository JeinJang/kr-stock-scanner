from __future__ import annotations

import json
from datetime import date

from sqlalchemy import (
    Column, Integer, String, Float, Date,
    create_engine, delete, select, text,
)
from sqlalchemy.orm import DeclarativeBase, Session

from src.fundamentals.models import FundamentalsMetrics, ScoreCard


class FundamentalsBase(DeclarativeBase):
    pass


class MetricsRow(FundamentalsBase):
    __tablename__ = "fundamentals_metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    as_of_date = Column(Date, nullable=False, index=True)
    current_ratio = Column(Float, nullable=True)
    interest_coverage = Column(Float, nullable=True)
    debt_ratio = Column(Float, nullable=True)
    roe = Column(Float, nullable=True)
    roic = Column(Float, nullable=True)
    operating_margin = Column(Float, nullable=True)
    revenue_cagr_3y = Column(Float, nullable=True)
    op_income_cagr_3y = Column(Float, nullable=True)
    ocf_to_ni_ratio = Column(Float, nullable=True)
    fcf_positive_years = Column(Integer, nullable=True)
    pe = Column(Float, nullable=True)
    pb = Column(Float, nullable=True)
    peg = Column(Float, nullable=True)
    eps = Column(Float, nullable=True)
    bps = Column(Float, nullable=True)
    psr = Column(Float, nullable=True)
    ocf = Column(Float, nullable=True)
    fcf = Column(Float, nullable=True)
    capex_to_revenue = Column(Float, nullable=True)
    dividend_yield = Column(Float, nullable=True)
    payout_ratio = Column(Float, nullable=True)
    consecutive_dividend_years = Column(Integer, nullable=True)
    fs_basis = Column(String(8), nullable=True)


class ScoreRow(FundamentalsBase):
    __tablename__ = "fundamentals_scores"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    as_of_date = Column(Date, nullable=False, index=True)
    liquidity_score = Column(Float, nullable=True)
    profitability_score = Column(Float, nullable=True)
    growth_score = Column(Float, nullable=True)
    cashflow_score = Column(Float, nullable=True)
    total_score = Column(Float, nullable=False)
    grade = Column(String(10), nullable=False)
    categories = Column(String(200), nullable=False)  # JSON-encoded list


def _migrate_add_enrichment_columns(engine) -> None:
    """Idempotent ALTER for the 9 enrichment columns. Safe to run repeatedly
    and safe to call even if the table does not exist yet."""
    from sqlalchemy import inspect
    insp = inspect(engine)
    try:
        existing = {col["name"] for col in insp.get_columns("fundamentals_metrics")}
    except Exception:
        # Table doesn't exist yet; create_all will create it with the columns directly.
        return
    to_add = [
        ("eps", "FLOAT"),
        ("bps", "FLOAT"),
        ("psr", "FLOAT"),
        ("ocf", "FLOAT"),
        ("fcf", "FLOAT"),
        ("capex_to_revenue", "FLOAT"),
        ("dividend_yield", "FLOAT"),
        ("payout_ratio", "FLOAT"),
        ("consecutive_dividend_years", "INTEGER"),
        ("fs_basis", "VARCHAR(8)"),
    ]
    with engine.begin() as conn:
        for name, sqltype in to_add:
            if name not in existing:
                conn.execute(text(f"ALTER TABLE fundamentals_metrics ADD COLUMN {name} {sqltype}"))


class FundamentalsDB:
    """Persistence for derived metrics and scores."""

    def __init__(self, url: str = "sqlite:///data/scanner.db"):
        self.engine = create_engine(url)
        FundamentalsBase.metadata.create_all(self.engine)
        _migrate_add_enrichment_columns(self.engine)

    def save_metrics(self, metrics: list[FundamentalsMetrics]) -> None:
        if not metrics:
            return
        with Session(self.engine) as session:
            as_of = metrics[0].as_of_date
            session.execute(delete(MetricsRow).where(MetricsRow.as_of_date == as_of))
            for m in metrics:
                session.add(MetricsRow(
                    ticker=m.ticker, as_of_date=m.as_of_date,
                    current_ratio=m.current_ratio, interest_coverage=m.interest_coverage,
                    debt_ratio=m.debt_ratio, roe=m.roe, roic=m.roic,
                    operating_margin=m.operating_margin,
                    revenue_cagr_3y=m.revenue_cagr_3y, op_income_cagr_3y=m.op_income_cagr_3y,
                    ocf_to_ni_ratio=m.ocf_to_ni_ratio, fcf_positive_years=m.fcf_positive_years,
                    pe=m.pe, pb=m.pb, peg=m.peg,
                    eps=m.eps, bps=m.bps, psr=m.psr,
                    ocf=m.ocf, fcf=m.fcf, capex_to_revenue=m.capex_to_revenue,
                    dividend_yield=m.dividend_yield, payout_ratio=m.payout_ratio,
                    consecutive_dividend_years=m.consecutive_dividend_years,
                    fs_basis=m.fs_basis,
                ))
            session.commit()

    def save_scores(self, scores: list[ScoreCard]) -> None:
        if not scores:
            return
        with Session(self.engine) as session:
            as_of = scores[0].as_of_date
            session.execute(delete(ScoreRow).where(ScoreRow.as_of_date == as_of))
            for s in scores:
                session.add(ScoreRow(
                    ticker=s.ticker, as_of_date=s.as_of_date,
                    liquidity_score=s.liquidity_score,
                    profitability_score=s.profitability_score,
                    growth_score=s.growth_score, cashflow_score=s.cashflow_score,
                    total_score=s.total_score, grade=s.grade,
                    categories=json.dumps(s.categories, ensure_ascii=False),
                ))
            session.commit()

    def load_scores(self, as_of_date: date) -> list[ScoreCard]:
        with Session(self.engine) as session:
            rows = session.execute(
                select(ScoreRow).where(ScoreRow.as_of_date == as_of_date)
            ).scalars().all()
            return [
                ScoreCard(
                    ticker=r.ticker, as_of_date=r.as_of_date,
                    liquidity_score=r.liquidity_score,
                    profitability_score=r.profitability_score,
                    growth_score=r.growth_score, cashflow_score=r.cashflow_score,
                    total_score=r.total_score, grade=r.grade,
                    categories=json.loads(r.categories),
                )
                for r in rows
            ]

    def load_metrics(self, as_of_date: date) -> list[FundamentalsMetrics]:
        with Session(self.engine) as session:
            rows = session.execute(
                select(MetricsRow).where(MetricsRow.as_of_date == as_of_date)
            ).scalars().all()
            return [
                FundamentalsMetrics(
                    ticker=r.ticker, as_of_date=r.as_of_date,
                    current_ratio=r.current_ratio, interest_coverage=r.interest_coverage,
                    debt_ratio=r.debt_ratio, roe=r.roe, roic=r.roic,
                    operating_margin=r.operating_margin,
                    revenue_cagr_3y=r.revenue_cagr_3y,
                    op_income_cagr_3y=r.op_income_cagr_3y,
                    ocf_to_ni_ratio=r.ocf_to_ni_ratio,
                    fcf_positive_years=r.fcf_positive_years,
                    pe=r.pe, pb=r.pb, peg=r.peg,
                    eps=r.eps, bps=r.bps, psr=r.psr,
                    ocf=r.ocf, fcf=r.fcf, capex_to_revenue=r.capex_to_revenue,
                    dividend_yield=r.dividend_yield, payout_ratio=r.payout_ratio,
                    consecutive_dividend_years=r.consecutive_dividend_years,
                    fs_basis=r.fs_basis,
                )
                for r in rows
            ]
