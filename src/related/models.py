from typing import Literal
from pydantic import BaseModel


RelationType = Literal["Supplier", "Customer", "Competitor", "Affiliate", "Subsidiary"]


class ReportSections(BaseModel):
    """Parsed sections of a DART business report (사업보고서)."""

    corp_code: str
    rcept_no: str                # 접수번호 — cache invalidation key
    business_content: str        # II. 사업의 내용
    affiliates: str              # IX. 계열회사 등에 관한 사항
    related_party: str           # X. 대주주 등과의 거래내용
    related_party_notes: str     # 재무제표 주석 — 특수관계자 거래


class ExtractedEdge(BaseModel):
    """A single relationship extracted by GPT."""

    name: str                    # company name as written in report
    ticker: str | None           # null = unlisted / foreign / unresolved
    relation: RelationType
    evidence: str                # quoted excerpt from report
