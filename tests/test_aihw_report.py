from datetime import date

import pytest

from src.aihw.models import AihwSummary, CompanySummary, GroupSummary
from src.aihw.report import build_caption


def _summary(ratio=0.762, status=None):
    return AihwSummary(
        as_of=date(2026, 8, 22),
        ratio=ratio,
        ratio_prev=0.754,
        change_pp=0.8,
        high_30d=0.781,
        low_30d=0.74,
        threshold=0.8,
        status=status,
        groups=[
            GroupSummary(name="AI HW", total_usd=6.82e12, companies=[
                CompanySummary(ticker="NVDA", name="엔비디아", cap_usd=4.21e12, day_change_pct=1.2),
                CompanySummary(ticker="AVGO", name="브로드컴", cap_usd=1.35e12, day_change_pct=-0.5),
            ]),
            GroupSummary(name="빅테크", total_usd=8.95e12, companies=[
                CompanySummary(ticker="MSFT", name="MS", cap_usd=3.12e12, day_change_pct=0.4),
                CompanySummary(ticker="META", name="메타", cap_usd=1.52e12, day_change_pct=None),
            ]),
        ],
    )


class TestBuildCaption:
    def test_header_lines(self):
        caption = build_caption(_summary())
        lines = caption.split("\n")
        assert lines[0] == "📊 AI HW / 빅테크 비율: 76.2% (경고선 80%)"
        assert lines[1] == "전일 대비 +0.8%p · 30일 최고 78.1%"

    def test_group_and_company_lines(self):
        caption = build_caption(_summary())
        assert "[AI HW] $6.82T" in caption
        assert "· 엔비디아 $4.21T (+1.2%)" in caption
        assert "· 브로드컴 $1.35T (-0.5%)" in caption
        assert "[빅테크] $8.95T" in caption
        assert "· 메타 $1.52T (-)" in caption  # 전일 데이터 없음

    def test_warning_when_at_or_above_threshold(self):
        caption = build_caption(_summary(ratio=0.81, status="above"))
        assert caption.startswith("⚠️")

    def test_cross_up_marks_warning(self):
        caption = build_caption(_summary(ratio=0.80, status="cross_up"))
        assert caption.startswith("⚠️")
        assert "상향 돌파" in caption

    def test_under_1024_chars(self):
        # 실제 구성(11종목)보다 많은 20종목으로도 한도 이내인지 확인
        companies = [
            CompanySummary(ticker=f"T{i}", name=f"종목이름{i}", cap_usd=1.0e12, day_change_pct=1.23)
            for i in range(10)
        ]
        s = _summary()
        s.groups[0].companies = companies
        s.groups[1].companies = companies
        assert len(build_caption(s)) <= 1024
