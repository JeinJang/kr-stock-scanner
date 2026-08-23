from datetime import date


from src.aihw.models import AihwSeries, AihwSummary, CompanySummary, GroupSummary
from src.aihw.report import build_caption, build_figures, generate_html


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

    def test_over_limit_truncated_with_ellipsis(self):
        # 1,024자 초과 시 절단 및 줄임표 추가 확인
        companies = [
            CompanySummary(ticker=f"T{i}", name=f"아주긴종목이름{i}", cap_usd=1.0e12, day_change_pct=1.23)
            for i in range(40)
        ]
        s = _summary()
        s.groups[0].companies = companies
        s.groups[1].companies = companies
        caption = build_caption(s)
        assert len(caption) <= 1024
        assert caption.endswith("…")

    def test_cross_down_message(self):
        # cross_down 상태에서 하향 이탈 메시지 표시, 경고 접두 없음
        caption = build_caption(_summary(ratio=0.79, status="cross_down"))
        assert "하향 이탈" in caption
        assert not caption.startswith("⚠️")


def _series():
    return AihwSeries(
        dates=[date(2026, 1, 10), date(2026, 1, 11)],
        ai_hw_total=[6.0e12, 6.8e12],
        big_tech_total=[8.8e12, 8.9e12],
        ratio=[0.682, 0.764],
        indexed={
            "AI HW": [100.0, 113.3],
            "빅테크": [100.0, 101.1],
            "SPY": [100.0, 101.0],
            "RSP": [100.0, 100.4],
        },
    )


class TestBuildFigures:
    def test_ratio_figure_has_threshold_line(self):
        ratio_fig, index_fig = build_figures(_series(), threshold=0.8)
        # 경고선은 hline shape로 추가됨
        assert any(s.type == "line" for s in ratio_fig.layout.shapes)
        assert len(ratio_fig.data) == 1  # 비율 트레이스 1개

    def test_index_figure_has_all_series(self):
        _, index_fig = build_figures(_series(), threshold=0.8)
        names = {t.name for t in index_fig.data}
        assert names == {"AI HW", "빅테크", "SPY", "RSP"}


class TestGenerateHtml:
    def test_writes_file_with_table(self, tmp_path):
        path = generate_html(_series(), _summary(), output_dir=str(tmp_path))
        assert path.endswith("aihw-2026-08-22.html")
        html = open(path, encoding="utf-8").read()
        assert "엔비디아" in html
        assert "76.2%" in html
