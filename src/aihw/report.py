"""src/aihw/report.py

AI HW/빅테크 지표 산출물: 텔레그램 캡션, HTML 리포트, 공유용 PNG.
"""
from __future__ import annotations

import os
from pathlib import Path

import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader
from loguru import logger

from src.aihw.models import AihwSeries, AihwSummary

CAPTION_LIMIT = 1024
GROUP_COLORS = {"AI HW": "#f5a623", "빅테크": "#7b61c4", "SPY": "#4a90d9", "RSP": "#3d9970"}
# 절대 시총 차트에서 개별 기업 선 색상 — 그룹별 색 계열 (순환 사용)
COMPANY_PALETTES = {
    "AI HW": ["#e8590c", "#f08c00", "#e67700", "#d9480f", "#c92a2a", "#a61e4d"],
    "빅테크": ["#5f3dc4", "#7048e8", "#9775fa", "#4263eb", "#3b5bdb", "#364fc7"],
}


def _fmt_t(cap_usd: float) -> str:
    return f"${cap_usd / 1e12:.2f}T"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "(-)"
    return f"({value:+.1f}%)"


def build_caption(summary: AihwSummary) -> str:
    warn = summary.status in ("above", "cross_up")
    head = "⚠️ " if warn else "📊 "
    lines = [
        f"{head}AI HW / 빅테크 비율: {summary.ratio * 100:.1f}% "
        f"(경고선 {summary.threshold * 100:.0f}%)"
    ]
    if summary.status == "cross_up":
        lines.append(f"🚨 경고선 {summary.threshold * 100:.0f}% 상향 돌파")
    elif summary.status == "cross_down":
        lines.append(f"경고선 {summary.threshold * 100:.0f}% 하향 이탈")

    parts = []
    if summary.change_pp is not None:
        parts.append(f"전일 대비 {summary.change_pp:+.1f}%p")
    parts.append(f"30일 최고 {summary.high_30d * 100:.1f}%")
    lines.append(" · ".join(parts))

    for group in summary.groups:
        lines.append("")
        lines.append(f"[{group.name}] {_fmt_t(group.total_usd)}")
        for c in group.companies:
            lines.append(f"· {c.name} {_fmt_t(c.cap_usd)} {_fmt_pct(c.day_change_pct)}")

    caption = "\n".join(lines)
    if len(caption) > CAPTION_LIMIT:
        caption = caption[: CAPTION_LIMIT - 1] + "…"
    return caption


def build_figures(series: AihwSeries, threshold: float) -> tuple[go.Figure, go.Figure]:
    ratio_fig = go.Figure()
    ratio_fig.add_trace(go.Scatter(
        x=series.dates, y=[r * 100 for r in series.ratio],
        mode="lines", name="AI HW / 빅테크",
        line=dict(color=GROUP_COLORS["AI HW"], width=2),
    ))
    ratio_fig.add_hline(
        y=threshold * 100, line_color="red", line_width=2,
        annotation_text=f"경고선 {threshold * 100:.0f}%",
    )
    ratio_fig.update_layout(
        title="AI HW 시총합 / 빅테크 시총합 비율 (%)",
        yaxis_title="%", template="plotly_white", height=420,
    )

    index_fig = go.Figure()
    for name, values in series.indexed.items():
        index_fig.add_trace(go.Scatter(
            x=series.dates, y=values, mode="lines", name=name,
            line=dict(color=GROUP_COLORS.get(name), width=2),
        ))
    base = series.dates[0].isoformat()
    index_fig.update_layout(
        title=f"시총 지수 비교 ({base} = 100)",
        template="plotly_white", height=420,
    )
    return ratio_fig, index_fig


def build_cap_figure(series: AihwSeries, names: dict[str, str] | None = None) -> go.Figure:
    """절대 시가총액($T) 차트: 그룹 합계(굵은 선) + 개별 기업(얇은 선)."""
    names = names or {}
    fig = go.Figure()
    for label, totals in (
        ("AI HW 합계", series.ai_hw_total),
        ("빅테크 합계", series.big_tech_total),
    ):
        group = label.replace(" 합계", "")
        fig.add_trace(go.Scatter(
            x=series.dates, y=[v / 1e12 for v in totals],
            mode="lines", name=label,
            line=dict(color=GROUP_COLORS[group], width=3),
        ))
    for group, companies in series.company_caps.items():
        palette = COMPANY_PALETTES.get(group, [])
        for i, (ticker, caps) in enumerate(companies.items()):
            fig.add_trace(go.Scatter(
                x=series.dates, y=[v / 1e12 for v in caps],
                mode="lines", name=names.get(ticker, ticker),
                line=dict(
                    color=palette[i % len(palette)] if palette else None,
                    width=1.2,
                ),
            ))
    fig.update_layout(
        title="시가총액 추이 (조 달러)",
        yaxis_title="$T", template="plotly_white", height=480,
        legend=dict(orientation="h", yanchor="top", y=-0.15),
    )
    return fig


def generate_html(
    series: AihwSeries,
    summary: AihwSummary,
    output_dir: str,
    names: dict[str, str] | None = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"aihw-{summary.as_of.isoformat()}.html")
    ratio_fig, index_fig = build_figures(series, summary.threshold)
    cap_fig = build_cap_figure(series, names)

    env = Environment(loader=FileSystemLoader(str(Path(__file__).parent / "templates")))
    html = env.get_template("report.html").render(
        summary=summary,
        ratio_pct=f"{summary.ratio * 100:.1f}%",
        ratio_div=ratio_fig.to_html(full_html=False, include_plotlyjs="cdn"),
        cap_div=cap_fig.to_html(full_html=False, include_plotlyjs=False),
        index_div=index_fig.to_html(full_html=False, include_plotlyjs=False),
        fmt_t=_fmt_t,
        fmt_pct=_fmt_pct,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"HTML 리포트 저장: {path}")
    return path


def _build_combined_figure(
    series: AihwSeries,
    summary: AihwSummary,
    names: dict[str, str] | None = None,
) -> go.Figure:
    """PNG용 3단 결합 차트. 각 차트의 범례를 해당 차트 오른쪽에 분리 배치한다."""
    from plotly.subplots import make_subplots

    ratio_fig, index_fig = build_figures(series, summary.threshold)
    cap_fig = build_cap_figure(series, names)

    combined = make_subplots(
        rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.08,
        subplot_titles=(
            "AI HW / 빅테크 시총 비율 (%)",
            "시가총액 추이 (조 달러)",
            f"시총 지수 비교 ({series.dates[0].isoformat()} = 100)",
        ),
    )
    row_legends = [
        (ratio_fig, 1, "legend"),
        (cap_fig, 2, "legend2"),
        (index_fig, 3, "legend3"),
    ]
    for fig, row, legend_id in row_legends:
        for trace in fig.data:
            trace.update(legend=legend_id)
            combined.add_trace(trace, row=row, col=1)
    combined.add_hline(y=summary.threshold * 100, line_color="red", line_width=2, row=1, col=1)

    # 각 행의 y-domain 상단에 해당 범례를 정렬 (3행 균등 분할 + spacing 0.08)
    legend_style = dict(xanchor="left", x=1.02, yanchor="top")
    combined.update_layout(
        template="plotly_white", height=1350, width=1150,
        title=f"AI HW / 빅테크 고점 지표 — {summary.as_of.isoformat()}",
        legend=dict(title_text="비율", y=1.0, **legend_style),
        legend2=dict(title_text="시가총액", y=0.64, **legend_style),
        legend3=dict(title_text="지수 비교", y=0.28, **legend_style),
    )
    return combined


def generate_png(
    series: AihwSeries,
    summary: AihwSummary,
    output_dir: str,
    names: dict[str, str] | None = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"aihw-{summary.as_of.isoformat()}.png")
    combined = _build_combined_figure(series, summary, names)
    combined.write_image(path, scale=2)
    logger.info(f"PNG 저장: {path}")
    return path
