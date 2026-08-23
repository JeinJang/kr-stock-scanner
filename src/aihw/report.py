"""src/aihw/report.py

AI HW/빅테크 지표 산출물: 텔레그램 캡션, HTML 리포트, 공유용 PNG.
"""
from __future__ import annotations

from src.aihw.models import AihwSummary

CAPTION_LIMIT = 1024


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
