from __future__ import annotations

from dataclasses import dataclass


class InvestingFetchError(RuntimeError):
    """investing.com 취득 실패(Cloudflare 챌린지/403 등)."""


class InvestingParseError(RuntimeError):
    """investing.com 페이지 구조 파싱 실패."""


@dataclass
class InvestingHighRow:
    name: str
    last_price: float
    change_pct: float
    volume: int


def _parse_volume(text: str) -> int:
    """'2.07M'/'617.58K'/'1,234'/''/'-' → int (없으면 0)."""
    if not text:
        return 0
    t = text.strip().upper().replace(",", "")
    if t in ("", "-", "N/A"):
        return 0
    mult = 1
    if t.endswith("K"):
        mult, t = 1_000, t[:-1]
    elif t.endswith("M"):
        mult, t = 1_000_000, t[:-1]
    elif t.endswith("B"):
        mult, t = 1_000_000_000, t[:-1]
    try:
        return int(round(float(t) * mult))
    except ValueError:
        return 0


def filter_tradeable(rows: list[InvestingHighRow]) -> list[InvestingHighRow]:
    """거래량 없는(0) 종목 제외 — 정지·유동성 없는 종목 제거."""
    return [r for r in rows if r.volume > 0]
