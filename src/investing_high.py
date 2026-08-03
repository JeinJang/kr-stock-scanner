from __future__ import annotations

import re
from dataclasses import dataclass
from bs4 import BeautifulSoup


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


def _clean_num(text: str) -> float:
    t = (text or "").strip().replace(",", "").replace("%", "").replace("+", "")
    try:
        return float(t)
    except ValueError:
        return 0.0


def parse_high_rows(html: str) -> tuple[list[InvestingHighRow], int | None]:
    """investing 신고가 HTML → (행 목록, total). 표 없으면 InvestingParseError."""
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        raise InvestingParseError("데이터 표를 찾지 못함(Cloudflare 챌린지 또는 구조 변경)")
    table = max(tables, key=lambda t: len(t.find_all("tr")))

    rows: list[InvestingHighRow] = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 7:
            continue  # 헤더/빈 행
        name = tds[1].get_text(strip=True)
        if not name:
            continue
        rows.append(InvestingHighRow(
            name=name,
            last_price=_clean_num(tds[2].get_text(strip=True)),
            change_pct=_clean_num(tds[5].get_text(strip=True)),
            volume=_parse_volume(tds[6].get_text(strip=True)),
        ))
    if not rows:
        raise InvestingParseError("표는 있으나 데이터 행이 없음")

    total: int | None = None
    m = re.search(r'"total"\s*:\s*(\d+)', html)
    if m:
        total = int(m.group(1))
    return rows, total
