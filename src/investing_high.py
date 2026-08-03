from __future__ import annotations

import re
from dataclasses import dataclass
from bs4 import BeautifulSoup
from loguru import logger


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


URL = "https://kr.investing.com/equities/52-week-high"
IMPERSONATE_TARGETS = ("chrome124", "safari17_0")
_HEADERS = {"Accept-Language": "ko-KR,ko;q=0.9"}


def _default_get(url, impersonate, timeout, headers):
    from curl_cffi import requests as cffi
    return cffi.get(url, impersonate=impersonate, timeout=timeout, headers=headers)


def _is_challenge(text: str) -> bool:
    return ("Just a moment" in text) and ("<table" not in text)


def _fetch_html(url: str, targets: tuple[str, ...], _get=None) -> str:
    get = _get or _default_get
    last = ""
    for target in targets:
        try:
            resp = get(url, impersonate=target, timeout=25, headers=_HEADERS)
        except Exception as e:  # noqa: BLE001 — 네트워크 예외는 다음 타깃으로
            last = f"{type(e).__name__}: {e}"
            continue
        text = resp.text
        if resp.status_code == 200 and "<table" in text and not _is_challenge(text):
            return text
        last = f"status={resp.status_code}, challenge={_is_challenge(text)}"
        logger.warning(f"investing 취득 실패(impersonate={target}): {last}")
    raise InvestingFetchError(f"모든 impersonate 타깃 실패: {last}")


def fetch_52w_high_rows(_get=None) -> tuple[list[InvestingHighRow], int | None]:
    html = _fetch_html(URL, IMPERSONATE_TARGETS, _get=_get)
    rows, total = parse_high_rows(html)
    if total is not None and total > len(rows):
        logger.warning(
            f"investing 신고가 커버리지 잘림: total={total} > 취득={len(rows)}. "
            f"페이지네이션 미구현으로 초기 배치만 사용합니다."
        )
    return rows, total
