from __future__ import annotations

import json
import re
from dataclasses import dataclass
from loguru import logger

from src.models import StockHigh


class InvestingFetchError(RuntimeError):
    """investing.com 취득 실패(Cloudflare 챌린지/403 등)."""


class InvestingParseError(RuntimeError):
    """investing.com 페이지 구조 파싱 실패."""


@dataclass
class InvestingHighRow:
    name: str
    ticker: str
    last_price: float
    change_pct: float
    volume: int


def filter_tradeable(rows: list[InvestingHighRow]) -> list[InvestingHighRow]:
    """거래량 없는(0) 종목 제외 — 정지·유동성 없는 종목 제거."""
    return [r for r in rows if r.volume > 0]


_NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)


def parse_high_rows(html: str) -> tuple[list[InvestingHighRow], int | None]:
    """investing 신고가 페이지의 __NEXT_DATA__ JSON → (행 목록, total).

    구조: props.pageProps.state.assetsCollectionStore.assetsCollection._collection
    스크립트가 없거나 JSON이 깨졌거나 _collection 경로가 없으면 InvestingParseError.
    """
    m = _NEXT_DATA_RE.search(html)
    if not m:
        raise InvestingParseError("__NEXT_DATA__ 스크립트를 찾지 못함(Cloudflare 챌린지 또는 구조 변경)")
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        raise InvestingParseError(f"__NEXT_DATA__ JSON 파싱 실패: {e}") from e

    try:
        collection = (
            data["props"]["pageProps"]["state"]["assetsCollectionStore"]
            ["assetsCollection"]["_collection"]
        )
    except (KeyError, TypeError) as e:
        raise InvestingParseError(f"_collection 경로를 찾지 못함: {e}") from e

    rows: list[InvestingHighRow] = []
    for item in collection:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        ticker = item.get("symbol")
        if not name or not ticker:
            continue
        rows.append(InvestingHighRow(
            name=name,
            ticker=ticker,
            last_price=float(item.get("last") or 0),
            change_pct=float(item.get("changeOneDayPercent") or 0),
            volume=int(item.get("volumeOneDay") or 0),
        ))
    if not rows:
        raise InvestingParseError("_collection은 있으나 유효한 데이터 행이 없음")

    total: int | None = None
    m_total = re.search(r'"total"\s*:\s*(\d+)', html)
    if m_total:
        total = int(m_total.group(1))
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


def resolve_to_krx(
    rows: list[InvestingHighRow],
    ticker_to_market: dict[str, str],
) -> tuple[list[tuple[InvestingHighRow, str, str]], list[str]]:
    """행의 symbol(ticker)로 직접 KRX 유니버스와 매칭 — 이름 정규화 불필요."""
    matched: list[tuple[InvestingHighRow, str, str]] = []
    unmatched: list[str] = []
    for row in rows:
        market = ticker_to_market.get(row.ticker)
        if market is None:
            unmatched.append(row.name)
            continue
        matched.append((row, row.ticker, market))
    if unmatched:
        logger.warning(f"investing 미매칭 {len(unmatched)}종목: {unmatched[:20]}")
    return matched, unmatched


def build_highs(
    matched: list[tuple[InvestingHighRow, str, str]],
    market_caps: dict[str, int],
    sector_map: dict[str, str],
) -> list[StockHigh]:
    highs: list[StockHigh] = []
    for row, ticker, market in matched:
        highs.append(StockHigh(
            ticker=ticker,
            name=row.name,
            market=market,
            sector=sector_map.get(ticker, "기타"),
            close_price=row.last_price,
            high_52w=row.last_price,
            prev_high_52w=0.0,
            breakout_pct=row.change_pct,
            volume=row.volume,
            avg_volume_20d=0,
        ))
    return highs


def collect_investing_highs(date_str, collector, corps, _get=None):
    """investing 신고가 → 거래량 필터 → KRX 매핑 → 시총/섹터 보강 → StockHigh 목록."""
    rows, _total = fetch_52w_high_rows(_get=_get)
    rows = filter_tradeable(rows)
    ticker_to_market = {c.ticker: c.market for c in corps}
    matched, _unmatched = resolve_to_krx(rows, ticker_to_market)
    market_caps = collector.get_market_caps(date_str)
    sector_map: dict[str, str] = {}
    for m in ("KOSPI", "KOSDAQ"):
        sector_map.update(collector.get_sector_map(date_str, m))
    highs = build_highs(matched, market_caps, sector_map)
    return highs, market_caps
