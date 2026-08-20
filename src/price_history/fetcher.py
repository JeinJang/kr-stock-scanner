"""KRX Open API 일별매매정보 취득.

일자별 전 종목 벌크만 쓴다. 종목별 기간 조회는 이 API에 없고, 로그인
클라이언트 쪽은 2년 상한이 있어 쓰지 않는다.
"""
from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import requests

BASE_URL = "https://data-dbg.krx.co.kr/svc/apis/sto"
MARKET_ENDPOINTS = {"KOSPI": "stk_bydd_trd", "KOSDAQ": "ksq_bydd_trd"}


class KrxApiError(RuntimeError):
    """KRX Open API 인증 실패 또는 비정상 응답."""


def _default_get(url, params, headers, timeout):
    return requests.get(url, params=params, headers=headers, timeout=timeout)


def _num(v) -> int:
    s = str(v or "").replace(",", "")
    try:
        return int(float(s))
    except ValueError:
        return 0


def parse_rows(payload: dict) -> list[tuple]:
    """OutBlock_1 -> [(ticker, high, close, chg)]. 티커 없는 행은 버린다."""
    out: list[tuple] = []
    for item in (payload or {}).get("OutBlock_1", []):
        ticker = item.get("ISU_CD", "")
        if not ticker:
            continue
        out.append((
            ticker,
            _num(item.get("TDD_HGPRC")),
            _num(item.get("TDD_CLSPRC")),
            _num(item.get("CMPPREVDD_PRC")),
        ))
    return out


def fetch_day(api_key: str, market: str, d: str, _get=None) -> list[tuple]:
    """하루치 전 종목. 휴장일은 빈 리스트."""
    get = _get or _default_get
    resp = get(
        f"{BASE_URL}/{MARKET_ENDPOINTS[market]}",
        params={"basDd": d},
        headers={"AUTH_KEY": api_key},
        timeout=60,
    )
    if resp.status_code == 401:
        raise KrxApiError(
            "KRX Open API 인증 실패(401). KRX_API_KEY와 "
            "openapi.krx.co.kr의 서비스 이용 신청 상태를 확인하세요."
        )
    if resp.status_code != 200:
        raise KrxApiError(f"KRX Open API 응답 이상: status={resp.status_code} ({market} {d})")
    return parse_rows(resp.json())


def fetch_many(
    api_key: str,
    jobs: list[tuple[str, str]],
    workers: int = 8,
    _get=None,
) -> Iterator[tuple[str, str, list[tuple]]]:
    """jobs = [(market, d)]. 완료 순서와 무관하게 (market, d, rows)를 흘려보낸다.

    workers를 올리지 않는다 — 과다요청은 이 프로젝트가 차단당한 원인이다.
    """
    def one(job):
        market, d = job
        return market, d, fetch_day(api_key, market, d, _get=_get)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        yield from ex.map(one, jobs)
