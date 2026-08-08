"""실제 DB 기준 연결/별도 회귀 테스트.

2026-08-08 72종목 전수 감사에서 실측한 값들입니다.
data/scanner.db 가 없으면 건너뜁니다.
"""

from pathlib import Path

import pytest

from src.dart.cache import DartCache
from src.fundamentals.basis import filter_to_basis

DB = Path("data/scanner.db")
pytestmark = pytest.mark.skipif(not DB.exists(), reason="scanner.db 없음")

# (티커, 연도, 계정, 연결 실제값[억원])
GOLDEN = [
    ("051910", 2025, "당기순이익", -9771),   # 별도는 +13,680 (부호 반전)
    ("051910", 2025, "영업이익", 11809),     # 별도는 -2,105
]


def _corp_code(cache, ticker):
    for c in cache.load_corp_info():
        if c.ticker == ticker:
            return c.corp_code
    pytest.skip(f"{ticker} 없음")


@pytest.mark.parametrize("ticker,year,account,expected_eok", GOLDEN)
def test_consolidated_value_is_selected(ticker, year, account, expected_eok):
    cache = DartCache()
    cc = _corp_code(cache, ticker)
    stmts = cache.load_financials(corp_codes=[cc])
    if all(s.fs_div is None for s in stmts):
        pytest.skip("재적재 전 데이터입니다. `fundamentals.cli refresh` 를 먼저 실행하세요.")

    picked, basis = filter_to_basis(stmts)
    assert basis in ("CFS", "MIXED")
    vals = [s.value for s in picked
            if s.year == year and s.quarter == 0 and s.account == account]
    assert len(vals) == 1, f"{ticker} {year} {account} 행이 {len(vals)}개입니다"
    assert vals[0] / 1e8 == pytest.approx(expected_eok, abs=1.0)


def test_no_duplicate_account_rows_after_basis_filter():
    """기준 필터 후에는 (year, quarter, account) 중복이 없어야 합니다."""
    cache = DartCache()
    cc = _corp_code(cache, "051910")
    picked, _ = filter_to_basis(cache.load_financials(corp_codes=[cc]))
    keys = [(s.year, s.quarter, s.account) for s in picked]
    assert len(keys) == len(set(keys))
