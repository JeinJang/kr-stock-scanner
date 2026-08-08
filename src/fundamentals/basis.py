"""연결(CFS)/별도(OFS) 재무제표 기준 선택.

DART는 같은 계정을 연결·별도 두 벌로 내려줍니다. 기준을 섞으면 매출은 연결,
영업이익은 별도인 "키메라" 실적이 만들어지므로, 계산 전에 한 기준으로 통일합니다.

폴백은 **(year, quarter) 단위**로만 일어납니다. 1차 기준(연결 우선)에 그
연도·분기 행이 하나도 없을 때만 그 연도·분기 전체를 2차 기준(별도)으로
채웁니다. 1차 기준에 그 연도·분기 행이 하나라도 있으면, 없는 계정은 다른
기준에서 가져오지 않고 그냥 비웁니다 — 계정별로 다른 재무제표를 섞으면
어느 재무제표에도 없는 수치(예: 연결 영업이익 ÷ 별도 매출액)가 나오기
때문입니다.
"""

from __future__ import annotations

from src.dart.models import FinancialStatement

CONSOLIDATED = "CFS"
SEPARATE = "OFS"
MIXED = "MIXED"
UNKNOWN = "UNKNOWN"


def filter_to_basis(
    statements: list[FinancialStatement],
) -> tuple[list[FinancialStatement], str]:
    """연결 우선으로 단일 기준을 골라 필터합니다.

    폴백은 (year, quarter) 단위입니다. 1차 기준에 그 연도·분기 행이 하나도
    없을 때만 2차 기준으로 그 연도·분기 전체를 채웁니다. 1차 기준에 행이
    있는 연도·분기라면, 없는 계정은 채우지 않고 비웁니다.

    Returns:
        (필터된 statements, 기준). 기준은 CFS / OFS / MIXED / UNKNOWN.
        MIXED 는 일부 (연도, 분기)가 1차 기준에 통째로 없어 다른 기준으로
        폴백됐다는 뜻입니다.
    """
    if not statements:
        return [], UNKNOWN

    if all(s.fs_div is None for s in statements):
        return list(statements), UNKNOWN

    has_cfs = any(s.fs_div == CONSOLIDATED for s in statements)
    primary = CONSOLIDATED if has_cfs else SEPARATE
    secondary = SEPARATE if primary == CONSOLIDATED else CONSOLIDATED

    primary_rows = {
        (s.year, s.quarter, s.account): s
        for s in statements
        if s.fs_div == primary
    }
    year_quarters_with_primary = {(y, q) for (y, q, _account) in primary_rows}

    secondary_rows = {
        (s.year, s.quarter, s.account): s
        for s in statements
        if s.fs_div == secondary and (s.year, s.quarter) not in year_quarters_with_primary
    }

    fell_back = bool(secondary_rows)
    basis = MIXED if fell_back else primary
    selected = list(primary_rows.values()) + list(secondary_rows.values())
    return selected, basis


def select_basis(statements: list[FinancialStatement]) -> str:
    """filter_to_basis 와 동일한 규칙으로 기준만 반환합니다."""
    return filter_to_basis(statements)[1]
