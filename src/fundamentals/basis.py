"""연결(CFS)/별도(OFS) 재무제표 기준 선택.

DART는 같은 계정을 연결·별도 두 벌로 내려줍니다. 기준을 섞으면 매출은 연결,
영업이익은 별도인 "키메라" 실적이 만들어지므로, 계산 전에 한 기준으로 통일합니다.
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

    Returns:
        (필터된 statements, 기준). 기준은 CFS / OFS / MIXED / UNKNOWN.
        MIXED 는 일부 연도가 1차 기준에 없어 다른 기준으로 메워졌다는 뜻입니다.
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
    selected = dict(primary_rows)
    fell_back = False
    for s in statements:
        if s.fs_div != secondary:
            continue
        key = (s.year, s.quarter, s.account)
        if key not in selected:
            selected[key] = s
            fell_back = True

    basis = MIXED if fell_back else primary
    return list(selected.values()), basis


def select_basis(statements: list[FinancialStatement]) -> str:
    """filter_to_basis 와 동일한 규칙으로 기준만 반환합니다."""
    return filter_to_basis(statements)[1]
