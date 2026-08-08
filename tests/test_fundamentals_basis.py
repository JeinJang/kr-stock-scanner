from src.dart.models import FinancialStatement
from src.fundamentals.basis import filter_to_basis, select_basis


def _s(year, account, value, fs_div, quarter=0):
    return FinancialStatement(corp_code="X", year=year, quarter=quarter,
                              account=account, value=value, fs_div=fs_div)


def test_prefers_consolidated_when_both_present():
    """LG화학 2025 실측: 연결 영업이익 11,809억 / 별도 -2,105억."""
    stmts = [
        _s(2025, "영업이익", 1180900000000.0, "CFS"),
        _s(2025, "영업이익", -210500000000.0, "OFS"),
        _s(2025, "당기순이익", -977100000000.0, "CFS"),
        _s(2025, "당기순이익", 1368000000000.0, "OFS"),
    ]
    out, basis = filter_to_basis(stmts)
    assert basis == "CFS"
    assert {s.account: s.value for s in out} == {
        "영업이익": 1180900000000.0,
        "당기순이익": -977100000000.0,
    }


def test_falls_back_to_separate_when_no_consolidated():
    """지주사·비연결 기업은 별도만 존재합니다."""
    stmts = [_s(2025, "영업이익", 100.0, "OFS")]
    out, basis = filter_to_basis(stmts)
    assert basis == "OFS"
    assert len(out) == 1


def test_marks_mixed_when_a_year_lacks_the_primary_basis():
    """2024년에 연결이 없으면 그 해만 별도로 메우고 MIXED로 표시합니다."""
    stmts = [
        _s(2025, "영업이익", 300.0, "CFS"),
        _s(2024, "영업이익", 200.0, "OFS"),
    ]
    out, basis = filter_to_basis(stmts)
    assert basis == "MIXED"
    assert {s.year: s.value for s in out} == {2025: 300.0, 2024: 200.0}


def test_unknown_when_fs_div_missing():
    """구버전 데이터(fs_div=None)는 필터하지 않고 UNKNOWN으로 표시합니다."""
    stmts = [_s(2025, "영업이익", 100.0, None)]
    out, basis = filter_to_basis(stmts)
    assert basis == "UNKNOWN"
    assert len(out) == 1


def test_select_basis_matches_filter_to_basis():
    stmts = [_s(2025, "영업이익", 1.0, "CFS"), _s(2025, "영업이익", 2.0, "OFS")]
    assert select_basis(stmts) == "CFS"
