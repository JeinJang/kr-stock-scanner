from src.dart.models import CorpInfo, FinancialStatement


def test_corp_info_creation():
    corp = CorpInfo(
        corp_code="00126380",
        ticker="005930",
        name="삼성전자",
        market="KOSPI",
    )
    assert corp.corp_code == "00126380"
    assert corp.ticker == "005930"
    assert corp.market == "KOSPI"


def test_financial_statement_creation():
    fs = FinancialStatement(
        corp_code="00126380",
        year=2025,
        quarter=0,
        account="매출액",
        value=300_000_000_000_000.0,
    )
    assert fs.year == 2025
    assert fs.quarter == 0  # annual report
    assert fs.account == "매출액"


def test_financial_statement_has_fs_div():
    s = FinancialStatement(
        corp_code="00356361", year=2025, quarter=0,
        account="영업이익", value=1180900000000.0, fs_div="CFS",
    )
    assert s.fs_div == "CFS"


def test_financial_statement_fs_div_defaults_to_none():
    """기존 데이터 호환: fs_div 없이도 생성돼야 합니다."""
    s = FinancialStatement(
        corp_code="00356361", year=2025, quarter=0,
        account="영업이익", value=1.0,
    )
    assert s.fs_div is None
