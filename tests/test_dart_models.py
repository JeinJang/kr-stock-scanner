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
