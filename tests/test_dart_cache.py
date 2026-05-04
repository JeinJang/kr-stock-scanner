from datetime import datetime
import pytest

from src.dart.cache import DartCache
from src.dart.models import CorpInfo, FinancialStatement


@pytest.fixture
def cache(tmp_path):
    db_url = f"sqlite:///{tmp_path}/test.db"
    return DartCache(url=db_url)


def test_save_and_load_corp_info(cache):
    corps = [
        CorpInfo(corp_code="00000001", ticker="005930", name="삼성전자", market="KOSPI"),
        CorpInfo(corp_code="00000002", ticker="000660", name="SK하이닉스", market="KOSPI"),
    ]
    cache.save_corp_info(corps)

    loaded = cache.load_corp_info()
    assert len(loaded) == 2
    assert loaded[0].ticker == "005930"


def test_save_and_load_financials(cache):
    statements = [
        FinancialStatement(corp_code="00000001", year=2025, quarter=0, account="매출액", value=300_000_000_000_000.0),
        FinancialStatement(corp_code="00000001", year=2025, quarter=0, account="영업이익", value=30_000_000_000_000.0),
    ]
    cache.save_financials(statements)

    loaded = cache.load_financials(corp_codes=["00000001"])
    assert len(loaded) == 2
    accounts = {s.account for s in loaded}
    assert "매출액" in accounts
    assert "영업이익" in accounts


def test_meta_last_updated(cache):
    assert cache.last_updated() is None
    cache.set_last_updated(datetime(2026, 4, 1, 12, 0))
    ts = cache.last_updated()
    assert ts is not None
    assert ts.year == 2026
