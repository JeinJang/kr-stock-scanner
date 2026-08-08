from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.dart.fetcher import DartFetcher
from src.dart.models import CorpInfo


@pytest.mark.asyncio
async def test_fetch_corp_universe_filters_listed():
    """Only corps with non-empty stock_code (listed) and target markets are returned."""
    mock_client = MagicMock()
    fetcher = DartFetcher(client=mock_client)

    # Mock the corp universe XML download
    fake_xml_zip = b"fake-zip-bytes"
    fake_corps_xml = """<?xml version="1.0" encoding="UTF-8"?>
<result>
  <list>
    <corp_code>00126380</corp_code>
    <corp_name>삼성전자</corp_name>
    <stock_code>005930</stock_code>
    <modify_date>20250101</modify_date>
  </list>
  <list>
    <corp_code>00264529</corp_code>
    <corp_name>비상장기업</corp_name>
    <stock_code></stock_code>
    <modify_date>20250101</modify_date>
  </list>
</result>"""

    async def mock_download(*args, **kwargs):
        return fake_xml_zip

    market_map = {"005930": "KOSPI"}

    with patch.object(fetcher, "_download_corp_zip", side_effect=mock_download), \
         patch.object(fetcher, "_extract_xml", return_value=fake_corps_xml):
        corps = await fetcher.fetch_corp_universe(
            markets=["KOSPI", "KOSDAQ"], market_map=market_map,
        )

    assert len(corps) == 1
    assert corps[0].ticker == "005930"
    assert corps[0].market == "KOSPI"


@pytest.mark.asyncio
async def test_fetch_financials_batches_by_100():
    """Multi-account API supports up to 100 corp_codes per call."""
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value={
        "status": "000",
        "list": [
            {
                "corp_code": "00000001", "bsns_year": "2025", "reprt_code": "11011",
                "account_nm": "매출액", "thstrm_amount": "1,000,000",
            },
        ],
    })
    fetcher = DartFetcher(client=mock_client)

    corp_codes = [f"{i:08d}" for i in range(1, 251)]  # 250 corps
    statements = await fetcher.fetch_financials(corp_codes, years=[2025], report_codes=["11011"])

    # 250 corps / 100 batch * 1 year * 1 report = 3 calls
    assert mock_client.get.call_count == 3


from src.dart.fetcher import ACCOUNT_NORMALIZE


def test_account_normalize_includes_cashflow_and_dividend():
    # OCF
    assert ACCOUNT_NORMALIZE.get("영업활동 현금흐름") == "영업활동현금흐름"
    assert ACCOUNT_NORMALIZE.get("영업활동으로 인한 현금흐름") == "영업활동현금흐름"
    # CAPEX
    assert ACCOUNT_NORMALIZE.get("유형자산의 취득") == "유형자산취득"
    # Dividend (paid)
    assert ACCOUNT_NORMALIZE.get("배당금지급") == "배당총액"


class _StubClient:
    def __init__(self, payload):
        self._payload = payload

    async def get(self, path, params=None):
        return self._payload


@pytest.mark.asyncio
async def test_fetch_financials_captures_fs_div():
    payload = {
        "status": "000",
        "list": [
            {"corp_code": "X", "fs_div": "CFS", "sj_div": "IS",
             "account_nm": "영업이익", "thstrm_amount": "1,180,900,000,000"},
            {"corp_code": "X", "fs_div": "OFS", "sj_div": "IS",
             "account_nm": "영업이익", "thstrm_amount": "-210,500,000,000"},
        ],
    }
    f = DartFetcher(client=_StubClient(payload))
    out = await f.fetch_financials(["X"], [2025], ["11011"])
    by_div = {s.fs_div: s.value for s in out}
    assert by_div["CFS"] == 1180900000000.0
    assert by_div["OFS"] == -210500000000.0


@pytest.mark.asyncio
async def test_fetch_financials_dedupes_repeated_account_within_same_basis():
    """DART는 당기순이익을 같은 fs_div 안에서 2번 반환합니다. 첫 행만 남겨야 합니다."""
    payload = {
        "status": "000",
        "list": [
            {"corp_code": "X", "fs_div": "CFS", "sj_div": "IS",
             "account_nm": "당기순이익(손실)", "thstrm_amount": "-977,100,000,000"},
            {"corp_code": "X", "fs_div": "CFS", "sj_div": "IS",
             "account_nm": "당기순이익(손실)", "thstrm_amount": "-977,100,000,000"},
        ],
    }
    f = DartFetcher(client=_StubClient(payload))
    out = await f.fetch_financials(["X"], [2025], ["11011"])
    assert len(out) == 1
    assert out[0].fs_div == "CFS"


@pytest.mark.asyncio
async def test_fetch_financials_normalizes_unknown_fs_div_to_none():
    """CFS/OFS 가 아닌 fs_div 값은 조용히 버리지 않고 None 으로 정규화해야 합니다.

    filter_to_basis 는 CFS/OFS 가 아닌 행을 1차·2차 어느 기준에도 매칭하지 않아
    버리므로, 새로운 구분값이 나오면 fs_div 를 None 으로 낮춰 UNKNOWN 경로로
    빠지게 해야 무음 데이터 손실을 막을 수 있습니다.
    """
    payload = {
        "status": "000",
        "list": [
            {"corp_code": "X", "fs_div": "CFS2", "sj_div": "IS",
             "account_nm": "영업이익", "thstrm_amount": "100"},
        ],
    }
    f = DartFetcher(client=_StubClient(payload))
    out = await f.fetch_financials(["X"], [2025], ["11011"])
    assert len(out) == 1
    assert out[0].fs_div is None
