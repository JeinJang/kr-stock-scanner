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
