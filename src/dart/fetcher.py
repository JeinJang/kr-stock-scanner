from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from typing import Iterable

import httpx
from loguru import logger

from src.dart.client import DartClient
from src.dart.models import CorpInfo, FinancialStatement


# DART report codes
REPORT_CODES = {
    "11011": "사업보고서",       # annual (Q4)
    "11012": "반기보고서",       # H1
    "11013": "1분기보고서",
    "11014": "3분기보고서",
}

# Map DART account_nm → our internal account names (use as-is for now)
ACCOUNT_WHITELIST = {
    "매출액", "영업이익", "당기순이익", "자산총계", "부채총계",
    "자본총계", "이익잉여금", "유동자산", "유동부채",
}


class DartFetcher:
    """Fetches corp universe and financial statements via DART API."""

    def __init__(self, client: DartClient):
        self._client = client

    async def _download_corp_zip(self) -> bytes:
        """Download the corpCode.xml ZIP file."""
        url = "https://opendart.fss.or.kr/api/corpCode.xml"
        async with httpx.AsyncClient(timeout=60.0) as http:
            resp = await http.get(url, params={"crtfc_key": self._client._api_key})
            resp.raise_for_status()
            return resp.content

    def _extract_xml(self, zip_bytes: bytes) -> str:
        """Extract CORPCODE.xml from ZIP."""
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            with zf.open("CORPCODE.xml") as f:
                return f.read().decode("utf-8")

    async def fetch_corp_universe(
        self,
        markets: list[str],
        market_map: dict[str, str] | None = None,
    ) -> list[CorpInfo]:
        """Fetch all listed corporations matching markets filter.

        Args:
            markets: Market codes to keep (e.g., ["KOSPI", "KOSDAQ"]).
            market_map: Optional ticker -> market mapping (from KRX).
                        If provided, only tickers in this map are kept.
                        If None, all listed corps are returned with market="UNKNOWN".
        """
        logger.info("Downloading DART corp universe...")
        zip_bytes = await self._download_corp_zip()
        xml_str = self._extract_xml(zip_bytes)
        root = ET.fromstring(xml_str)

        corps: list[CorpInfo] = []
        for item in root.findall("list"):
            stock_code = (item.findtext("stock_code") or "").strip()
            if not stock_code:
                continue  # skip unlisted

            corp_code = (item.findtext("corp_code") or "").strip()
            name = (item.findtext("corp_name") or "").strip()

            if market_map is not None:
                market = market_map.get(stock_code)
                if market is None or market not in markets:
                    continue
            else:
                market = "UNKNOWN"

            corps.append(CorpInfo(
                corp_code=corp_code, ticker=stock_code, name=name, market=market,
            ))

        logger.info(f"Fetched {len(corps)} listed corporations")
        return corps

    async def fetch_financials(
        self,
        corp_codes: list[str],
        years: list[int],
        report_codes: list[str],
    ) -> list[FinancialStatement]:
        """Fetch financial statements via fnlttMultiAcnt.json (batch up to 100 corps)."""
        statements: list[FinancialStatement] = []

        # Quarter mapping
        report_to_quarter = {"11011": 0, "11012": 2, "11013": 1, "11014": 3}

        total_batches = sum(
            (len(corp_codes) + 99) // 100 for _ in years for _ in report_codes
        )
        batch_idx = 0
        ok_count = 0
        skip_count = 0

        for year in years:
            for report_code in report_codes:
                for batch_start in range(0, len(corp_codes), 100):
                    batch_idx += 1
                    batch = corp_codes[batch_start:batch_start + 100]
                    params = {
                        "corp_code": ",".join(batch),
                        "bsns_year": str(year),
                        "reprt_code": report_code,
                    }
                    data = await self._client.get("/api/fnlttMultiAcnt.json", params=params)

                    status = data.get("status")
                    if status != "000":
                        skip_count += 1
                        if batch_idx % 50 == 0 or status == "999":
                            logger.info(
                                f"  [{batch_idx}/{total_batches}] year={year} skip "
                                f"(status={status}: {data.get('message', '')[:60]})"
                            )
                        continue

                    ok_count += 1
                    for row in data.get("list", []):
                        account_nm = row.get("account_nm", "")
                        if account_nm not in ACCOUNT_WHITELIST:
                            continue

                        amount_str = row.get("thstrm_amount", "0").replace(",", "")
                        try:
                            value = float(amount_str)
                        except ValueError:
                            continue

                        statements.append(FinancialStatement(
                            corp_code=row.get("corp_code", ""),
                            year=year,
                            quarter=report_to_quarter[report_code],
                            account=account_nm,
                            value=value,
                        ))

                    if batch_idx % 20 == 0:
                        logger.info(
                            f"  [{batch_idx}/{total_batches}] ok={ok_count} skip={skip_count} "
                            f"statements={len(statements)}"
                        )

        logger.info(f"Fetched {len(statements)} financial statement entries")
        return statements
