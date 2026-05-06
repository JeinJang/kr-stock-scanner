from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, AsyncMock
import pytest

from src.dart.models import CorpInfo, FinancialStatement
from src.fundamentals.pipeline import Pipeline


@pytest.mark.asyncio
async def test_pipeline_uses_cache_when_fresh():
    """If last_updated is recent, skip fetch."""
    cache = MagicMock()
    cache.last_updated.return_value = datetime.now() - timedelta(days=5)
    cache.load_corp_info.return_value = [
        CorpInfo(corp_code="001", ticker="005930", name="삼성전자", market="KOSPI"),
    ]
    cache.load_financials.return_value = [
        FinancialStatement(corp_code="001", year=2025, quarter=0, account="자본총계", value=1000),
    ]
    fetcher = MagicMock()
    fetcher.fetch_corp_universe = AsyncMock()
    fetcher.fetch_financials = AsyncMock()

    pipeline = Pipeline(cache=cache, fetcher=fetcher, ttl_days=30)

    await pipeline.refresh_data(force=False, years=[2025])

    fetcher.fetch_corp_universe.assert_not_called()
    fetcher.fetch_financials.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_refreshes_when_force():
    cache = MagicMock()
    cache.last_updated.return_value = datetime.now()
    fetcher = MagicMock()
    fetcher.fetch_corp_universe = AsyncMock(return_value=[
        CorpInfo(corp_code="001", ticker="005930", name="삼성전자", market="KOSPI"),
    ])
    fetcher.fetch_financials = AsyncMock(return_value=[])

    pipeline = Pipeline(cache=cache, fetcher=fetcher, ttl_days=30)

    await pipeline.refresh_data(force=True, years=[2025])

    fetcher.fetch_corp_universe.assert_called_once()
    fetcher.fetch_financials.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_refreshes_when_ttl_expired():
    cache = MagicMock()
    cache.last_updated.return_value = datetime.now() - timedelta(days=60)
    fetcher = MagicMock()
    fetcher.fetch_corp_universe = AsyncMock(return_value=[
        CorpInfo(corp_code="001", ticker="005930", name="삼성전자", market="KOSPI"),
    ])
    fetcher.fetch_financials = AsyncMock(return_value=[])

    pipeline = Pipeline(cache=cache, fetcher=fetcher, ttl_days=30)

    await pipeline.refresh_data(force=False, years=[2025])

    fetcher.fetch_corp_universe.assert_called_once()
