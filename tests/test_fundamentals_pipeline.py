from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, AsyncMock
import pytest

from src.dart.models import CorpInfo, FinancialStatement
from src.fundamentals.pipeline import Pipeline
from src.market_data.models import MarketYearly


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


def test_compute_all_keys_market_yearly_by_corp_code():
    """compute_all looks up market_yearly_map by corp.corp_code, not corp.ticker."""
    cache = MagicMock()
    cache.load_corp_info.return_value = [
        CorpInfo(corp_code="001", ticker="005930", name="삼성전자", market="KOSPI"),
    ]
    cache.load_financials.return_value = [
        FinancialStatement(corp_code="001", year=2024, quarter=0, account="자본총계", value=1_000_000_000_000),
        FinancialStatement(corp_code="001", year=2024, quarter=0, account="당기순이익", value=100_000_000_000),
    ]
    fetcher = MagicMock()
    pipeline = Pipeline(cache=cache, fetcher=fetcher, ttl_days=30)

    market_yearly_map = {
        "001": [MarketYearly(
            corp_code="001", ticker="005930", year=2024,
            as_of_date=date(2024, 12, 30),
            market_cap=2_000_000_000_000, shares_outstanding=1_000_000_000,
        )],
    }
    metrics, scores = pipeline.compute_all(market_yearly_map=market_yearly_map)

    assert len(metrics) == 1
    # If the lookup keyed by corp_code worked, EPS will be populated.
    assert metrics[0].eps is not None
    assert metrics[0].pe is not None


def test_compute_all_handles_empty_market_yearly_map():
    """compute_all works with no market data — share/valuation metrics stay None."""
    cache = MagicMock()
    cache.load_corp_info.return_value = [
        CorpInfo(corp_code="001", ticker="005930", name="삼성전자", market="KOSPI"),
    ]
    cache.load_financials.return_value = []
    fetcher = MagicMock()
    pipeline = Pipeline(cache=cache, fetcher=fetcher, ttl_days=30)

    metrics, scores = pipeline.compute_all(market_yearly_map=None)
    assert len(metrics) == 1
    assert metrics[0].eps is None
