from datetime import date
from unittest.mock import MagicMock
import pandas as pd

from src.market_data.db import MarketDB
from src.market_data.models import MarketYearly
from src.market_data.pipeline import MarketDataPipeline


def _fake_df(rows):
    """rows: list of (ticker, close, market_cap, volume, value, shares)."""
    df = pd.DataFrame(
        rows,
        columns=["티커", "종가", "시가총액", "거래량", "거래대금", "상장주식수"],
    ).set_index("티커")
    return df


def _stub_client(kospi_df, kosdaq_df=None):
    c = MagicMock()
    kosdaq_df = kosdaq_df if kosdaq_df is not None else _fake_df([])
    c.get_market_cap_by_ticker.side_effect = lambda d, market="KOSPI": (
        kospi_df if market == "KOSPI" else kosdaq_df
    )
    return c


def test_refresh_persists_rows_and_returns_report(tmp_path):
    client = _stub_client(_fake_df([
        ("000660", 173_900, 126_599_611_273_500, 1, 1, 728_002_365),
    ]))
    db = MarketDB(url=f"sqlite:///{tmp_path/'t.db'}")
    pipeline = MarketDataPipeline(db=db, client=client)

    report = pipeline.refresh(
        years=[2024],
        tickers=["000660"],
        corp_code_map={"000660": "00126380"},
    )

    assert report.successful_rows == 1
    loaded = db.load_for_corp("00126380")
    assert len(loaded) == 1
    assert loaded[0].market_cap == 126_599_611_273_500


def test_refresh_is_idempotent(tmp_path):
    client = _stub_client(_fake_df([
        ("000660", 173_900, 126_599_611_273_500, 1, 1, 728_002_365),
    ]))
    db = MarketDB(url=f"sqlite:///{tmp_path/'t.db'}")
    pipeline = MarketDataPipeline(db=db, client=client)

    pipeline.refresh(years=[2024], tickers=["000660"], corp_code_map={"000660": "00126380"})
    pipeline.refresh(years=[2024], tickers=["000660"], corp_code_map={"000660": "00126380"})
    rows = db.load_for_corp("00126380")
    assert len(rows) == 1  # not duplicated
