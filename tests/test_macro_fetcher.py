from unittest.mock import patch, MagicMock
from src.forecast.macro_fetcher import MacroFetcher


def _mock_ecos_response(values: list[tuple[str, str]]) -> dict:
    """Build a fake ECOS API response."""
    return {
        "StatisticSearch": {
            "row": [
                {"TIME": date, "DATA_VALUE": val}
                for date, val in values
            ]
        }
    }


def test_fetch_ecos_kospi():
    fetcher = MacroFetcher(ecos_api_key="test-key", fred_api_key="")
    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_ecos_response([
        ("20260101", "2800"),
        ("20260102", "2810"),
        ("20260103", "2820"),
    ])
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp) as mock_get:
        dates, values = fetcher.fetch_ecos_series(
            stat_code="802Y001", item_code="0001000",
            start_date="20260101", end_date="20260103",
        )

    assert len(dates) == 3
    assert values == [2800.0, 2810.0, 2820.0]
    assert "ecos.bok.or.kr" in mock_get.call_args[0][0]


def test_fetch_ecos_empty_response():
    fetcher = MacroFetcher(ecos_api_key="test-key", fred_api_key="")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"StatisticSearch": {"row": []}}
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        dates, values = fetcher.fetch_ecos_series(
            stat_code="802Y001", item_code="0001000",
            start_date="20260101", end_date="20260103",
        )
    assert dates == []
    assert values == []
