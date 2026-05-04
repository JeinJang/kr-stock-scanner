from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from src.dart.client import DartClient


@pytest.mark.asyncio
async def test_get_returns_json():
    """Test that DartClient.get parses JSON response."""
    client = DartClient(api_key="test-key")

    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "000", "list": [{"a": 1}]}
    mock_response.raise_for_status = MagicMock()

    mock_async_client = MagicMock()
    mock_async_client.get = AsyncMock(return_value=mock_response)
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        result = await client.get("/api/foo.json", params={"p": "v"})

    assert result == {"status": "000", "list": [{"a": 1}]}
    call_args = mock_async_client.get.call_args
    assert "crtfc_key" in call_args.kwargs["params"]
    assert call_args.kwargs["params"]["crtfc_key"] == "test-key"


@pytest.mark.asyncio
async def test_get_handles_dart_error_status():
    """DART returns status='013' meaning no data; should not raise but return empty."""
    client = DartClient(api_key="test-key")

    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "013", "message": "조회된 데이타가 없습니다."}
    mock_response.raise_for_status = MagicMock()

    mock_async_client = MagicMock()
    mock_async_client.get = AsyncMock(return_value=mock_response)
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        result = await client.get("/api/foo.json", params={})

    assert result == {"status": "013", "message": "조회된 데이타가 없습니다."}
