from __future__ import annotations

import asyncio

import httpx
from loguru import logger


DART_BASE_URL = "https://opendart.fss.or.kr"
RATE_LIMIT_PER_MINUTE = 1000


class DartClient:
    """Open DART API client with rate limiting and retries."""

    def __init__(self, api_key: str, max_concurrency: int = 10):
        if not api_key:
            raise ValueError("OPENDART_API_KEY is required")
        self._api_key = api_key
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def get(self, path: str, params: dict, max_retries: int = 3) -> dict:
        """GET request to DART API. Adds crtfc_key automatically."""
        url = f"{DART_BASE_URL}{path}"
        full_params = {**params, "crtfc_key": self._api_key}

        async with self._semaphore:
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        resp = await client.get(url, params=full_params)
                        resp.raise_for_status()
                        return resp.json()
                except httpx.HTTPStatusError as e:
                    if e.response.status_code >= 500 and attempt < max_retries - 1:
                        wait = 2 ** attempt
                        logger.warning(f"DART {e.response.status_code}, retry in {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    raise
                except httpx.RequestError as e:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt
                        logger.warning(f"DART request error: {e}, retry in {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    raise

        raise RuntimeError("Unreachable")
