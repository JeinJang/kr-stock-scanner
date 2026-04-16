# TimesFM Forecast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a TimesFM-based forecasting module that predicts macro indices and individual stock prices for 52-week high scan results, outputting an interactive HTML report.

**Architecture:** New `src/forecast/` package within the existing project. Reads scan results from the shared SQLite DB, fetches historical data via ECOS/FRED/KRX APIs, runs TimesFM batch predictions, and generates a self-contained Plotly HTML report. Separate CLI entrypoint: `python -m src.forecast.cli`.

**Tech Stack:** TimesFM 2.5 (PyTorch), ECOS API, FRED API (fredapi), Plotly, Jinja2, Typer, Pydantic

**Spec:** `docs/superpowers/specs/2026-04-16-timesfm-forecast-design.md`

---

### Task 1: Dependencies & Configuration

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/config.py`
- Modify: `config.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Add dependencies to pyproject.toml**

Add forecast dependencies to the main dependencies list:

```toml
dependencies = [
    # ... existing dependencies ...
    "timesfm>=2.0.0",
    "torch>=2.0.0",
    "plotly>=5.18.0",
    "jinja2>=3.1.0",
    "fredapi>=0.5.0",
]
```

- [ ] **Step 2: Add forecast config section to src/config.py**

Add `ECOS_API_KEY` and `FRED_API_KEY` to `Settings`, add `ForecastSection` and include it in `ScannerConfig`:

```python
# In Settings class, add:
    ecos_api_key: str = ""
    fred_api_key: str = ""

# New section after TelegramSection:
class ForecastSection(BaseModel):
    horizon: int = 60
    model: str = "google/timesfm-2.5-200m-pytorch"
    report_dir: str = "reports"

# In ScannerConfig, add:
    forecast: ForecastSection = ForecastSection()
```

- [ ] **Step 3: Add forecast section to config.yaml**

```yaml
forecast:
  horizon: 60
  model: "google/timesfm-2.5-200m-pytorch"
  report_dir: "reports"
```

- [ ] **Step 4: Write test for new config fields**

```python
# In tests/test_config.py, add:
def test_forecast_config_defaults():
    from src.config import ForecastSection, ScannerConfig
    config = ScannerConfig()
    assert config.forecast.horizon == 60
    assert config.forecast.model == "google/timesfm-2.5-200m-pytorch"
    assert config.forecast.report_dir == "reports"


def test_settings_has_ecos_fred_keys():
    import os
    os.environ["ECOS_API_KEY"] = "test-ecos"
    os.environ["FRED_API_KEY"] = "test-fred"
    from src.config import Settings
    s = Settings()
    assert s.ecos_api_key == "test-ecos"
    assert s.fred_api_key == "test-fred"
    os.environ.pop("ECOS_API_KEY", None)
    os.environ.pop("FRED_API_KEY", None)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/config.py config.yaml tests/test_config.py
git commit -m "feat(forecast): add dependencies and config for TimesFM forecast module"
```

---

### Task 2: Forecast Data Models

**Files:**
- Create: `src/forecast/__init__.py`
- Create: `src/forecast/models.py`
- Create: `tests/test_forecast_models.py`

- [ ] **Step 1: Create package init**

```python
# src/forecast/__init__.py
```

Empty file to make it a Python package.

- [ ] **Step 2: Write failing test for ForecastResult model**

```python
# tests/test_forecast_models.py
from src.forecast.models import ForecastResult


def test_forecast_result_creation():
    result = ForecastResult(
        ticker="005930",
        name="Samsung Electronics",
        category="stock",
        history=[100.0, 101.0, 102.0],
        dates_history=["2026-01-01", "2026-01-02", "2026-01-03"],
        forecast=[103.0, 104.0],
        dates_forecast=["2026-01-04", "2026-01-05"],
        quantile_low=[101.0, 102.0],
        quantile_high=[105.0, 106.0],
        predicted_return=3.0,
        uncertainty=4.0,
    )
    assert result.ticker == "005930"
    assert result.category == "stock"
    assert len(result.forecast) == 2
    assert result.predicted_return == 3.0


def test_forecast_result_macro():
    result = ForecastResult(
        ticker="KOSPI",
        name="KOSPI",
        category="macro",
        history=[2800.0, 2810.0],
        dates_history=["2026-01-01", "2026-01-02"],
        forecast=[2820.0],
        dates_forecast=["2026-01-03"],
        quantile_low=[2790.0],
        quantile_high=[2850.0],
        predicted_return=0.71,
        uncertainty=2.14,
    )
    assert result.category == "macro"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_forecast_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.forecast.models'`

- [ ] **Step 4: Write ForecastResult model**

```python
# src/forecast/models.py
from pydantic import BaseModel


class ForecastResult(BaseModel):
    """Prediction result for a single ticker or macro indicator."""

    ticker: str
    name: str
    category: str  # "macro" | "stock"
    sector: str = ""
    history: list[float]
    dates_history: list[str]
    forecast: list[float]
    dates_forecast: list[str]
    quantile_low: list[float]
    quantile_high: list[float]
    predicted_return: float
    uncertainty: float
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_forecast_models.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/forecast/__init__.py src/forecast/models.py tests/test_forecast_models.py
git commit -m "feat(forecast): add ForecastResult data model"
```

---

### Task 3: Macro Fetcher — ECOS API

**Files:**
- Create: `src/forecast/macro_fetcher.py`
- Create: `tests/test_macro_fetcher.py`

The ECOS (한국은행 경제통계시스템) API returns JSON with statistical data. Key endpoint: `https://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/1/250/{stat_code}/{freq}/{start_date}/{end_date}/{item_code}`

Stat codes:
- KOSPI: stat_code=`802Y001`, item_code=`0001000`
- KOSDAQ: stat_code=`802Y002`, item_code=`0001000`
- USD/KRW: stat_code=`731Y003`, item_code=`0000001`
- 한국 기준금리: stat_code=`722Y001`, item_code=`0101000`

- [ ] **Step 1: Write failing test with mocked HTTP**

```python
# tests/test_macro_fetcher.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_macro_fetcher.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement MacroFetcher (ECOS part)**

```python
# src/forecast/macro_fetcher.py
from __future__ import annotations

from datetime import datetime, timedelta

import requests
from loguru import logger


# ECOS stat codes for Korean macro indicators
ECOS_INDICATORS = {
    "KOSPI": {"stat_code": "802Y001", "item_code": "0001000", "freq": "D"},
    "KOSDAQ": {"stat_code": "802Y002", "item_code": "0001000", "freq": "D"},
    "USD_KRW": {"stat_code": "731Y003", "item_code": "0000001", "freq": "D"},
    "KR_RATE": {"stat_code": "722Y001", "item_code": "0101000", "freq": "D"},
}

ECOS_BASE_URL = "https://ecos.bok.or.kr/api/StatisticSearch"

# Display names for indicators
INDICATOR_NAMES = {
    "KOSPI": "KOSPI",
    "KOSDAQ": "KOSDAQ",
    "USD_KRW": "USD/KRW",
    "KR_RATE": "한국 기준금리",
    "SP500": "S&P 500",
    "NASDAQ": "NASDAQ",
    "US_RATE": "미국 기준금리",
}


class MacroFetcher:
    """Fetches macro indicator data from ECOS and FRED APIs."""

    def __init__(self, ecos_api_key: str, fred_api_key: str):
        self._ecos_key = ecos_api_key
        self._fred_key = fred_api_key

    def fetch_ecos_series(
        self,
        stat_code: str,
        item_code: str,
        start_date: str,
        end_date: str,
        freq: str = "D",
    ) -> tuple[list[str], list[float]]:
        """Fetch a single time series from ECOS API.

        Returns (dates, values) as parallel lists.
        """
        url = (
            f"{ECOS_BASE_URL}/{self._ecos_key}/json/kr/1/1000"
            f"/{stat_code}/{freq}/{start_date}/{end_date}/{item_code}"
        )
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()

        rows = data.get("StatisticSearch", {}).get("row", [])
        dates = []
        values = []
        for row in rows:
            val_str = row.get("DATA_VALUE", "")
            if val_str == "" or val_str == "-":
                continue
            dates.append(row["TIME"])
            values.append(float(val_str))
        return dates, values

    def fetch_all_ecos(self, lookback_days: int = 400) -> dict[str, tuple[list[str], list[float]]]:
        """Fetch all ECOS indicators for the given lookback period."""
        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")

        results: dict[str, tuple[list[str], list[float]]] = {}
        for name, params in ECOS_INDICATORS.items():
            logger.info(f"Fetching ECOS: {name}")
            dates, values = self.fetch_ecos_series(
                stat_code=params["stat_code"],
                item_code=params["item_code"],
                start_date=start_str,
                end_date=end_str,
                freq=params["freq"],
            )
            results[name] = (dates, values)
            logger.info(f"  -> {len(values)} data points")
        return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_macro_fetcher.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/forecast/macro_fetcher.py tests/test_macro_fetcher.py
git commit -m "feat(forecast): add MacroFetcher with ECOS API support"
```

---

### Task 4: Macro Fetcher — FRED API

**Files:**
- Modify: `src/forecast/macro_fetcher.py`
- Modify: `tests/test_macro_fetcher.py`

The `fredapi` library provides a clean Python interface. Series IDs:
- S&P 500: `SP500`
- NASDAQ: `NASDAQCOM`
- Federal Funds Rate: `FEDFUNDS`

- [ ] **Step 1: Write failing test for FRED fetching**

```python
# Add to tests/test_macro_fetcher.py:
import pandas as pd


def test_fetch_fred_series():
    fetcher = MacroFetcher(ecos_api_key="", fred_api_key="test-fred-key")

    mock_series = pd.Series(
        [4500.0, 4510.0, 4520.0],
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
    )

    with patch("fredapi.Fred") as MockFred:
        mock_fred_instance = MagicMock()
        mock_fred_instance.get_series.return_value = mock_series
        MockFred.return_value = mock_fred_instance

        dates, values = fetcher.fetch_fred_series("SP500", lookback_days=400)

    assert len(dates) == 3
    assert values == [4500.0, 4510.0, 4520.0]
    mock_fred_instance.get_series.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_macro_fetcher.py::test_fetch_fred_series -v`
Expected: FAIL — `AttributeError: 'MacroFetcher' object has no attribute 'fetch_fred_series'`

- [ ] **Step 3: Add FRED methods to MacroFetcher**

Add to `src/forecast/macro_fetcher.py`:

```python
# Add at top of file:
import pandas as pd

# Add FRED indicator mapping after ECOS_INDICATORS:
FRED_INDICATORS = {
    "SP500": "SP500",
    "NASDAQ": "NASDAQCOM",
    "US_RATE": "FEDFUNDS",
}

# Add methods to MacroFetcher class:
    def fetch_fred_series(
        self, series_id: str, lookback_days: int = 400,
    ) -> tuple[list[str], list[float]]:
        """Fetch a single time series from FRED API."""
        from fredapi import Fred

        fred = Fred(api_key=self._fred_key)
        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        series: pd.Series = fred.get_series(series_id, start, end)
        series = series.dropna()

        dates = [d.strftime("%Y%m%d") for d in series.index]
        values = [float(v) for v in series.values]
        return dates, values

    def fetch_all_fred(self, lookback_days: int = 400) -> dict[str, tuple[list[str], list[float]]]:
        """Fetch all FRED indicators."""
        results: dict[str, tuple[list[str], list[float]]] = {}
        for name, series_id in FRED_INDICATORS.items():
            logger.info(f"Fetching FRED: {name} ({series_id})")
            dates, values = self.fetch_fred_series(series_id, lookback_days)
            results[name] = (dates, values)
            logger.info(f"  -> {len(values)} data points")
        return results

    def fetch_all(self, lookback_days: int = 400) -> dict[str, tuple[list[str], list[float]]]:
        """Fetch all macro indicators from both ECOS and FRED."""
        results = self.fetch_all_ecos(lookback_days)
        results.update(self.fetch_all_fred(lookback_days))
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_macro_fetcher.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/forecast/macro_fetcher.py tests/test_macro_fetcher.py
git commit -m "feat(forecast): add FRED API support to MacroFetcher"
```

---

### Task 5: Stock Fetcher

**Files:**
- Create: `src/forecast/stock_fetcher.py`
- Create: `tests/test_stock_fetcher.py`

Reuses the existing `KrxClient` to fetch per-ticker historical close prices.

- [ ] **Step 1: Write failing test**

```python
# tests/test_stock_fetcher.py
from unittest.mock import MagicMock
import pandas as pd
from src.forecast.stock_fetcher import StockFetcher


def test_fetch_stock_history():
    mock_client = MagicMock()
    mock_client.supports_history = True
    mock_df = pd.DataFrame(
        {"종가": [50000, 51000, 52000]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
    )
    mock_client.get_market_ohlcv_by_date.return_value = mock_df

    fetcher = StockFetcher(client=mock_client)
    dates, values = fetcher.fetch_history("005930", end_date="20260103", lookback_days=250)

    assert len(values) == 3
    assert values == [50000.0, 51000.0, 52000.0]
    mock_client.get_market_ohlcv_by_date.assert_called_once()


def test_fetch_multiple_stocks():
    mock_client = MagicMock()
    mock_client.supports_history = True

    def make_df(prices):
        return pd.DataFrame(
            {"종가": prices},
            index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
        )

    mock_client.get_market_ohlcv_by_date.side_effect = [
        make_df([50000, 51000]),
        make_df([10000, 10500]),
    ]

    fetcher = StockFetcher(client=mock_client)
    results = fetcher.fetch_histories(
        tickers=["005930", "035720"],
        end_date="20260103",
        lookback_days=250,
    )
    assert "005930" in results
    assert "035720" in results
    assert len(results["005930"][1]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stock_fetcher.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement StockFetcher**

```python
# src/forecast/stock_fetcher.py
from __future__ import annotations

from datetime import datetime, timedelta

from loguru import logger

from src.krx_client import KrxClient


class StockFetcher:
    """Fetches historical close prices for individual stocks via KRX."""

    def __init__(self, client: KrxClient):
        self._client = client

    def fetch_history(
        self, ticker: str, end_date: str, lookback_days: int = 250,
    ) -> tuple[list[str], list[float]]:
        """Fetch historical close prices for a single ticker.

        Returns (dates, close_prices) as parallel lists.
        """
        end = datetime.strptime(end_date, "%Y%m%d")
        start = end - timedelta(days=int(lookback_days * 1.5))
        df = self._client.get_market_ohlcv_by_date(
            start.strftime("%Y%m%d"), end_date, ticker,
        )
        if df.empty or "종가" not in df.columns:
            return [], []

        dates = [d.strftime("%Y%m%d") for d in df.index]
        values = [float(v) for v in df["종가"].values]
        return dates, values

    def fetch_histories(
        self,
        tickers: list[str],
        end_date: str,
        lookback_days: int = 250,
    ) -> dict[str, tuple[list[str], list[float]]]:
        """Fetch historical close prices for multiple tickers."""
        results: dict[str, tuple[list[str], list[float]]] = {}
        for i, ticker in enumerate(tickers):
            logger.info(f"Fetching stock history [{i+1}/{len(tickers)}]: {ticker}")
            dates, values = self.fetch_history(ticker, end_date, lookback_days)
            if values:
                results[ticker] = (dates, values)
            else:
                logger.warning(f"  -> No data for {ticker}, skipping")
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stock_fetcher.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/forecast/stock_fetcher.py tests/test_stock_fetcher.py
git commit -m "feat(forecast): add StockFetcher for historical price data"
```

---

### Task 6: Predictor — TimesFM Wrapper

**Files:**
- Create: `src/forecast/predictor.py`
- Create: `tests/test_predictor.py`

- [ ] **Step 1: Write failing test with mocked TimesFM**

```python
# tests/test_predictor.py
from unittest.mock import patch, MagicMock
import numpy as np

from src.forecast.predictor import Predictor
from src.forecast.models import ForecastResult


def test_predict_single():
    """Test prediction for a single time series."""
    predictor = Predictor.__new__(Predictor)
    predictor._model = MagicMock()
    predictor._horizon = 5

    # Mock TimesFM forecast output
    # point_forecast shape: (1, horizon)
    # quantile_forecast shape: (1, horizon, 10) — mean + 9 quantile levels
    point = np.array([[105.0, 106.0, 107.0, 108.0, 109.0]])
    quantiles = np.zeros((1, 5, 10))
    quantiles[0, :, 0] = [105.0, 106.0, 107.0, 108.0, 109.0]  # mean
    quantiles[0, :, 1] = [102.0, 103.0, 104.0, 105.0, 106.0]  # 10th pct
    quantiles[0, :, 9] = [108.0, 109.0, 110.0, 111.0, 112.0]  # 90th pct

    predictor._model.forecast.return_value = (point, quantiles)

    history = [100.0, 101.0, 102.0, 103.0, 104.0]
    dates_hist = ["20260101", "20260102", "20260103", "20260106", "20260107"]

    result = predictor.predict_single(
        ticker="005930",
        name="Samsung",
        category="stock",
        history=history,
        dates_history=dates_hist,
    )

    assert isinstance(result, ForecastResult)
    assert result.ticker == "005930"
    assert len(result.forecast) == 5
    assert len(result.quantile_low) == 5
    assert len(result.quantile_high) == 5
    assert result.predicted_return > 0  # price went up


def test_predict_batch():
    """Test batch prediction for multiple series."""
    predictor = Predictor.__new__(Predictor)
    predictor._model = MagicMock()
    predictor._horizon = 3

    point = np.array([
        [105.0, 106.0, 107.0],
        [2850.0, 2860.0, 2870.0],
    ])
    quantiles = np.zeros((2, 3, 10))
    # Series 1
    quantiles[0, :, 0] = [105.0, 106.0, 107.0]
    quantiles[0, :, 1] = [103.0, 104.0, 105.0]
    quantiles[0, :, 9] = [107.0, 108.0, 109.0]
    # Series 2
    quantiles[1, :, 0] = [2850.0, 2860.0, 2870.0]
    quantiles[1, :, 1] = [2830.0, 2840.0, 2850.0]
    quantiles[1, :, 9] = [2870.0, 2880.0, 2890.0]

    predictor._model.forecast.return_value = (point, quantiles)

    items = [
        {
            "ticker": "005930", "name": "Samsung", "category": "stock",
            "history": [100.0, 101.0, 102.0, 103.0, 104.0],
            "dates_history": ["20260101", "20260102", "20260103", "20260106", "20260107"],
        },
        {
            "ticker": "KOSPI", "name": "KOSPI", "category": "macro",
            "history": [2800.0, 2810.0, 2820.0, 2830.0, 2840.0],
            "dates_history": ["20260101", "20260102", "20260103", "20260106", "20260107"],
        },
    ]

    results = predictor.predict_batch(items)
    assert len(results) == 2
    assert results[0].ticker == "005930"
    assert results[1].ticker == "KOSPI"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_predictor.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement Predictor**

```python
# src/forecast/predictor.py
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from loguru import logger

from src.forecast.models import ForecastResult


class Predictor:
    """Wraps TimesFM model for time-series forecasting."""

    def __init__(self, model_name: str, horizon: int = 60):
        import timesfm

        self._horizon = horizon
        logger.info(f"Loading TimesFM model: {model_name}")
        self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_name)
        self._model.compile(timesfm.ForecastConfig(
            max_context=512,
            max_horizon=horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
        ))
        logger.info("TimesFM model loaded and compiled")

    def _generate_forecast_dates(self, last_date_str: str, n: int) -> list[str]:
        """Generate n future business dates starting from last_date."""
        last = datetime.strptime(last_date_str, "%Y%m%d")
        dates = []
        current = last
        while len(dates) < n:
            current += timedelta(days=1)
            if current.weekday() < 5:  # skip weekends
                dates.append(current.strftime("%Y%m%d"))
        return dates

    def predict_single(
        self,
        ticker: str,
        name: str,
        category: str,
        history: list[float],
        dates_history: list[str],
    ) -> ForecastResult:
        """Run prediction for a single time series."""
        inputs = [np.array(history)]
        point_forecast, quantile_forecast = self._model.forecast(
            horizon=self._horizon,
            inputs=inputs,
        )

        forecast_values = point_forecast[0].tolist()
        q_low = quantile_forecast[0, :, 1].tolist()   # 10th percentile
        q_high = quantile_forecast[0, :, 9].tolist()   # 90th percentile

        last_price = history[-1]
        final_price = forecast_values[-1]
        predicted_return = ((final_price - last_price) / last_price) * 100

        avg_spread = np.mean(np.array(q_high) - np.array(q_low))
        uncertainty = (avg_spread / last_price) * 100

        dates_forecast = self._generate_forecast_dates(dates_history[-1], self._horizon)

        return ForecastResult(
            ticker=ticker,
            name=name,
            category=category,
            history=history,
            dates_history=dates_history,
            forecast=forecast_values,
            dates_forecast=dates_forecast,
            quantile_low=q_low,
            quantile_high=q_high,
            predicted_return=round(predicted_return, 2),
            uncertainty=round(uncertainty, 2),
        )

    def predict_batch(
        self, items: list[dict],
    ) -> list[ForecastResult]:
        """Run batch prediction for multiple time series at once."""
        if not items:
            return []

        inputs = [np.array(item["history"]) for item in items]
        point_forecast, quantile_forecast = self._model.forecast(
            horizon=self._horizon,
            inputs=inputs,
        )

        results = []
        for i, item in enumerate(items):
            forecast_values = point_forecast[i].tolist()
            q_low = quantile_forecast[i, :, 1].tolist()
            q_high = quantile_forecast[i, :, 9].tolist()

            last_price = item["history"][-1]
            final_price = forecast_values[-1]
            predicted_return = ((final_price - last_price) / last_price) * 100

            avg_spread = np.mean(np.array(q_high) - np.array(q_low))
            uncertainty = (avg_spread / last_price) * 100

            dates_forecast = self._generate_forecast_dates(
                item["dates_history"][-1], self._horizon,
            )

            results.append(ForecastResult(
                ticker=item["ticker"],
                name=item["name"],
                category=item["category"],
                sector=item.get("sector", ""),
                history=item["history"],
                dates_history=item["dates_history"],
                forecast=forecast_values,
                dates_forecast=dates_forecast,
                quantile_low=q_low,
                quantile_high=q_high,
                predicted_return=round(predicted_return, 2),
                uncertainty=round(uncertainty, 2),
            ))

        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_predictor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/forecast/predictor.py tests/test_predictor.py
git commit -m "feat(forecast): add Predictor wrapping TimesFM model"
```

---

### Task 7: HTML Report — Jinja2 Template

**Files:**
- Create: `src/forecast/templates/report.html`
- Create: `src/forecast/report.py`
- Create: `tests/test_forecast_report.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_forecast_report.py
import os
from src.forecast.report import ReportGenerator
from src.forecast.models import ForecastResult


def _make_result(ticker: str, name: str, category: str) -> ForecastResult:
    return ForecastResult(
        ticker=ticker,
        name=name,
        category=category,
        history=[100.0, 101.0, 102.0],
        dates_history=["20260101", "20260102", "20260103"],
        forecast=[103.0, 104.0, 105.0],
        dates_forecast=["20260106", "20260107", "20260108"],
        quantile_low=[101.0, 102.0, 103.0],
        quantile_high=[105.0, 106.0, 107.0],
        predicted_return=2.94,
        uncertainty=3.92,
    )


def test_generate_report(tmp_path):
    macro_results = [
        _make_result("KOSPI", "KOSPI", "macro"),
        _make_result("SP500", "S&P 500", "macro"),
    ]
    stock_results = [
        _make_result("005930", "삼성전자", "stock"),
        _make_result("035720", "카카오", "stock"),
    ]

    generator = ReportGenerator()
    path = generator.generate(
        macro_results=macro_results,
        stock_results=stock_results,
        ai_analyses={},
        scan_date="2026-04-16",
        output_dir=str(tmp_path),
    )

    assert os.path.exists(path)
    with open(path) as f:
        html = f.read()
    assert "KOSPI" in html
    assert "삼성전자" in html
    assert "plotly" in html.lower() or "Plotly" in html


def test_report_with_ai_analysis(tmp_path):
    stock_results = [_make_result("005930", "삼성전자", "stock")]
    ai_analyses = {
        "005930": {
            "ai_analysis": "[상승 원인] 반도체 호황\n[핵심 뉴스] HBM 수주 확대\n[투자 포인트] AI 수요",
            "news_summary": "삼성전자 실적 호조",
        }
    }

    generator = ReportGenerator()
    path = generator.generate(
        macro_results=[],
        stock_results=stock_results,
        ai_analyses=ai_analyses,
        scan_date="2026-04-16",
        output_dir=str(tmp_path),
    )

    with open(path) as f:
        html = f.read()
    assert "반도체 호황" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_forecast_report.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create the Jinja2 HTML template**

```html
<!-- src/forecast/templates/report.html -->
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>주가 예측 리포트 — {{ scan_date }}</title>
    <script>{{ plotly_js }}</script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; color: #333; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        h1 { font-size: 24px; margin-bottom: 24px; padding-bottom: 12px; border-bottom: 2px solid #333; }
        h2 { font-size: 20px; margin: 32px 0 16px; color: #1a1a2e; }
        .grid-2x3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
        .grid-2x2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
        .card { background: #fff; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .chart-container { width: 100%; min-height: 300px; }
        table { width: 100%; border-collapse: collapse; margin-top: 12px; }
        th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; font-size: 13px; color: #666; }
        td { font-size: 14px; }
        .up { color: #e74c3c; }
        .down { color: #2980b9; }
        .flat { color: #7f8c8d; }
        .risk-low { background: #d5f5e3; color: #27ae60; padding: 2px 8px; border-radius: 4px; }
        .risk-mid { background: #fdebd0; color: #e67e22; padding: 2px 8px; border-radius: 4px; }
        .risk-high { background: #fadbd8; color: #e74c3c; padding: 2px 8px; border-radius: 4px; }
        .highlight { background: #eafaf1; }
        .ai-box { background: #f8f9fa; border-left: 3px solid #3498db; padding: 12px; margin-top: 8px; font-size: 13px; white-space: pre-wrap; }
        .collapsible { cursor: pointer; user-select: none; }
        .collapsible::before { content: "▸ "; }
        .collapsible.open::before { content: "▾ "; }
        .collapsible-content { display: none; }
        .collapsible-content.open { display: block; }
        .section { margin-bottom: 32px; }
        @media (max-width: 900px) {
            .grid-2x3 { grid-template-columns: repeat(2, 1fr); }
            .grid-2x2 { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
<div class="container">
    <h1>주가 예측 리포트 — {{ scan_date }}</h1>

    <!-- Section 1: Macro Dashboard -->
    {% if macro_results %}
    <div class="section">
        <h2>1. 매크로 대시보드</h2>
        <div class="grid-2x3">
            {% for m in macro_results %}
            <div class="card">
                <div id="macro-chart-{{ loop.index }}" class="chart-container"></div>
            </div>
            {% endfor %}
        </div>
        <div class="card" style="margin-top: 16px;">
            <table>
                <thead>
                    <tr><th>지표</th><th>현재값</th><th>예측값 ({{ horizon }}일 후)</th><th>변동률</th><th>방향</th></tr>
                </thead>
                <tbody>
                {% for m in macro_results %}
                    <tr>
                        <td>{{ indicator_names[m.ticker] or m.name }}</td>
                        <td>{{ "%.2f"|format(m.history[-1]) }}</td>
                        <td>{{ "%.2f"|format(m.forecast[-1]) }}</td>
                        <td class="{{ 'up' if m.predicted_return > 0.5 else ('down' if m.predicted_return < -0.5 else 'flat') }}">
                            {{ "%+.2f"|format(m.predicted_return) }}%
                        </td>
                        <td>{{ "▲" if m.predicted_return > 0.5 else ("▼" if m.predicted_return < -0.5 else "►") }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
    {% endif %}

    <!-- Section 2: Stock Forecast Charts -->
    {% if stock_results %}
    <div class="section">
        <h2>2. 종목별 예측 차트</h2>
        {% for s in stock_results %}
        <div class="card" style="margin-bottom: 12px;">
            <h3 class="collapsible{% if loop.index <= 5 %} open{% endif %}"
                onclick="this.classList.toggle('open'); this.nextElementSibling.classList.toggle('open');">
                {{ s.name }} ({{ s.ticker }}) — 예측 수익률: <span class="{{ 'up' if s.predicted_return > 0 else 'down' }}">{{ "%+.2f"|format(s.predicted_return) }}%</span>
            </h3>
            <div class="collapsible-content{% if loop.index <= 5 %} open{% endif %}">
                <div id="stock-chart-{{ loop.index }}" class="chart-container"></div>
                {% if s.ticker in ai_analyses %}
                <div class="ai-box">{{ ai_analyses[s.ticker].ai_analysis }}</div>
                {% endif %}
            </div>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <!-- Section 3: Stock Ranking -->
    {% if stock_results %}
    <div class="section">
        <h2>3. 종목 랭킹 (예측 수익률 기준)</h2>
        <div class="card">
            <table>
                <thead>
                    <tr><th>#</th><th>종목명</th><th>현재가</th><th>예측가</th><th>예측 수익률</th><th>불확실성</th><th>섹터</th><th>위험등급</th></tr>
                </thead>
                <tbody>
                {% for s in ranked_stocks %}
                    <tr class="{{ 'highlight' if s.predicted_return > 5 else '' }}">
                        <td>{{ loop.index }}</td>
                        <td>{{ s.name }} ({{ s.ticker }})</td>
                        <td>{{ "{:,.0f}"|format(s.history[-1]) }}</td>
                        <td>{{ "{:,.0f}"|format(s.forecast[-1]) }}</td>
                        <td class="{{ 'up' if s.predicted_return > 0 else 'down' }}">{{ "%+.2f"|format(s.predicted_return) }}%</td>
                        <td>{{ "%.1f"|format(s.uncertainty) }}%</td>
                        <td>{{ s.sector }}</td>
                        <td>
                            {% if s.uncertainty < 5 %}
                                <span class="risk-low">낮음</span>
                            {% elif s.uncertainty < 15 %}
                                <span class="risk-mid">보통</span>
                            {% else %}
                                <span class="risk-high">높음</span>
                            {% endif %}
                        </td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
    {% endif %}

    <!-- Section 4: Risk Scatter Plot -->
    {% if stock_results %}
    <div class="section">
        <h2>4. 리스크 지표</h2>
        <div class="card">
            <div id="risk-scatter" class="chart-container" style="min-height: 400px;"></div>
        </div>
    </div>
    {% endif %}

    <!-- Section 5: AI Analysis (standalone) -->
    {% if ai_only_stocks %}
    <div class="section">
        <h2>5. AI 분석 결과</h2>
        {% for ticker, analysis in ai_only_stocks.items() %}
        <div class="card" style="margin-bottom: 8px;">
            <strong>{{ analysis.ticker_name or ticker }}</strong>
            <div class="ai-box">{{ analysis.ai_analysis }}</div>
        </div>
        {% endfor %}
    </div>
    {% endif %}
</div>

<script>
// Macro charts
{% for m in macro_results %}
(function() {
    var hist = {{ m.history | tojson }};
    var dHist = {{ m.dates_history | tojson }};
    var fc = {{ m.forecast | tojson }};
    var dFc = {{ m.dates_forecast | tojson }};
    var qLow = {{ m.quantile_low | tojson }};
    var qHigh = {{ m.quantile_high | tojson }};

    var traces = [
        {x: dHist, y: hist, mode: 'lines', name: '실제', line: {color: '#2c3e50'}},
        {x: dFc, y: fc, mode: 'lines', name: '예측', line: {color: '#e74c3c', dash: 'dot'}},
        {x: dFc.concat(dFc.slice().reverse()), y: qHigh.concat(qLow.slice().reverse()),
         fill: 'toself', fillcolor: 'rgba(231,76,60,0.1)', line: {color: 'transparent'}, name: '신뢰구간', showlegend: false},
    ];
    Plotly.newPlot('macro-chart-{{ loop.index }}', traces, {
        title: '{{ indicator_names[m.ticker] or m.name }}',
        margin: {t: 40, r: 20, b: 40, l: 60},
        height: 280,
        xaxis: {showgrid: false},
        yaxis: {showgrid: true, gridcolor: '#eee'},
        legend: {x: 0, y: 1, font: {size: 10}},
    }, {responsive: true});
})();
{% endfor %}

// Stock charts
{% for s in stock_results %}
(function() {
    var hist = {{ s.history | tojson }};
    var dHist = {{ s.dates_history | tojson }};
    var fc = {{ s.forecast | tojson }};
    var dFc = {{ s.dates_forecast | tojson }};
    var qLow = {{ s.quantile_low | tojson }};
    var qHigh = {{ s.quantile_high | tojson }};

    var traces = [
        {x: dHist, y: hist, mode: 'lines', name: '실제', line: {color: '#2c3e50'}},
        {x: dFc, y: fc, mode: 'lines', name: '예측', line: {color: '#e74c3c', dash: 'dot'}},
        {x: dFc.concat(dFc.slice().reverse()), y: qHigh.concat(qLow.slice().reverse()),
         fill: 'toself', fillcolor: 'rgba(231,76,60,0.1)', line: {color: 'transparent'}, name: '신뢰구간', showlegend: false},
    ];
    Plotly.newPlot('stock-chart-{{ loop.index }}', traces, {
        title: '{{ s.name }}',
        margin: {t: 40, r: 20, b: 40, l: 60},
        height: 300,
        xaxis: {showgrid: false},
        yaxis: {showgrid: true, gridcolor: '#eee'},
        legend: {x: 0, y: 1, font: {size: 10}},
    }, {responsive: true});
})();
{% endfor %}

// Risk scatter plot
{% if stock_results %}
(function() {
    var tickers = {{ stock_results | map(attribute='name') | list | tojson }};
    var returns = {{ stock_results | map(attribute='predicted_return') | list | tojson }};
    var uncertainties = {{ stock_results | map(attribute='uncertainty') | list | tojson }};

    var colors = returns.map(function(r) { return r > 0 ? '#e74c3c' : '#2980b9'; });

    Plotly.newPlot('risk-scatter', [{
        x: uncertainties,
        y: returns,
        mode: 'markers+text',
        text: tickers,
        textposition: 'top center',
        textfont: {size: 9},
        marker: {size: 10, color: colors, opacity: 0.7},
        type: 'scatter',
    }], {
        title: '수익률 vs 불확실성',
        xaxis: {title: '불확실성 (%)', showgrid: true, gridcolor: '#eee'},
        yaxis: {title: '예측 수익률 (%)', showgrid: true, gridcolor: '#eee', zeroline: true, zerolinecolor: '#ccc'},
        margin: {t: 40, r: 20, b: 60, l: 60},
        height: 400,
        shapes: [{
            type: 'line', x0: 0, x1: 0, y0: 0, y1: 1,
            xref: 'x', yref: 'paper',
            line: {color: '#ccc', dash: 'dot'},
        }],
    }, {responsive: true});
})();
{% endif %}
</script>
</body>
</html>
```

- [ ] **Step 4: Implement ReportGenerator**

```python
# src/forecast/report.py
from __future__ import annotations

import os
from pathlib import Path

import plotly
from jinja2 import Environment, FileSystemLoader
from loguru import logger

from src.forecast.models import ForecastResult

# Re-export indicator names for the template
from src.forecast.macro_fetcher import INDICATOR_NAMES


class ReportGenerator:
    """Generates an interactive HTML forecast report."""

    def __init__(self):
        template_dir = Path(__file__).parent / "templates"
        self._env = Environment(loader=FileSystemLoader(str(template_dir)))

    def generate(
        self,
        macro_results: list[ForecastResult],
        stock_results: list[ForecastResult],
        ai_analyses: dict[str, dict],
        scan_date: str,
        output_dir: str = "reports",
        horizon: int = 60,
    ) -> str:
        """Generate HTML report and return the file path."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"forecast-{scan_date}.html")

        ranked_stocks = sorted(stock_results, key=lambda r: r.predicted_return, reverse=True)

        # Stocks that have AI analysis but aren't in the stock forecast list
        forecasted_tickers = {s.ticker for s in stock_results}
        ai_only_stocks = {
            ticker: data for ticker, data in ai_analyses.items()
            if ticker not in forecasted_tickers
        }

        # Inline plotly.js for offline use
        plotly_js = plotly.offline.get_plotlyjs()

        template = self._env.get_template("report.html")
        html = template.render(
            scan_date=scan_date,
            horizon=horizon,
            macro_results=macro_results,
            stock_results=stock_results,
            ranked_stocks=ranked_stocks,
            ai_analyses=ai_analyses,
            ai_only_stocks=ai_only_stocks,
            indicator_names=INDICATOR_NAMES,
            plotly_js=plotly_js,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"Report generated: {output_path}")
        return output_path
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_forecast_report.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/forecast/templates/report.html src/forecast/report.py tests/test_forecast_report.py
git commit -m "feat(forecast): add HTML report generator with Plotly charts"
```

---

### Task 8: Forecast CLI

**Files:**
- Create: `src/forecast/cli.py`
- Create: `tests/test_forecast_cli.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_forecast_cli.py
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from src.forecast.cli import app

runner = CliRunner()


def test_run_no_scan_result():
    """If no scan results exist, CLI should print an error."""
    with patch("src.forecast.cli.Database") as MockDB:
        mock_db = MagicMock()
        mock_db.get_scan_result_full.return_value = None
        MockDB.return_value = mock_db

        result = runner.invoke(app, ["run"])

    assert result.exit_code == 0
    assert "먼저" in result.stdout or "스캔" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_forecast_cli.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement forecast CLI**

```python
# src/forecast/cli.py
from __future__ import annotations

import webbrowser
from datetime import date, datetime

import typer
from rich.console import Console

from src.config import Settings, load_scanner_config
from src.db import Database

app = typer.Typer(help="TimesFM-based stock price forecast")
console = Console()


def _resolve_scan_date(target_date: str | None) -> date:
    if target_date:
        return datetime.strptime(target_date, "%Y%m%d").date()
    return date.today()


@app.command()
def run(
    target_date: str = typer.Option(None, "--date", "-d", help="Scan date to use (YYYYMMDD)"),
    horizon: int = typer.Option(None, "--horizon", "-h", help="Forecast horizon in trading days"),
):
    """Run forecast pipeline for scanned 52-week high stocks."""
    settings = Settings()
    config = load_scanner_config()
    forecast_horizon = horizon or config.forecast.horizon

    scan_date = _resolve_scan_date(target_date)
    date_str = scan_date.strftime("%Y%m%d")

    console.print(f"[bold]주가 예측 시작: {date_str} (horizon={forecast_horizon}일)[/bold]")

    # Step 1: Load scan results from DB
    db = Database()
    scan_result = db.get_scan_result_full(scan_date)
    if scan_result is None:
        console.print(
            f"[red]{date_str} 스캔 결과가 없습니다. "
            f"먼저 `python -m src.cli run --date {date_str}`을 실행하세요.[/red]"
        )
        return

    highs = scan_result.highs
    console.print(f"[dim]스캔 결과 로드: {len(highs)}개 종목[/dim]")

    # Load AI analyses
    all_ai = db.get_all_ai_analyses(scan_date)
    ai_map = {a.ticker: {"ai_analysis": a.ai_analysis, "news_summary": a.news_summary} for a in all_ai}

    # Step 2: Fetch data
    console.print("[dim]1/4 매크로 데이터 수집 중...[/dim]")
    from src.forecast.macro_fetcher import MacroFetcher
    macro_fetcher = MacroFetcher(
        ecos_api_key=settings.ecos_api_key,
        fred_api_key=settings.fred_api_key,
    )
    macro_data = macro_fetcher.fetch_all()

    console.print("[dim]2/4 종목 과거 데이터 수집 중...[/dim]")
    from src.forecast.stock_fetcher import StockFetcher
    from src.krx_client import create_krx_client
    client = create_krx_client(
        krx_id=settings.krx_id,
        krx_pw=settings.krx_pw,
        krx_api_key=settings.krx_api_key,
    )
    stock_fetcher = StockFetcher(client=client)
    tickers = [h.ticker for h in highs]
    stock_data = stock_fetcher.fetch_histories(tickers, date_str)

    # Step 3: Run predictions
    console.print("[dim]3/4 TimesFM 예측 실행 중...[/dim]")
    from src.forecast.predictor import Predictor
    from src.forecast.macro_fetcher import INDICATOR_NAMES

    predictor = Predictor(model_name=config.forecast.model, horizon=forecast_horizon)

    # Macro predictions
    macro_items = [
        {
            "ticker": name,
            "name": INDICATOR_NAMES.get(name, name),
            "category": "macro",
            "history": values,
            "dates_history": dates,
        }
        for name, (dates, values) in macro_data.items()
        if values
    ]
    macro_results = predictor.predict_batch(macro_items)

    # Stock predictions
    name_map = {h.ticker: h.name for h in highs}
    sector_map = {h.ticker: h.sector for h in highs}
    stock_items = [
        {
            "ticker": ticker,
            "name": name_map.get(ticker, ticker),
            "category": "stock",
            "sector": sector_map.get(ticker, ""),
            "history": values,
            "dates_history": dates,
        }
        for ticker, (dates, values) in stock_data.items()
    ]
    stock_results = predictor.predict_batch(stock_items)

    # Step 4: Generate report
    console.print("[dim]4/4 HTML 리포트 생성 중...[/dim]")
    from src.forecast.report import ReportGenerator

    generator = ReportGenerator()
    report_path = generator.generate(
        macro_results=macro_results,
        stock_results=stock_results,
        ai_analyses=ai_map,
        scan_date=str(scan_date),
        output_dir=config.forecast.report_dir,
        horizon=forecast_horizon,
    )

    console.print(f"[bold green]완료! 리포트: {report_path}[/bold green]")
    webbrowser.open(f"file://{os.path.abspath(report_path)}")


if __name__ == "__main__":
    app()
```

Add the missing import at the top:
```python
import os
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_forecast_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/forecast/cli.py tests/test_forecast_cli.py
git commit -m "feat(forecast): add CLI entrypoint for forecast pipeline"
```

---

### Task 9: Integration Test & Final Wiring

**Files:**
- Create: `tests/test_forecast_integration.py`
- Modify: `src/forecast/cli.py` (if needed)

- [ ] **Step 1: Write integration test (fully mocked external deps)**

```python
# tests/test_forecast_integration.py
"""End-to-end integration test with all external services mocked."""
from unittest.mock import patch, MagicMock
from datetime import date

import numpy as np
import pandas as pd

from src.forecast.models import ForecastResult
from src.forecast.macro_fetcher import MacroFetcher
from src.forecast.stock_fetcher import StockFetcher
from src.forecast.report import ReportGenerator


def test_full_pipeline_mocked(tmp_path):
    """Test the full pipeline: fetch -> predict -> report."""
    # 1. Mock macro data
    macro_fetcher = MacroFetcher(ecos_api_key="test", fred_api_key="test")

    mock_ecos_resp = MagicMock()
    mock_ecos_resp.json.return_value = {
        "StatisticSearch": {
            "row": [
                {"TIME": f"2026010{i}", "DATA_VALUE": str(2800 + i * 10)}
                for i in range(1, 6)
            ]
        }
    }
    mock_ecos_resp.raise_for_status = MagicMock()

    mock_fred_series = pd.Series(
        [4500.0 + i * 10 for i in range(5)],
        index=pd.to_datetime([f"2026-01-0{i}" for i in range(1, 6)]),
    )

    with patch("requests.get", return_value=mock_ecos_resp):
        ecos_data = macro_fetcher.fetch_all_ecos(lookback_days=400)

    with patch("fredapi.Fred") as MockFred:
        mock_fred = MagicMock()
        mock_fred.get_series.return_value = mock_fred_series
        MockFred.return_value = mock_fred
        fred_data = macro_fetcher.fetch_all_fred(lookback_days=400)

    all_macro = {**ecos_data, **fred_data}

    # 2. Build ForecastResults directly (skip actual TimesFM)
    macro_results = []
    for name, (dates, values) in all_macro.items():
        if not values:
            continue
        macro_results.append(ForecastResult(
            ticker=name,
            name=name,
            category="macro",
            history=values,
            dates_history=dates,
            forecast=[values[-1] + i for i in range(1, 4)],
            dates_forecast=["20260106", "20260107", "20260108"],
            quantile_low=[values[-1] - 10 + i for i in range(1, 4)],
            quantile_high=[values[-1] + 10 + i for i in range(1, 4)],
            predicted_return=1.5,
            uncertainty=3.0,
        ))

    stock_results = [
        ForecastResult(
            ticker="005930", name="삼성전자", category="stock",
            history=[50000.0, 51000.0, 52000.0],
            dates_history=["20260101", "20260102", "20260103"],
            forecast=[53000.0, 54000.0, 55000.0],
            dates_forecast=["20260106", "20260107", "20260108"],
            quantile_low=[51000.0, 52000.0, 53000.0],
            quantile_high=[55000.0, 56000.0, 57000.0],
            predicted_return=5.77,
            uncertainty=7.69,
        ),
    ]

    # 3. Generate report
    ai_analyses = {
        "005930": {
            "ai_analysis": "[상승 원인] 반도체 호황",
            "news_summary": "삼성전자 실적 발표",
        }
    }

    generator = ReportGenerator()
    path = generator.generate(
        macro_results=macro_results,
        stock_results=stock_results,
        ai_analyses=ai_analyses,
        scan_date="2026-04-16",
        output_dir=str(tmp_path),
    )

    assert path.endswith(".html")
    with open(path) as f:
        html = f.read()
    assert "삼성전자" in html
    assert "반도체 호황" in html
    assert len(html) > 1000  # should have substantial content
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_forecast_integration.py -v`
Expected: PASS

- [ ] **Step 3: Run all tests to verify nothing is broken**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_forecast_integration.py
git commit -m "test(forecast): add integration test for full forecast pipeline"
```

---

### Task 10: Documentation & .env Template

**Files:**
- Modify: `README.md`
- Modify: `.env.example` (or create if not exists)

- [ ] **Step 1: Add forecast section to README.md**

Add after the existing usage section:

```markdown
## 주가 예측 (Forecast)

52주 신고가 스캔 결과를 기반으로 TimesFM 모델을 이용한 주가/매크로 예측 및 HTML 리포트 생성.

### 추가 설정

`.env` 파일에 추가:
```
ECOS_API_KEY=your_ecos_api_key    # 한국은행 ECOS API
FRED_API_KEY=your_fred_api_key    # FRED API
```

### 사용법

```bash
# 스캐너 먼저 실행
python -m src.cli run

# 예측 실행 (가장 최근 스캔 결과 기반)
python -m src.forecast.cli run

# 특정 날짜 & 예측 기간 지정
python -m src.forecast.cli run --date 20260416 --horizon 40
```

리포트는 `reports/forecast-YYYY-MM-DD.html`에 생성되며 브라우저에서 자동으로 열립니다.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add forecast module usage to README"
```
