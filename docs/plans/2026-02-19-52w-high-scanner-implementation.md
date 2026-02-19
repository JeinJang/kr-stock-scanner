# 52-Week High Stock Scanner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a daily Korean stock market 52-week high scanner that collects data from KRX, identifies new highs across KOSPI/KOSDAQ/ETF, analyzes rise reasons with OpenAI GPT, and delivers reports via Telegram.

**Architecture:** Monolithic Python project in `kr-stock-scanner/` with modular separation. Each module handles one concern (collection, scanning, news, AI analysis, reporting). SQLite for persistence, typer for CLI, pykrx for market data.

**Tech Stack:** Python 3.11+, pykrx, SQLAlchemy, pydantic-settings, typer, openai, python-telegram-bot, httpx, rich, loguru, beautifulsoup4, pytest

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `config.yaml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `data/.gitkeep`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "kr-stock-scanner"
version = "0.1.0"
description = "Korean stock market 52-week high scanner with AI analysis"
requires-python = ">=3.11"

dependencies = [
    "pykrx>=1.0.45",
    "sqlalchemy>=2.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
    "typer>=0.9.0",
    "openai>=1.0.0",
    "python-telegram-bot>=20.0",
    "httpx>=0.27.0",
    "beautifulsoup4>=4.12.0",
    "rich>=13.0.0",
    "loguru>=0.7.0",
    "pandas>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.4.0",
    "mypy>=1.4.0",
]

[project.scripts]
kr-scanner = "src.cli:app"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

**Step 2: Create config.yaml**

```yaml
# KRX Scanner Configuration
scanner:
  markets: ["KOSPI", "KOSDAQ", "ETF"]
  lookback_days: 250  # ~52 weeks of trading days
  max_ai_analyze: 50  # Max stocks for AI analysis (cost cap)

news:
  provider: "naver_api"  # "naver_api" or "naver_finance"
  max_articles_per_stock: 5
  search_days: 3

ai:
  model: "gpt-4o-mini"
  max_tokens: 300

telegram:
  enabled: true

schedule:
  run_time: "16:00"
  timezone: "Asia/Seoul"
```

**Step 3: Create .env.example**

```
OPENAI_API_KEY=sk-...
TELEGRAM_BOT_TOKEN=7123456789:AAH1bGzR...
TELEGRAM_CHAT_ID=123456789
NAVER_CLIENT_ID=your_naver_client_id
NAVER_CLIENT_SECRET=your_naver_client_secret
```

**Step 4: Create .gitignore**

```
.env
*.db
__pycache__/
.venv/
*.pyc
data/scanner.db
.mypy_cache/
.pytest_cache/
.ruff_cache/
dist/
*.egg-info/
```

**Step 5: Create empty init files and data dir**

```bash
mkdir -p src tests data
touch src/__init__.py tests/__init__.py data/.gitkeep
```

**Step 6: Create virtual environment and install**

```bash
cd kr-stock-scanner
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Step 7: Verify installation**

Run: `python -c "import pykrx; import sqlalchemy; import typer; print('OK')"`
Expected: `OK`

**Step 8: Commit**

```bash
git add pyproject.toml config.yaml .env.example .gitignore src/__init__.py tests/__init__.py data/.gitkeep
git commit -m "chore: scaffold kr-stock-scanner project with dependencies"
```

---

## Task 2: Config Module

**Files:**
- Create: `src/config.py`
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
import os
import pytest
from pathlib import Path


def test_settings_loads_from_env(monkeypatch, tmp_path):
    """Settings should load API keys from environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-bot-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
    monkeypatch.setenv("NAVER_CLIENT_ID", "test-naver-id")
    monkeypatch.setenv("NAVER_CLIENT_SECRET", "test-naver-secret")

    from src.config import Settings
    settings = Settings()

    assert settings.openai_api_key == "sk-test-key"
    assert settings.telegram_bot_token == "test-bot-token"
    assert settings.telegram_chat_id == 123456
    assert settings.naver_client_id == "test-naver-id"
    assert settings.naver_client_secret == "test-naver-secret"


def test_scanner_config_loads_from_yaml(tmp_path):
    """ScannerConfig should load scanner settings from YAML."""
    yaml_content = """
scanner:
  markets: ["KOSPI", "KOSDAQ"]
  lookback_days: 200
  max_ai_analyze: 30
news:
  provider: "naver_api"
  max_articles_per_stock: 3
  search_days: 2
ai:
  model: "gpt-4o-mini"
  max_tokens: 200
telegram:
  enabled: false
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    from src.config import load_scanner_config
    config = load_scanner_config(config_file)

    assert config.scanner.markets == ["KOSPI", "KOSDAQ"]
    assert config.scanner.lookback_days == 200
    assert config.news.max_articles_per_stock == 3
    assert config.ai.model == "gpt-4o-mini"
    assert config.telegram.enabled is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (ModuleNotFoundError)

**Step 3: Write implementation**

```python
# src/config.py
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Secret settings loaded from environment variables / .env file."""

    openai_api_key: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: int = 0
    naver_client_id: str = ""
    naver_client_secret: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


class ScannerSection(BaseModel):
    markets: list[str] = ["KOSPI", "KOSDAQ", "ETF"]
    lookback_days: int = 250
    max_ai_analyze: int = 50


class NewsSection(BaseModel):
    provider: str = "naver_api"
    max_articles_per_stock: int = 5
    search_days: int = 3


class AISection(BaseModel):
    model: str = "gpt-4o-mini"
    max_tokens: int = 300


class TelegramSection(BaseModel):
    enabled: bool = True


class ScannerConfig(BaseModel):
    scanner: ScannerSection = ScannerSection()
    news: NewsSection = NewsSection()
    ai: AISection = AISection()
    telegram: TelegramSection = TelegramSection()


def load_scanner_config(path: Path | None = None) -> ScannerConfig:
    """Load scanner configuration from YAML file."""
    if path is None:
        path = Path("config.yaml")
    if not path.exists():
        return ScannerConfig()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return ScannerConfig(**data)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add config module with pydantic-settings"
```

---

## Task 3: Pydantic Data Models

**Files:**
- Create: `src/models.py`
- Test: `tests/test_models.py`

**Step 1: Write the failing test**

```python
# tests/test_models.py
from datetime import date


def test_stock_high_model():
    from src.models import StockHigh

    stock = StockHigh(
        ticker="005930",
        name="삼성전자",
        market="KOSPI",
        sector="전기전자",
        close_price=78500,
        high_52w=79000,
        prev_high_52w=77000,
        breakout_pct=2.60,
        volume=15000000,
        avg_volume_20d=12000000,
    )
    assert stock.ticker == "005930"
    assert stock.name == "삼성전자"
    assert stock.breakout_pct == 2.60


def test_scan_result_model():
    from src.models import ScanResult, StockHigh, MarketStats

    result = ScanResult(
        scan_date=date(2026, 2, 19),
        stats=MarketStats(
            total_stocks=2500,
            new_high_count=32,
            kospi_count=15,
            kosdaq_count=12,
            etf_count=5,
        ),
        highs=[],
        sector_breakdown={},
    )
    assert result.stats.new_high_count == 32
    assert result.scan_date == date(2026, 2, 19)


def test_ai_analysis_model():
    from src.models import AIAnalysisResult

    analysis = AIAnalysisResult(
        ticker="005930",
        news_summary="HBM4 수주 확대 기대감",
        ai_analysis="반도체 업황 회복과 HBM4 수주 확대에 따른 실적 개선 기대.",
    )
    assert analysis.ticker == "005930"
    assert "HBM4" in analysis.ai_analysis
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/models.py
from datetime import date
from pydantic import BaseModel


class StockHigh(BaseModel):
    """A single stock that hit a 52-week high."""

    ticker: str
    name: str
    market: str  # KOSPI, KOSDAQ, ETF
    sector: str
    close_price: float
    high_52w: float
    prev_high_52w: float
    breakout_pct: float  # % above previous 52w high
    volume: int
    avg_volume_20d: int


class MarketStats(BaseModel):
    """Aggregate market statistics for a scan."""

    total_stocks: int
    new_high_count: int
    kospi_count: int = 0
    kosdaq_count: int = 0
    etf_count: int = 0


class ScanResult(BaseModel):
    """Complete result of a daily 52-week high scan."""

    scan_date: date
    stats: MarketStats
    highs: list[StockHigh]
    sector_breakdown: dict[str, list[StockHigh]]  # sector -> stocks


class NewsArticle(BaseModel):
    """A single news article."""

    title: str
    link: str
    description: str = ""
    source: str = ""
    pub_date: str = ""


class AIAnalysisResult(BaseModel):
    """AI analysis result for a single stock."""

    ticker: str
    news_summary: str
    ai_analysis: str
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models.py tests/test_models.py
git commit -m "feat: add pydantic data models for scan results"
```

---

## Task 4: Database Module

**Files:**
- Create: `src/db.py`
- Test: `tests/test_db.py`

**Step 1: Write the failing test**

```python
# tests/test_db.py
from datetime import date
import pytest


@pytest.fixture
def db(tmp_path):
    from src.db import Database
    db_path = tmp_path / "test.db"
    return Database(f"sqlite:///{db_path}")


def test_save_and_load_scan_result(db):
    from src.models import ScanResult, StockHigh, MarketStats

    result = ScanResult(
        scan_date=date(2026, 2, 19),
        stats=MarketStats(
            total_stocks=2500,
            new_high_count=2,
            kospi_count=1,
            kosdaq_count=1,
            etf_count=0,
        ),
        highs=[
            StockHigh(
                ticker="005930", name="삼성전자", market="KOSPI",
                sector="전기전자", close_price=78500, high_52w=79000,
                prev_high_52w=77000, breakout_pct=2.60,
                volume=15000000, avg_volume_20d=12000000,
            ),
            StockHigh(
                ticker="035420", name="NAVER", market="KOSDAQ",
                sector="서비스업", close_price=220000, high_52w=222000,
                prev_high_52w=218000, breakout_pct=1.83,
                volume=3000000, avg_volume_20d=2500000,
            ),
        ],
        sector_breakdown={},
    )

    db.save_scan_result(result)
    loaded = db.get_scan_result(date(2026, 2, 19))

    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0]["ticker"] == "005930"


def test_get_new_high_count_history(db):
    from src.models import ScanResult, StockHigh, MarketStats

    for day_offset, count in [(1, 10), (2, 15), (3, 12)]:
        d = date(2026, 2, day_offset)
        result = ScanResult(
            scan_date=d,
            stats=MarketStats(
                total_stocks=2500, new_high_count=count,
                kospi_count=count, kosdaq_count=0, etf_count=0,
            ),
            highs=[],
            sector_breakdown={},
        )
        db.save_scan_result(result)

    history = db.get_high_count_history(days=3)
    assert len(history) == 3


def test_save_ai_analysis(db):
    from src.models import AIAnalysisResult

    analysis = AIAnalysisResult(
        ticker="005930",
        news_summary="HBM4 관련 뉴스",
        ai_analysis="반도체 업황 호조",
    )
    db.save_ai_analysis(date(2026, 2, 19), analysis)
    loaded = db.get_ai_analysis(date(2026, 2, 19), "005930")
    assert loaded is not None
    assert "반도체" in loaded["ai_analysis"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_db.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/db.py
from datetime import date, datetime

from sqlalchemy import (
    Column, Integer, String, Float, BigInteger, Date, DateTime, Text,
    create_engine, select, func,
)
from sqlalchemy.orm import DeclarativeBase, Session

from src.models import ScanResult, AIAnalysisResult


class Base(DeclarativeBase):
    pass


class DailyScan(Base):
    __tablename__ = "daily_scans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_date = Column(Date, nullable=False)
    total_stocks = Column(Integer, nullable=False)
    new_high_count = Column(Integer, nullable=False)
    market_type = Column(String(10), nullable=False)  # KOSPI/KOSDAQ/ETF/ALL


class NewHigh(Base):
    __tablename__ = "new_highs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_date = Column(Date, nullable=False, index=True)
    ticker = Column(String(10), nullable=False)
    name = Column(String(100), nullable=False)
    market = Column(String(10), nullable=False)
    sector = Column(String(50), nullable=False)
    close_price = Column(Float, nullable=False)
    high_52w = Column(Float, nullable=False)
    prev_high_52w = Column(Float, nullable=False)
    breakout_pct = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)
    avg_volume_20d = Column(BigInteger, nullable=False)


class AIAnalysis(Base):
    __tablename__ = "ai_analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False)
    scan_date = Column(Date, nullable=False, index=True)
    news_summary = Column(Text, nullable=False)
    ai_analysis = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now)


class Database:
    def __init__(self, url: str = "sqlite:///data/scanner.db"):
        self.engine = create_engine(url)
        Base.metadata.create_all(self.engine)

    def save_scan_result(self, result: ScanResult) -> None:
        with Session(self.engine) as session:
            # Save aggregate stats
            session.add(DailyScan(
                scan_date=result.scan_date,
                total_stocks=result.stats.total_stocks,
                new_high_count=result.stats.new_high_count,
                market_type="ALL",
            ))
            # Save individual new highs
            for stock in result.highs:
                session.add(NewHigh(
                    scan_date=result.scan_date,
                    ticker=stock.ticker,
                    name=stock.name,
                    market=stock.market,
                    sector=stock.sector,
                    close_price=stock.close_price,
                    high_52w=stock.high_52w,
                    prev_high_52w=stock.prev_high_52w,
                    breakout_pct=stock.breakout_pct,
                    volume=stock.volume,
                    avg_volume_20d=stock.avg_volume_20d,
                ))
            session.commit()

    def get_scan_result(self, scan_date: date) -> list[dict]:
        with Session(self.engine) as session:
            rows = session.execute(
                select(NewHigh).where(NewHigh.scan_date == scan_date)
            ).scalars().all()
            return [
                {
                    "ticker": r.ticker, "name": r.name, "market": r.market,
                    "sector": r.sector, "close_price": r.close_price,
                    "high_52w": r.high_52w, "breakout_pct": r.breakout_pct,
                }
                for r in rows
            ]

    def get_high_count_history(self, days: int = 5) -> list[dict]:
        with Session(self.engine) as session:
            rows = session.execute(
                select(DailyScan)
                .order_by(DailyScan.scan_date.desc())
                .limit(days)
            ).scalars().all()
            return [
                {"date": r.scan_date, "count": r.new_high_count}
                for r in reversed(rows)
            ]

    def save_ai_analysis(self, scan_date: date, analysis: AIAnalysisResult) -> None:
        with Session(self.engine) as session:
            session.add(AIAnalysis(
                ticker=analysis.ticker,
                scan_date=scan_date,
                news_summary=analysis.news_summary,
                ai_analysis=analysis.ai_analysis,
            ))
            session.commit()

    def get_ai_analysis(self, scan_date: date, ticker: str) -> dict | None:
        with Session(self.engine) as session:
            row = session.execute(
                select(AIAnalysis).where(
                    AIAnalysis.scan_date == scan_date,
                    AIAnalysis.ticker == ticker,
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            return {
                "ticker": row.ticker,
                "news_summary": row.news_summary,
                "ai_analysis": row.ai_analysis,
            }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_db.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/db.py tests/test_db.py
git commit -m "feat: add SQLite database module with SQLAlchemy ORM"
```

---

## Task 5: Data Collector Module

**Files:**
- Create: `src/collector.py`
- Test: `tests/test_collector.py`

**Step 1: Write the failing test**

```python
# tests/test_collector.py
from datetime import date
from unittest.mock import patch, MagicMock
import pandas as pd


def test_collect_daily_ohlcv():
    """Collector should fetch OHLCV for KOSPI, KOSDAQ, and ETF."""
    from src.collector import Collector

    # Mock pykrx responses
    kospi_df = pd.DataFrame(
        {"시가": [70000], "고가": [72000], "저가": [69000], "종가": [71500],
         "거래량": [15000000], "거래대금": [1000000000], "등락률": [1.5]},
        index=["005930"],
    )
    kosdaq_df = pd.DataFrame(
        {"시가": [50000], "고가": [52000], "저가": [49000], "종가": [51000],
         "거래량": [3000000], "거래대금": [150000000], "등락률": [2.0]},
        index=["035420"],
    )
    etf_df = pd.DataFrame(
        {"NAV": [50000], "시가": [49800], "고가": [50500], "저가": [49500],
         "종가": [50200], "거래량": [500000], "거래대금": [25000000], "기초지수": [2500]},
        index=["069500"],
    )

    with patch("src.collector.stock") as mock_stock:
        mock_stock.get_market_ohlcv_by_ticker.side_effect = [kospi_df, kosdaq_df]
        mock_stock.get_etf_ohlcv_by_ticker.return_value = etf_df

        collector = Collector()
        result = collector.collect_daily("20260219", markets=["KOSPI", "KOSDAQ", "ETF"])

    assert "005930" in result
    assert result["005930"]["market"] == "KOSPI"
    assert result["005930"]["close"] == 71500
    assert result["005930"]["high"] == 72000
    assert "069500" in result
    assert result["069500"]["market"] == "ETF"


def test_collect_historical_high():
    """Collector should fetch 52-week high for a ticker."""
    from src.collector import Collector

    hist_df = pd.DataFrame(
        {"시가": [65000, 68000, 70000], "고가": [66000, 69000, 72000],
         "저가": [64000, 67000, 69000], "종가": [65500, 68500, 71500],
         "거래량": [10000000, 12000000, 15000000]},
        index=pd.to_datetime(["2025-06-01", "2025-12-01", "2026-02-19"]),
    )

    with patch("src.collector.stock") as mock_stock:
        mock_stock.get_market_ohlcv_by_date.return_value = hist_df

        collector = Collector()
        high_52w = collector.get_52w_high("005930", "20260219", lookback=250)

    assert high_52w == 72000


def test_collect_sector_info():
    """Collector should fetch sector classifications."""
    from src.collector import Collector

    sector_df = pd.DataFrame(
        {"종목명": ["삼성전자", "NAVER"], "업종명": ["전기전자", "서비스업"],
         "종가": [71500, 220000], "대비": [1000, 3000],
         "등락률": [1.42, 1.38], "시가총액": [450000000000000, 36000000000000]},
        index=["005930", "035420"],
    )

    with patch("src.collector.stock") as mock_stock:
        mock_stock.get_market_sector_classifications.return_value = sector_df

        collector = Collector()
        sectors = collector.get_sector_map("20260219", "KOSPI")

    assert sectors["005930"] == "전기전자"
    assert sectors["035420"] == "서비스업"


def test_collect_market_cap():
    """Collector should fetch market cap data."""
    from src.collector import Collector

    cap_df = pd.DataFrame(
        {"시가총액": [450000000000000, 36000000000000],
         "거래량": [15000000, 3000000], "거래대금": [1000000000, 600000000],
         "상장주식수": [5969782550, 163750000], "외국인보유주식수": [0, 0]},
        index=["005930", "035420"],
    )

    with patch("src.collector.stock") as mock_stock:
        mock_stock.get_market_cap_by_ticker.return_value = cap_df

        collector = Collector()
        caps = collector.get_market_caps("20260219")

    assert caps["005930"] == 450000000000000
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_collector.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/collector.py
import time

import pandas as pd
from loguru import logger
from pykrx import stock


class Collector:
    """Collects stock market data from KRX via pykrx."""

    def collect_daily(
        self, date_str: str, markets: list[str] | None = None
    ) -> dict[str, dict]:
        """Fetch daily OHLCV for all tickers across specified markets.

        Returns dict: ticker -> {market, open, high, low, close, volume, change_pct}
        """
        if markets is None:
            markets = ["KOSPI", "KOSDAQ", "ETF"]

        result = {}

        for market in markets:
            if market == "ETF":
                df = stock.get_etf_ohlcv_by_ticker(date_str)
                for ticker in df.index:
                    result[ticker] = {
                        "market": "ETF",
                        "open": int(df.loc[ticker, "시가"]),
                        "high": int(df.loc[ticker, "고가"]),
                        "low": int(df.loc[ticker, "저가"]),
                        "close": int(df.loc[ticker, "종가"]),
                        "volume": int(df.loc[ticker, "거래량"]),
                        "change_pct": 0.0,
                    }
            else:
                df = stock.get_market_ohlcv_by_ticker(date_str, market=market)
                for ticker in df.index:
                    result[ticker] = {
                        "market": market,
                        "open": int(df.loc[ticker, "시가"]),
                        "high": int(df.loc[ticker, "고가"]),
                        "low": int(df.loc[ticker, "저가"]),
                        "close": int(df.loc[ticker, "종가"]),
                        "volume": int(df.loc[ticker, "거래량"]),
                        "change_pct": float(df.loc[ticker, "등락률"]),
                    }

        logger.info(f"Collected daily data for {len(result)} tickers on {date_str}")
        return result

    def get_52w_high(self, ticker: str, date_str: str, lookback: int = 250) -> float:
        """Get the 52-week high price for a ticker.

        Uses get_market_ohlcv_by_date to fetch historical data.
        Returns the max '고가' (high) over the lookback period.
        """
        from datetime import datetime, timedelta

        end = datetime.strptime(date_str, "%Y%m%d")
        start = end - timedelta(days=int(lookback * 1.5))  # calendar days buffer
        start_str = start.strftime("%Y%m%d")

        df = stock.get_market_ohlcv_by_date(start_str, date_str, ticker)
        if df.empty:
            return 0.0
        return float(df["고가"].max())

    def get_sector_map(self, date_str: str, market: str) -> dict[str, str]:
        """Get ticker -> sector name mapping for a market.

        Returns dict: ticker -> sector_name
        """
        df = stock.get_market_sector_classifications(date_str, market)
        return {ticker: row["업종명"] for ticker, row in df.iterrows()}

    def get_market_caps(self, date_str: str, market: str = "ALL") -> dict[str, int]:
        """Get ticker -> market cap mapping.

        Returns dict: ticker -> market_cap (KRW)
        """
        df = stock.get_market_cap_by_ticker(date_str, market=market)
        return {ticker: int(row["시가총액"]) for ticker, row in df.iterrows()}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_collector.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/collector.py tests/test_collector.py
git commit -m "feat: add KRX data collector module using pykrx"
```

---

## Task 6: Scanner Module (52-Week High Detection)

**Files:**
- Create: `src/scanner.py`
- Test: `tests/test_scanner.py`

**Step 1: Write the failing test**

```python
# tests/test_scanner.py
from datetime import date
from unittest.mock import MagicMock, patch


def test_find_new_highs():
    """Scanner should identify stocks at 52-week highs."""
    from src.scanner import Scanner
    from src.models import StockHigh

    daily_data = {
        "005930": {
            "market": "KOSPI", "open": 70000, "high": 79000,
            "low": 69000, "close": 78500, "volume": 15000000, "change_pct": 2.0,
        },
        "000660": {
            "market": "KOSPI", "open": 200000, "high": 210000,
            "low": 199000, "close": 205000, "volume": 5000000, "change_pct": 1.5,
        },
        "035420": {
            "market": "KOSDAQ", "open": 210000, "high": 215000,
            "low": 208000, "close": 212000, "volume": 3000000, "change_pct": -0.5,
        },
    }

    sector_map = {
        "005930": "전기전자",
        "000660": "전기전자",
        "035420": "서비스업",
    }

    name_map = {
        "005930": "삼성전자",
        "000660": "SK하이닉스",
        "035420": "NAVER",
    }

    mock_collector = MagicMock()
    # 005930: high=79000, 52w_prev=77000 -> NEW HIGH
    # 000660: high=210000, 52w_prev=215000 -> NOT new high
    # 035420: high=215000, 52w_prev=214000 -> NEW HIGH
    mock_collector.get_52w_high.side_effect = lambda t, d, **kw: {
        "005930": 79000,   # today's high matches -> new high (prev was 77000)
        "000660": 215000,  # 52w high is higher than today
        "035420": 215000,  # today's high matches
    }[t]

    scanner = Scanner(collector=mock_collector)
    highs = scanner.find_new_highs(
        daily_data=daily_data,
        date_str="20260219",
        sector_map=sector_map,
        name_map=name_map,
        prev_highs={"005930": 77000, "000660": 215000, "035420": 214000},
    )

    tickers = [h.ticker for h in highs]
    assert "005930" in tickers
    assert "035420" in tickers
    assert "000660" not in tickers


def test_build_scan_result():
    """Scanner should produce a complete ScanResult with stats."""
    from src.scanner import Scanner
    from src.models import StockHigh

    mock_collector = MagicMock()
    scanner = Scanner(collector=mock_collector)

    highs = [
        StockHigh(
            ticker="005930", name="삼성전자", market="KOSPI", sector="전기전자",
            close_price=78500, high_52w=79000, prev_high_52w=77000,
            breakout_pct=2.60, volume=15000000, avg_volume_20d=12000000,
        ),
        StockHigh(
            ticker="035420", name="NAVER", market="KOSDAQ", sector="서비스업",
            close_price=212000, high_52w=215000, prev_high_52w=214000,
            breakout_pct=0.47, volume=3000000, avg_volume_20d=2500000,
        ),
    ]

    result = scanner.build_scan_result(
        scan_date=date(2026, 2, 19),
        highs=highs,
        total_stocks=2500,
    )

    assert result.stats.new_high_count == 2
    assert result.stats.kospi_count == 1
    assert result.stats.kosdaq_count == 1
    assert "전기전자" in result.sector_breakdown
    assert len(result.sector_breakdown["전기전자"]) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_scanner.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/scanner.py
from collections import defaultdict
from datetime import date

from loguru import logger

from src.collector import Collector
from src.models import StockHigh, ScanResult, MarketStats


class Scanner:
    """Identifies 52-week high stocks from daily market data."""

    def __init__(self, collector: Collector | None = None):
        self.collector = collector or Collector()

    def find_new_highs(
        self,
        daily_data: dict[str, dict],
        date_str: str,
        sector_map: dict[str, str],
        name_map: dict[str, str],
        prev_highs: dict[str, float] | None = None,
        lookback: int = 250,
    ) -> list[StockHigh]:
        """Find stocks that hit 52-week highs on the given date.

        Args:
            daily_data: ticker -> {market, high, close, volume, ...}
            date_str: Date string in YYYYMMDD format
            sector_map: ticker -> sector name
            name_map: ticker -> stock name
            prev_highs: ticker -> previous 52w high (before today).
                If None, fetched via collector for each ticker.
            lookback: Number of trading days for 52-week calculation
        """
        highs = []

        for ticker, data in daily_data.items():
            today_high = data["high"]
            if today_high == 0:
                continue

            # Get 52-week high (includes today)
            high_52w = self.collector.get_52w_high(ticker, date_str, lookback=lookback)

            if today_high >= high_52w and high_52w > 0:
                prev_high = prev_highs.get(ticker, high_52w) if prev_highs else high_52w
                breakout_pct = (
                    ((today_high - prev_high) / prev_high * 100)
                    if prev_high > 0 else 0.0
                )

                highs.append(StockHigh(
                    ticker=ticker,
                    name=name_map.get(ticker, ticker),
                    market=data["market"],
                    sector=sector_map.get(ticker, "기타"),
                    close_price=data["close"],
                    high_52w=today_high,
                    prev_high_52w=prev_high,
                    breakout_pct=round(breakout_pct, 2),
                    volume=data["volume"],
                    avg_volume_20d=0,  # filled later if needed
                ))

        logger.info(f"Found {len(highs)} stocks at 52-week high on {date_str}")
        return highs

    def build_scan_result(
        self,
        scan_date: date,
        highs: list[StockHigh],
        total_stocks: int,
    ) -> ScanResult:
        """Build a ScanResult with stats and sector breakdown."""
        kospi = [h for h in highs if h.market == "KOSPI"]
        kosdaq = [h for h in highs if h.market == "KOSDAQ"]
        etf = [h for h in highs if h.market == "ETF"]

        sector_breakdown: dict[str, list[StockHigh]] = defaultdict(list)
        for h in highs:
            sector_breakdown[h.sector].append(h)

        return ScanResult(
            scan_date=scan_date,
            stats=MarketStats(
                total_stocks=total_stocks,
                new_high_count=len(highs),
                kospi_count=len(kospi),
                kosdaq_count=len(kosdaq),
                etf_count=len(etf),
            ),
            highs=highs,
            sector_breakdown=dict(sector_breakdown),
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_scanner.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/scanner.py tests/test_scanner.py
git commit -m "feat: add 52-week high scanner module"
```

---

## Task 7: News Fetcher Module

**Files:**
- Create: `src/news_fetcher.py`
- Test: `tests/test_news_fetcher.py`

**Step 1: Write the failing test**

```python
# tests/test_news_fetcher.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx


@pytest.mark.asyncio
async def test_fetch_naver_api_news():
    """NewsFetcher should parse Naver Search API response correctly."""
    from src.news_fetcher import NewsFetcher

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "items": [
            {
                "title": "<b>삼성전자</b> HBM4 수주 확대",
                "originallink": "https://example.com/1",
                "link": "https://n.news.naver.com/1",
                "description": "<b>삼성전자</b>가 HBM4 양산 준비에 박차를 가하고 있다.",
                "pubDate": "Wed, 19 Feb 2026 10:00:00 +0900",
            },
            {
                "title": "반도체 업황 개선 신호",
                "originallink": "https://example.com/2",
                "link": "https://n.news.naver.com/2",
                "description": "글로벌 반도체 시장이 회복세를 보이고 있다.",
                "pubDate": "Wed, 19 Feb 2026 09:00:00 +0900",
            },
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("src.news_fetcher.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        fetcher = NewsFetcher(
            naver_client_id="test-id",
            naver_client_secret="test-secret",
        )
        articles = await fetcher.fetch_news("삼성전자", max_articles=5)

    assert len(articles) == 2
    assert "삼성전자" in articles[0].title  # HTML tags stripped
    assert "<b>" not in articles[0].title
    assert articles[0].link == "https://n.news.naver.com/1"


@pytest.mark.asyncio
async def test_fetch_news_for_stocks():
    """Should fetch news for multiple stocks."""
    from src.news_fetcher import NewsFetcher
    from src.models import NewsArticle

    fetcher = NewsFetcher(naver_client_id="test", naver_client_secret="test")

    async def mock_fetch(query, max_articles=5):
        return [NewsArticle(title=f"{query} 뉴스", link="https://test.com", description="test")]

    fetcher.fetch_news = mock_fetch

    results = await fetcher.fetch_news_for_stocks(["삼성전자", "SK하이닉스"])

    assert "삼성전자" in results
    assert "SK하이닉스" in results
    assert len(results["삼성전자"]) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_news_fetcher.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/news_fetcher.py
import asyncio
import re

import httpx
from loguru import logger

from src.models import NewsArticle


def _strip_html(text: str) -> str:
    """Remove HTML tags from Naver API response text."""
    return re.sub(r"<[^>]+>", "", text).strip()


class NewsFetcher:
    """Fetches stock news from Naver Search API."""

    def __init__(self, naver_client_id: str, naver_client_secret: str):
        self.client_id = naver_client_id
        self.client_secret = naver_client_secret

    async def fetch_news(self, query: str, max_articles: int = 5) -> list[NewsArticle]:
        """Fetch news articles for a stock name from Naver Search API."""
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
        }
        params = {
            "query": query,
            "display": max_articles,
            "sort": "date",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()

        data = response.json()
        articles = []
        for item in data.get("items", []):
            articles.append(NewsArticle(
                title=_strip_html(item["title"]),
                link=item.get("link", item.get("originallink", "")),
                description=_strip_html(item.get("description", "")),
                source="",
                pub_date=item.get("pubDate", ""),
            ))

        logger.debug(f"Fetched {len(articles)} articles for '{query}'")
        return articles

    async def fetch_news_for_stocks(
        self, stock_names: list[str], max_articles: int = 5, delay: float = 0.2
    ) -> dict[str, list[NewsArticle]]:
        """Fetch news for multiple stocks with rate limiting."""
        results = {}
        for name in stock_names:
            try:
                articles = await self.fetch_news(name, max_articles)
                results[name] = articles
            except Exception as e:
                logger.warning(f"Failed to fetch news for '{name}': {e}")
                results[name] = []
            await asyncio.sleep(delay)
        return results
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_news_fetcher.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/news_fetcher.py tests/test_news_fetcher.py
git commit -m "feat: add Naver News fetcher module"
```

---

## Task 8: AI Analyst Module (OpenAI GPT)

**Files:**
- Create: `src/ai_analyst.py`
- Test: `tests/test_ai_analyst.py`

**Step 1: Write the failing test**

```python
# tests/test_ai_analyst.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_analyze_stock():
    """AIAnalyst should call OpenAI API and return analysis."""
    from src.ai_analyst import AIAnalyst
    from src.models import StockHigh, NewsArticle

    stock = StockHigh(
        ticker="005930", name="삼성전자", market="KOSPI", sector="전기전자",
        close_price=78500, high_52w=79000, prev_high_52w=77000,
        breakout_pct=2.60, volume=15000000, avg_volume_20d=12000000,
    )
    news = [
        NewsArticle(title="삼성전자 HBM4 수주 확대", link="https://test.com",
                     description="HBM4 양산 준비 박차"),
    ]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(
        content="HBM4 수주 확대에 따른 실적 개선 기대감이 52주 신고가를 견인했습니다."
    ))]

    with patch("src.ai_analyst.openai.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_cls.return_value = mock_client

        analyst = AIAnalyst(api_key="test-key", model="gpt-4o-mini")
        result = await analyst.analyze_stock(stock, news)

    assert result.ticker == "005930"
    assert "HBM4" in result.ai_analysis
    assert "HBM4" in result.news_summary


@pytest.mark.asyncio
async def test_analyze_stocks_respects_max_limit():
    """Should only analyze up to max_analyze stocks by market cap."""
    from src.ai_analyst import AIAnalyst
    from src.models import StockHigh, NewsArticle

    stocks = [
        StockHigh(
            ticker=f"00{i}000", name=f"Stock{i}", market="KOSPI", sector="테스트",
            close_price=10000 * i, high_52w=10000 * i, prev_high_52w=9000 * i,
            breakout_pct=11.1, volume=1000000, avg_volume_20d=800000,
        )
        for i in range(1, 6)
    ]
    news_map = {s.name: [] for s in stocks}
    market_caps = {s.ticker: s.close_price * 1000000 for s in stocks}

    with patch("src.ai_analyst.openai.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="분석 결과"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_cls.return_value = mock_client

        analyst = AIAnalyst(api_key="test-key", model="gpt-4o-mini")
        results = await analyst.analyze_stocks(
            stocks, news_map, market_caps, max_analyze=3
        )

    assert len(results) == 3  # Only top 3 by market cap
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ai_analyst.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/ai_analyst.py
import asyncio

import openai
from loguru import logger

from src.models import StockHigh, NewsArticle, AIAnalysisResult


class AIAnalyst:
    """Analyzes stock rise reasons using OpenAI GPT."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", max_tokens: int = 300):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    async def analyze_stock(
        self, stock: StockHigh, news: list[NewsArticle]
    ) -> AIAnalysisResult:
        """Analyze why a stock hit its 52-week high using news context."""
        news_text = "\n".join(
            f"- {a.title}: {a.description}" for a in news
        ) if news else "관련 뉴스 없음"

        prompt = f"""다음 종목이 52주 신고가를 기록했습니다. 관련 뉴스를 바탕으로 상승 이유를 1-2문장으로 분석해주세요.

종목: {stock.name} ({stock.ticker})
시장: {stock.market} / 섹터: {stock.sector}
종가: {stock.close_price:,.0f}원
52주 신고가: {stock.high_52w:,.0f}원 (전고점 대비 +{stock.breakout_pct:.1f}%)
거래량: {stock.volume:,}주

최근 뉴스:
{news_text}

분석 (1-2문장, 한국어):"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.3,
        )

        analysis = response.choices[0].message.content.strip()

        return AIAnalysisResult(
            ticker=stock.ticker,
            news_summary=news_text[:500],
            ai_analysis=analysis,
        )

    async def analyze_stocks(
        self,
        stocks: list[StockHigh],
        news_map: dict[str, list[NewsArticle]],
        market_caps: dict[str, int],
        max_analyze: int = 50,
    ) -> list[AIAnalysisResult]:
        """Analyze multiple stocks, limited by max_analyze (sorted by market cap)."""
        # Sort by market cap descending, take top N
        sorted_stocks = sorted(
            stocks,
            key=lambda s: market_caps.get(s.ticker, 0),
            reverse=True,
        )[:max_analyze]

        results = []
        for stock in sorted_stocks:
            try:
                news = news_map.get(stock.name, [])
                result = await self.analyze_stock(stock, news)
                results.append(result)
            except Exception as e:
                logger.warning(f"AI analysis failed for {stock.ticker}: {e}")
            await asyncio.sleep(0.5)  # Rate limiting

        logger.info(f"Completed AI analysis for {len(results)}/{len(stocks)} stocks")
        return results
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ai_analyst.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ai_analyst.py tests/test_ai_analyst.py
git commit -m "feat: add OpenAI GPT-based AI analyst module"
```

---

## Task 9: Telegram Reporter Module

**Files:**
- Create: `src/reporter.py`
- Test: `tests/test_reporter.py`

**Step 1: Write the failing test**

```python
# tests/test_reporter.py
import pytest
from datetime import date
from unittest.mock import AsyncMock, patch, MagicMock


def test_format_report():
    """Reporter should format a ScanResult into a readable Telegram message."""
    from src.reporter import Reporter
    from src.models import ScanResult, StockHigh, MarketStats, AIAnalysisResult

    result = ScanResult(
        scan_date=date(2026, 2, 19),
        stats=MarketStats(
            total_stocks=2500, new_high_count=2,
            kospi_count=1, kosdaq_count=1, etf_count=0,
        ),
        highs=[
            StockHigh(
                ticker="005930", name="삼성전자", market="KOSPI", sector="전기전자",
                close_price=78500, high_52w=79000, prev_high_52w=77000,
                breakout_pct=2.60, volume=15000000, avg_volume_20d=12000000,
            ),
            StockHigh(
                ticker="035420", name="NAVER", market="KOSDAQ", sector="서비스업",
                close_price=212000, high_52w=215000, prev_high_52w=214000,
                breakout_pct=0.47, volume=3000000, avg_volume_20d=2500000,
            ),
        ],
        sector_breakdown={
            "전기전자": [StockHigh(
                ticker="005930", name="삼성전자", market="KOSPI", sector="전기전자",
                close_price=78500, high_52w=79000, prev_high_52w=77000,
                breakout_pct=2.60, volume=15000000, avg_volume_20d=12000000,
            )],
            "서비스업": [StockHigh(
                ticker="035420", name="NAVER", market="KOSDAQ", sector="서비스업",
                close_price=212000, high_52w=215000, prev_high_52w=214000,
                breakout_pct=0.47, volume=3000000, avg_volume_20d=2500000,
            )],
        },
    )

    ai_analyses = [
        AIAnalysisResult(
            ticker="005930",
            news_summary="HBM4 수주 확대",
            ai_analysis="HBM4 수주 확대에 따른 실적 개선 기대감.",
        ),
    ]
    trend = [{"date": date(2026, 2, i), "count": c} for i, c in
             [(17, 18), (18, 24), (19, 32)]]

    reporter = Reporter(bot_token="test", chat_id=123)
    text = reporter.format_report(result, ai_analyses, trend)

    assert "52주 신고가 리포트" in text
    assert "2026-02-19" in text
    assert "삼성전자" in text
    assert "NAVER" in text
    assert "전기전자" in text
    assert "HBM4" in text


def test_split_message():
    """Should split long messages respecting 4096 char limit."""
    from src.reporter import split_message

    short = "Hello"
    assert split_message(short) == ["Hello"]

    long_text = "line\n" * 2000  # ~10000 chars
    chunks = split_message(long_text, max_length=4096)
    assert all(len(c) <= 4096 for c in chunks)
    assert len(chunks) > 1


@pytest.mark.asyncio
async def test_send_report():
    """Reporter should send message via Telegram Bot."""
    from src.reporter import Reporter

    with patch("src.reporter.Bot") as mock_bot_cls:
        mock_bot = AsyncMock()
        mock_bot_cls.return_value = mock_bot

        reporter = Reporter(bot_token="test-token", chat_id=123456)
        await reporter.send("Test message")

        mock_bot.send_message.assert_called_once()
        call_kwargs = mock_bot.send_message.call_args[1]
        assert call_kwargs["chat_id"] == 123456
        assert call_kwargs["text"] == "Test message"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_reporter.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/reporter.py
import asyncio
from datetime import date

from loguru import logger
from telegram import Bot
from telegram.constants import ParseMode

from src.models import ScanResult, AIAnalysisResult


def split_message(text: str, max_length: int = 4096) -> list[str]:
    """Split a long message into chunks that fit Telegram's limit."""
    if len(text) <= max_length:
        return [text]

    chunks = []
    current = ""

    for line in text.split("\n"):
        if len(current) + len(line) + 1 > max_length:
            if current:
                chunks.append(current)
                current = ""
            while len(line) > max_length:
                chunks.append(line[:max_length])
                line = line[max_length:]
            current = line
        else:
            current = f"{current}\n{line}" if current else line

    if current:
        chunks.append(current)
    return chunks


class Reporter:
    """Formats and sends reports via Telegram."""

    def __init__(self, bot_token: str, chat_id: int):
        self.bot_token = bot_token
        self.chat_id = chat_id

    def format_report(
        self,
        result: ScanResult,
        ai_analyses: list[AIAnalysisResult],
        trend: list[dict],
    ) -> str:
        """Format a ScanResult into a Telegram message string."""
        s = result.stats
        lines = []

        # Header
        lines.append(f"📊 52주 신고가 리포트 ({result.scan_date})")
        lines.append("")

        # Market Summary
        lines.append("■ 시장 요약")
        lines.append(f"• 신고가 종목: {s.new_high_count}개 "
                      f"(KOSPI {s.kospi_count} / KOSDAQ {s.kosdaq_count} / ETF {s.etf_count})")
        if len(trend) >= 2:
            prev = trend[-2]["count"]
            diff = s.new_high_count - prev
            sign = "+" if diff >= 0 else ""
            lines.append(f"• 전일 대비: {sign}{diff}개")
        if trend:
            trend_str = "→".join(str(t["count"]) for t in trend)
            lines.append(f"• 최근 추이: {trend_str}")
        lines.append("")

        # Sector breakdown
        sorted_sectors = sorted(
            result.sector_breakdown.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )
        lines.append("■ 섹터별 TOP")
        for i, (sector, stocks) in enumerate(sorted_sectors[:5], 1):
            names = ", ".join(s.name for s in stocks[:3])
            suffix = "..." if len(stocks) > 3 else ""
            lines.append(f"{i}. {sector} ({len(stocks)}종목): {names}{suffix}")
        lines.append("")

        # AI Analysis section
        ai_map = {a.ticker: a for a in ai_analyses}
        if ai_analyses:
            lines.append("■ 주요 종목 AI 분석")
            for stock in result.highs:
                if stock.ticker in ai_map:
                    a = ai_map[stock.ticker]
                    lines.append(
                        f"🔹 {stock.name} ({stock.ticker}) | "
                        f"{stock.close_price:,.0f}원 | +{stock.breakout_pct:.1f}%"
                    )
                    lines.append(f"   📰 {a.ai_analysis}")
                    lines.append("")
            lines.append("")

        # Full list
        lines.append("■ 전체 52주 신고가 목록")
        for stock in sorted(result.highs, key=lambda h: h.breakout_pct, reverse=True):
            lines.append(
                f"  {stock.name} | {stock.close_price:,.0f}원 | "
                f"+{stock.breakout_pct:.1f}% | {stock.sector}"
            )

        return "\n".join(lines)

    async def send(self, text: str) -> None:
        """Send a message via Telegram Bot, splitting if needed."""
        bot = Bot(token=self.bot_token)
        for chunk in split_message(text):
            await bot.send_message(
                chat_id=self.chat_id,
                text=chunk,
            )
            await asyncio.sleep(0.5)

    async def send_report(
        self,
        result: ScanResult,
        ai_analyses: list[AIAnalysisResult],
        trend: list[dict],
    ) -> None:
        """Format and send a complete scan report."""
        text = self.format_report(result, ai_analyses, trend)
        await self.send(text)
        logger.info(f"Report sent to Telegram chat {self.chat_id}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_reporter.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/reporter.py tests/test_reporter.py
git commit -m "feat: add Telegram reporter module with message formatting"
```

---

## Task 10: CLI Module

**Files:**
- Create: `src/cli.py`
- Test: `tests/test_cli.py`

**Step 1: Write the failing test**

```python
# tests/test_cli.py
from unittest.mock import patch, MagicMock, AsyncMock
from typer.testing import CliRunner


runner = CliRunner()


def test_cli_run_command_exists():
    """CLI should have a 'run' command."""
    from src.cli import app
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "run" in result.output.lower() or "full" in result.output.lower()


def test_cli_collect_command_exists():
    """CLI should have a 'collect' command."""
    from src.cli import app
    result = runner.invoke(app, ["collect", "--help"])
    assert result.exit_code == 0


def test_cli_history_command_exists():
    """CLI should have a 'history' command."""
    from src.cli import app
    result = runner.invoke(app, ["history", "--help"])
    assert result.exit_code == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/cli.py
import asyncio
from datetime import date, datetime

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.config import Settings, load_scanner_config

app = typer.Typer(help="Korean Stock Market 52-Week High Scanner")
console = Console()


def _get_date_str(target_date: date | None = None) -> str:
    """Convert date to YYYYMMDD string."""
    d = target_date or date.today()
    return d.strftime("%Y%m%d")


@app.command()
def run(
    target_date: str = typer.Option(None, "--date", "-d", help="Target date (YYYYMMDD)"),
):
    """Run the full pipeline: collect → scan → news → analyze → report."""
    date_str = target_date or _get_date_str()
    scan_date = datetime.strptime(date_str, "%Y%m%d").date()

    settings = Settings()
    config = load_scanner_config()

    console.print(f"[bold]52주 신고가 스캔 시작: {date_str}[/bold]")

    from src.collector import Collector
    from src.scanner import Scanner
    from src.news_fetcher import NewsFetcher
    from src.ai_analyst import AIAnalyst
    from src.reporter import Reporter
    from src.db import Database

    db = Database()
    collector = Collector()
    scanner = Scanner(collector=collector)

    # Step 1: Collect daily data
    console.print("[dim]1/5 데이터 수집 중...[/dim]")
    daily_data = collector.collect_daily(date_str, markets=config.scanner.markets)

    # Step 2: Get sector info
    console.print("[dim]2/5 섹터 정보 수집 중...[/dim]")
    sector_map = {}
    name_map = {}
    for market in ["KOSPI", "KOSDAQ"]:
        sector_map.update(collector.get_sector_map(date_str, market))
    from pykrx import stock as pykrx_stock
    for ticker in daily_data:
        if ticker not in name_map:
            try:
                name_map[ticker] = pykrx_stock.get_market_ticker_name(ticker)
            except Exception:
                try:
                    name_map[ticker] = pykrx_stock.get_etf_ticker_name(ticker)
                except Exception:
                    name_map[ticker] = ticker

    # Step 3: Scan for 52-week highs
    console.print("[dim]3/5 52주 신고가 스캔 중...[/dim]")
    market_caps = collector.get_market_caps(date_str)
    highs = scanner.find_new_highs(
        daily_data=daily_data,
        date_str=date_str,
        sector_map=sector_map,
        name_map=name_map,
        lookback=config.scanner.lookback_days,
    )
    result = scanner.build_scan_result(scan_date, highs, len(daily_data))
    db.save_scan_result(result)

    # Step 4: Fetch news and AI analysis
    console.print("[dim]4/5 뉴스 수집 및 AI 분석 중...[/dim]")
    stock_names = [h.name for h in highs]
    fetcher = NewsFetcher(settings.naver_client_id, settings.naver_client_secret)
    news_map = asyncio.run(
        fetcher.fetch_news_for_stocks(stock_names, config.news.max_articles_per_stock)
    )

    analyst = AIAnalyst(
        api_key=settings.openai_api_key,
        model=config.ai.model,
        max_tokens=config.ai.max_tokens,
    )
    ai_results = asyncio.run(
        analyst.analyze_stocks(highs, news_map, market_caps, config.scanner.max_ai_analyze)
    )
    for ar in ai_results:
        db.save_ai_analysis(scan_date, ar)

    # Step 5: Send report
    console.print("[dim]5/5 리포트 전송 중...[/dim]")
    trend = db.get_high_count_history(days=5)

    if config.telegram.enabled and settings.telegram_bot_token:
        reporter = Reporter(settings.telegram_bot_token, settings.telegram_chat_id)
        asyncio.run(reporter.send_report(result, ai_results, trend))
        console.print("[green]텔레그램 리포트 전송 완료![/green]")
    else:
        reporter = Reporter(bot_token="", chat_id=0)
        text = reporter.format_report(result, ai_results, trend)
        console.print(text)

    console.print(f"[bold green]완료! {result.stats.new_high_count}개 신고가 종목 발견[/bold green]")


@app.command()
def collect(
    target_date: str = typer.Option(None, "--date", "-d", help="Target date (YYYYMMDD)"),
):
    """Collect daily market data only."""
    date_str = target_date or _get_date_str()
    config = load_scanner_config()

    from src.collector import Collector
    collector = Collector()
    daily_data = collector.collect_daily(date_str, markets=config.scanner.markets)
    console.print(f"[green]수집 완료: {len(daily_data)}개 종목[/green]")


@app.command()
def history(
    target_date: str = typer.Option(None, "--date", "-d", help="Date to query (YYYYMMDD)"),
):
    """Query historical scan results."""
    from src.db import Database
    db = Database()

    if target_date:
        scan_date = datetime.strptime(target_date, "%Y%m%d").date()
        results = db.get_scan_result(scan_date)
        if not results:
            console.print(f"[yellow]{target_date} 데이터 없음[/yellow]")
            return

        table = Table(title=f"52주 신고가 ({target_date})")
        table.add_column("종목코드")
        table.add_column("종목명")
        table.add_column("시장")
        table.add_column("섹터")
        table.add_column("종가", justify="right")
        table.add_column("돌파율", justify="right")

        for r in results:
            table.add_row(
                r["ticker"], r["name"], r["market"], r["sector"],
                f"{r['close_price']:,.0f}", f"+{r['breakout_pct']:.1f}%",
            )
        console.print(table)
    else:
        # Show recent trend
        trend = db.get_high_count_history(days=10)
        if not trend:
            console.print("[yellow]저장된 데이터 없음[/yellow]")
            return
        table = Table(title="최근 52주 신고가 추이")
        table.add_column("날짜")
        table.add_column("신고가 종목 수", justify="right")
        for t in trend:
            table.add_row(str(t["date"]), str(t["count"]))
        console.print(table)


@app.command()
def stats(
    days: int = typer.Option(30, "--days", "-n", help="Number of days"),
):
    """Show historical statistics."""
    from src.db import Database
    db = Database()
    trend = db.get_high_count_history(days=days)
    if not trend:
        console.print("[yellow]저장된 데이터 없음[/yellow]")
        return

    counts = [t["count"] for t in trend]
    avg = sum(counts) / len(counts)
    console.print(f"[bold]최근 {len(trend)}일 통계[/bold]")
    console.print(f"  평균 신고가 종목 수: {avg:.1f}")
    console.print(f"  최대: {max(counts)} / 최소: {min(counts)}")


if __name__ == "__main__":
    app()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/cli.py tests/test_cli.py
git commit -m "feat: add typer-based CLI with run/collect/history/stats commands"
```

---

## Task 11: Integration Test & Final Polish

**Files:**
- Create: `tests/test_integration.py`
- Modify: `src/cli.py` (if needed)

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""Integration test: verifies the full pipeline works end-to-end with mocks."""
import pytest
from datetime import date
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
    monkeypatch.setenv("NAVER_CLIENT_ID", "test-naver-id")
    monkeypatch.setenv("NAVER_CLIENT_SECRET", "test-naver-secret")


def test_full_pipeline_with_mocks(mock_env, tmp_path):
    """Full pipeline should collect, scan, analyze, and produce a report."""
    from src.collector import Collector
    from src.scanner import Scanner
    from src.news_fetcher import NewsFetcher
    from src.ai_analyst import AIAnalyst
    from src.reporter import Reporter
    from src.db import Database
    from src.config import Settings, load_scanner_config

    # Setup
    db = Database(f"sqlite:///{tmp_path}/test.db")
    settings = Settings()
    config = load_scanner_config()

    # Mock collector
    daily_data = {
        "005930": {
            "market": "KOSPI", "open": 70000, "high": 79000,
            "low": 69000, "close": 78500, "volume": 15000000, "change_pct": 2.0,
        },
    }
    sector_map = {"005930": "전기전자"}
    name_map = {"005930": "삼성전자"}
    market_caps = {"005930": 450000000000000}

    mock_collector = MagicMock(spec=Collector)
    mock_collector.collect_daily.return_value = daily_data
    mock_collector.get_sector_map.return_value = sector_map
    mock_collector.get_market_caps.return_value = market_caps
    mock_collector.get_52w_high.return_value = 79000

    # Scan
    scanner = Scanner(collector=mock_collector)
    highs = scanner.find_new_highs(
        daily_data=daily_data, date_str="20260219",
        sector_map=sector_map, name_map=name_map,
        prev_highs={"005930": 77000},
    )
    result = scanner.build_scan_result(date(2026, 2, 19), highs, 1)

    assert result.stats.new_high_count == 1
    assert result.highs[0].ticker == "005930"

    # Save to DB
    db.save_scan_result(result)
    loaded = db.get_scan_result(date(2026, 2, 19))
    assert len(loaded) == 1

    # Format report (without actually sending)
    reporter = Reporter(bot_token="test", chat_id=123)
    text = reporter.format_report(result, [], [])
    assert "삼성전자" in text
    assert "52주 신고가 리포트" in text
```

**Step 2: Run test**

Run: `pytest tests/test_integration.py -v`
Expected: PASS

**Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test for full pipeline"
```

---

## Task 12: Copy .env.example → .env and Setup Instructions

**Step 1: Create .env from example**

```bash
cp .env.example .env
```

**Step 2: Update .env with real credentials**

User fills in:
- `OPENAI_API_KEY` - from OpenAI platform
- `TELEGRAM_BOT_TOKEN` - from @BotFather
- `TELEGRAM_CHAT_ID` - from Telegram getUpdates API
- `NAVER_CLIENT_ID` - from developers.naver.com
- `NAVER_CLIENT_SECRET` - from developers.naver.com

**Step 3: Test with real data (manual)**

```bash
# Quick smoke test: collect today's data
python -m src.cli collect

# Full run
python -m src.cli run
```

**Step 4: Set up crontab (optional)**

```bash
# Edit crontab
crontab -e

# Add line (adjust paths):
0 16 * * 1-5 cd /path/to/kr-stock-scanner && /path/to/.venv/bin/python -m src.cli run >> /tmp/scanner.log 2>&1
```

**Step 5: Final commit**

```bash
git add -A
git commit -m "docs: finalize project setup and configuration"
```

---

## Summary

| Task | Module | Description |
|------|--------|-------------|
| 1 | Scaffolding | pyproject.toml, config.yaml, .gitignore, venv |
| 2 | Config | pydantic-settings for env vars + YAML config |
| 3 | Models | Pydantic data models (StockHigh, ScanResult, etc.) |
| 4 | Database | SQLite + SQLAlchemy ORM |
| 5 | Collector | pykrx data collection (OHLCV, sector, market cap) |
| 6 | Scanner | 52-week high detection logic |
| 7 | News | Naver News API fetcher |
| 8 | AI Analyst | OpenAI GPT analysis |
| 9 | Reporter | Telegram message formatting + sending |
| 10 | CLI | typer CLI (run/collect/history/stats) |
| 11 | Integration | End-to-end integration test |
| 12 | Setup | .env setup, crontab configuration |
