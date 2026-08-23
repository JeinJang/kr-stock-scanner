from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Secret settings loaded from environment variables / .env file."""

    krx_api_key: str = ""
    krx_id: str = ""
    krx_pw: str = ""
    openai_api_key: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: int = 0
    aihw_telegram_chat_id: int = 0
    naver_client_id: str = ""
    naver_client_secret: str = ""
    ecos_api_key: str = ""
    fred_api_key: str = ""
    opendart_api_key: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class ScannerSection(BaseModel):
    markets: list[str] = ["KOSPI", "KOSDAQ"]
    # markets: list[str] = ["KOSPI", "KOSDAQ", "ETF"]
    lookback_days: int = 250
    max_ai_analyze: int = 50


class NewsSection(BaseModel):
    provider: str = "naver_api"
    max_articles_per_stock: int = 5
    search_days: int = 3


class AISection(BaseModel):
    model: str = "gpt-5-nano"
    max_tokens: int = 300


class TelegramSection(BaseModel):
    enabled: bool = True


class ForecastSection(BaseModel):
    horizon: int = 60
    model: str = "google/timesfm-2.5-200m-pytorch"
    report_dir: str = "reports"


class FundamentalsSection(BaseModel):
    years_lookback: int = 10
    cache_ttl_days: int = 30
    report_dir: str = "reports"
    market_filter: list[str] = ["KOSPI", "KOSDAQ"]


class RelatedSection(BaseModel):
    model: str = "gpt-5-nano"
    report_dir: str = "reports"
    max_tokens_per_section: int = 8000


class AihwSection(BaseModel):
    ai_hw_tickers: dict[str, str] = {
        "NVDA": "엔비디아",
        "AVGO": "브로드컴",
        "005930.KS": "삼성전자",
        "000660.KS": "SK하이닉스",
        "MU": "마이크론",
        "SNDK": "샌디스크",
    }
    big_tech_tickers: dict[str, str] = {
        "AMZN": "아마존",
        "TSLA": "테슬라",
        "MSFT": "MS",
        "META": "메타",
        "GOOGL": "구글",
    }
    benchmarks: list[str] = ["SPY", "RSP"]
    base_date: str = "2026-01-10"
    threshold: float = 0.8
    report_dir: str = "reports"
    auto_send: bool = True


class ScannerConfig(BaseModel):
    scanner: ScannerSection = ScannerSection()
    news: NewsSection = NewsSection()
    ai: AISection = AISection()
    telegram: TelegramSection = TelegramSection()
    forecast: ForecastSection = ForecastSection()
    fundamentals: FundamentalsSection = FundamentalsSection()
    related: RelatedSection = RelatedSection()
    aihw: AihwSection = AihwSection()


def load_scanner_config(path: Path | None = None) -> ScannerConfig:
    """Load scanner configuration from YAML file."""
    if path is None:
        path = Path("config.yaml")
    if not path.exists():
        return ScannerConfig()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return ScannerConfig(**data)
