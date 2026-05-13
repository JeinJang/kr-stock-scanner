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


class ScannerConfig(BaseModel):
    scanner: ScannerSection = ScannerSection()
    news: NewsSection = NewsSection()
    ai: AISection = AISection()
    telegram: TelegramSection = TelegramSection()
    forecast: ForecastSection = ForecastSection()
    fundamentals: FundamentalsSection = FundamentalsSection()
    related: RelatedSection = RelatedSection()


def load_scanner_config(path: Path | None = None) -> ScannerConfig:
    """Load scanner configuration from YAML file."""
    if path is None:
        path = Path("config.yaml")
    if not path.exists():
        return ScannerConfig()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return ScannerConfig(**data)
