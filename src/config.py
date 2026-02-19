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
    model: str = "gpt-5-nano"
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
