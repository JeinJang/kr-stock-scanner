# tests/test_config.py


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
  model: "gpt-5-nano"
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
    assert config.ai.model == "gpt-5-nano"
    assert config.telegram.enabled is False


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


def test_fundamentals_config_defaults():
    from src.config import FundamentalsSection, ScannerConfig
    config = ScannerConfig()
    assert config.fundamentals.years_lookback == 10
    assert config.fundamentals.cache_ttl_days == 30
    assert config.fundamentals.report_dir == "reports"
    assert config.fundamentals.market_filter == ["KOSPI", "KOSDAQ"]


def test_settings_has_opendart_key():
    import os
    os.environ["OPENDART_API_KEY"] = "test-dart-key"
    from src.config import Settings
    s = Settings()
    assert s.opendart_api_key == "test-dart-key"
    os.environ.pop("OPENDART_API_KEY", None)


def test_related_config_defaults():
    from src.config import RelatedSection, ScannerConfig
    config = ScannerConfig()
    assert config.related.model == "gpt-5-nano"
    assert config.related.report_dir == "reports"
    assert config.related.max_tokens_per_section == 8000
