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
