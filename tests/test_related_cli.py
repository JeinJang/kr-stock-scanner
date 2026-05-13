from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from src.related.cli import app


def test_stats_command_empty(tmp_path):
    runner = CliRunner()
    with patch("src.related.cli.RelatedDB") as MockDB:
        m = MagicMock()
        m.stats.return_value = {"sources": 0, "edges": 0, "by_relation": {}}
        MockDB.return_value = m
        result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "0" in result.stdout


def test_show_missing_ticker_in_corp_info():
    """show with unknown ticker prints helpful message."""
    runner = CliRunner()
    with patch("src.related.cli.DartCache") as MockCache:
        c = MagicMock()
        c.load_corp_info.return_value = []
        MockCache.return_value = c
        result = runner.invoke(app, ["show", "999999"])
    assert result.exit_code == 0
    assert "999999" in result.stdout
