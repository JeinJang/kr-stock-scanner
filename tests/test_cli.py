# tests/test_cli.py
from typer.testing import CliRunner

runner = CliRunner()


def test_cli_run_command_exists():
    """CLI should have a 'run' command."""
    from src.cli import app
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "run" in result.output.lower() or "full" in result.output.lower() or "pipeline" in result.output.lower()


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


def test_cli_stats_command_exists():
    """CLI should have a 'stats' command."""
    from src.cli import app
    result = runner.invoke(app, ["stats", "--help"])
    assert result.exit_code == 0
