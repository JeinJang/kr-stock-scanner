# tests/test_cli.py
from datetime import date

from typer.testing import CliRunner

runner = CliRunner()


def test_cli_run_command_exists():
    """CLI should have a 'run' command."""
    from src.cli import app
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "run" in result.output.lower() or "full" in result.output.lower() or "pipeline" in result.output.lower()


def test_cli_run_has_force_flag():
    """run should expose a --force flag to bypass the per-date cache."""
    from src.cli import app
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--force" in result.output


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


def test_collection_blocked_reason_none_when_today():
    """scan_date가 today와 같으면 수집을 막을 이유가 없다."""
    from src.cli import _collection_blocked_reason
    today = date(2026, 8, 20)
    assert _collection_blocked_reason(today, today=today) is None


def test_collection_blocked_reason_past_date():
    """과거 날짜는 차단되고, 메시지에 그 날짜가 언급된다."""
    from src.cli import _collection_blocked_reason
    today = date(2026, 8, 20)
    past = date(2026, 6, 2)
    reason = _collection_blocked_reason(past, today=today)
    assert reason is not None
    assert "20260602" in reason


def test_collection_blocked_reason_future_date():
    """미래 날짜도 차단된다."""
    from src.cli import _collection_blocked_reason
    today = date(2026, 8, 20)
    future = date(2026, 9, 1)
    reason = _collection_blocked_reason(future, today=today)
    assert reason is not None
    assert "20260901" in reason


def test_collection_blocked_reason_mentions_alternatives():
    """대안 안내(collect --date, history --date)가 메시지에 포함되어야 한다."""
    from src.cli import _collection_blocked_reason
    today = date(2026, 8, 20)
    past = date(2026, 6, 2)
    reason = _collection_blocked_reason(past, today=today)
    assert reason is not None
    assert "collect --date" in reason
    assert "history --date" in reason
