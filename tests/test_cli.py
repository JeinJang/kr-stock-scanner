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


def test_cli_prices_subcommands_exist():
    """prices backfill / sync / status 가 노출된다."""
    from src.cli import app
    result = runner.invoke(app, ["prices", "--help"])
    assert result.exit_code == 0
    for sub in ("backfill", "sync", "status"):
        assert sub in result.output


def test_cli_prices_backfill_has_years_option():
    from src.cli import app
    result = runner.invoke(app, ["prices", "backfill", "--help"])
    assert result.exit_code == 0
    assert "--years" in result.output


def test_prices_sync_command_has_key_guard():
    """키가 없을 때 prices sync가 트레이스백 대신 깔끔히 종료한다."""
    from src.cli import app
    result = runner.invoke(app, ["prices", "sync", "--help"])
    assert result.exit_code == 0


def test_cli_prices_refetch_command_has_date_option():
    from src.cli import app
    result = runner.invoke(app, ["prices", "refetch", "--help"])
    assert result.exit_code == 0
    assert "--date" in result.output


def test_sync_price_store_or_warn_swallows_krx_api_error():
    """run이 동기화 실패에도 계속 진행하도록, 헬퍼가 KrxApiError를 삼키고
    {"rows": 0, "same_day_rows": 0}을 반환한다 — 잃는 것은 돌파 신선도 배지뿐이어야 한다."""
    from src.cli import _sync_price_store_or_warn
    from src.price_history.fetcher import KrxApiError

    def fake_sync_fn(price_db, api_key, krx_client=None):
        raise KrxApiError("401 unauthorized")

    result = _sync_price_store_or_warn(fake_sync_fn, price_db=None, api_key="bad-key")
    assert result == {"rows": 0, "same_day_rows": 0}


def test_sync_price_store_or_warn_passes_through_on_success():
    """성공 시에는 sync_fn의 반환값을 그대로 돌려준다."""
    from src.cli import _sync_price_store_or_warn

    def fake_sync_fn(price_db, api_key, krx_client=None):
        return {"rows": 42, "requested": 2}

    result = _sync_price_store_or_warn(fake_sync_fn, price_db=None, api_key="good-key")
    assert result == {"rows": 42, "requested": 2}


def test_sync_price_store_or_warn_passes_krx_client_through():
    """krx_client 인자가 sync_fn에 그대로 전달돼야 run에서 로그인 클라이언트를 넘길 수 있다."""
    from src.cli import _sync_price_store_or_warn

    received = {}

    def fake_sync_fn(price_db, api_key, krx_client=None):
        received["krx_client"] = krx_client
        return {"rows": 0, "same_day_rows": 0}

    sentinel = object()
    _sync_price_store_or_warn(fake_sync_fn, price_db=None, api_key="good-key", krx_client=sentinel)
    assert received["krx_client"] is sentinel
