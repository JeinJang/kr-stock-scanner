from datetime import date
from unittest.mock import patch

from typer.testing import CliRunner

from src.aihw.models import AihwSummary, CompanySummary, GroupSummary
from src.aihw.pipeline import AihwResult
from src.cli import _run_aihw_step, app
from src.config import ScannerConfig, Settings

runner = CliRunner()


def _result():
    summary = AihwSummary(
        as_of=date(2026, 8, 22), ratio=0.762, ratio_prev=0.754, change_pp=0.8,
        high_30d=0.781, low_30d=0.74, threshold=0.8, status=None,
        groups=[
            GroupSummary(name="AI HW", total_usd=6.82e12, companies=[
                CompanySummary(ticker="NVDA", name="엔비디아", cap_usd=4.21e12, day_change_pct=1.2),
            ]),
            GroupSummary(name="빅테크", total_usd=8.95e12, companies=[
                CompanySummary(ticker="MSFT", name="MS", cap_usd=3.12e12, day_change_pct=0.4),
            ]),
        ],
    )
    return AihwResult(
        summary=summary, html_path="reports/aihw-2026-08-22.html",
        png_path="reports/aihw-2026-08-22.png", caption="캡션",
    )


class TestAihwCommand:
    @patch("src.aihw.pipeline.run_aihw")
    def test_aihw_prints_summary(self, mock_run):
        mock_run.return_value = _result()
        result = runner.invoke(app, ["aihw"])
        assert result.exit_code == 0
        assert "76.2%" in result.output
        assert "aihw-2026-08-22.html" in result.output
        mock_run.assert_called_once()

    @patch("src.cli.asyncio.run")
    @patch("src.aihw.pipeline.run_aihw")
    def test_aihw_send_flag_sends_photo(self, mock_run, mock_asyncio, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
        mock_run.return_value = _result()
        result = runner.invoke(app, ["aihw", "--send"])
        assert result.exit_code == 0
        mock_asyncio.assert_called_once()  # send_photo 코루틴 실행

    @patch("src.aihw.pipeline.run_aihw")
    def test_aihw_send_without_token_exits_1(self, mock_run, monkeypatch):
        # Settings는 .env 파일도 읽으므로(pydantic-settings), delenv만으로는
        # 리포지토리의 실제 .env에 남아 있는 토큰이 그대로 남을 수 있다.
        # 빈 문자열로 env var를 설정하면 .env 값보다 우선해 빈 토큰이 강제된다.
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
        mock_run.return_value = _result()
        result = runner.invoke(app, ["aihw", "--send"])
        assert result.exit_code == 1

    @patch("src.aihw.pipeline.run_aihw")
    def test_aihw_days_overrides_base_date(self, mock_run):
        mock_run.return_value = _result()
        result = runner.invoke(app, ["aihw", "--days", "30"])
        assert result.exit_code == 0
        passed_config = mock_run.call_args.args[0]
        from datetime import timedelta
        assert passed_config.base_date == (date.today() - timedelta(days=30)).isoformat()


class TestRunAihwStep:
    @patch("src.cli.asyncio.run")
    @patch("src.aihw.pipeline.run_aihw")
    def test_sends_when_auto_send_enabled(self, mock_run, mock_asyncio):
        mock_run.return_value = _result()
        config = ScannerConfig()
        config.aihw.auto_send = True
        settings = Settings(telegram_bot_token="token", telegram_chat_id=123)
        _run_aihw_step(config, settings)
        mock_run.assert_called_once()
        mock_asyncio.assert_called_once()

    @patch("src.aihw.pipeline.run_aihw")
    def test_skips_when_disabled(self, mock_run):
        config = ScannerConfig()
        config.aihw.auto_send = False
        _run_aihw_step(config, Settings())
        mock_run.assert_not_called()

    @patch("src.aihw.pipeline.run_aihw", side_effect=RuntimeError("yfinance down"))
    def test_failure_does_not_raise(self, mock_run):
        config = ScannerConfig()
        config.aihw.auto_send = True
        settings = Settings(telegram_bot_token="token", telegram_chat_id=123)
        _run_aihw_step(config, settings)  # 예외가 전파되면 테스트 실패
