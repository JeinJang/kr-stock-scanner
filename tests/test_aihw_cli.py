from datetime import date
from unittest.mock import patch

from typer.testing import CliRunner

from src.aihw.models import AihwSummary, CompanySummary, GroupSummary
from src.aihw.pipeline import AihwResult
from src.cli import app

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
