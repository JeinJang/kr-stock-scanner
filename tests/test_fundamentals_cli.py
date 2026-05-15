from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from src.fundamentals.cli import app

runner = CliRunner()


def test_show_command_no_data():
    """show command on missing ticker prints message."""
    with patch("src.fundamentals.cli.FundamentalsDB") as MockDB:
        mock = MagicMock()
        mock.load_scores.return_value = []
        MockDB.return_value = mock

        result = runner.invoke(app, ["show", "999999"])
    assert result.exit_code == 0
    assert "999999" in result.stdout


def test_run_with_skip_market_and_skip_dart_short_circuits():
    """Both --skip-dart and --skip-market should still let compute_all run."""
    from unittest.mock import patch, MagicMock
    from src.fundamentals.cli import app

    runner = CliRunner()
    with patch("src.fundamentals.cli._make_pipeline") as make_pipe, \
         patch("src.fundamentals.cli._load_market_data", return_value=({}, {})) as load_md, \
         patch("src.fundamentals.cli._build_market_pipeline") as build_market, \
         patch("src.market_data.db.MarketDB") as market_db_cls, \
         patch("src.fundamentals.report.ReportGenerator"):
        pipe = MagicMock()
        pipe.compute_all.return_value = ([], [])
        pipe._cache.load_corp_info.return_value = []
        make_pipe.return_value = pipe
        market_db_cls.return_value.load_all.return_value = {}

        result = runner.invoke(app, ["run", "--skip-dart", "--skip-market"])
        assert result.exit_code == 0, result.output
        build_market.assert_not_called()
        # DART refresh not called either
        pipe.refresh_data.assert_not_called()
