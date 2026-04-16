from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from src.forecast.cli import app

runner = CliRunner()


def test_run_no_scan_result():
    """If no scan results exist, CLI should print an error."""
    with patch("src.forecast.cli.Database") as MockDB:
        mock_db = MagicMock()
        mock_db.get_scan_result_full.return_value = None
        MockDB.return_value = mock_db

        result = runner.invoke(app, ["run"])

    assert result.exit_code == 0
    assert "먼저" in result.stdout or "스캔" in result.stdout
