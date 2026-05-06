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
