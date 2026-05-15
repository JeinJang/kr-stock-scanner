from datetime import date
from src.fundamentals.db import FundamentalsDB
from src.fundamentals.models import FundamentalsMetrics


def test_save_load_round_trips_new_fields(tmp_path):
    db = FundamentalsDB(url=f"sqlite:///{tmp_path/'t.db'}")
    m = FundamentalsMetrics(
        ticker="000660", as_of_date=date(2026, 5, 15),
        eps=1900.5, bps=120000.0, psr=2.3,
        ocf=470000.0, fcf=320000.0, capex_to_revenue=15.4,
        dividend_yield=1.8, payout_ratio=22.5, consecutive_dividend_years=7,
    )
    db.save_metrics([m])
    loaded = db.load_metrics(date(2026, 5, 15))
    assert len(loaded) == 1
    r = loaded[0]
    assert r.eps == 1900.5
    assert r.bps == 120000.0
    assert r.psr == 2.3
    assert r.ocf == 470000.0
    assert r.fcf == 320000.0
    assert r.capex_to_revenue == 15.4
    assert r.dividend_yield == 1.8
    assert r.payout_ratio == 22.5
    assert r.consecutive_dividend_years == 7


def test_migration_helper_handles_missing_table(tmp_path):
    """If table doesn't exist yet, migration returns silently (no exception)."""
    from sqlalchemy import create_engine
    from src.fundamentals.db import _migrate_add_enrichment_columns
    engine = create_engine(f"sqlite:///{tmp_path/'empty.db'}")
    # No create_all run; table doesn't exist
    _migrate_add_enrichment_columns(engine)  # must not raise
