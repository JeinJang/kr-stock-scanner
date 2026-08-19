# tests/test_db.py
from datetime import date
import pytest


@pytest.fixture
def db(tmp_path):
    from src.db import Database
    db_path = tmp_path / "test.db"
    return Database(f"sqlite:///{db_path}")


def test_save_and_load_scan_result(db):
    from src.models import ScanResult, StockHigh, MarketStats

    result = ScanResult(
        scan_date=date(2026, 2, 19),
        stats=MarketStats(
            total_stocks=2500,
            new_high_count=2,
            kospi_count=1,
            kosdaq_count=1,
            etf_count=0,
        ),
        highs=[
            StockHigh(
                ticker="005930", name="삼성전자", market="KOSPI",
                sector="전기전자", close_price=78500, high_52w=79000,
                prev_high_52w=77000, breakout_pct=2.60,
                volume=15000000, avg_volume_20d=12000000,
            ),
            StockHigh(
                ticker="035420", name="NAVER", market="KOSDAQ",
                sector="서비스업", close_price=220000, high_52w=222000,
                prev_high_52w=218000, breakout_pct=1.83,
                volume=3000000, avg_volume_20d=2500000,
            ),
        ],
        sector_breakdown={},
    )

    db.save_scan_result(result)
    loaded = db.get_scan_result(date(2026, 2, 19))

    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0]["ticker"] == "005930"


def test_get_new_high_count_history(db):
    from src.models import ScanResult, MarketStats

    for day_offset, count in [(1, 10), (2, 15), (3, 12)]:
        d = date(2026, 2, day_offset)
        result = ScanResult(
            scan_date=d,
            stats=MarketStats(
                total_stocks=2500, new_high_count=count,
                kospi_count=count, kosdaq_count=0, etf_count=0,
            ),
            highs=[],
            sector_breakdown={},
        )
        db.save_scan_result(result)

    history = db.get_high_count_history(days=3)
    assert len(history) == 3


def test_save_ai_analysis(db):
    from src.models import AIAnalysisResult

    analysis = AIAnalysisResult(
        ticker="005930",
        news_summary="HBM4 관련 뉴스",
        ai_analysis="반도체 업황 호조",
    )
    db.save_ai_analysis(date(2026, 2, 19), analysis)
    loaded = db.get_ai_analysis(date(2026, 2, 19), "005930")
    assert loaded is not None
    assert "반도체" in loaded["ai_analysis"]


def test_migration_helper_handles_missing_table(tmp_path):
    """new_highs 테이블이 없어도 마이그레이션은 조용히 반환한다."""
    from sqlalchemy import create_engine
    from src.db import _migrate_add_recency_columns

    engine = create_engine(f"sqlite:///{tmp_path}/empty.db")
    _migrate_add_recency_columns(engine)  # 예외가 나면 안 됨


def test_migration_adds_recency_columns_to_legacy_table(tmp_path):
    """구 스키마 테이블에 4개 컬럼이 추가되고, 두 번 실행해도 안전하다."""
    from sqlalchemy import create_engine, inspect, text
    from src.db import _migrate_add_recency_columns

    db_path = f"{tmp_path}/legacy.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE new_highs ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " scan_date DATE NOT NULL, ticker VARCHAR(10) NOT NULL,"
            " name VARCHAR(100) NOT NULL, market VARCHAR(10) NOT NULL,"
            " sector VARCHAR(50) NOT NULL, close_price FLOAT NOT NULL,"
            " high_52w FLOAT NOT NULL, prev_high_52w FLOAT NOT NULL,"
            " breakout_pct FLOAT NOT NULL, volume BIGINT NOT NULL,"
            " avg_volume_20d BIGINT NOT NULL)"
        ))

    _migrate_add_recency_columns(engine)
    _migrate_add_recency_columns(engine)  # 멱등

    cols = {c["name"] for c in inspect(engine).get_columns("new_highs")}
    assert {
        "days_since_prev_new_high", "days_since_price_above",
        "history_span_days", "change_pct",
    } <= cols


def test_scan_result_roundtrip_preserves_recency_fields(tmp_path):
    """저장 후 복원했을 때 신선도 필드가 그대로 살아온다."""
    from datetime import date
    from src.db import Database
    from src.models import ScanResult, StockHigh, MarketStats

    db = Database(url=f"sqlite:///{tmp_path}/rt.db")
    stock = StockHigh(
        ticker="005930", name="삼성전자", market="KOSPI", sector="전기전자",
        close_price=78500, high_52w=79000, prev_high_52w=77000,
        breakout_pct=2.6, volume=1000, avg_volume_20d=0,
        days_since_prev_new_high=1170, days_since_price_above=None,
        history_span_days=4017, change_pct=3.1,
    )
    result = ScanResult(
        scan_date=date(2026, 8, 19),
        stats=MarketStats(total_stocks=1, new_high_count=1, kospi_count=1),
        highs=[stock], sector_breakdown={"전기전자": [stock]},
    )
    db.save_scan_result(result)

    loaded = db.get_scan_result_full(date(2026, 8, 19))
    assert loaded is not None
    got = loaded.highs[0]
    assert got.days_since_prev_new_high == 1170
    assert got.days_since_price_above is None
    assert got.history_span_days == 4017
    assert got.change_pct == 3.1


def _create_legacy_new_highs(engine, rows):
    """change_pct 컬럼이 없던 구 스키마 테이블을 만들고 행을 넣는다."""
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE new_highs ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " scan_date DATE NOT NULL, ticker VARCHAR(10) NOT NULL,"
            " name VARCHAR(100) NOT NULL, market VARCHAR(10) NOT NULL,"
            " sector VARCHAR(50) NOT NULL, close_price FLOAT NOT NULL,"
            " high_52w FLOAT NOT NULL, prev_high_52w FLOAT NOT NULL,"
            " breakout_pct FLOAT NOT NULL, volume BIGINT NOT NULL,"
            " avg_volume_20d BIGINT NOT NULL)"
        ))
        conn.execute(text("CREATE TABLE IF NOT EXISTS daily_scans ("
                          " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                          " scan_date DATE NOT NULL, total_stocks INTEGER NOT NULL,"
                          " new_high_count INTEGER NOT NULL, market_type VARCHAR(10) NOT NULL)"))
        conn.execute(text(
            "INSERT INTO daily_scans (scan_date, total_stocks, new_high_count, market_type)"
            " VALUES ('2026-08-18', 2500, 1, 'ALL')"
        ))
        for ticker, name, breakout_pct in rows:
            conn.execute(text(
                "INSERT INTO new_highs (scan_date, ticker, name, market, sector,"
                " close_price, high_52w, prev_high_52w, breakout_pct, volume, avg_volume_20d)"
                f" VALUES ('2026-08-18', '{ticker}', '{name}', 'KOSPI', '전기전자',"
                f" 1000, 1000, 0, {breakout_pct}, 1, 0)"
            ))


def test_legacy_rows_move_day_change_out_of_breakout_pct(tmp_path):
    """구 스키마의 breakout_pct는 당일 등락률이므로 change_pct로 옮기고 0으로 되돌린다."""
    from datetime import date
    from sqlalchemy import create_engine
    from src.db import Database

    db_path = f"{tmp_path}/legacy_backfill.db"
    _create_legacy_new_highs(create_engine(f"sqlite:///{db_path}"), [("005930", "삼성전자", 5.2)])

    db = Database(url=f"sqlite:///{db_path}")
    result = db.get_scan_result_full(date(2026, 8, 18))

    assert result is not None
    stock = result.highs[0]
    assert stock.change_pct == 5.2      # 당일 등락률로 복원
    assert stock.breakout_pct == 0      # 돌파율은 알 수 없으므로 0


def test_legacy_backfill_is_not_applied_twice(tmp_path):
    """이미 보정된 행은 change_pct가 채워져 재실행해도 다시 걸리지 않는다."""
    from datetime import date
    from sqlalchemy import create_engine
    from src.db import Database, _migrate_add_recency_columns

    db_path = f"{tmp_path}/legacy_twice.db"
    _create_legacy_new_highs(create_engine(f"sqlite:///{db_path}"), [("005930", "삼성전자", 5.2)])

    db = Database(url=f"sqlite:///{db_path}")
    _migrate_add_recency_columns(db.engine)   # 재기동 상황

    stock = db.get_scan_result_full(date(2026, 8, 18)).highs[0]
    assert stock.change_pct == 5.2
    assert stock.breakout_pct == 0


def test_migration_leaves_new_rows_untouched(tmp_path):
    """change_pct가 채워진 신규 행은 백필 대상이 아니다."""
    from datetime import date
    from src.db import Database, _migrate_add_recency_columns
    from src.models import ScanResult, StockHigh, MarketStats

    db = Database(url=f"sqlite:///{tmp_path}/fresh.db")
    stock = StockHigh(
        ticker="005930", name="삼성전자", market="KOSPI", sector="전기전자",
        close_price=1000, high_52w=1000, prev_high_52w=990,
        breakout_pct=1.0, volume=1, avg_volume_20d=0, change_pct=3.5,
    )
    db.save_scan_result(ScanResult(
        scan_date=date(2026, 8, 19),
        stats=MarketStats(total_stocks=1, new_high_count=1, kospi_count=1),
        highs=[stock], sector_breakdown={"전기전자": [stock]},
    ))

    _migrate_add_recency_columns(db.engine)

    loaded = db.get_scan_result_full(date(2026, 8, 19)).highs[0]
    assert loaded.breakout_pct == 1.0
    assert loaded.change_pct == 3.5
