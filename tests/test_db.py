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
    from src.models import ScanResult, StockHigh, MarketStats

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
