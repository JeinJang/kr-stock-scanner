from datetime import date


def test_stock_high_model():
    from src.models import StockHigh

    stock = StockHigh(
        ticker="005930",
        name="삼성전자",
        market="KOSPI",
        sector="전기전자",
        close_price=78500,
        high_52w=79000,
        prev_high_52w=77000,
        breakout_pct=2.60,
        volume=15000000,
        avg_volume_20d=12000000,
    )
    assert stock.ticker == "005930"
    assert stock.name == "삼성전자"
    assert stock.breakout_pct == 2.60


def test_scan_result_model():
    from src.models import ScanResult, MarketStats

    result = ScanResult(
        scan_date=date(2026, 2, 19),
        stats=MarketStats(
            total_stocks=2500,
            new_high_count=32,
            kospi_count=15,
            kosdaq_count=12,
            etf_count=5,
        ),
        highs=[],
        sector_breakdown={},
    )
    assert result.stats.new_high_count == 32
    assert result.scan_date == date(2026, 2, 19)


def test_ai_analysis_model():
    from src.models import AIAnalysisResult

    analysis = AIAnalysisResult(
        ticker="005930",
        news_summary="HBM4 수주 확대 기대감",
        ai_analysis="반도체 업황 회복과 HBM4 수주 확대에 따른 실적 개선 기대.",
    )
    assert analysis.ticker == "005930"
    assert "HBM4" in analysis.ai_analysis
