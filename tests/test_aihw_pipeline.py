from datetime import date

from src.aihw.db import AihwDB
from src.aihw.models import DailyCap
from src.aihw.pipeline import run_aihw
from src.config import AihwSection

D1, D2 = date(2026, 1, 12), date(2026, 1, 13)

CFG = AihwSection(
    ai_hw_tickers={"NVDA": "엔비디아"},
    big_tech_tickers={"MSFT": "MS"},
    benchmarks=["SPY"],
    base_date="2026-01-10",
    threshold=0.8,
)


def _fake_fetch(cap_tickers, benchmark_tickers, start, end):
    rows = []
    for d, nvda, msft, spy in [(D1, 3.0e12, 4.0e12, 500.0), (D2, 3.3e12, 4.0e12, 505.0)]:
        source = "snapshot" if d == D2 else "backfill"
        rows.append(DailyCap(date=d, ticker="NVDA", close=100.0, shares=10,
                             market_cap_usd=nvda, source=source))
        rows.append(DailyCap(date=d, ticker="MSFT", close=100.0, shares=10,
                             market_cap_usd=msft, source=source))
        rows.append(DailyCap(date=d, ticker="SPY", close=spy, shares=None,
                             market_cap_usd=None, source=source))
    return rows


class TestRunAihw:
    def test_full_pipeline(self, tmp_path):
        cfg = CFG.model_copy(update={"report_dir": str(tmp_path)})
        db = AihwDB(url="sqlite:///:memory:")
        result = run_aihw(cfg, db=db, fetch=_fake_fetch)
        assert result.summary.as_of == D2
        assert result.summary.ratio == 3.3e12 / 4.0e12
        assert result.html_path.endswith("aihw-2026-01-13.html")
        assert result.png_path.endswith("aihw-2026-01-13.png")
        assert "엔비디아" in result.caption
        # DB에 저장됐는지
        assert len(db.load_caps(D1, D2)) == 6

    def test_stale_trailing_db_rows_are_excluded(self, tmp_path):
        # 이전 실행이 남긴 하이브리드 행(미국 종가 ffill)이 DB에 있어도,
        # 이번 fetch의 마지막 날짜 이후 행은 리포트에 포함되지 않아야 한다.
        cfg = CFG.model_copy(update={"report_dir": str(tmp_path)})
        db = AihwDB(url="sqlite:///:memory:")
        stale = date(2026, 1, 14)  # fetch 범위(D2=1/13) 이후의 잔존 행
        db.save_caps([
            DailyCap(date=stale, ticker="NVDA", close=100.0, shares=10,
                     market_cap_usd=3.3e12, source="backfill"),
            DailyCap(date=stale, ticker="MSFT", close=100.0, shares=10,
                     market_cap_usd=4.0e12, source="backfill"),
        ])
        result = run_aihw(cfg, db=db, fetch=_fake_fetch)
        assert result.summary.as_of == D2  # stale(1/14)이 아니라 fetch 마지막 날

    def test_snapshot_persists_across_runs(self, tmp_path):
        cfg = CFG.model_copy(update={"report_dir": str(tmp_path)})
        db = AihwDB(url="sqlite:///:memory:")
        run_aihw(cfg, db=db, fetch=_fake_fetch)

        def _fetch_backfill_only(cap_tickers, benchmark_tickers, start, end):
            rows = _fake_fetch(cap_tickers, benchmark_tickers, start, end)
            # 두 번째 실행이 과거를 다른 값의 backfill로 다시 준다고 가정
            return [r.model_copy(update={"source": "backfill", "market_cap_usd":
                    (r.market_cap_usd or 0) * 2 or None}) for r in rows]

        run_aihw(cfg, db=db, fetch=_fetch_backfill_only)
        d2_nvda = [r for r in db.load_caps(D2, D2) if r.ticker == "NVDA"][0]
        # D2는 첫 실행에서 snapshot이었으므로 값이 유지된다
        assert d2_nvda.market_cap_usd == 3.3e12
