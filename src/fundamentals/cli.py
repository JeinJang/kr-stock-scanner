from __future__ import annotations

import asyncio
import os
import webbrowser
from datetime import date

import typer
from loguru import logger
from rich.console import Console

from src.config import Settings, load_scanner_config
from src.fundamentals.db import FundamentalsDB

app = typer.Typer(help="Open DART fundamentals screener", no_args_is_help=True)
console = Console()


def _make_pipeline(settings: Settings, ttl_days: int):
    from src.dart.cache import DartCache
    from src.dart.client import DartClient
    from src.dart.fetcher import DartFetcher
    from src.fundamentals.db import FundamentalsDB
    from src.fundamentals.pipeline import Pipeline

    cache = DartCache()
    client = DartClient(api_key=settings.opendart_api_key)
    fetcher = DartFetcher(client=client)
    db = FundamentalsDB()
    return Pipeline(cache=cache, fetcher=fetcher, ttl_days=ttl_days, fundamentals_db=db)


def _build_market_pipeline(settings: Settings):
    """Create MarketDataPipeline + MarketDB pair using project's KRX client."""
    from src.krx_client import create_krx_client
    from src.market_data.db import MarketDB
    from src.market_data.pipeline import MarketDataPipeline

    client = create_krx_client(
        krx_id=settings.krx_id, krx_pw=settings.krx_pw, krx_api_key=settings.krx_api_key,
    )
    db = MarketDB()
    pipeline = MarketDataPipeline(db=db, client=client)
    return pipeline, db


def _load_market_data(settings: Settings) -> tuple[dict[str, float], dict[str, str]]:
    """Load latest market caps and market info (KOSPI/KOSDAQ) from KRX.

    Searches back up to 10 days to find a trading day with non-zero data,
    skipping weekends and Korean holidays automatically.

    Returns:
        (market_caps: ticker -> 시가총액, market_map: ticker -> "KOSPI"|"KOSDAQ")
    """
    from src.krx_client import create_krx_client
    from datetime import datetime, timedelta

    client = create_krx_client(
        krx_id=settings.krx_id, krx_pw=settings.krx_pw, krx_api_key=settings.krx_api_key,
    )

    market_caps: dict[str, float] = {}
    market_map: dict[str, str] = {}

    # Try yesterday first; walk back up to 10 days if data is empty/zero
    end = datetime.now()
    for offset in range(1, 11):
        candidate = end - timedelta(days=offset)
        if candidate.weekday() >= 5:
            continue  # skip weekends
        target_date = candidate.strftime("%Y%m%d")

        attempt_caps: dict[str, float] = {}
        attempt_market: dict[str, str] = {}
        for market in ["KOSPI", "KOSDAQ"]:
            try:
                df = client.get_market_cap_by_ticker(target_date, market=market)
                for ticker, row in df.iterrows():
                    cap = float(row["시가총액"])
                    if cap > 0:
                        attempt_caps[ticker] = cap
                        attempt_market[ticker] = market
            except Exception:
                continue

        if attempt_caps:
            logger.info(f"KRX market data loaded from {target_date} ({len(attempt_caps)} tickers)")
            return attempt_caps, attempt_market

    return market_caps, market_map


@app.command()
def run(
    refresh: bool = typer.Option(False, "--refresh", help="Force refresh of DART data"),
    skip_dart: bool = typer.Option(False, "--skip-dart", help="Skip DART refresh"),
    skip_market: bool = typer.Option(False, "--skip-market", help="Skip pykrx-equivalent market data refresh"),
    years_opt: list[int] | None = typer.Option(
        None, "--years", help="Specific years to fetch (default: lookback range)"
    ),
):
    """Run the full screening pipeline and generate report."""
    settings = Settings()
    config = load_scanner_config()

    console.print(f"[bold]펀더멘털 스크리너 시작[/bold]")

    pipeline = _make_pipeline(settings, ttl_days=config.fundamentals.cache_ttl_days)

    # Step 1: KRX 시장 라벨 (KOSPI/KOSDAQ) 수집 — DART 호출에 필요
    console.print("[dim]1/5 KRX 시장 라벨 수집 중...[/dim]")
    market_caps_now, market_map = _load_market_data(settings)
    console.print(f"[dim]   {len(market_map)}개 종목 시장 정보 수집[/dim]")

    # Years range (current year inclusive for in-progress year)
    if years_opt:
        years = years_opt
    else:
        end_year = date.today().year
        years = list(range(end_year - config.fundamentals.years_lookback, end_year + 1))

    # Step 2: DART refresh
    if not skip_dart:
        console.print("[dim]2/5 DART 데이터 확인/수집 중...[/dim]")
        asyncio.run(pipeline.refresh_data(
            force=refresh, years=years,
            markets=config.fundamentals.market_filter,
            market_map=market_map,
        ))
    else:
        console.print("[dim]2/5 DART skip[/dim]")

    # Step 3: pykrx-equivalent 연도별 시장 데이터 refresh
    market_yearly_map: dict[str, list] = {}
    if not skip_market:
        console.print(f"[dim]3/5 KRX 연도별 시장 데이터 수집 중 (years={years})...[/dim]")
        market_pipeline, market_db = _build_market_pipeline(settings)

        # corps loaded from DART cache so we have corp_code↔ticker mapping
        corps = pipeline._cache.load_corp_info(markets=config.fundamentals.market_filter)
        corp_code_map = {c.ticker: c.corp_code for c in corps}
        tickers = list(corp_code_map.keys())

        report = market_pipeline.refresh(
            years=years, tickers=tickers, corp_code_map=corp_code_map,
        )
        console.print(
            f"[dim]   완료: {report.successful_rows}행, {report.duration_seconds:.1f}s[/dim]"
        )
        market_yearly_map = market_db.load_all()
    else:
        console.print("[dim]3/5 시장 데이터 skip[/dim]")
        # Still try to load existing rows from DB so compute_all has data
        from src.market_data.db import MarketDB
        try:
            market_yearly_map = MarketDB().load_all()
        except Exception:
            pass

    # Step 4: Compute metrics + scores
    console.print("[dim]4/5 지표 계산 + 점수 산정 중...[/dim]")
    metrics, scores = pipeline.compute_all(
        market_yearly_map=market_yearly_map,
        markets=config.fundamentals.market_filter,
    )
    console.print(f"[dim]   완료: {len(scores)}개 종목 점수 산정[/dim]")

    # Step 5: Generate report
    console.print("[dim]5/5 HTML 리포트 생성 중...[/dim]")
    corps = pipeline._cache.load_corp_info(markets=config.fundamentals.market_filter)
    name_map = {c.ticker: c.name for c in corps}
    rendered_market_map = {c.ticker: c.market for c in corps}
    from src.fundamentals.report import ReportGenerator
    gen = ReportGenerator()
    path = gen.generate(
        metrics=metrics, scores=scores,
        name_map=name_map, market_map=rendered_market_map,
        as_of_date=str(date.today()),
        output_dir=config.fundamentals.report_dir,
    )

    console.print(f"[bold green]완료! 리포트: {path}[/bold green]")
    webbrowser.open(f"file://{os.path.abspath(path)}")


@app.command()
def refresh():
    """Refresh DART data only (no analysis)."""
    settings = Settings()
    config = load_scanner_config()

    console.print(f"[bold]DART 데이터 갱신[/bold]")
    _, market_map = _load_market_data(settings)
    pipeline = _make_pipeline(settings, ttl_days=config.fundamentals.cache_ttl_days)
    years = list(range(date.today().year - config.fundamentals.years_lookback, date.today().year))
    asyncio.run(pipeline.refresh_data(
        force=True, years=years,
        markets=config.fundamentals.market_filter,
        market_map=market_map,
    ))
    console.print("[bold green]완료![/bold green]")


@app.command()
def show(ticker: str):
    """Show scorecard for a single ticker."""
    db = FundamentalsDB()
    today = date.today()
    scores = db.load_scores(today)
    found = [s for s in scores if s.ticker == ticker]
    if not found:
        console.print(f"[yellow]{ticker} 점수 없음. 먼저 `python -m src.fundamentals.cli run` 실행하세요.[/yellow]")
        return
    s = found[0]
    console.print(f"[bold]{ticker}[/bold] — 등급: {s.grade}, 종합: {s.total_score}")
    console.print(f"  안정성: {s.liquidity_score}, 수익성: {s.profitability_score}")
    console.print(f"  성장성: {s.growth_score}, 현금흐름: {s.cashflow_score}")
    console.print(f"  카테고리: {', '.join(s.categories) if s.categories else '없음'}")


if __name__ == "__main__":
    app()
