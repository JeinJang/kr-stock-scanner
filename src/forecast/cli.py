from __future__ import annotations

import os
import webbrowser
from datetime import date, datetime

import typer
from rich.console import Console

from src.config import Settings, load_scanner_config
from src.db import Database

app = typer.Typer(help="TimesFM-based stock price forecast", no_args_is_help=True)
console = Console()


def _resolve_scan_date(target_date: str | None) -> date:
    if target_date:
        return datetime.strptime(target_date, "%Y%m%d").date()
    return date.today()


@app.command()
def run(
    target_date: str = typer.Option(None, "--date", "-d", help="Scan date to use (YYYYMMDD)"),
    horizon: int = typer.Option(None, "--horizon", "-H", help="Forecast horizon in trading days"),
    covariates: bool = typer.Option(False, "--covariates", help="Use macro indicators as covariates (experimental)"),
):
    """Run forecast pipeline for scanned 52-week high stocks."""
    settings = Settings()
    config = load_scanner_config()
    forecast_horizon = horizon or config.forecast.horizon

    scan_date = _resolve_scan_date(target_date)
    date_str = scan_date.strftime("%Y%m%d")

    console.print(f"[bold]주가 예측 시작: {date_str} (horizon={forecast_horizon}일)[/bold]")

    # Step 1: Load scan results from DB
    db = Database()
    scan_result = db.get_scan_result_full(scan_date)
    if scan_result is None:
        console.print(
            f"[red]{date_str} 스캔 결과가 없습니다. "
            f"먼저 `python -m src.cli run --date {date_str}`을 실행하세요.[/red]"
        )
        return

    highs = scan_result.highs
    console.print(f"[dim]스캔 결과 로드: {len(highs)}개 종목[/dim]")

    # Load AI analyses
    all_ai = db.get_all_ai_analyses(scan_date)
    ai_map = {a.ticker: {"ai_analysis": a.ai_analysis, "news_summary": a.news_summary} for a in all_ai}

    # Step 2: Fetch data
    console.print("[dim]1/4 매크로 데이터 수집 중...[/dim]")
    from src.forecast.macro_fetcher import MacroFetcher
    macro_fetcher = MacroFetcher(
        ecos_api_key=settings.ecos_api_key,
        fred_api_key=settings.fred_api_key,
    )
    macro_data = macro_fetcher.fetch_all()

    console.print("[dim]2/4 종목 과거 데이터 수집 중...[/dim]")
    from src.forecast.stock_fetcher import StockFetcher
    from src.krx_client import create_krx_client
    client = create_krx_client(
        krx_id=settings.krx_id,
        krx_pw=settings.krx_pw,
        krx_api_key=settings.krx_api_key,
    )
    stock_fetcher = StockFetcher(client=client)
    tickers = [h.ticker for h in highs]
    stock_data = stock_fetcher.fetch_histories(tickers, date_str)

    # Step 3: Run predictions
    console.print("[dim]3/4 TimesFM 예측 실행 중...[/dim]")
    from src.forecast.predictor import Predictor
    from src.forecast.macro_fetcher import INDICATOR_NAMES

    predictor = Predictor(model_name=config.forecast.model, horizon=forecast_horizon)

    # Macro predictions
    macro_items = [
        {
            "ticker": name,
            "name": INDICATOR_NAMES.get(name, name),
            "category": "macro",
            "history": values,
            "dates_history": dates,
        }
        for name, (dates, values) in macro_data.items()
        if values
    ]
    macro_results = predictor.predict_macro(macro_items)

    # Stock predictions with macro covariates
    name_map = {h.ticker: h.name for h in highs}
    sector_map = {h.ticker: h.sector for h in highs}
    stock_items = [
        {
            "ticker": ticker,
            "name": name_map.get(ticker, ticker),
            "category": "stock",
            "sector": sector_map.get(ticker, ""),
            "history": values,
            "dates_history": dates,
        }
        for ticker, (dates, values) in stock_data.items()
    ]

    # Build macro covariates: indices only (no rates/FX — policy-driven, adds noise)
    COV_INDICATORS = {"KOSPI", "KOSDAQ", "SP500", "NASDAQ"}
    macro_cov: dict[str, tuple[list[float], list[float]]] = {}
    macro_forecast_map = {r.ticker: r for r in macro_results}
    for indicator_name, (_, hist_values) in macro_data.items():
        if indicator_name not in COV_INDICATORS or not hist_values:
            continue
        macro_r = macro_forecast_map.get(indicator_name)
        if macro_r:
            macro_cov[indicator_name] = (hist_values, macro_r.forecast)

    if covariates and macro_cov:
        console.print(f"[dim]   공변량 {len(macro_cov)}개: {', '.join(macro_cov.keys())} (experimental)[/dim]")
        stock_results = predictor.predict_with_covariates(stock_items, macro_cov)
    else:
        stock_results = predictor.predict_batch(stock_items)

    # Step 4: Generate report
    console.print("[dim]4/4 HTML 리포트 생성 중...[/dim]")
    from src.forecast.report import ReportGenerator

    generator = ReportGenerator()
    report_path = generator.generate(
        macro_results=macro_results,
        stock_results=stock_results,
        ai_analyses=ai_map,
        scan_date=str(scan_date),
        output_dir=config.forecast.report_dir,
        horizon=forecast_horizon,
    )

    console.print(f"[bold green]완료! 리포트: {report_path}[/bold green]")
    webbrowser.open(f"file://{os.path.abspath(report_path)}")


@app.command()
def list_reports(
    target_date: str = typer.Option(None, "--date", "-d", help="Scan date to filter (YYYYMMDD)"),
):
    """List available forecast reports."""
    import glob as glob_mod

    config = load_scanner_config()
    report_dir = config.forecast.report_dir
    pattern = os.path.join(report_dir, "forecast_*.html")
    files = sorted(glob_mod.glob(pattern), reverse=True)

    if not files:
        console.print(f"[yellow]{report_dir} 에 리포트가 없습니다.[/yellow]")
        return

    for f in files:
        console.print(f)


if __name__ == "__main__":
    app()
