# src/cli.py
import asyncio
from datetime import date, datetime

import typer
from rich.console import Console
from rich.table import Table

from src.config import Settings, load_scanner_config

app = typer.Typer(help="Korean Stock Market 52-Week High Scanner")
console = Console()


def _get_date_str(target_date: date | None = None) -> str:
    """Convert date to YYYYMMDD string."""
    d = target_date or date.today()
    return d.strftime("%Y%m%d")


@app.command()
def run(
    target_date: str = typer.Option(None, "--date", "-d", help="Target date (YYYYMMDD)"),
):
    """Run the full pipeline: collect, scan, news, analyze, report."""
    date_str = target_date or _get_date_str()
    scan_date = datetime.strptime(date_str, "%Y%m%d").date()

    settings = Settings()
    config = load_scanner_config()

    console.print(f"[bold]52주 신고가 스캔 시작: {date_str}[/bold]")

    from src.collector import Collector
    from src.scanner import Scanner
    from src.news_fetcher import NewsFetcher
    from src.ai_analyst import AIAnalyst
    from src.reporter import Reporter
    from src.db import Database

    db = Database()
    collector = Collector()
    scanner = Scanner(collector=collector)

    # Step 1: Collect daily data
    console.print("[dim]1/5 데이터 수집 중...[/dim]")
    daily_data = collector.collect_daily(date_str, markets=config.scanner.markets)

    # Step 2: Get sector info and stock names
    console.print("[dim]2/5 섹터 정보 수집 중...[/dim]")
    sector_map = {}
    name_map = {}
    for market in ["KOSPI", "KOSDAQ"]:
        sector_map.update(collector.get_sector_map(date_str, market))
    from pykrx import stock as pykrx_stock
    for ticker in daily_data:
        if ticker not in name_map:
            try:
                name = pykrx_stock.get_market_ticker_name(ticker)
                if isinstance(name, str) and name:
                    name_map[ticker] = name
                    continue
            except Exception:
                pass
            try:
                name = pykrx_stock.get_etf_ticker_name(ticker)
                if isinstance(name, str) and name:
                    name_map[ticker] = name
                    continue
            except Exception:
                pass
            name_map[ticker] = ticker

    # Step 3: Scan for 52-week highs
    console.print("[dim]3/5 52주 신고가 스캔 중...[/dim]")
    market_caps = collector.get_market_caps(date_str)
    highs = scanner.find_new_highs(
        daily_data=daily_data,
        date_str=date_str,
        sector_map=sector_map,
        name_map=name_map,
        lookback=config.scanner.lookback_days,
    )
    result = scanner.build_scan_result(scan_date, highs, len(daily_data))
    db.save_scan_result(result)

    # Step 4: Fetch news and AI analysis
    console.print("[dim]4/5 뉴스 수집 및 AI 분석 중...[/dim]")
    stock_names = [h.name for h in highs]
    fetcher = NewsFetcher(settings.naver_client_id, settings.naver_client_secret)
    news_map = asyncio.run(
        fetcher.fetch_news_for_stocks(stock_names, config.news.max_articles_per_stock)
    )

    analyst = AIAnalyst(
        api_key=settings.openai_api_key,
        model=config.ai.model,
        max_tokens=config.ai.max_tokens,
    )
    ai_results = asyncio.run(
        analyst.analyze_stocks(highs, news_map, market_caps, config.scanner.max_ai_analyze)
    )
    for ar in ai_results:
        db.save_ai_analysis(scan_date, ar)

    # Step 5: Send report
    console.print("[dim]5/5 리포트 전송 중...[/dim]")
    trend = db.get_high_count_history(days=5)

    if config.telegram.enabled and settings.telegram_bot_token:
        reporter = Reporter(settings.telegram_bot_token, settings.telegram_chat_id)
        asyncio.run(reporter.send_report(result, ai_results, trend))
        console.print("[green]텔레그램 리포트 전송 완료![/green]")
    else:
        reporter = Reporter(bot_token="", chat_id=0)
        text = reporter.format_report(result, ai_results, trend)
        console.print(text)

    console.print(f"[bold green]완료! {result.stats.new_high_count}개 신고가 종목 발견[/bold green]")


@app.command()
def collect(
    target_date: str = typer.Option(None, "--date", "-d", help="Target date (YYYYMMDD)"),
):
    """Collect daily market data only."""
    date_str = target_date or _get_date_str()
    config = load_scanner_config()

    from src.collector import Collector
    collector = Collector()
    daily_data = collector.collect_daily(date_str, markets=config.scanner.markets)
    console.print(f"[green]수집 완료: {len(daily_data)}개 종목[/green]")


@app.command()
def history(
    target_date: str = typer.Option(None, "--date", "-d", help="Date to query (YYYYMMDD)"),
):
    """Query historical scan results."""
    from src.db import Database
    db = Database()

    if target_date:
        scan_date = datetime.strptime(target_date, "%Y%m%d").date()
        results = db.get_scan_result(scan_date)
        if not results:
            console.print(f"[yellow]{target_date} 데이터 없음[/yellow]")
            return

        table = Table(title=f"52주 신고가 ({target_date})")
        table.add_column("종목코드")
        table.add_column("종목명")
        table.add_column("시장")
        table.add_column("섹터")
        table.add_column("종가", justify="right")
        table.add_column("돌파율", justify="right")

        for r in results:
            table.add_row(
                r["ticker"], r["name"], r["market"], r["sector"],
                f"{r['close_price']:,.0f}", f"+{r['breakout_pct']:.1f}%",
            )
        console.print(table)
    else:
        trend = db.get_high_count_history(days=10)
        if not trend:
            console.print("[yellow]저장된 데이터 없음[/yellow]")
            return
        table = Table(title="최근 52주 신고가 추이")
        table.add_column("날짜")
        table.add_column("신고가 종목 수", justify="right")
        for t in trend:
            table.add_row(str(t["date"]), str(t["count"]))
        console.print(table)


@app.command()
def stats(
    days: int = typer.Option(30, "--days", "-n", help="Number of days"),
):
    """Show historical statistics."""
    from src.db import Database
    db = Database()
    trend = db.get_high_count_history(days=days)
    if not trend:
        console.print("[yellow]저장된 데이터 없음[/yellow]")
        return

    counts = [t["count"] for t in trend]
    avg = sum(counts) / len(counts)
    console.print(f"[bold]최근 {len(trend)}일 통계[/bold]")
    console.print(f"  평균 신고가 종목 수: {avg:.1f}")
    console.print(f"  최대: {max(counts)} / 최소: {min(counts)}")


@app.command(name="test-ai")
def test_ai():
    """Test OpenAI API connection and model response."""
    settings = Settings()
    config = load_scanner_config()

    if not settings.openai_api_key:
        console.print("[red]OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.[/red]")
        return

    console.print(f"[dim]모델: {config.ai.model}[/dim]")
    console.print(f"[dim]API Key: {settings.openai_api_key[:8]}...{settings.openai_api_key[-4:]}[/dim]")
    console.print("[dim]테스트 요청 전송 중...[/dim]")

    import openai
    try:
        client = openai.OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=config.ai.model,
            messages=[{"role": "user", "content": "삼성전자가 52주 신고가를 기록한 이유를 한 문장으로 분석해주세요."}],
            max_completion_tokens=200,
        )
        content = response.choices[0].message.content
        console.print(f"\n[green]API 연결 성공![/green]")
        console.print(f"[dim]응답 content 타입: {type(content).__name__}[/dim]")
        console.print(f"[dim]응답 content 값: {repr(content)}[/dim]")
        if content:
            console.print(f"\n[bold]AI 응답:[/bold]\n{content}")
        else:
            refusal = response.choices[0].message.refusal
            console.print(f"[yellow]content가 비어있습니다.[/yellow]")
            console.print(f"[dim]refusal: {repr(refusal)}[/dim]")
            console.print(f"[dim]전체 message: {response.choices[0].message}[/dim]")
    except Exception as e:
        console.print(f"\n[red]API 호출 실패: {type(e).__name__}[/red]")
        console.print(f"[red]{e}[/red]")


if __name__ == "__main__":
    app()
