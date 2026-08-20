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


def _collection_blocked_reason(scan_date: date, today: date | None = None) -> str | None:
    """investing.com은 '오늘'의 52주 신고가 목록만 제공한다.

    scan_date가 오늘이 아니면 수집을 막아야 하는 이유를 반환한다(오늘이면 None).
    today를 주입받아 시계를 직접 읽지 않는다 — 단위 테스트를 위해서다.
    """
    ref = today or date.today()
    if scan_date == ref:
        return None
    date_str = scan_date.strftime("%Y%m%d")
    return (
        f"investing.com은 오늘 하루치 52주 신고가 목록만 제공하므로, "
        f"{date_str}로 수집하면 오늘 종목·가격이 그 날짜로 저장됩니다. "
        f"원시 일별 데이터가 필요하면 'collect --date {date_str}'를, "
        f"이미 저장된 과거 스캔을 보려면 'history --date {date_str}'를 사용하세요."
    )


def _make_client(settings: Settings):
    """Create a KrxClient from settings."""
    from src.krx_client import create_krx_client
    return create_krx_client(
        krx_id=getattr(settings, "krx_id", ""),
        krx_pw=getattr(settings, "krx_pw", ""),
        krx_api_key=settings.krx_api_key,
    )


@app.command()
def run(
    target_date: str = typer.Option(None, "--date", "-d", help="Target date (YYYYMMDD)"),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="캐시 무시하고 해당 날짜를 완전히 다시 수집·스캔·분석",
    ),
):
    """Run the full pipeline: collect, scan, news, analyze, report."""
    date_str = target_date or _get_date_str()
    scan_date = datetime.strptime(date_str, "%Y%m%d").date()

    settings = Settings()
    config = load_scanner_config()

    console.print(f"[bold]52주 신고가 스캔 시작: {date_str}[/bold]")

    from src.news_fetcher import NewsFetcher
    from src.ai_analyst import AIAnalyst
    from src.reporter import Reporter
    from src.db import Database

    db = Database()

    # Step 1-3: 수집/스캔 (DB에 이전 결과 있으면 스킵; --force면 무시하고 재수집)
    existing = None if force else db.get_scan_result_full(scan_date)
    market_caps: dict[str, int] = {}
    if existing:
        console.print("[dim]1-3/5 이전 스캔 결과 사용 (DB에서 로드)[/dim]")
        result = existing
        highs = result.highs
    else:
        reason = _collection_blocked_reason(scan_date)
        if reason:
            console.print(f"[red]{reason}[/red]")
            raise typer.Exit(code=1)

        from src.dart.cache import DartCache
        from src.collector import Collector
        from src.scanner import Scanner
        from src.investing_high import collect_investing_highs, InvestingFetchError, InvestingParseError
        from src.recency_source import enrich_highs
        from src.krx_login_client import KrxBlockedError

        client = _make_client(settings)
        collector = Collector(client=client)
        scanner = Scanner(collector=collector)
        corps = DartCache().load_corp_info(markets=["KOSPI", "KOSDAQ"])

        console.print("[dim]1-3/5 investing.com 52주 신고가 수집 중...[/dim]")
        try:
            highs, market_caps = collect_investing_highs(date_str, collector, corps)
        except (InvestingFetchError, InvestingParseError) as e:
            console.print(f"[red]investing 신고가 수집 실패: {e}[/red]")
            raise typer.Exit(code=1)

        console.print(f"[dim]1-3/5 돌파 신선도 계산 중... ({len(highs)}종목)[/dim]")
        try:
            enrich_highs(client, highs, scan_date)
        except KrxBlockedError:
            # 차단은 신선도 지표만 잃는다. 스캔·뉴스·AI·리포트는 그대로 진행한다.
            console.print("[yellow]KRX 차단으로 돌파 신선도 일부/전체 누락[/yellow]")

        result = scanner.build_scan_result(scan_date, highs, len(highs))
        db.save_scan_result(result)

    # Step 4: 뉴스 수집 및 AI 분석 (이미 분석된 티커 스킵; --force면 전체 재분석)
    if force:
        db.delete_ai_analyses(scan_date)
    done_tickers = set() if force else db.get_ai_analyzed_tickers(scan_date)
    remaining = [h for h in highs if h.ticker not in done_tickers]

    if remaining:
        skipped = len(highs) - len(remaining)
        if skipped:
            console.print(f"[dim]4/5 AI 분석 중... ({skipped}개 스킵, {len(remaining)}개 남음)[/dim]")
        else:
            console.print("[dim]4/5 뉴스 수집 및 AI 분석 중...[/dim]")

        stock_names = [h.name for h in remaining]
        fetcher = NewsFetcher(settings.naver_client_id, settings.naver_client_secret)
        news_map = asyncio.run(
            fetcher.fetch_news_for_stocks(stock_names, config.news.max_articles_per_stock)
        )

        analyst = AIAnalyst(
            api_key=settings.openai_api_key,
            model=config.ai.model,
        )
        ai_results = asyncio.run(
            analyst.analyze_stocks(remaining, news_map, market_caps, config.scanner.max_ai_analyze)
        )
        for ar in ai_results:
            db.save_ai_analysis(scan_date, ar)
    else:
        console.print("[dim]4/5 AI 분석 스킵 (모두 완료됨)[/dim]")

    # Step 5: DB에서 전체 AI 결과 로드하여 리포트 전송
    console.print("[dim]5/5 리포트 전송 중...[/dim]")
    all_ai = db.get_all_ai_analyses(scan_date)
    trend = db.get_high_count_history(days=5)

    if config.telegram.enabled and settings.telegram_bot_token:
        reporter = Reporter(settings.telegram_bot_token, settings.telegram_chat_id)
        asyncio.run(reporter.send_report(result, all_ai, trend))
        console.print("[green]텔레그램 리포트 전송 완료![/green]")
        db.delete_ai_analyses(scan_date)
    else:
        reporter = Reporter(bot_token="", chat_id=0)
        text = reporter.format_report(result, all_ai, trend)
        console.print(text)

    console.print(f"[bold green]완료! {result.stats.new_high_count}개 신고가 종목 발견[/bold green]")


@app.command()
def collect(
    target_date: str = typer.Option(None, "--date", "-d", help="Target date (YYYYMMDD)"),
):
    """Collect daily market data only."""
    date_str = target_date or _get_date_str()
    settings = Settings()
    config = load_scanner_config()

    from src.collector import Collector
    client = _make_client(settings)
    collector = Collector(client=client)
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
