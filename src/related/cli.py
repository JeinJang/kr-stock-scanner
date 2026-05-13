from __future__ import annotations

import asyncio
import os
import webbrowser
from datetime import date, datetime

import typer
from rich.console import Console

from src.config import Settings, load_scanner_config
from src.dart.cache import DartCache
from src.related.db import RelatedDB

app = typer.Typer(help="Related companies discovery via DART business reports",
                  no_args_is_help=True)
console = Console()


async def _process_one(
    ticker: str,
    refresh: bool,
    settings: Settings,
    config,
    cache: DartCache,
    db: RelatedDB,
) -> bool:
    """Fetch + extract + save edges for a single ticker. Returns True on success."""
    from src.dart.client import DartClient
    from src.related.extractor import RelationExtractor
    from src.related.report_fetcher import ReportFetcher

    corps = cache.load_corp_info()
    by_ticker = {c.ticker: c for c in corps}
    name_to_ticker = {c.name: c.ticker for c in corps}
    if ticker not in by_ticker:
        console.print(
            f"[yellow]{ticker} 가 DART corp_info에 없습니다. "
            f"먼저 `python -m src.fundamentals.cli refresh` 를 실행하세요.[/yellow]"
        )
        return False
    corp = by_ticker[ticker]

    client = DartClient(api_key=settings.opendart_api_key)
    fetcher = ReportFetcher(client=client)

    rcept_no = await fetcher.latest_rcept_no(corp.corp_code)
    if rcept_no is None:
        console.print(f"[yellow]{ticker} ({corp.name}): 사업보고서 없음[/yellow]")
        return False

    if not refresh and not db.needs_refresh(ticker, rcept_no):
        console.print(f"[dim]  {ticker} ({corp.name}): 캐시 사용 (rcept_no={rcept_no})[/dim]")
        return True

    console.print(f"[dim]  {ticker} ({corp.name}): 사업보고서 다운로드 중...[/dim]")
    xml_text = await fetcher.download_document(rcept_no)
    sections = fetcher.parse_sections(xml_text, corp.corp_code, rcept_no)

    console.print(f"[dim]  {ticker}: GPT 추출 중...[/dim]")
    extractor = RelationExtractor(
        api_key=settings.openai_api_key, model=config.related.model,
    )
    edges = await extractor.extract(
        target_name=corp.name, target_ticker=ticker,
        sections=sections, name_to_ticker=name_to_ticker,
    )
    now = datetime.now()
    db.save_edges(ticker, edges, now)
    db.set_meta(ticker, rcept_no, now)
    console.print(f"[green]  {ticker}: {len(edges)}개 관계 추출[/green]")
    return True


def _load_scores(db_url: str = "sqlite:///data/scanner.db") -> dict[str, dict]:
    """Load fundamentals scores keyed by ticker. Returns empty if module/data missing."""
    try:
        from src.fundamentals.db import FundamentalsDB
        fdb = FundamentalsDB(url=db_url)
        scores = fdb.load_scores(date.today())
        return {
            s.ticker: {
                "total_score": s.total_score, "grade": s.grade,
                "categories": s.categories,
            } for s in scores
        }
    except Exception:
        return {}


@app.command()
def show(
    ticker: str,
    depth: int = typer.Option(1, "--depth", "-d", help="Multi-hop expansion depth"),
    refresh: bool = typer.Option(False, "--refresh", help="Force re-extraction"),
):
    """Build related-companies graph for one ticker and open HTML report."""
    from src.related.graph import build_graph, expand
    from src.related.report import ReportGenerator

    settings = Settings()
    config = load_scanner_config()
    cache = DartCache()
    db = RelatedDB()

    corps = cache.load_corp_info()
    by_ticker = {c.ticker: c for c in corps}
    if ticker not in by_ticker:
        console.print(f"[yellow]{ticker} 가 DART corp_info에 없습니다.[/yellow]")
        return

    console.print(f"[bold]연관 기업 발굴: {by_ticker[ticker].name} ({ticker})[/bold]")
    ok = asyncio.run(_process_one(ticker, refresh, settings, config, cache, db))
    if not ok:
        return

    if depth > 1:
        first_edges = db.load_edges(source_tickers=[ticker])
        neighbors = [e.target_ticker for e in first_edges
                     if e.target_ticker and e.target_ticker != ticker]
        for n in neighbors:
            asyncio.run(_process_one(n, refresh=False, settings=settings,
                                      config=config, cache=cache, db=db))

    seed = [ticker]
    if depth > 1:
        seed += [e.target_ticker for e in db.load_edges(source_tickers=[ticker])
                 if e.target_ticker]
    full_edges = db.load_edges(source_tickers=list(set(seed)))
    g = build_graph(full_edges)
    sub = expand(g, ticker, depth=depth)

    meta = db.get_meta(ticker)
    rcept_no = meta.rcept_no if meta else "unknown"

    name_map = {c.ticker: c.name for c in corps}
    market_map = {c.ticker: c.market for c in corps}
    scores = _load_scores()

    gen = ReportGenerator()
    path = gen.generate(
        root_ticker=ticker, graph=sub, edges=full_edges,
        name_map=name_map, market_map=market_map, scores=scores,
        depth=depth, rcept_no=rcept_no,
        as_of_date=str(date.today()),
        output_dir=config.related.report_dir,
    )
    console.print(f"[bold green]완료! 리포트: {path}[/bold green]")
    webbrowser.open(f"file://{os.path.abspath(path)}")


@app.command()
def batch(
    tickers: str = typer.Option(..., "--tickers", help="Comma-separated tickers"),
    refresh: bool = typer.Option(False, "--refresh", help="Force re-extraction"),
):
    """Run extraction (no report) for an explicit list of tickers."""
    settings = Settings()
    config = load_scanner_config()
    cache = DartCache()
    db = RelatedDB()
    ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
    console.print(f"[bold]배치 추출: {len(ticker_list)}개 종목[/bold]")
    for t in ticker_list:
        asyncio.run(_process_one(t, refresh, settings, config, cache, db))


@app.command()
def stats():
    """Print stored edge stats."""
    db = RelatedDB()
    s = db.stats()
    console.print(f"[bold]저장된 관계 통계[/bold]")
    console.print(f"  source 종목: {s['sources']}")
    console.print(f"  전체 엣지: {s['edges']}")
    for rel, cnt in s.get("by_relation", {}).items():
        console.print(f"    {rel}: {cnt}")


if __name__ == "__main__":
    app()
