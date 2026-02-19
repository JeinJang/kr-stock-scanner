# 52-Week High Stock Scanner Design

**Date:** 2026-02-19
**Status:** Approved

## Overview

Korean stock market daily 52-week high scanner. Collects post-market data from KRX, identifies 52-week high stocks across KOSPI/KOSDAQ/ETF, analyzes rise reasons using OpenAI GPT, and delivers reports via Telegram.

## Requirements

- **Target:** KOSPI + KOSDAQ + ETF (all listed instruments)
- **Data source:** pykrx (KRX public data, free)
- **Storage:** SQLite local DB
- **AI analysis:** OpenAI GPT-4o/4o-mini for stock rise reason analysis
- **News:** Naver News crawling/search for each high stock
- **Report delivery:** Telegram Bot
- **Execution:** CLI manual + cron automated (weekdays 16:00)

## Architecture

Monolithic Python project with modular separation.

```
kr-stock-scanner/
├── pyproject.toml
├── config.yaml              # Settings (Telegram token, API keys)
├── .env                     # Secrets (gitignored)
├── data/
│   └── scanner.db           # SQLite DB
├── src/
│   ├── __init__.py
│   ├── cli.py               # typer-based CLI entry point
│   ├── config.py            # pydantic-settings config loader
│   ├── collector.py         # pykrx data collection
│   ├── scanner.py           # 52-week high filtering logic
│   ├── sector.py            # Sector/industry classification
│   ├── news_fetcher.py      # News collection (Naver News)
│   ├── ai_analyst.py        # OpenAI GPT analysis
│   ├── reporter.py          # Telegram report formatting & sending
│   ├── db.py                # SQLite ORM (SQLAlchemy)
│   └── models.py            # Pydantic data models
└── tests/
```

## Data Collection

1. `collector.py` uses pykrx to fetch daily OHLCV for all KOSPI+KOSDAQ+ETF stocks
2. Uses `stock.get_market_ohlcv_by_ticker()` to get all tickers' prices at once
3. Compares today's high/close against 52-week (250 trading days) maximum

## 52-Week High Detection

- Fetch 250 trading days of OHLCV data per ticker
- Condition: today's high >= max high over past 250 days
- Calculate breakout percentage vs previous 52-week high
- Track both new breakouts and continued highs

## Sector Classification

- Fetch sector/industry data from KRX via pykrx
- Group new-high stocks by sector
- Calculate sector-level statistics (count, percentage)

## Market Statistics

- Daily new-high count by market (KOSPI / KOSDAQ / ETF)
- Day-over-day change in new-high count
- N-day trend of new-high counts (market sentiment indicator)
- Top 5 sectors by new-high stock ratio

## Database Schema (SQLite + SQLAlchemy)

### daily_scans
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| scan_date | DATE | Scan date |
| total_stocks | INTEGER | Total stocks scanned |
| new_high_count | INTEGER | Stocks at 52w high |
| market_type | VARCHAR | KOSPI/KOSDAQ/ETF/ALL |

### new_highs
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| scan_date | DATE | Scan date |
| ticker | VARCHAR | Stock ticker code |
| name | VARCHAR | Stock name |
| market | VARCHAR | KOSPI/KOSDAQ/ETF |
| sector | VARCHAR | Industry sector |
| close_price | FLOAT | Closing price |
| high_52w | FLOAT | 52-week high (today) |
| prev_high_52w | FLOAT | Previous 52-week high |
| breakout_pct | FLOAT | Breakout % vs prev high |
| volume | BIGINT | Today's volume |
| avg_volume_20d | BIGINT | 20-day average volume |

### ai_analyses
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| ticker | VARCHAR | Stock ticker code |
| scan_date | DATE | Scan date |
| news_summary | TEXT | Collected news summary |
| ai_analysis | TEXT | GPT analysis result |
| created_at | DATETIME | Creation timestamp |

## News Fetching

- Search Naver News by stock name, last 1-3 days
- Extract title + summary for each article
- Limit 3-5 articles per stock (cost management)

## AI Analysis (OpenAI GPT)

- Use GPT-4o-mini for cost efficiency
- Prompt: stock data + news → 1-2 sentence analysis of why stock hit 52w high
- Cost cap: if >50 new-high stocks, only analyze top stocks by market cap
- Remaining stocks get news links only
- Estimated daily cost: $0.01~$0.05

## Telegram Report Format

```
52-week High Report (YYYY-MM-DD)

Market Summary
- New highs: N (KOSPI X / KOSDAQ Y / ETF Z)
- vs Previous day: +/- N
- 5-day trend: a->b->c->d->e

Sector TOP
1. Semiconductors (8 stocks): Samsung, SK Hynix...
2. Battery (5 stocks): LG Energy...

Key Stock AI Analysis
Stock Name (ticker) | price | change%
  52w high breakout (+X% new high)
  "News-based AI analysis summary"

Full 52-Week High List
[Name | Close | Change% | Sector] table
```

## CLI Commands

```bash
python -m src.cli run              # Full pipeline
python -m src.cli collect          # Data collection only
python -m src.cli analyze          # Analysis only (reads from DB)
python -m src.cli report           # Report generation/send only
python -m src.cli history --date   # Query past data
python -m src.cli stats --days 30  # Historical statistics
```

## Scheduling

- crontab: `0 16 * * 1-5` (weekdays 16:00 KST)
- Configurable via config.yaml

## Tech Stack

- Python 3.11+
- pykrx (KRX data)
- SQLAlchemy (ORM)
- pydantic / pydantic-settings (config & models)
- typer (CLI)
- openai (GPT API)
- python-telegram-bot (Telegram)
- httpx (HTTP client for news)
- rich (console output)
- loguru (logging)
