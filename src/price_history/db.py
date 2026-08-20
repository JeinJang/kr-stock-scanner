"""data/prices.db — 원주가 일봉과 수정 이벤트 저장소.

이 패키지는 sqlite3를 직접 쓴다. 700만 행 규모의 단순 적재·범위 조회라
SQLAlchemy 계층이 이득 없이 비용만 된다(저장소의 다른 DB는 SQLAlchemy 사용).
"""
from __future__ import annotations

import os
import sqlite3
from datetime import date

from src.price_history.adjust import AdjustEvent, PxRow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_px (
    d      TEXT NOT NULL,
    ticker TEXT NOT NULL,
    market TEXT NOT NULL,
    high   INTEGER NOT NULL,
    close  INTEGER NOT NULL,
    chg    INTEGER NOT NULL,
    PRIMARY KEY (d, ticker)
);
CREATE INDEX IF NOT EXISTS idx_px_ticker_d ON daily_px(ticker, d);
CREATE INDEX IF NOT EXISTS idx_px_market_d ON daily_px(market, d);
CREATE TABLE IF NOT EXISTS px_adjust (
    ticker TEXT NOT NULL,
    d      TEXT NOT NULL,
    factor REAL NOT NULL,
    PRIMARY KEY (ticker, d)
);
CREATE TABLE IF NOT EXISTS px_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _to_date(s: str) -> date:
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


class PriceDB:
    """원주가 일봉 저장소. 스키마는 생성 시 보장한다."""

    def __init__(self, path: str = "data/prices.db"):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.path = path
        self.con = sqlite3.connect(path)
        self.con.executescript(_SCHEMA)
        self.con.commit()

    # -- 적재 ---------------------------------------------------------------

    def save_day(self, d: str, market: str, records: list[tuple]) -> int:
        """records = [(ticker, high, close, chg)]. 같은 (d,ticker)는 덮어쓴다."""
        rows = [(d, tk, market, int(h), int(c), int(ch)) for tk, h, c, ch in records]
        self.con.executemany(
            "INSERT OR REPLACE INTO daily_px (d,ticker,market,high,close,chg) "
            "VALUES (?,?,?,?,?,?)",
            rows,
        )
        self.con.commit()
        return len(rows)

    def loaded_dates(self, market: str) -> set[str]:
        cur = self.con.execute(
            "SELECT DISTINCT d FROM daily_px WHERE market = ?", (market,)
        )
        return {r[0] for r in cur}

    def last_loaded_date(self) -> str | None:
        r = self.con.execute("SELECT MAX(d) FROM daily_px").fetchone()
        return r[0] if r and r[0] else None

    # -- 메타 ---------------------------------------------------------------

    def set_meta(self, key: str, value: str) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO px_meta (key,value) VALUES (?,?)", (key, value)
        )
        self.con.commit()

    def get_meta(self, key: str) -> str | None:
        r = self.con.execute("SELECT value FROM px_meta WHERE key = ?", (key,)).fetchone()
        return r[0] if r else None

    # -- 조회 ---------------------------------------------------------------

    def load_rows(self, ticker: str, since: str) -> list[PxRow]:
        cur = self.con.execute(
            "SELECT d, high, close, chg FROM daily_px "
            "WHERE ticker = ? AND d >= ? ORDER BY d",
            (ticker, since),
        )
        return [
            PxRow(d=_to_date(d), high=float(h), close=float(c), chg=float(ch))
            for d, h, c, ch in cur
        ]

    def tickers(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT DISTINCT ticker FROM daily_px")]

    # -- 수정 이벤트 --------------------------------------------------------

    def save_events(self, ticker: str, events: list[AdjustEvent]) -> None:
        """해당 티커의 이벤트를 통째로 교체한다(재계산 결과로 덮어쓰기)."""
        self.con.execute("DELETE FROM px_adjust WHERE ticker = ?", (ticker,))
        self.con.executemany(
            "INSERT INTO px_adjust (ticker,d,factor) VALUES (?,?,?)",
            [(ticker, e.d.strftime("%Y%m%d"), e.factor) for e in events],
        )
        self.con.commit()

    def load_events(self, ticker: str) -> list[AdjustEvent]:
        cur = self.con.execute(
            "SELECT d, factor FROM px_adjust WHERE ticker = ? ORDER BY d", (ticker,)
        )
        return [AdjustEvent(d=_to_date(d), factor=float(f)) for d, f in cur]
