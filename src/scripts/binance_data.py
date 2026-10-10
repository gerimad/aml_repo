"""Download and load Binance spot candles (klines) from data.binance.vision.

Binance publishes its full candle history as zipped CSV files, one per symbol,
interval and month (and one per day for the current month). No API key needed.

Usage, from the repository root:

    uv run python src/scripts/binance_data.py                 # default symbols, 1h
    uv run python src/scripts/binance_data.py --symbols BTCUSDT ETHUSDT --interval 1h

This writes one Parquet file per symbol to data/binance_spot_<interval>/.
Load one in a notebook with:

    from binance_data import load_klines
    df = load_klines("BTCUSDT", "1h")
"""

from __future__ import annotations

import argparse
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://data.binance.vision/data/spot"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]

# Column order of the Binance kline CSV files.
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]
KEEP_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
]


def month_range(start: date, end: date) -> list[tuple[int, int]]:
    """Return every (year, month) from the month of `start` to the month of `end`."""
    months = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def monthly_url(symbol: str, interval: str, year: int, month: int) -> str:
    """URL of the zip file holding one month of candles."""
    name = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    return f"{BASE_URL}/monthly/klines/{symbol}/{interval}/{name}"


def daily_url(symbol: str, interval: str, day: date) -> str:
    """URL of the zip file holding one day of candles."""
    name = f"{symbol}-{interval}-{day.isoformat()}.zip"
    return f"{BASE_URL}/daily/klines/{symbol}/{interval}/{name}"


def parse_kline_csv(raw: bytes) -> pd.DataFrame:
    """Parse one Binance kline CSV into a frame indexed by UTC open time.

    Handles two quirks of the files: some have a header row and some do not,
    and timestamps are in milliseconds before 2025 and microseconds from 2025.
    """
    df = pd.read_csv(io.BytesIO(raw), header=None, names=KLINE_COLUMNS, dtype=str)
    if not df.empty and not df["open_time"].iloc[0].isdigit():
        df = df.iloc[1:]
    open_time = df["open_time"].astype("int64")
    # Millisecond timestamps have 13 digits until the year 2286, microsecond
    # ones have 16, so 1e14 separates the two units.
    in_ms = open_time.where(open_time < 10**14, open_time // 1000)
    out = df[KEEP_COLUMNS].astype("float64")
    out.index = pd.DatetimeIndex(pd.to_datetime(in_ms, unit="ms", utc=True), name="open_time")
    return out


def fetch_zip(url: str, session: requests.Session | None = None) -> bytes | None:
    """Download one zip file and return the CSV inside it, or None if it is not published."""
    response = (session or requests).get(url, timeout=60)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        return archive.read(archive.namelist()[0])


def download_klines(
    symbol: str,
    interval: str = "1h",
    start: date = date(2017, 8, 1),
    end: date | None = None,
    workers: int = 8,
) -> pd.DataFrame:
    """Download all candles of one symbol between `start` and `end` (inclusive).

    Uses the monthly files, then the daily files for the days after the last
    published month. Months before the symbol was listed are skipped.
    """
    end = end or date.today() - timedelta(days=1)
    months = month_range(start, end)
    session = requests.Session()
    with ThreadPoolExecutor(workers) as pool:
        raws = list(
            pool.map(lambda ym: fetch_zip(monthly_url(symbol, interval, *ym), session), months)
        )
    frames = [parse_kline_csv(raw) for raw in raws if raw is not None]

    # Days after the last published month only exist as daily files.
    published = [ym for ym, raw in zip(months, raws) if raw is not None]
    if published:
        year, month = published[-1]
        first_missing = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    else:
        first_missing = start
    days = [first_missing + timedelta(days=i) for i in range((end - first_missing).days + 1)]
    with ThreadPoolExecutor(workers) as pool:
        raws = list(pool.map(lambda d: fetch_zip(daily_url(symbol, interval, d), session), days))
    frames += [parse_kline_csv(raw) for raw in raws if raw is not None]

    if not frames:
        raise ValueError(f"no data published for {symbol} {interval}")
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]


def klines_path(symbol: str, interval: str = "1h", data_dir: Path = DATA_DIR) -> Path:
    """Path of the local Parquet file for one symbol and interval."""
    return data_dir / f"binance_spot_{interval}" / f"{symbol}_{interval}.parquet"


def load_klines(symbol: str, interval: str = "1h", data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load previously downloaded candles for one symbol."""
    return pd.read_parquet(klines_path(symbol, interval, data_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2017, 8, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    args = parser.parse_args()

    for symbol in args.symbols:
        df = download_klines(symbol, args.interval, args.start, args.end)
        path = klines_path(symbol, args.interval)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        print(f"{symbol}: {len(df):,} rows, {df.index[0]} to {df.index[-1]} -> {path}")


if __name__ == "__main__":
    main()
