"""Hourly order-flow tables of new tokens for launch momentum (implementation plan 12.8).

Usage, from the repository root:

    uv run python src/scripts/build_launch_tables.py 2026-07-02 2026-09-10 <out file>

One table per hour (strategy.launch_table), all written to one Parquet file.
Only the price columns the tables need are loaded, to keep memory low.
"""

from __future__ import annotations

import sys

import pandas as pd

import prices as px
import strategy as st

COLUMNS = ["timestamp", "tx_from", "side", "volume_eth", "token", "pool", "block", "tx_index", "log_index"]


def main(start: str, end: str, out: str) -> None:
    prices = px.load_prices()[COLUMNS]
    for column in ("tx_from", "token", "pool", "side"):
        prices[column] = prices[column].astype("category")
    launch = st.launch_times(prices)
    hours = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="1h", inclusive="left")
    parts = []
    for i, s in enumerate(hours):
        window = prices[(prices["timestamp"] >= s - pd.Timedelta(hours=7)) & (prices["timestamp"] < s)]
        table = st.launch_table(window, s, launch=launch)
        for column in ("token", "pool"):
            table[column] = table[column].astype(str)
        parts.append(table)
        if i % 24 == 0:
            print(f"  {s:%d %b}: {len(table)} young tokens", flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(out)
    print(f"Written {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
