"""Build the strategy signal for every 6-hour decision time in a period (work package 8).

Usage, from the repository root:

    uv run python src/scripts/build_signals.py 2026-07-02 2026-09-10 [extension]

Each decision is saved to data/project/signal/ as soon as it is built, so a
stopped run resumes where it stopped. With "extension", the legs and tokens
of the held-out extension are used (token_data.build_extension) in addition
to the project's; decisions inside the project's period give the same signal
either way, since the signal uses only data before the decision time.
"""

from __future__ import annotations

import json
import sys
import time

import pandas as pd

import prices as px
import protocol as pc
import strategy as st
import token_data as td


def main(start: str, end: str, use_extension: bool) -> None:
    legs = td.load_all_legs() if use_extension else td.load_legs()
    tokens = td.load_all_tokens() if use_extension else td.load_tokens()
    users_dir = td.EXTENSION_DIR if use_extension else td.PROJECT_DIR
    users = pd.read_parquet(users_dir / "users.parquet")
    frozen = json.loads((pc.RESULTS_DIR.parent / "configs" / "frozen.json").read_text())
    settings = frozen["models"]["fixed blend"]["settings"]
    times = st.decision_times(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))
    # Prices from 14 days before the first decision (account volumes) to the last one.
    first_ts = times[0] - pd.Timedelta(days=15)
    # Only the columns the signal uses, with repeated text as categories: the machine has 8 GB.
    prices = px.load_prices()[["timestamp", "tx_from", "side", "volume_eth", "token", "pool"]]
    prices = prices[(prices["timestamp"] >= first_ts) & (prices["timestamp"] < times[-1])].reset_index(drop=True)
    for column in ("tx_from", "side", "token", "pool"):
        prices[column] = prices[column].astype("category")
    print(f"{len(times)} decision times from {times[0]} to {times[-1]}; {len(prices):,} priced swaps loaded.", flush=True)
    for s in times:
        started = time.time()
        table = st.load_or_build_signal(legs, prices, tokens, users, s, settings)
        print(f"  {s:%d %b %H:%M}: {len(table):,} tail tokens ({time.time() - started:.0f} s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], len(sys.argv) > 3 and sys.argv[3] == "extension")
