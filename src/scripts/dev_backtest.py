"""Development backtests of work package 8 (implementation plan 12.4, 12.7 step 9 and 12.8).

Every tail version and every launch version is backtested over its grid of
settings, together with its benchmarks, on the development data:

  tuning period   Dev A and Dev B, decisions from 2 July to 13 August
  choice period   the project's test period, decisions from 13 August on

Only prices up to the end of the project data (10 September, block
25,948,799) are used, so nothing of the held-out weeks is read. A position
needs prices up to its holding time plus 24 hours of grace to exit, so the
last decisions are dropped: a tail decision must lie at least 48 + 24 hours
before 10 September, a launch decision 72 + 24 hours (the longest holding
time of each grid). The same decisions are used for every setting.

Usage, from src/notebooks (writes data/project/backtest_dev/):

    uv run python -u ../scripts/dev_backtest.py

The notebook 18_backtest_development reads what this script writes.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import backtest as bt
import performance as pf
import prices as px
import strategy as st
import token_data as td

OUT_DIR = td.PROJECT_DIR / "backtest_dev"
CHOICE_START = pd.Timestamp("2026-08-13", tz="UTC")  # the project's test period: used to choose, never to tune
DATA_END = td.DATA_END  # 10 September: the held-out weeks start here and are not read
GRACE_HOURS = 24

TAIL_VERSIONS = ["surprise", "filtered_visible", "unexplained_flow"]
TAIL_BENCHMARKS = ["tail_popularity", "tail_flow", "random", "predicted"]  # "predicted" is reported, not a candidate
TAIL_K = [3, 5, 10]
TAIL_HOURS = [6, 24, 48]
TAIL_STOPS = [None, 0.5]
TAIL_WEIGHTINGS = ["accounts", "eth"]

LAUNCH_N = [10, 30]
LAUNCH_B = [0.5, 0.7]
LAUNCH_K = [1, 3]
LAUNCH_HOURS = [6, 24, 72]
LAUNCH_LADDERS = [(3, 10), (5, 20), None]
LAUNCH_POSITION_ETH = 0.05  # tiny and equal (plan 12.8); capped at 1% of the buy volume since launch


# --------------------------------------------------------------------------
# The grids
# --------------------------------------------------------------------------


def tail_grid() -> pd.DataFrame:
    """Every tail setting: the three candidate versions over K, H, stop and weighting, and the benchmarks over K, H and stop.

    Benchmarks pick by visible demand or at random, so the weighting of
    predicted demand does not change them; they get weighting "accounts".
    """
    rows = []
    for version in TAIL_VERSIONS + TAIL_BENCHMARKS:
        weightings = TAIL_WEIGHTINGS if version in TAIL_VERSIONS else ["accounts"]
        for k, hours, stop, weighting in itertools.product(TAIL_K, TAIL_HOURS, TAIL_STOPS, weightings):
            rows.append({"version": version, "k": k, "holding_hours": hours, "stop": stop, "weighting": weighting})
    grid = pd.DataFrame(rows)
    grid["candidate"] = grid["version"].isin(TAIL_VERSIONS)
    return grid


def launch_grid() -> pd.DataFrame:
    """Every launch setting: launch momentum over N, b, K, H and the ladder, and launch_random over N, K, H and the ladder.

    launch_random draws from the same universe (at least N buyers since
    launch) and does not look at the buy share, so b does not apply to it.
    """
    rows = []
    for n, b, k, hours, ladder in itertools.product(LAUNCH_N, LAUNCH_B, LAUNCH_K, LAUNCH_HOURS, LAUNCH_LADDERS):
        rows.append({"version": "launch_momentum", "min_buyers": n, "min_buy_share": b, "k": k, "holding_hours": hours, "ladder": ladder})
    for n, k, hours, ladder in itertools.product(LAUNCH_N, LAUNCH_K, LAUNCH_HOURS, LAUNCH_LADDERS):
        rows.append({"version": "launch_random", "min_buyers": n, "min_buy_share": np.nan, "k": k, "holding_hours": hours, "ladder": ladder})
    grid = pd.DataFrame(rows)
    grid["candidate"] = grid["version"] == "launch_momentum"
    return grid


def setting_id(row) -> str:
    """A short readable name of one setting, used as the key of its trades and daily returns."""
    if row["version"].startswith("launch"):
        ladder = "none" if row["ladder"] is None else f"{row['ladder'][0]}-{row['ladder'][1]}"
        share = "" if pd.isna(row["min_buy_share"]) else f"_b{row['min_buy_share']:g}"
        return f"{row['version']}_N{row['min_buyers']}{share}_K{row['k']}_H{row['holding_hours']}_ladder{ladder}"
    stop = "none" if row["stop"] is None or pd.isna(row["stop"]) else f"{row['stop']:g}"
    return f"{row['version']}_K{row['k']}_H{row['holding_hours']}_stop{stop}_{row['weighting']}"


# --------------------------------------------------------------------------
# Decisions and selections
# --------------------------------------------------------------------------


def last_decision(longest_hours: float, end: pd.Timestamp = DATA_END) -> pd.Timestamp:
    """The last decision time whose positions can still exit before `end`: end minus the longest holding time and the grace."""
    return end - pd.Timedelta(hours=longest_hours + GRACE_HOURS)


def tail_key(row) -> tuple:
    """The part of a tail setting that decides which tokens are picked."""
    return (row["version"], int(row["k"]), row["weighting"])


def launch_key(row) -> tuple:
    """The part of a launch setting that decides which tokens are picked (b is None for launch_random)."""
    b = None if pd.isna(row["min_buy_share"]) else float(row["min_buy_share"])
    return (row["version"], int(row["min_buyers"]), b, int(row["k"]))


def tail_selections(signal_files: list[Path], until: pd.Timestamp) -> dict[tuple, pd.DataFrame]:
    """The tokens each tail rule picks at each decision time up to `until`, by (version, k, weighting)."""
    keys = sorted({tail_key(row) for row in tail_grid().to_dict("records")})
    picked = {key: [] for key in keys}
    for path in signal_files:
        signal = pd.read_parquet(path)
        if signal.empty or signal["decision"].iloc[0] > until:
            continue
        for version, k, weighting in keys:
            chosen = st.select(signal, version, k=k, weighting=weighting)
            picked[(version, k, weighting)].append(chosen[["decision", "token", "pool", "buy_eth_24h"]])
    return {key: pd.concat(parts, ignore_index=True) for key, parts in picked.items() if parts}


def launch_selections(launch: pd.DataFrame, until: pd.Timestamp) -> dict[tuple, pd.DataFrame]:
    """The tokens each launch rule picks at each hourly decision up to `until`, by (version, N, b, k); a token is bought at most once."""
    keys = sorted({launch_key(row) for row in launch_grid().to_dict("records")}, key=str)
    picked = {key: [] for key in keys}
    bought = {key: set() for key in keys}
    for decision, table in launch[launch["decision"] <= until].groupby("decision"):
        for key in keys:
            version, n, b, k = key
            chosen = st.select_launch(table, version, k=k, min_buyers=n, min_buy_share=0.0 if b is None else b, already=bought[key])
            bought[key].update(chosen["token"])
            picked[key].append(chosen[["decision", "token", "pool", "buy_eth_since_launch"]])
    return {key: pd.concat(parts, ignore_index=True) for key, parts in picked.items() if parts}


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------

PRICE_COLUMNS = ["block", "tx_index", "log_index", "timestamp", "pool", "side", "price_eth", "volume_eth"]


def load_pool_prices(pools: set[str], last_block: int = td.PROJECT_LAST_BLOCK, out_dir: Path = px.PRICES_DIR) -> pd.DataFrame:
    """The priced swaps of the given pools up to last_block, read file by file with only the columns the backtest needs.

    Reading only these pools keeps memory low (8 GB machine). It does not
    change the backtest: a pool's entry and exits use its own swaps, and the
    decision block is found among swaps at or after s, of which every pool
    that trades after s has one (backtest.PoolPrices.first_block_at).
    """
    wanted = pd.Series(sorted(pools), dtype="string").to_numpy()
    parts = []
    for path in sorted(out_dir.glob("prices_*.parquet")):
        start = int(path.stem.split("_")[1])
        if start > last_block:
            continue
        table = pq.read_table(path, columns=PRICE_COLUMNS)
        table = table.filter(pc.is_in(table["pool"], value_set=pa.array(wanted)))
        table = table.filter(pc.less_equal(table["block"], last_block))
        parts.append(table.to_pandas())
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# Running the grids
# --------------------------------------------------------------------------


def tail_settings(row) -> bt.Settings:
    stop = None if row["stop"] is None or pd.isna(row["stop"]) else float(row["stop"])
    return bt.Settings(holding_hours=row["holding_hours"], stop=stop)


def launch_settings(row) -> bt.Settings:
    return bt.Settings(
        holding_hours=row["holding_hours"], take_profit=row["ladder"], position_eth=LAUNCH_POSITION_ETH, volume_column="buy_eth_since_launch"
    )


def run_grid(grid: pd.DataFrame, selections: dict, pool_prices: bt.PoolPrices, family: str) -> pd.DataFrame:
    """All trades of every setting of a grid, one table with the setting's id in column "setting"."""
    parts = []
    for i, row in enumerate(grid.to_dict("records")):
        if family == "tail":
            key, settings = tail_key(row), tail_settings(row)
        else:
            key, settings = launch_key(row), launch_settings(row)
        chosen = selections.get(key)
        if chosen is None or chosen.empty:
            continue
        trades = bt.run(chosen, pool_prices, settings)
        trades.insert(0, "setting", setting_id(row))
        parts.append(trades)
        if i % 10 == 0:
            print(f"  {family} {i + 1}/{len(grid)}: {setting_id(row)}, {len(trades):,} positions", flush=True)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# Daily returns and the choice rules
# --------------------------------------------------------------------------

def in_period(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """The positions of decisions in [start, end)."""
    decision = pd.to_datetime(trades["decision"], utc=True)
    return trades[(decision >= start) & (decision < end)]


def daily_returns(trades: pd.DataFrame, holding_hours: float, interval_hours: float) -> pd.Series:
    """Daily returns of the portfolio of sleeves (performance.equity_curve), net of costs."""
    return pf.daily_returns_from_equity(pf.equity_curve(trades, holding_hours, interval_hours=interval_hours))


def paired_difference(a: pd.Series, b: pd.Series) -> pd.Series:
    """Day-by-day a - b. A day missing from one curve had no change there, so its return counts as 0."""
    days = a.index.union(b.index)
    return a.reindex(days, fill_value=0.0) - b.reindex(days, fill_value=0.0)


def choose(table: pd.DataFrame, score: str, standard_error: str, simpler: list[tuple[str, bool]]) -> pd.Series:
    """The chosen row: the highest score, except that rows within one standard error of the best go to the simplest.

    simpler lists (column, ascending) pairs in order of priority, e.g.
    [("k", True), ("weighting_is_eth", True), ("has_stop", True)]: a smaller
    K first, then counting accounts, then no stop. Among equally simple rows
    the higher score wins.
    """
    best = table.loc[table[score].idxmax()]
    close = table[table[score] >= best[score] - best[standard_error]]
    columns = [c for c, _ in simpler] + [score]
    ascending = [a for _, a in simpler] + [False]
    return close.sort_values(columns, ascending=ascending).iloc[0]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tail_until = last_decision(max(TAIL_HOURS))
    launch_until = last_decision(max(LAUNCH_HOURS))
    print(f"Tail decisions up to {tail_until:%d %b %H:%M}, launch decisions up to {launch_until:%d %b %H:%M} (exits end before 10 Sep)")

    tail_picks = tail_selections(sorted(st.SIGNAL_DIR.glob("signal_*.parquet")), tail_until)
    launch = pd.read_parquet(td.PROJECT_DIR / "launch_dev.parquet")
    launch_picks = launch_selections(launch, launch_until)
    print(f"Selections: {len(tail_picks)} tail rules, {len(launch_picks)} launch rules", flush=True)

    pools = set()
    for picks in list(tail_picks.values()) + list(launch_picks.values()):
        pools.update(picks["pool"].dropna().astype(str))
    prices = load_pool_prices(pools)
    print(f"Prices: {len(prices):,} swaps in {prices['pool'].nunique():,} pools", flush=True)
    pool_prices = bt.PoolPrices(prices)
    del prices

    for family, grid, picks in (("tail", tail_grid(), tail_picks), ("launch", launch_grid(), launch_picks)):
        path = OUT_DIR / f"trades_{family}.parquet"
        if path.exists():
            print(f"{path.name} exists, skipped", flush=True)
            continue
        trades = run_grid(grid, picks, pool_prices, family)
        trades.to_parquet(path)
        print(f"Written {path.name}: {len(trades):,} rows, {trades['setting'].nunique()} settings", flush=True)


if __name__ == "__main__":
    sys.exit(main())
