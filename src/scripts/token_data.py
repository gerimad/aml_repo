"""Build the project's working data from the downloaded swaps (work package 1).

The project uses the 12 weeks of blocks 25,344,000 to 25,948,799 (18 June to
10 September 2026). This script turns their swaps into *trade legs*: one row
per (transaction, token, side), with the tokens that were only passed through
a route dropped and the base currencies removed. See the implementation plan,
section 3.3.

Usage, from the repository root, after the download has finished:

    uv run python src/scripts/token_data.py

It writes to data/project/:

    legs.parquet          one row per trade leg (users, tokens and pools as integer ids)
    users.parquet         user_id -> address
    tokens.parquet        token_id -> address, first trade, launch pool, launch base
    pools.parquet         pool_id -> address, token0, token1
    base_tokens.json      the removed base currencies and how they were chosen
    validation.csv        data-quality checks per week

Users, tokens and pools are stored as integer ids, not as 42-character
addresses, so that the legs of 12 weeks fit comfortably in memory. Every id
table maps back to the address.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import dex_swaps as ds
import pool_recsys as pr

PROJECT_DIR = Path(__file__).resolve().parents[2] / "data" / "project"
# The plan names blocks 25,344,000 to 25,948,799. Block 25,344,000 turned out to be
# at 18 June 10:58 UTC, not midnight, so 4,000 earlier blocks are added to cover
# the whole first day (recorded in project_direction.md, "Changes to the plan").
# Legs outside [DATA_START, DATA_END) are dropped, so the dates are exactly the plan's.
PROJECT_FIRST_BLOCK = 25_340_000
PROJECT_LAST_BLOCK = 25_948_799

# The project's calendar (implementation plan, section 1.1). All times UTC.
DATA_START = pd.Timestamp("2026-06-18", tz="UTC")
BURN_IN_END = pd.Timestamp("2026-07-02", tz="UTC")
DATA_END = pd.Timestamp("2026-09-10", tz="UTC")

BASE_TOKEN_THRESHOLD = 0.10  # share of users above which a token counts as a base currency

# Well-known addresses, used only to print readable names. Not used for any decision.
KNOWN_SYMBOLS = {
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "WETH",
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "USDC",
    "0xdac17f958d2ee523a2206206994597c13d831ec7": "USDT",
    "0x6b175474e89094c44da98b954eedeac495271d0f": "DAI",
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": "WBTC",
}


def symbol(address: str) -> str:
    """A readable name for a token address: its symbol if well known, else the shortened address."""
    return KNOWN_SYMBOLS.get(address, address[:6] + "…" + address[-4:])


# --------------------------------------------------------------------------
# Step 1: trade legs with time and pool
# --------------------------------------------------------------------------


def legs_with_time_and_pool(swaps: pd.DataFrame, pool_tokens: pd.DataFrame) -> pd.DataFrame:
    """Trade legs of a set of swaps, with the block time and the pool of each leg.

    Uses pool_recsys.trade_legs (one row per transaction, token and side,
    intermediate tokens dropped) and adds two columns:
      timestamp   the time of the block
      pool        the pool in which the token was bought or sold

    Assumption: when a transaction trades the same token in several pools
    (a split route), the pool of its first swap (lowest log_index) is kept.
    The pool is only used to find the launch pool of a token, where the first
    pool is the one that matters.
    """
    legs = pr.trade_legs(swaps, pool_tokens)

    block_time = swaps.drop_duplicates("block").set_index("block")["timestamp"]
    legs["timestamp"] = legs["block"].map(block_time)

    tokens = pool_tokens.dropna().set_index("pool")
    known = swaps[swaps["pool"].isin(tokens.index)].sort_values(["block", "tx_index", "log_index"])
    pool_of_token = pd.concat(
        [
            pd.DataFrame({"tx_hash": known["tx_hash"], "token": known["pool"].map(tokens["token0"]), "pool": known["pool"]}),
            pd.DataFrame({"tx_hash": known["tx_hash"], "token": known["pool"].map(tokens["token1"]), "pool": known["pool"]}),
        ],
        ignore_index=True,
    )
    # concat puts all token0 rows before all token1 rows; restore chain order before keeping the first
    pool_of_token["order"] = np.concatenate([np.arange(len(known)), np.arange(len(known))])
    pool_of_token = pool_of_token.sort_values("order", kind="stable").drop_duplicates(["tx_hash", "token"])
    legs = legs.merge(pool_of_token[["tx_hash", "token", "pool"]], on=["tx_hash", "token"], how="left")

    assert legs["timestamp"].notna().all(), "every leg must have a block time"
    assert legs["pool"].notna().all(), "every leg must have a pool"
    return legs


# --------------------------------------------------------------------------
# Step 2: base currencies
# --------------------------------------------------------------------------


def base_tokens(legs: pd.DataFrame, start, end, threshold: float = BASE_TOKEN_THRESHOLD) -> pd.DataFrame:
    """Tokens traded by more than `threshold` of the users active in [start, end).

    Returns one row per token of the period, most used first, with columns
    token, users, share and is_base. The caller removes the tokens with
    is_base == True. Only the period given is looked at, which for the
    project is the burn-in, so the list does not use any later data.
    """
    period = legs[(legs["timestamp"] >= start) & (legs["timestamp"] < end)]
    n_users = period["user"].nunique()
    users_per_token = period.groupby("token")["user"].nunique().sort_values(ascending=False)
    table = users_per_token.rename("users").reset_index()
    table["share"] = table["users"] / n_users
    table["is_base"] = table["share"] > threshold
    return table


# --------------------------------------------------------------------------
# Step 3: integer ids and the token table
# --------------------------------------------------------------------------


class IdMap:
    """Gives each distinct string (an address) a stable integer id, in order of first appearance."""

    def __init__(self):
        self.ids: dict[str, int] = {}

    def encode(self, values: pd.Series) -> np.ndarray:
        """Integer ids of the values, adding the values not seen before."""
        for value in pd.unique(values):
            if value not in self.ids:
                self.ids[value] = len(self.ids)
        return values.map(self.ids).to_numpy(np.int32)

    def table(self, id_name: str) -> pd.DataFrame:
        """The mapping as a frame with columns id_name and address."""
        return pd.DataFrame({id_name: np.arange(len(self.ids), dtype=np.int32), "address": list(self.ids)})

    @classmethod
    def from_table(cls, table: pd.DataFrame, id_name: str) -> IdMap:
        """An IdMap that continues an existing table: known addresses keep their ids, new ones get the next."""
        ordered = table.sort_values(id_name)
        assert (ordered[id_name].to_numpy() == np.arange(len(ordered))).all(), "ids must be 0, 1, 2, ... without gaps"
        id_map = cls()
        id_map.ids = dict(zip(ordered["address"], ordered[id_name].astype(int)))
        return id_map


def token_table(legs: pd.DataFrame, pools: pd.DataFrame, data_start=DATA_START) -> pd.DataFrame:
    """One row per token: when it was first traded and in which pool.

    `legs` must have integer columns token and pool and a timestamp, and be
    sorted by time. `pools` has pool_id, token0_id and token1_id. Columns:
      token_id, first_block, first_timestamp
      launch_pool_id   the pool of the token's first leg
      launch_base_id   the other token of that pool (often a base currency)
      censored         first traded in the first 24 hours of the data, so the
                       token probably existed before and its age is unknown
    """
    assert legs["timestamp"].is_monotonic_increasing, "legs must be sorted by time"
    first = legs.drop_duplicates("token", keep="first")
    table = pd.DataFrame({
        "token_id": first["token"].to_numpy(),
        "first_block": first["block"].to_numpy(),
        "first_timestamp": first["timestamp"].to_numpy(),
        "launch_pool_id": first["pool"].to_numpy(),
    })  # fmt: skip
    pool_index = pools.set_index("pool_id")
    token0 = table["launch_pool_id"].map(pool_index["token0_id"])
    token1 = table["launch_pool_id"].map(pool_index["token1_id"])
    table["launch_base_id"] = np.where(token0 == table["token_id"], token1, token0).astype(np.int32)
    table["censored"] = table["first_timestamp"] < data_start + pd.Timedelta(hours=24)
    return table.sort_values("token_id", ignore_index=True)


# --------------------------------------------------------------------------
# The whole pipeline
# --------------------------------------------------------------------------


def build_project_data(out_dir: Path = PROJECT_DIR, chunks_per_batch: int = 50) -> None:
    """Run work package 1 from the downloaded chunks to the files in data/project/.

    Steps, each printed with its counts:
      1. Load the project's chunks in batches of `chunks_per_batch` (about a
         week of blocks), validate them, turn them into trade legs with time
         and pool, and replace addresses by integer ids. Transactions never
         span two chunks (a transaction is in one block), so batching does
         not change any leg.
      2. Decide the base currencies on the burn-in period.
      3. Remove them and write the legs, the id tables and the token table.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    files = ds.chunk_files(ds.DATA_DIR, PROJECT_FIRST_BLOCK, PROJECT_LAST_BLOCK)
    print(f"Project blocks {PROJECT_FIRST_BLOCK:,} to {PROJECT_LAST_BLOCK:,}: {len(files):,} chunk files, no gaps.")

    # Step 1
    users, tokens, pools = IdMap(), IdMap(), IdMap()
    leg_batches, validations = [], []
    for start in range(0, len(files), chunks_per_batch):
        batch = files[start : start + chunks_per_batch]
        swaps = pd.concat([pd.read_parquet(f) for f in batch], ignore_index=True)
        swaps = swaps.sort_values(["block", "tx_index", "log_index"], ignore_index=True)
        pool_tokens = ds.load_pool_tokens(swaps["pool"].unique())
        validations.append(ds.validate_swaps(swaps, pool_tokens))
        legs = legs_with_time_and_pool(swaps, pool_tokens)
        leg_batches.append(pd.DataFrame({
            "block": legs["block"].to_numpy(np.int64),
            "timestamp": legs["timestamp"].to_numpy(),
            "user": users.encode(legs["user"]),
            "token": tokens.encode(legs["token"]),
            "side": pd.Categorical(legs["side"], categories=["buy", "sell"]),
            "pool": pools.encode(legs["pool"]),
        }))  # fmt: skip
        print(
            f"  batch {start // chunks_per_batch + 1:>2}: {len(swaps):>9,} swaps -> {len(legs):>9,} legs "
            f"({swaps['timestamp'].min():%d %b %H:%M} to {swaps['timestamp'].max():%d %b %H:%M})"
        )
        del swaps, legs

    validation = merge_week_rows(pd.concat(validations, ignore_index=True))
    validation.to_csv(out_dir / "validation.csv", index=False)
    print(f"Validation: {int(validation['passed'].sum())} of {len(validation)} weeks passed.")

    legs = pd.concat(leg_batches, ignore_index=True)
    del leg_batches
    legs = legs.sort_values(["block", "user", "token"], kind="stable", ignore_index=True)

    # The blocks must cover the plan's dates; then legs outside them are dropped.
    first_time, last_time = legs["timestamp"].min(), legs["timestamp"].max()
    print(f"The blocks cover {first_time:%d %b %H:%M} to {last_time:%d %b %H:%M} (UTC).")
    assert first_time <= DATA_START + pd.Timedelta(minutes=5), "the data must start by 18 June 00:00"
    assert last_time >= DATA_END - pd.Timedelta(minutes=5), "the data must reach 10 September 00:00"
    inside = (legs["timestamp"] >= DATA_START) & (legs["timestamp"] < DATA_END)
    print(f"Keeping the legs from {DATA_START:%d %b} to {DATA_END:%d %b} 00:00: {len(legs):,} -> {int(inside.sum()):,}.")
    legs = legs[inside].reset_index(drop=True)
    print(f"All legs: {len(legs):,} rows, {legs['user'].nunique():,} users, {legs['token'].nunique():,} tokens.")

    # Step 2
    address_of_token = tokens.table("token_id").set_index("token_id")["address"]
    base = base_tokens(legs, DATA_START, BURN_IN_END)
    base["address"] = base["token"].map(address_of_token)
    removed = base[base["is_base"]]
    print("Most used tokens of the burn-in (share of its users):")
    for row in base.head(5).itertuples():
        print(f"  {symbol(row.address):>14}  {row.share:6.1%}  {'removed' if row.is_base else 'kept'}")

    def describe(rows):
        return [{"address": a, "symbol": symbol(a), "share": round(float(s), 4)} for a, s in zip(rows["address"], rows["share"])]

    with open(out_dir / "base_tokens.json", "w") as f:
        json.dump(
            {
                "rule": f"tokens traded by more than {BASE_TOKEN_THRESHOLD:.0%} of the users of the burn-in",
                "period": [str(DATA_START), str(BURN_IN_END)],
                "removed": describe(removed),
                "top_10": describe(base.head(10)),
            },
            f,
            indent=2,
        )

    # Step 3
    before = len(legs)
    legs = legs[~legs["token"].isin(set(removed["token"]))].reset_index(drop=True)
    print(f"Removing base currencies: {before:,} -> {len(legs):,} legs.")

    # Pool table with both tokens of each pool as token ids. A pool's other
    # token may never appear in a leg (for example a base currency of an
    # untraded side); it still gets an id so that a launch base can be named.
    pool_table = pools.table("pool_id")
    pool_tokens = ds.load_pool_tokens(pool_table["address"].tolist()).set_index("pool")
    pool_table["token0_id"] = tokens.encode(pool_table["address"].map(pool_tokens["token0"]))
    pool_table["token1_id"] = tokens.encode(pool_table["address"].map(pool_tokens["token1"]))

    token_ids = tokens.table("token_id")
    token_ids["symbol"] = token_ids["address"].map(symbol)
    token_ids["is_base"] = token_ids["token_id"].isin(set(removed["token"]))
    traded = token_table(legs, pool_table)
    token_ids = token_ids.merge(traded, on="token_id", how="left")

    legs.to_parquet(out_dir / "legs.parquet")
    users.table("user_id").to_parquet(out_dir / "users.parquet")
    token_ids.to_parquet(out_dir / "tokens.parquet")
    pool_table.to_parquet(out_dir / "pools.parquet")
    print(
        f"Written to {out_dir}: {len(legs):,} legs, {legs['user'].nunique():,} users, "
        f"{len(traded):,} traded tokens, {len(pool_table):,} pools."
    )


def merge_week_rows(validation: pd.DataFrame) -> pd.DataFrame:
    """Combine validation rows of the same week that came from two neighbouring batches.

    A batch boundary can fall inside a week; its two halves are then checked
    separately. This keeps one row per week: counts are added, shares are
    weighted by swaps, minima and maxima are taken over both halves, and the
    week passes only if both halves pass. Distinct users cannot be added, so
    the weekly user count is left out; the daily counts remain. A day cut by
    the boundary is partial in both halves and therefore not a "full day".
    """
    rows = []
    for week, part in validation.groupby("week", sort=True):
        weights = part["swaps"] / part["swaps"].sum()
        rows.append({
            "week": week,
            "days": part["days"].sum() - (len(part) - 1),  # a split day is counted in both halves
            "swaps": part["swaps"].sum(),
            "blocks_with_swap": (part["blocks_with_swap"] * weights).sum(),
            "longest_gap_blocks": part["longest_gap_blocks"].max(),
            "duplicate_events": part["duplicate_events"].sum(),
            "missing_values": part["missing_values"].sum(),
            "swaps_with_known_tokens": (part["swaps_with_known_tokens"] * weights).sum(),
            "swaps_per_day_min": part["swaps_per_day_min"].min(),
            "swaps_per_day_max": part["swaps_per_day_max"].max(),
            "users_per_day_min": part["users_per_day_min"].min(),
            "users_per_day_max": part["users_per_day_max"].max(),
            "low_days": part["low_days"].sum(),
            "passed": part["passed"].all(),
        })  # fmt: skip
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The extension: weeks after the project (work package 8)
# --------------------------------------------------------------------------

EXTENSION_DIR = PROJECT_DIR / "extension"


def build_extension(
    first_block: int, last_block: int, start, end, out_dir: Path = EXTENSION_DIR, chunks_per_batch: int = 50
) -> None:
    """Trade legs for weeks after the project, with the project's ids and base currencies.

    The same steps as build_project_data, except that:
      - the id maps continue the project's tables, so an account or token
        known in the project keeps its id and new ones get the next ids;
      - the base currencies are the project's (decided on the burn-in), not
        decided again;
      - only legs with start <= timestamp < end are kept, so the extension
        joins the project's legs (which end at 10 September 00:00) without
        overlap. first_block may lie before `start` to cover it.
    Writes legs.parquet (the extension's legs only), users, tokens and pools
    (the full, extended tables) and validation.csv to out_dir.
    """
    # Times without a time zone are taken as UTC, like every time in the project.
    start, end = (pd.Timestamp(t) if pd.Timestamp(t).tzinfo else pd.Timestamp(t, tz="UTC") for t in (start, end))
    out_dir.mkdir(parents=True, exist_ok=True)
    users = IdMap.from_table(pd.read_parquet(PROJECT_DIR / "users.parquet"), "user_id")
    project_tokens = load_tokens()
    tokens = IdMap.from_table(project_tokens[["token_id", "address"]], "token_id")
    pools = IdMap.from_table(pd.read_parquet(PROJECT_DIR / "pools.parquet")[["pool_id", "address"]], "pool_id")
    removed = set(project_tokens.loc[project_tokens["is_base"], "token_id"])

    files = ds.chunk_files(ds.DATA_DIR, first_block, last_block)
    print(f"Extension blocks {first_block:,} to {last_block:,}: {len(files):,} chunk files, no gaps.")
    leg_batches, validations = [], []
    for i in range(0, len(files), chunks_per_batch):
        swaps = pd.concat([pd.read_parquet(f) for f in files[i : i + chunks_per_batch]], ignore_index=True)
        swaps = swaps.sort_values(["block", "tx_index", "log_index"], ignore_index=True)
        pool_tokens = ds.load_pool_tokens(swaps["pool"].unique())
        validations.append(ds.validate_swaps(swaps, pool_tokens))
        legs = legs_with_time_and_pool(swaps, pool_tokens)
        legs = legs[(legs["timestamp"] >= start) & (legs["timestamp"] < end)]
        leg_batches.append(pd.DataFrame({
            "block": legs["block"].to_numpy(np.int64),
            "timestamp": legs["timestamp"].to_numpy(),
            "user": users.encode(legs["user"]),
            "token": tokens.encode(legs["token"]),
            "side": pd.Categorical(legs["side"], categories=["buy", "sell"]),
            "pool": pools.encode(legs["pool"]),
        }))  # fmt: skip
        print(f"  batch {i // chunks_per_batch + 1}: {len(swaps):,} swaps -> {len(legs):,} legs", flush=True)

    merge_week_rows(pd.concat(validations, ignore_index=True)).to_csv(out_dir / "validation.csv", index=False)
    legs = pd.concat(leg_batches, ignore_index=True).sort_values(["block", "user", "token"], kind="stable", ignore_index=True)
    first_time, last_time = legs["timestamp"].min(), legs["timestamp"].max()
    assert first_time <= start + pd.Timedelta(minutes=5) and last_time >= end - pd.Timedelta(minutes=5), (
        f"the blocks cover {first_time} to {last_time}, not {start} to {end}"
    )
    legs = legs[~legs["token"].isin(removed)].reset_index(drop=True)

    pool_table = pools.table("pool_id")
    pool_tokens = ds.load_pool_tokens(pool_table["address"].tolist()).set_index("pool")
    pool_table["token0_id"] = tokens.encode(pool_table["address"].map(pool_tokens["token0"]))
    pool_table["token1_id"] = tokens.encode(pool_table["address"].map(pool_tokens["token1"]))

    # Tokens first traded in the extension get their first trade from it; project tokens keep theirs.
    token_ids = tokens.table("token_id")
    token_ids["symbol"] = token_ids["address"].map(symbol)
    token_ids["is_base"] = token_ids["token_id"].isin(removed)
    new_legs = legs[~legs["token"].isin(set(project_tokens["token_id"]))].sort_values("timestamp", kind="stable")
    new_traded = token_table(new_legs, pool_table, data_start=DATA_START) if len(new_legs) else pd.DataFrame()
    old_traded = project_tokens.drop(columns=["address", "symbol", "is_base"])
    token_ids = token_ids.merge(pd.concat([old_traded, new_traded], ignore_index=True), on="token_id", how="left")

    legs.to_parquet(out_dir / "legs.parquet")
    users.table("user_id").to_parquet(out_dir / "users.parquet")
    token_ids.to_parquet(out_dir / "tokens.parquet")
    pool_table.to_parquet(out_dir / "pools.parquet")
    print(f"Written to {out_dir}: {len(legs):,} legs, {legs['user'].nunique():,} users, {len(new_traded):,} new tokens.")


def load_all_legs() -> pd.DataFrame:
    """The project's legs followed by the extension's, one timeline with one set of ids."""
    return pd.concat([load_legs(), load_legs(EXTENSION_DIR)], ignore_index=True)


def load_all_tokens() -> pd.DataFrame:
    """The extended token table (a superset of the project's, with the same ids)."""
    return load_tokens(EXTENSION_DIR)


def load_legs(data_dir: Path = PROJECT_DIR) -> pd.DataFrame:
    """The project's trade legs, sorted by block: block, timestamp, user, token, side, pool (ids)."""
    path = data_dir / "legs.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run token_data.py first")
    return pd.read_parquet(path)


def load_tokens(data_dir: Path = PROJECT_DIR) -> pd.DataFrame:
    """The token table (one row per token id), see token_table."""
    return pd.read_parquet(data_dir / "tokens.parquet")


if __name__ == "__main__":
    build_project_data()
