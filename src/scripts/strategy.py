"""The demand signal of the trading strategy (work package 8, implementation plan section 12).

At a decision time s, the frozen fixed blend makes a list of 10 tokens for
every active account. Counting, for each token outside the popular head (a
"tail" token), the lists that contain it gives the predicted new demand.
The signal table of one decision time has one row per tail token with:

    predicted_<group>     lists containing the token, by the history size of
                          the account (1, 2-4, 5+ tokens), so that the hit
                          rate of each group can weight them later
    predicted_eth_<group> the same, each list weighted by the account's buy
                          volume in ETH over the 14 days before s (capped at
                          the 99th percentile of active accounts)
    buyers_24h            accounts that bought the token in the 24 hours before s
    sellers_24h           accounts that sold it in the 24 hours before s
    buy_eth_24h           its buy volume in ETH in those 24 hours
    age_hours             hours since its first trade
    pool                  its most active base pool in those 24 hours (by ETH volume)

Everything is computed from data before s. The selection rules
(versions of the signal) work on these tables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import pool_recsys as pr
import protocol as pc

GROUPS = ["1", "2-4", "5+"]
SIGNAL_DIR = pc.RESULTS_DIR.parent / "data" / "project" / "signal"


def decision_times(start, end, hours: int = 6) -> list[pd.Timestamp]:
    """start, start + hours, ... strictly before end."""
    return list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=f"{hours}h", inclusive="left"))


def active_rows(task: pc.Task) -> np.ndarray:
    """Accounts with a token in the 14 days before s and at least one trade in the last 7 days."""
    traded_last_week = np.asarray(task.history(7).sum(axis=1)).ravel() > 0
    return np.flatnonzero((task.window_size >= 1) & traded_last_week)


def account_buy_volume(prices: pd.DataFrame, users: pd.DataFrame, s: pd.Timestamp, days: int = 14) -> pd.Series:
    """Buy volume in ETH per account id over the `days` before s, from the priced swaps (signing account = tx_from)."""
    window = prices[(prices["timestamp"] >= s - pd.Timedelta(days=days)) & (prices["timestamp"] < s) & (prices["side"] == "buy")]
    by_address = window.groupby("tx_from", observed=True)["volume_eth"].sum()
    user_id = users.set_index("address")["user_id"]
    by_user = by_address.groupby(by_address.index.astype(str).map(user_id)).sum()
    return by_user[by_user.index.notna()]


def recent_activity(legs: pd.DataFrame, prices: pd.DataFrame, token_address: pd.Series, s: pd.Timestamp) -> pd.DataFrame:
    """Per token id, over the 24 hours before s: buyers, sellers, buy volume in ETH and the most active base pool."""
    day = legs[(legs["timestamp"] >= s - pd.Timedelta(hours=24)) & (legs["timestamp"] < s)]
    table = pd.DataFrame({
        "buyers_24h": day[day["side"] == "buy"].groupby("token", observed=True)["user"].nunique(),
        "sellers_24h": day[day["side"] == "sell"].groupby("token", observed=True)["user"].nunique(),
    }).fillna(0)  # fmt: skip
    priced = prices[(prices["timestamp"] >= s - pd.Timedelta(hours=24)) & (prices["timestamp"] < s)]
    token_id = pd.Series(token_address.index, index=token_address.to_numpy())
    priced = priced.assign(token_id=priced["token"].astype(str).map(token_id).astype(float))
    volume = priced[priced["side"] == "buy"].groupby("token_id", observed=True)["volume_eth"].sum().rename("buy_eth_24h")
    by_pool = priced.groupby(["token_id", "pool"], observed=True)["volume_eth"].sum().reset_index()
    best_pool = by_pool.sort_values("volume_eth").drop_duplicates("token_id", keep="last").set_index("token_id")["pool"]
    return table.join(volume, how="outer").join(best_pool.rename("pool"), how="outer").fillna({"buyers_24h": 0, "sellers_24h": 0, "buy_eth_24h": 0.0})


def signal_table(
    legs: pd.DataFrame, prices: pd.DataFrame, tokens: pd.DataFrame, users: pd.DataFrame, s: pd.Timestamp, blend_settings: dict
) -> pd.DataFrame:
    """The signal of one decision time (module docstring), one row per tail token on at least one list or with recent buyers."""
    task = pc.make_task(legs, s)
    window = task.history(pc.REFERENCE_DAYS)
    popularity = pr.Popularity().fit(window)
    neighbours = pr.ItemKNN(blend_settings["k"], blend_settings["shrink"]).fit(window)
    blend = pr.Blend(popularity, neighbours, blend_settings["weight"])

    rows = active_rows(task)
    groups = pc.history_group(task.window_size[rows])
    volume = account_buy_volume(prices, users, s).reindex(task.users[rows]).fillna(0.0).to_numpy()
    volume = np.minimum(volume, np.quantile(volume, 0.99)) if len(volume) else volume

    n_tokens = len(task.tokens)
    counts = {g: np.zeros(n_tokens) for g in GROUPS}
    eth = {g: np.zeros(n_tokens) for g in GROUPS}
    batch = max(64, 25_000_000 // n_tokens)
    for start in range(0, len(rows), batch):
        part = rows[start : start + batch]
        top = pr.top_k(blend.scores(window[part]), pc.K, exclude=task.past[part])
        part_groups, part_volume = groups[start : start + batch], volume[start : start + batch]
        for g in GROUPS:
            in_group = part_groups == g
            np.add.at(counts[g], top[in_group].ravel(), 1)
            np.add.at(eth[g], top[in_group].ravel(), np.repeat(part_volume[in_group], pc.K))

    head = np.zeros(n_tokens, dtype=bool)
    head[pc.head_tokens(window)] = True
    table = pd.DataFrame({"token": task.tokens, "in_head": head})
    for g in GROUPS:
        table[f"predicted_{g}"] = counts[g]
        table[f"predicted_eth_{g}"] = eth[g]
    activity = recent_activity(legs, prices, tokens.set_index("token_id")["address"], s)
    table = table.join(activity, on="token")
    table[["buyers_24h", "sellers_24h", "buy_eth_24h"]] = table[["buyers_24h", "sellers_24h", "buy_eth_24h"]].fillna(0)
    first = tokens.set_index("token_id")["first_timestamp"]
    table["age_hours"] = (s - table["token"].map(first)).dt.total_seconds() / 3600
    predicted_any = table[[f"predicted_{g}" for g in GROUPS]].sum(axis=1) > 0
    table = table[~table["in_head"] & (predicted_any | (table["buyers_24h"] > 0))].drop(columns="in_head")
    table.insert(0, "decision", s)
    table.attrs["active_accounts"] = int(len(rows))
    return table.reset_index(drop=True)


def load_or_build_signal(legs, prices, tokens, users, s, blend_settings, out_dir: Path = SIGNAL_DIR) -> pd.DataFrame:
    """The signal of one decision time, read from out_dir if already built (one Parquet file per decision)."""
    path = out_dir / f"signal_{s:%Y%m%d_%H}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    table = signal_table(legs, prices, tokens, users, s, blend_settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path)
    return table


# --------------------------------------------------------------------------
# Selection rules
# --------------------------------------------------------------------------


def universe(signal: pd.DataFrame, min_sellers: int = 20, min_age_hours: float = 24) -> pd.DataFrame:
    """Tradable tail tokens: a base pool, at least a day old, sold by at least `min_sellers` accounts in the last 24 hours."""
    return signal[signal["pool"].notna() & (signal["age_hours"] >= min_age_hours) & (signal["sellers_24h"] >= min_sellers)]


def predicted(signal: pd.DataFrame, weighting: str, hit_rates: dict[str, float] | None = None) -> pd.Series:
    """Predicted new demand: lists counted ("accounts") or weighted by buy volume ("eth"), each history group times its hit rate."""
    rates = hit_rates or {g: 1.0 for g in GROUPS}
    prefix = "predicted_" if weighting == "accounts" else "predicted_eth_"
    return sum(rates[g] * signal[prefix + g] for g in GROUPS)


def select(signal: pd.DataFrame, version: str, k: int, weighting: str = "accounts", hit_rates=None) -> pd.DataFrame:
    """The k tokens a selection rule picks from the universe of one decision time.

    Versions:
      surprise          predicted / (1 + visible), visible in the same unit (accounts or ETH)
      filtered_visible  of the 3k tokens with the most visible demand, the k with the highest predicted demand
      unexplained_flow  visible / (1 + predicted): demand the recommender did not expect (a catalyst, or a push)
      predicted         predicted demand alone (reported, not a candidate for the strategy)
      tail_popularity   the k tokens with the most buyers in the last 24 hours (the primary benchmark)
      tail_flow         the k tokens with the largest buy volume in ETH in the last 24 hours
      random            k tokens at random (seeded by the decision time)
    """
    pool = universe(signal)
    if pool.empty:
        return pool
    demand = predicted(pool, weighting, hit_rates)
    visible = pool["buyers_24h"] if weighting == "accounts" else pool["buy_eth_24h"]
    if version == "surprise":
        score = demand / (1 + visible)
    elif version == "filtered_visible":
        shortlist = visible.nlargest(3 * k).index
        score = demand.where(demand.index.isin(shortlist), -np.inf)
    elif version == "unexplained_flow":
        score = visible / (1 + demand)
    elif version == "predicted":
        score = demand
    elif version == "tail_popularity":
        score = pool["buyers_24h"].astype(float)
    elif version == "tail_flow":
        score = pool["buy_eth_24h"].astype(float)
    elif version == "random":
        seed = int(pd.Timestamp(pool["decision"].iloc[0]).timestamp())
        score = pd.Series(np.random.default_rng(seed).random(len(pool)), index=pool.index)
    else:
        raise ValueError(f"unknown version {version!r}")
    # Ties broken by visible demand, then by token id, so that the choice is reproducible.
    order = pd.DataFrame({"score": score, "visible": visible, "token": pool["token"]}).sort_values(
        ["score", "visible", "token"], ascending=[False, False, True]
    )
    chosen = order[np.isfinite(order["score"])].head(k).index
    return pool.loc[chosen]


# --------------------------------------------------------------------------
# Launch momentum (implementation plan 12.8): order flow of new tokens, no recommender
# --------------------------------------------------------------------------


def launch_times(prices: pd.DataFrame) -> pd.Series:
    """Each token's launch: its first priced swap. Computed once over all prices, so that launch_table need not.

    Assumption: a token's first priced swap in the data is its launch. For
    tokens traded before the data starts (18 June) this is too late, but they
    are never young at a decision time after 25 June, so the launch universe
    (at most 6 hours old) never contains them.
    """
    return prices.groupby("token", observed=True)["timestamp"].min()


def launch_table(
    prices: pd.DataFrame, s: pd.Timestamp, min_age_hours: float = 0.25, max_age_hours: float = 6, launch: pd.Series | None = None
) -> pd.DataFrame:
    """Tokens between min_age_hours and max_age_hours old at s, with their order flow, from priced swaps before s.

    A token's launch is its first priced swap (against a base currency).
    Columns: decision, token (address), pool (its most active base pool since
    launch, by ETH volume), age_hours, buyers_since_launch, buyers_1h,
    buy_share_1h (share of the last hour's swaps that are buys), buy_eth_since_launch.
    """
    since = prices[(prices["timestamp"] >= s - pd.Timedelta(hours=max_age_hours + 1)) & (prices["timestamp"] < s)]
    if launch is None:
        launch = launch_times(prices[prices["timestamp"] < s])
    launch = launch[launch < s]
    age = (s - launch).dt.total_seconds() / 3600
    young = age[(age >= min_age_hours) & (age <= max_age_hours)].index
    flow = since[since["token"].isin(young)]
    flow = flow[flow["timestamp"] >= flow["token"].map(launch)]
    # Plain strings from here on: the flow of young tokens is small, and pandas fails to align category-keyed
    # results when one of them is empty (an hour in which no young token traded).
    flow = flow.assign(token=flow["token"].astype(str), pool=flow["pool"].astype(str))
    age.index = age.index.astype(str)
    last_hour = flow[flow["timestamp"] >= s - pd.Timedelta(hours=1)]
    by_pool = flow.groupby(["token", "pool"], observed=True)["volume_eth"].sum().reset_index()
    table = pd.DataFrame({
        "buyers_since_launch": flow[flow["side"] == "buy"].groupby("token", observed=True)["tx_from"].nunique(),
        "buyers_1h": last_hour[last_hour["side"] == "buy"].groupby("token", observed=True)["tx_from"].nunique(),
        "buy_share_1h": (last_hour["side"] == "buy").groupby(last_hour["token"], observed=True).mean(),
        "buy_eth_since_launch": flow[flow["side"] == "buy"].groupby("token", observed=True)["volume_eth"].sum(),
        "pool": by_pool.sort_values("volume_eth").drop_duplicates("token", keep="last").set_index("token")["pool"],
    }).fillna({"buyers_since_launch": 0, "buyers_1h": 0, "buy_share_1h": 0.0, "buy_eth_since_launch": 0.0})  # fmt: skip
    table["age_hours"] = age.reindex(table.index)
    table.index.name = "token"
    table = table.reset_index()
    table.insert(0, "decision", s)
    return table


def select_launch(table: pd.DataFrame, version: str, k: int, min_buyers: int, min_buy_share: float, already: set) -> pd.DataFrame:
    """The k tokens a launch rule picks at one decision; tokens in `already` (bought before) are skipped.

    launch_momentum   the most buyers in the last hour, among tokens with at least min_buyers since launch
                      and a buy share of at least min_buy_share in the last hour
    launch_random     k tokens at random from the same universe (the benchmark), seeded by the decision time
    """
    pool = table[(table["buyers_since_launch"] >= min_buyers) & table["pool"].notna() & ~table["token"].isin(already)]
    if pool.empty:
        return pool
    if version == "launch_momentum":
        eligible = pool[pool["buy_share_1h"] >= min_buy_share]
        return eligible.sort_values(["buyers_1h", "token"], ascending=[False, True]).head(k)
    if version == "launch_random":
        seed = int(pd.Timestamp(pool["decision"].iloc[0]).timestamp())
        chosen = np.random.default_rng(seed).choice(len(pool), size=min(k, len(pool)), replace=False)
        return pool.iloc[np.sort(chosen)]
    raise ValueError(f"unknown launch version {version!r}")
