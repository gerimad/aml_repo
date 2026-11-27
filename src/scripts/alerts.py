"""Unusual-activity alerts for accounts (work package 14, implementation_plan_applications.md section 14).

Activity that fits a compromised key: the account suddenly picks up tokens far
outside its own pattern, sells what it holds, and sends the proceeds to an
address it never used before.

Decision blocks: the 2-day target window after each evaluation point t, cut
into 6-hour blocks (the windows of consecutive points join up, so a period is
covered without gaps). Per (account, block), for accounts with at least 5
tokens before t:

  deviation    mean log(1 + rank) of the account's new (warm) tokens in the block
               (WP9 ranks, model at t) minus the median log(1 + rank) of its
               ranked pick-ups at earlier points; needs at least 3 of those.
               (The raw rank, as in WP11: WP9 found percentiles compressed.)
  liquidation  share of the account's 14-day-window tokens it sold in the block
  exit         base tokens (WETH, USDC, USDT) sent by the account, from its
               first sale in the block to one hour after its last, to an address
               it never sent base tokens to before and that is not a DEX pool
  silence      no trade at all in the 7 days after the block (only to describe
               alerts after the fact, never an input)

Accounts without a trade in the 14 days before t are not ranked by WP9, so a
dormant account that suddenly trades (a typical compromise) is not covered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import protocol as pc

BLOCK = pd.Timedelta(hours=6)
MIN_TOKENS = 5
MIN_EARLIER = 3
EXIT_WINDOW = pd.Timedelta(hours=1)
SILENCE = pd.Timedelta(days=7)


def blocks_of_point(t: pd.Timestamp) -> list[pd.Timestamp]:
    """Starts of the 6-hour blocks of the point's 2-day target window."""
    return [t + i * BLOCK for i in range(int(pc.HORIZON / BLOCK))]


def deviation(pairs_at_point: pd.DataFrame, earlier_pairs: pd.DataFrame) -> pd.DataFrame:
    """Per (user, block): the deviation of the docstring and the number of ranked pick-ups it uses.

    pairs_at_point: WP9 pairs at one point (warm pairs carry a rank); earlier_pairs: WP9 pairs of earlier points.
    """
    warm = pairs_at_point[~pairs_at_point["cold"]].copy()
    warm["block"] = warm["point"] + ((warm["first_trade_time"] - warm["point"]) // BLOCK) * BLOCK
    warm["log_rank"] = np.log1p(warm["rank"])
    earlier = earlier_pairs[~earlier_pairs["cold"]]
    usual = np.log1p(earlier["rank"]).groupby(earlier["user"]).agg(["median", "size"])
    usual = usual[usual["size"] >= MIN_EARLIER]["median"]
    now = warm.groupby(["user", "block"])["log_rank"].agg(["mean", "size"]).reset_index()
    now = now[now["user"].isin(usual.index)]
    now["deviation"] = now["mean"] - now["user"].map(usual)
    return now.rename(columns={"size": "ranked_pickups"})[["user", "block", "deviation", "ranked_pickups"]]


def liquidation(legs: pd.DataFrame, t: pd.Timestamp, users) -> pd.DataFrame:
    """Per (user, block) with a sale: the share of its 14-day-window tokens sold, and the first and last sale times."""
    window = legs[(legs["timestamp"] < t) & (legs["timestamp"] >= t - pd.Timedelta(days=pc.REFERENCE_DAYS)) & legs["user"].isin(users)]
    held = window.groupby("user")["token"].nunique()
    sales = legs[(legs["timestamp"] >= t) & (legs["timestamp"] < t + pc.HORIZON) & (legs["side"] == "sell") & legs["user"].isin(held.index)]
    sales = sales.assign(block=t + ((sales["timestamp"] - t) // BLOCK) * BLOCK)
    sales = sales[sales["token"].isin(set(window["token"]))]
    known = sales.merge(window[["user", "token"]].drop_duplicates(), on=["user", "token"])  # only tokens of its window
    out = known.groupby(["user", "block"]).agg(tokens_sold=("token", "nunique"), first_sale=("timestamp", "min"), last_sale=("timestamp", "max")).reset_index()
    out["liquidation"] = out["tokens_sold"] / out["user"].map(held)
    return out


def exits(sales: pd.DataFrame, transfers: pd.DataFrame, addresses: pd.Series, pools: set[str]) -> pd.Series:
    """For each row of `sales` (user, first_sale, last_sale): base-token amount sent to new, non-pool addresses
    between the first sale and one hour after the last. transfers: sender, recipient, timestamp, amount_usd;
    addresses: user id -> address."""
    out = np.zeros(len(sales))
    sender = sales["user"].map(addresses).to_numpy()
    by_sender = {s: part.sort_values("timestamp") for s, part in transfers[transfers["sender"].isin(set(sender))].groupby("sender")}
    for i, (who, start, end) in enumerate(zip(sender, sales["first_sale"], sales["last_sale"] + EXIT_WINDOW)):
        sent = by_sender.get(who)
        if sent is None:
            continue
        before = set(sent.loc[sent["timestamp"] < start, "recipient"])
        during = sent[(sent["timestamp"] >= start) & (sent["timestamp"] <= end)]
        new = during[~during["recipient"].isin(before) & ~during["recipient"].isin(pools)]
        out[i] = new["amount_usd"].sum()
    return pd.Series(out, index=sales.index)


def silent(legs_users_times: dict, users, block_ends) -> np.ndarray:
    """True where the user has no trade in the 7 days after the block's end. legs_users_times: user -> sorted times (ns)."""
    out = np.zeros(len(users), dtype=bool)
    for i, (u, end) in enumerate(zip(users, block_ends)):
        times = legs_users_times.get(u, np.array([], dtype="datetime64[ns]"))
        e = end.tz_convert("UTC").tz_localize(None).to_datetime64().astype("datetime64[ns]")
        out[i] = np.searchsorted(times, e + np.timedelta64(7, "D")) == np.searchsorted(times, e)
    return out


def fire(table: pd.DataFrame, deviation_min: float, liquidation_min: float, exit_min: float = 0.0) -> pd.Series:
    """Alert flags: deviation, liquidation and exit all above their thresholds."""
    return (table["deviation"] >= deviation_min) & (table["liquidation"] >= liquidation_min) & (table["exit_usd"] > exit_min)
