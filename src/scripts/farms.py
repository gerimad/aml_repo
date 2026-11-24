"""Farm detection from behaviour (work package 10, implementation_plan_applications.md section 10).

A farm is a group of accounts that act as one: they pick up the same tokens
at nearly the same moments. This module finds such groups from the trade
legs alone and checks them on later data.

  first trades     each account's first trade of each token (its first ever
                   in the data), kept when it falls in a window
  co-action graph  two accounts are linked when at least m of the tokens they
                   first-traded in the window were first-traded by both within
                   delta minutes of each other; the weight is the number of
                   such tokens
  groups           connected components of the graph; a component larger than
                   max_component accounts is split into Louvain communities, so
                   that one weak link does not merge two farms
  co-action rate   for a group, on a later window: of the pairs of members that
                   both first-trade some token there, the share that first-trade
                   a common token within delta minutes
  matched pairs    the same rate for random accounts that look like the members
                   (same number of first trades and same first day in the build
                   window, and also active in the later window), the comparison
                   the success criterion of section 10.2 uses

Assumption: the account of a leg is the transaction's signer. The legs do not
carry the contract called (`tx_to`), so the plan's group feature "distinct
signing contracts" is not computed (recorded in project_direction.md).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph

import protocol as pc

DELTAS_MINUTES = [5, 30, 120]
MIN_SHARED = [2, 3, 5]
MAX_COMPONENT = 500


def first_trades(legs: pd.DataFrame, start, end) -> pd.DataFrame:
    """Each (user, token) pair whose first trade in the data falls in [start, end), with its time.

    The first trade is looked up over all legs before `end`, so a token an
    account already traded before `start` is not a first trade in the window.
    Columns: user, token, timestamp, sorted by token and time.
    """
    earlier = legs[legs["timestamp"] < end]
    first = earlier.groupby(["user", "token"], observed=True)["timestamp"].min().reset_index()
    first = first[first["timestamp"] >= start]
    return first.sort_values(["token", "timestamp", "user"], ignore_index=True)


def _seconds(times: pd.Series) -> np.ndarray:
    """Seconds since 2026-01-01 (float), for time differences."""
    return (times - pd.Timestamp("2026-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def _pair_key(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """One int64 per unordered pair of user ids (smaller id in the high bits)."""
    low, high = np.minimum(a, b).astype(np.int64), np.maximum(a, b).astype(np.int64)
    return (low << 32) | high


def _key_users(key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two user ids of each pair key."""
    return (key >> 32).astype(np.int64), (key & 0xFFFFFFFF).astype(np.int64)


def close_pairs(first: pd.DataFrame, delta_minutes: float) -> np.ndarray:
    """Pair keys, one per (token, pair of accounts) whose first trades of the token are at most delta apart.

    A sorted sweep: within each token the first trades are in time order, and
    each one is paired with the following ones up to delta later. All pairs of
    accounts are never formed.
    """
    if len(first) < 2:
        return np.empty(0, dtype=np.int64)
    first = first.sort_values(["token", "timestamp"], kind="stable")
    token_rank = np.unique(first["token"].to_numpy(), return_inverse=True)[1].astype(np.float64)
    seconds = _seconds(first["timestamp"])
    # One sorted line: tokens are 1e9 seconds (about 30 years) apart, so a sweep never crosses a token.
    line = token_rank * 1e9 + (seconds - seconds.min())
    end = np.searchsorted(line, line + delta_minutes * 60, side="right")
    partners = end - np.arange(len(line)) - 1  # the following first trades of the same token within delta
    left = np.repeat(np.arange(len(line)), partners)
    offset = np.arange(len(left)) - np.repeat(np.cumsum(partners) - partners, partners)
    right = left + 1 + offset
    users = first["user"].to_numpy()
    return _pair_key(users[left], users[right])


def coaction_edges(first: pd.DataFrame, delta_minutes: float, min_shared: int) -> pd.DataFrame:
    """The co-action graph: columns user_a, user_b, weight (shared close first trades), weight >= min_shared."""
    keys, weight = np.unique(close_pairs(first, delta_minutes), return_counts=True)
    keep = weight >= min_shared
    a, b = _key_users(keys[keep])
    return pd.DataFrame({"user_a": a, "user_b": b, "weight": weight[keep]})


def groups(edges: pd.DataFrame, max_component: int = MAX_COMPONENT, seed: int = 0) -> pd.Series:
    """Group id of every account in the graph (index: user), numbered from the largest group.

    Connected components; a component with more than max_component accounts is
    split into Louvain communities (networkx, weighted, fixed seed).
    """
    if edges.empty:
        return pd.Series(dtype=int, name="group")
    users = np.unique(np.concatenate([edges["user_a"], edges["user_b"]]))
    a, b = np.searchsorted(users, edges["user_a"]), np.searchsorted(users, edges["user_b"])
    graph = sparse.coo_matrix((edges["weight"].astype(float), (a, b)), shape=(len(users), len(users))).tocsr()
    _, component = csgraph.connected_components(graph, directed=False)

    label = component.astype(np.int64)
    next_label = label.max() + 1
    sizes = np.bincount(component)
    for big in np.flatnonzero(sizes > max_component):
        members = np.flatnonzero(component == big)
        inside = np.isin(a, members) & np.isin(b, members)
        g = nx.Graph()
        g.add_weighted_edges_from(zip(a[inside], b[inside], edges["weight"].to_numpy()[inside]))
        for community in nx.community.louvain_communities(g, weight="weight", seed=seed):
            label[list(community)] = next_label
            next_label += 1

    # Renumber from the largest group (0) down.
    _, dense = np.unique(label, return_inverse=True)
    order = np.argsort(-np.bincount(dense), kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return pd.Series(rank[dense], index=pd.Index(users, name="user"), name="group")


def group_table(group_of: pd.Series, first: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """One row per group: size, share of members with one first trade in the window, median tokens per member,
    tokens shared by at least half the members, and the median time spread (minutes) of members' first trades
    of those tokens.
    """
    members = first[first["user"].isin(group_of.index)].assign(group=lambda f: f["user"].map(group_of))
    per_member = members.groupby("user").size()
    rows = []
    for g, part in members.groupby("group"):
        size = part["user"].nunique()
        counts = part.groupby("token")["user"].nunique()
        common = counts[counts >= max(2, size / 2)].index
        spread = part[part["token"].isin(common)].groupby("token")["timestamp"].agg(lambda s: (s.max() - s.min()).total_seconds() / 60)
        own = per_member.reindex(part["user"].unique())
        rows.append({
            "group": int(g), "size": size, "one_token_share": float((own == 1).mean()),
            "median_tokens": float(own.median()), "common_tokens": len(common),
            "median_spread_minutes": float(spread.median()) if len(spread) else np.nan,
        })  # fmt: skip
    return pd.DataFrame(rows).sort_values("size", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------
# The check on later data (section 10.2)
# --------------------------------------------------------------------------


def coaction_rate(group_of: pd.Series, first_later: pd.DataFrame, delta_minutes: float) -> pd.DataFrame:
    """Per group, on a later window: active members, pairs of active members, linked pairs, rate.

    Active: the member first-trades at least one token in the later window.
    Linked: the two first-trade at least one common token there within delta.
    """
    active = np.unique(first_later["user"].to_numpy())
    linked = np.unique(close_pairs(first_later, delta_minutes))
    a, b = _key_users(linked)
    group_a = group_of.reindex(a).to_numpy()
    group_b = group_of.reindex(b).to_numpy()
    same = ~np.isnan(group_a) & (group_a == group_b)
    linked_per_group = pd.Series(group_a[same].astype(int)).value_counts()

    n_active = group_of[group_of.index.isin(active)].groupby(group_of[group_of.index.isin(active)]).size()
    table = pd.DataFrame({"size": group_of.groupby(group_of).size()})
    table["active"] = n_active.reindex(table.index).fillna(0).astype(int)
    table["pairs"] = table["active"] * (table["active"] - 1) // 2
    table["linked"] = linked_per_group.reindex(table.index).fillna(0).astype(int)
    table["rate"] = table["linked"] / table["pairs"].where(table["pairs"] > 0)
    table.index.name = "group"
    return table.reset_index()


def strata(first: pd.DataFrame) -> pd.DataFrame:
    """Per account in a build window: number of first trades (exact up to 10, then 11-20, 21-50, >50) and first day."""
    per_user = first.groupby("user").agg(n=("token", "size"), first_time=("timestamp", "min"))
    n = per_user["n"].to_numpy()
    bucket = np.select([n <= 10, n <= 20, n <= 50], [n, 20, 50], 999)
    return pd.DataFrame({"trades": bucket, "day": per_user["first_time"].dt.floor("D")}, index=per_user.index)


def matched_pairs(
    group_of: pd.Series,
    first_build: pd.DataFrame,
    first_later: pd.DataFrame,
    pairs_per_group: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """For each group, random pairs of non-members that look like a pair of its active members.

    A pair of active members (a, b) is drawn at random; a is replaced by a
    random account with the same stratum (strata(): number of first trades in
    the build window and first day) that is active in the later window and not
    a member of the group, and the same for b. Returns columns group, user_a,
    user_b, member_a, member_b (the members they stand for). A member whose
    stratum has no such account is not used.
    """
    rng = np.random.default_rng(seed)
    stratum = strata(first_build)
    active = set(np.unique(first_later["user"].to_numpy()).tolist())
    key_of = dict(zip(stratum.index, zip(stratum["trades"], stratum["day"])))
    stratum = stratum[stratum.index.isin(list(active))]
    pool = {key: part.index.to_numpy() for key, part in stratum.groupby(["trades", "day"])}

    def stand_in(member, member_set):
        """A random active non-member with the member's stratum (None if there is none; 20 tries)."""
        choices = pool.get(key_of[member], np.empty(0, dtype=int))
        for _ in range(20):
            if len(choices) == 0:
                return None
            pick = int(choices[rng.integers(len(choices))])
            if pick not in member_set:
                return pick
        return None

    rows = []
    for g, members in group_of.groupby(group_of):
        candidates = np.array([u for u in members.index if u in active and u in key_of])
        if len(candidates) < 2:
            continue
        member_set = set(members.index.tolist())
        n_pairs = min(pairs_per_group, len(candidates) * (len(candidates) - 1) // 2)
        for _ in range(n_pairs):
            a, b = rng.choice(candidates, size=2, replace=False)
            x, y = stand_in(a, member_set), stand_in(b, member_set)
            if x is None or y is None or x == y:
                continue
            rows.append((int(g), x, y, int(a), int(b)))
    return pd.DataFrame(rows, columns=["group", "user_a", "user_b", "member_a", "member_b"])


def pairs_linked(user_a: np.ndarray, user_b: np.ndarray, first_later: pd.DataFrame, delta_minutes: float) -> np.ndarray:
    """For each pair: do the two first-trade a common token in the later window within delta?"""
    linked = np.unique(close_pairs(first_later, delta_minutes))
    return np.isin(_pair_key(np.asarray(user_a), np.asarray(user_b)), linked)


def check(
    group_of: pd.Series,
    first_build: pd.DataFrame,
    first_later: pd.DataFrame,
    delta_minutes: float,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[dict, pd.DataFrame]:
    """The check of section 10.2: (summary, per-group table).

    ratio = median co-action rate over groups / rate of their matched random
    pairs (all matched pairs pooled). The interval resamples groups (with their
    matched pairs). Success: ratio >= 10 and the interval's lower end > 10.
    """
    rates = coaction_rate(group_of, first_later, delta_minutes)
    rates = rates[rates["pairs"] > 0].reset_index(drop=True)
    matched = matched_pairs(group_of[group_of.isin(rates["group"])], first_build, first_later, seed=seed)
    matched["linked"] = pairs_linked(matched["user_a"], matched["user_b"], first_later, delta_minutes)
    by_group = matched.groupby("group")["linked"].agg(["sum", "size"])
    rates = rates.join(by_group.rename(columns={"sum": "matched_linked", "size": "matched_pairs"}), on="group")
    rates[["matched_linked", "matched_pairs"]] = rates[["matched_linked", "matched_pairs"]].fillna(0).astype(int)
    rates = rates[rates["matched_pairs"] > 0].reset_index(drop=True)

    def ratio(table: pd.DataFrame) -> float:
        matched_rate = table["matched_linked"].sum() / table["matched_pairs"].sum()
        return table["rate"].median() / matched_rate if matched_rate > 0 else np.inf

    rng = np.random.default_rng(seed)
    boot = [ratio(rates.iloc[rng.integers(0, len(rates), len(rates))]) for _ in range(n_boot)] if len(rates) else [np.nan]
    summary = {
        "groups_checked": len(rates),
        "members_checked": int(rates["active"].sum()),
        "median_group_rate": float(rates["rate"].median()) if len(rates) else np.nan,
        "matched_rate": float(rates["matched_linked"].sum() / max(rates["matched_pairs"].sum(), 1)),
        "ratio": ratio(rates) if len(rates) else np.nan,
        "ratio_low": float(np.quantile(boot, 0.025)),
        "ratio_high": float(np.quantile(boot, 0.975)),
    }
    summary["success"] = bool(summary["ratio"] >= 10 and summary["ratio_low"] > 10)
    return summary, rates


def choose_setting(grid: pd.DataFrame) -> pd.Series | None:
    """The choice rule (project_direction.md, 24 Nov 2026): among the settings that pass the check, the one with
    the most grouped accounts; ties go to larger min_shared, then smaller delta. None if no setting passes.

    grid: one row per setting with columns delta, min_shared, accounts, success.
    """
    passing = grid[grid["success"]]
    if passing.empty:
        return None
    return passing.sort_values(["accounts", "min_shared", "delta"], ascending=[False, False, True]).iloc[0]


# --------------------------------------------------------------------------
# The baseline method of notebook 11, with members
# --------------------------------------------------------------------------


def identical_set_members(task: pc.Task, min_size: int = 20) -> pd.Series:
    """Like protocol.same_history_groups, but the group of each member: evaluated users with exactly the same
    14-day token set, in groups of at least min_size (index: user; values: a label unique to the point)."""
    window = task.history(pc.REFERENCE_DAYS)[task.eval_rows]
    token_sets = pd.Series([tuple(window[i].indices) for i in range(window.shape[0])], index=task.users[task.eval_rows])
    sizes = token_sets.value_counts()
    big = sizes[sizes >= min_size]
    label = {token_set: f"{task.t:%m%d}-{i}" for i, token_set in enumerate(big.index)}
    members = token_sets[token_sets.isin(big.index)]
    return members.map(label).rename("identical_set")
