"""Buyers of brand-new tokens (work package 19, implementation_plan_applications.md section 19).

WP15 ranks accounts for a token the model knows. A token launched after the
evaluation point t has no column in the similarity matrix, and new tokens are
29% of what traders pick up (the project's cold start). Here the token's first
buyers stand in for its column:

  launch     the token's first trade ever, in [t, t + 24 h)
  seeds      audience accounts that trade it in the first hour after launch
  buyers     audience accounts whose first trade of it falls in
             (launch + 1 h, launch + 25 h]; seeds are never buyers
  audience   the accounts with at least one token in the 14 days before t
             (as WP15), minus the seeds

Everything a score uses is known one hour after launch: the account histories
before t and the seeds. Scores (variants of the plan, chosen on Dev A):

  seed profile        p_j = number of seeds with token j in their window;
                      score = W p (W: accounts x tokens, 14-day window, binary)
  seed profile, idf   p_j divided by log(1 + accounts with j in their window)
  recommender         p passed through the frozen blend's similarity matrix S:
                      score = W (S p), the recommender's co-trading read for
                      the new token

Baselines: activity (window size, as WP15), launch hunters (the number of
tokens the account first traded in the 14 days before t within 24 hours of
their launch), random. Ties go to activity, then a fixed random order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

import audience as au
import expectedness as ex
import protocol as pc

SEED_WINDOW = pd.Timedelta(hours=1)
FOLLOW = pd.Timedelta(hours=24)
LAUNCH_WINDOW = pd.Timedelta(hours=24)
LATE = pd.Timedelta(hours=6)
MIN_SEEDS = 3
MIN_BUYERS = 10
EARLY = pd.Timedelta(hours=24)


def launch_times(legs: pd.DataFrame) -> pd.Series:
    """Each token's first trade ever (token -> timestamp)."""
    return legs.groupby("token")["timestamp"].min()


def launch_hunting(legs: pd.DataFrame, t: pd.Timestamp, launches: pd.Series) -> pd.Series:
    """Per user: tokens first traded in the 14 days before t within 24 hours of the token's launch."""
    before = legs[legs["timestamp"] < t]
    first = before.groupby(["user", "token"])["timestamp"].min().reset_index()
    first = first[first["timestamp"] >= t - pd.Timedelta(days=pc.REFERENCE_DAYS)]
    early = first["timestamp"] - first["token"].map(launches) < EARLY
    return first[early].groupby("user").size()


def new_tokens(legs: pd.DataFrame, t: pd.Timestamp, launches: pd.Series, user_rows: pd.Series) -> pd.DataFrame:
    """Tokens launched in [t, t + 24 h) with their seeds and buyers among the audience.

    user_rows: audience user id -> row position. Returns one row per token with launch, seed_rows, buyer_rows,
    buyer_times (hours after launch), only for tokens with at least MIN_SEEDS seeds and MIN_BUYERS buyers.
    """
    launched = launches[(launches >= t) & (launches < t + LAUNCH_WINDOW)]
    if launched.empty:
        return pd.DataFrame()
    window = legs[legs["token"].isin(launched.index) & (legs["timestamp"] < t + LAUNCH_WINDOW + SEED_WINDOW + FOLLOW)]
    first = window.groupby(["token", "user"])["timestamp"].min().reset_index()
    first = first[first["user"].isin(user_rows.index)]
    first["after"] = first["timestamp"] - first["token"].map(launched)
    rows = []
    for token, part in first.groupby("token"):
        seeds = part[part["after"] <= SEED_WINDOW]
        buyers = part[(part["after"] > SEED_WINDOW) & (part["after"] <= SEED_WINDOW + FOLLOW)]
        if len(seeds) >= MIN_SEEDS and len(buyers) >= MIN_BUYERS:
            rows.append({"token": token, "launch": launched[token], "seed_rows": user_rows[seeds["user"]].to_numpy(),
                         "buyer_rows": user_rows[buyers["user"]].to_numpy(),
                         "buyer_hours": (buyers["after"] / pd.Timedelta(hours=1)).to_numpy()})  # fmt: skip
    return pd.DataFrame(rows)


def method_scores(window: sparse.csr_matrix, similarity: sparse.csr_matrix, seed_rows: np.ndarray) -> dict[str, np.ndarray]:
    """The three seed-based scores for one token, one value per audience row."""
    profile = np.asarray(window[seed_rows].sum(axis=0)).ravel()
    popularity = np.asarray(window.sum(axis=0)).ravel()
    return {
        "seed profile": window @ profile,
        "seed profile, idf": window @ (profile / np.log1p(np.maximum(popularity, 1))),
        "recommender": window @ (similarity @ profile),
    }


def point_table(task: pc.Task, legs: pd.DataFrame, launches: pd.Series, exclude_users=(), seed: int = 0) -> pd.DataFrame:
    """One row per (new token, method) at the task's point: token, method, seeds, buyers, recall_1000,
    precision_100, late_buyers and recall_1000_late (buyers arriving more than 6 hours after launch)."""
    rows = au.audience_rows(task)
    rows = rows[~np.isin(task.users[rows], np.asarray(list(exclude_users)))]
    user_rows = pd.Series(np.arange(len(rows)), index=task.users[rows])
    tokens = new_tokens(legs, task.t, launches, user_rows)
    if tokens.empty:
        return pd.DataFrame()
    _, similarity, window = ex.blend_model(task)
    window = (window[rows] > 0).astype(np.float32).tocsr()
    similarity = similarity.astype(np.float32).tocsr()
    tiebreak = au.tiebreak_order(task, rows, seed)
    hunting = task.users[rows]
    hunting = pd.Series(hunting).map(launch_hunting(legs, task.t, launches)).fillna(0).to_numpy()
    rng = np.random.default_rng(seed + 1)

    n = len(rows)
    buyers = sparse.lil_matrix((n, len(tokens)), dtype=np.int8)
    late = sparse.lil_matrix((n, len(tokens)), dtype=np.int8)
    columns: dict[str, list[np.ndarray]] = {}
    for j, token in enumerate(tokens.itertuples()):
        buyers[token.buyer_rows, j] = 1
        late[token.buyer_rows[token.buyer_hours > LATE / pd.Timedelta(hours=1)], j] = 1
        scores = method_scores(window, similarity, token.seed_rows)
        scores.update({"activity": tiebreak.astype(np.float32), "launch hunters": hunting.astype(np.float32) + tiebreak / 1e6,
                       "random": rng.random(n, dtype=np.float32)})  # fmt: skip
        for name, score in scores.items():
            score = np.asarray(score, dtype=np.float64).copy()
            score[token.seed_rows] = -np.inf  # seeds are not in the token's audience
            columns.setdefault(name, []).append(score)

    tables = []
    for name, cols in columns.items():
        order = au.rank_accounts(np.column_stack(cols), tiebreak, au.K_RECALL)
        metrics = au.audience_metrics(order, buyers.tocsc())
        late_metrics = au.audience_metrics(order, late.tocsc())
        metrics["late_buyers"] = late_metrics["buyers"].to_numpy()
        metrics["recall_1000_late"] = late_metrics["recall_1000"].to_numpy()
        metrics.insert(0, "method", name)
        metrics.insert(0, "seeds", [len(s) for s in tokens["seed_rows"]])
        metrics.insert(0, "token", tokens["token"].to_numpy())
        tables.append(metrics)
    table = pd.concat(tables, ignore_index=True)
    table["audience"] = n
    table.insert(0, "point", task.t)
    return table
