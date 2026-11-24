"""Expectedness scores (work package 9, implementation_plan_applications.md section 9).

For every positive pair (account, token) at an evaluation point t, how
expected the trade was under the recommender:

  rank              the token's rank among the tokens the account has not
                    traded before t (0 = first choice): the number of these
                    candidate tokens with a strictly higher score, so tied
                    tokens share the best rank among them. The pair is a hit
                    of the top-10 list exactly when rank < 10 (without ties).
  n_candidates      the number of those candidate tokens
  rank_percentile   rank / (n_candidates - 1): 0 is the first choice, 1 the last
  window_size       the account's tokens in the 14-day window (history group)
  cold              the token did not exist before t, so it has no rank
  in_head           the token is in the popular head (protocol.head_tokens)
  max_similarity    the largest similarity from any of the account's tokens
                    in the 14-day window to this token, in the neighbour model
                    the blend uses (0 means no co-trading path)
  popularity_rank   0 for the token with the most accounts in the 14-day window
  token_age_days    age of the token at t (from its first trade in the data)
  first_trade_time  the account's first trade of the token in the target window

The scorer is the frozen fixed blend (configs/frozen.json, "fixed blend":
popularity and neighbours on the 14-day window, k = 400, shrink = 200,
weight 0.5). It scores every token of the universe, unlike the ranker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

import pool_recsys as pr
import protocol as pc

FROZEN_BLEND = {"k": 400, "shrink": 200.0, "weight": 0.5}


def blend_model(task: pc.Task, settings: dict = FROZEN_BLEND):
    """The frozen fixed blend at one point: (scorer, similarity matrix of its neighbour part, the 14-day window)."""
    window = task.history(pc.REFERENCE_DAYS)
    popularity = pr.Popularity().fit(window)
    neighbours = pr.ItemKNN(settings["k"], settings["shrink"]).fit(window)
    blend = pr.Blend(popularity, neighbours, settings["weight"])
    return (lambda rows: blend.scores(window[rows])), neighbours.similarity, window


def ranks(task: pc.Task, scorer: pc.Scorer, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Rank of each pair (rows[i], cols[i]) of the task among the account's candidate tokens; pairs must not be past pairs.

    Past tokens of the account are removed exactly as in protocol.evaluate_point
    (score -inf), in the same float32 precision as pool_recsys.top_k. Users are
    scored in batches of the same size as there.
    """
    rows, cols = np.asarray(rows), np.asarray(cols)
    out = np.full(len(rows), np.nan)
    past = task.past
    batch = max(64, 25_000_000 // max(len(task.tokens), 1))
    unique_rows = np.unique(rows)
    for start in range(0, len(unique_rows), batch):
        users = unique_rows[start : start + batch]
        scores = scorer(users).astype(np.float32, copy=True)
        excluded_rows, excluded_cols = past[users].nonzero()
        scores[excluded_rows, excluded_cols] = -np.inf
        descending = -np.sort(scores, axis=1)[:, ::-1]  # each row from the highest score down, as negatives ascending
        position = np.searchsorted(users, rows)
        in_batch = (position < len(users)) & (users[np.minimum(position, len(users) - 1)] == rows)
        for i in np.flatnonzero(in_batch):
            r = position[i]
            own = scores[r, cols[i]]
            out[i] = np.searchsorted(descending[r], -own, side="left")  # tokens with a strictly higher score
    return out


def candidates(task: pc.Task, rows: np.ndarray) -> np.ndarray:
    """The number of tokens each account has not traded before t (the tokens it can be recommended)."""
    return len(task.tokens) - np.asarray(task.past[rows].sum(axis=1)).ravel()


def rank_percentiles(task: pc.Task, scorer: pc.Scorer, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """rank / (n_candidates - 1) of each pair: 0 is the model's first choice, 1 its last."""
    return ranks(task, scorer, rows, cols) / np.maximum(candidates(task, rows) - 1, 1)


def max_similarity(similarity: sparse.csr_matrix, window: sparse.csr_matrix, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """For each pair: the largest similarity from one of the account's window tokens to the pair's token (0 if none)."""
    by_column = similarity.tocsc()
    out = np.zeros(len(rows))
    for i, (r, c) in enumerate(zip(rows, cols)):
        own_tokens = window.indices[window.indptr[r] : window.indptr[r + 1]]
        start, end = by_column.indptr[c], by_column.indptr[c + 1]
        sources, values = by_column.indices[start:end], by_column.data[start:end]
        hit = np.isin(sources, own_tokens)
        out[i] = float(values[hit].max()) if hit.any() else 0.0
    return out


def first_trades_in_target(legs: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Each (user, token) pair traded in the target window [t, t + 2 days), with its first trade time there."""
    target = legs[(legs["timestamp"] >= t) & (legs["timestamp"] < t + pc.HORIZON)]
    return target.groupby(["user", "token"], observed=True)["timestamp"].min().rename("first_trade_time").reset_index()


def pair_table(legs: pd.DataFrame, task: pc.Task, tokens: pd.DataFrame, settings: dict = FROZEN_BLEND) -> pd.DataFrame:
    """One row per positive pair of the evaluated accounts at the task's point (module docstring).

    tokens: the token table (token_data.load_tokens) with token_id and first_timestamp.
    """
    t = task.t
    scorer, similarity, window = blend_model(task, settings)
    evaluated = np.zeros(len(task.users), dtype=bool)
    evaluated[task.eval_rows] = True

    # Warm pairs: the task's positive matrix, evaluated rows only.
    new = task.new.tocoo()
    keep = evaluated[new.row]
    rows, cols = new.row[keep], new.col[keep]
    warm = pd.DataFrame({"user": task.users[rows], "token": task.tokens[cols], "cold": False})
    warm["rank"] = ranks(task, scorer, rows, cols)
    warm["n_candidates"] = candidates(task, rows)
    warm["rank_percentile"] = warm["rank"] / np.maximum(warm["n_candidates"] - 1, 1)
    head = np.zeros(len(task.tokens), dtype=bool)
    head[pc.head_tokens(window)] = True
    warm["in_head"] = head[cols]
    warm["max_similarity"] = max_similarity(similarity, window, rows, cols)
    users_per_token = np.asarray(window.sum(axis=0)).ravel()
    popularity_rank = np.empty(len(task.tokens), dtype=int)
    popularity_rank[np.argsort(-users_per_token, kind="stable")] = np.arange(len(task.tokens))
    warm["popularity_rank"] = popularity_rank[cols]

    # Cold pairs: evaluated accounts' target tokens that did not exist before t.
    first = first_trades_in_target(legs, t)
    evaluated_users = task.users[task.eval_rows]
    cold = first[first["user"].isin(evaluated_users) & ~first["token"].isin(task.tokens)][["user", "token"]].copy()
    cold["cold"] = True

    table = pd.concat([warm, cold], ignore_index=True)
    table = table.merge(first, on=["user", "token"], how="left")
    window_size = pd.Series(task.window_size, index=task.users)
    table["window_size"] = table["user"].map(window_size).astype(int)
    age = tokens.set_index("token_id")["first_timestamp"]
    table["token_age_days"] = (t - table["token"].map(age)).dt.total_seconds() / 86400
    table["token_age_days"] = table["token_age_days"].where(~table["cold"])  # a cold token is not born yet at t
    table.insert(0, "point", t)
    return table
