"""Forecasting a token's new buyers (work package 17, implementation_plan_applications.md section 17).

At each evaluation point t, for every token traded by at least one account in
the 14 days before t (an active token):

  n_known       target: accounts with a trade before t whose first trade of
                the token ever falls in [t, t + 48 h) (the accounts the
                recommender can see; equal to the token's column of task.new)
  n_all         the same, counting brand-new accounts too (reported alongside)
  persistence   accounts whose first trade of the token ever falls in the 48
                hours before t
  users_24h     accounts that traded the token in the 24 hours before t
  age_days      days since the token's first trade in the data
  list_count    accounts with the token in their top-50 list under the frozen
                fixed blend; the lists go to every account with a token in its
                14-day window (not only those evaluated, which would use the
                future), and leave out the account's past tokens as in the
                protocol

Models (linear regressions of log(1 + target), fitted on the Dev A points):
  (a) baseline:    log(1 + persistence), log(1 + users_24h), log(1 + age_days)
  (b) with lists:  the same plus log(1 + list_count)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import expectedness as ex
import pool_recsys as pr
import protocol as pc

LIST_DEPTH = 50
BASELINE_FEATURES = ["log_persistence", "log_users_24h", "log_age_days"]
WITH_LISTS = BASELINE_FEATURES + ["log_list_count"]


def first_ever(legs: pd.DataFrame) -> pd.DataFrame:
    """Each (user, token) pair with the time of its first trade in the data."""
    return legs.groupby(["user", "token"], observed=True)["timestamp"].min().rename("first_time").reset_index()


def new_buyers(first: pd.DataFrame, start, end, known_users=None) -> pd.Series:
    """Per token: accounts whose first trade of it falls in [start, end); only known_users if given."""
    window = first[(first["first_time"] >= start) & (first["first_time"] < end)]
    if known_users is not None:
        window = window[window["user"].isin(known_users)]
    return window.groupby("token").size()


def list_counts(task: pc.Task, depth: int = LIST_DEPTH, settings: dict = ex.FROZEN_BLEND) -> np.ndarray:
    """Per universe token: accounts (with a token in their 14-day window) that have it in their top-`depth` list."""
    scorer, _, _ = ex.blend_model(task, settings)
    rows = np.flatnonzero(task.window_size >= 1)
    past = task.past
    counts = np.zeros(len(task.tokens), dtype=np.int64)
    batch = max(64, 25_000_000 // max(len(task.tokens), 1))
    for start in range(0, len(rows), batch):
        part = rows[start : start + batch]
        lists = pr.top_k(scorer(part), depth, exclude=past[part])
        counts += np.bincount(lists.ravel(), minlength=len(task.tokens))
    return counts


def point_table(legs: pd.DataFrame, first: pd.DataFrame, task: pc.Task, tokens: pd.DataFrame, depth: int = LIST_DEPTH) -> pd.DataFrame:
    """One row per active token at the task's point: the target, the features and in_head (module docstring)."""
    t = task.t
    window = task.history(pc.REFERENCE_DAYS)
    active = np.flatnonzero(np.asarray(window.sum(axis=0)).ravel() >= 1)
    ids = task.tokens[active]
    horizon = pc.HORIZON

    table = pd.DataFrame({"point": t, "token": ids})
    table["n_known"] = np.asarray(task.new[:, active].sum(axis=0)).ravel().astype(int)
    table["n_all"] = new_buyers(first, t, t + horizon).reindex(ids).fillna(0).astype(int).to_numpy()
    table["persistence"] = new_buyers(first, t - horizon, t).reindex(ids).fillna(0).astype(int).to_numpy()
    recent = legs[(legs["timestamp"] >= t - pd.Timedelta(hours=24)) & (legs["timestamp"] < t)]
    table["users_24h"] = recent.groupby("token")["user"].nunique().reindex(ids).fillna(0).astype(int).to_numpy()
    born = tokens.set_index("token_id")["first_timestamp"]
    table["age_days"] = ((t - table["token"].map(born)).dt.total_seconds() / 86400).to_numpy()
    table["list_count"] = list_counts(task, depth)[active]
    head = np.zeros(len(task.tokens), dtype=bool)
    head[pc.head_tokens(window)] = True
    table["in_head"] = head[active]
    return table


def add_logs(table: pd.DataFrame) -> pd.DataFrame:
    """log(1 + x) of the features and targets, as the models use them."""
    out = table.copy()
    for column in ["persistence", "users_24h", "age_days", "list_count", "n_known", "n_all"]:
        out[f"log_{column}"] = np.log1p(out[column].clip(lower=0))
    return out


def fit_and_score(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str = "log_n_known") -> pd.DataFrame:
    """Fit a linear regression on train, predict test; per point: mean absolute error, Spearman, error on tail tokens."""
    model = LinearRegression().fit(train[features], train[target])
    scored = test.assign(prediction=model.predict(test[features]))
    scored["error"] = (scored["prediction"] - scored[target]).abs()
    rows = []
    for t, part in scored.groupby("point"):
        rows.append({
            "point": t,
            "mae": part["error"].mean(),
            "mae_tail": part.loc[~part["in_head"], "error"].mean(),
            "spearman": part["prediction"].corr(part[target], method="spearman"),
        })  # fmt: skip
    return pd.DataFrame(rows)
