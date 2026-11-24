"""Scam-push detection (work package 13, implementation_plan_applications.md section 13).

Is adoption that the recommender cannot explain a warning of collapse?

Decision windows. At each evaluation point t the frozen blend ranks every
account's candidate tokens (WP9). Two 24-hour windows follow each point:
w = [s, s + 24 h) with s = t and s = t + 24 h. In a window, a token's new
accounts are the accounts whose first trade of it ever falls in w. The score
is known at d = s + 24 h; the label looks at the 72 hours after d.

Recommender features (from WP9's pairs of the evaluated accounts that are
among the new accounts):
  unexplained_share  share with no co-trading path to the token (max
                     similarity 0) for which the token is not in the head
  low_rank_share     share for which the token ranks in the worst 10% among
                     the tokens active in the 14 days before t (project_direction.md,
                     24 Nov 2026: inactive tokens all score 0, so the rank among
                     active tokens is the WP9 rank, and the candidates are the
                     active tokens the account has not traded)
  burst              new accounts in the window's last hour / new accounts in the window
  farm_share         share of new accounts in WP10 groups built on the 14 days before t

Baseline features (anyone can see them): token age at d (days), accounts that
ever traded it before d, accounts in the hour before d, buy share of the
window's trades.

Only tokens that existed before t (warm tokens; a newer token has no rank) and
had at least MIN_NEW_ACCOUNTS new accounts in the window are scored.

Label (from prices only, section 13.2): in the token's most active pool in
[d, d + 72 h) (by ETH volume over swaps of at least 0.01 ETH), the price falls
more than 90% below its running peak, and the token has fewer than 5 trades
(any pool, from the legs) in [d + 48 h, d + 72 h). A token whose trading has
stopped and that has no swap of at least 0.01 ETH against a base in
[d, d + 72 h) is labelled collapsed (it is dead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import protocol as pc

MIN_NEW_ACCOUNTS = 20
WINDOW = pd.Timedelta(hours=24)
LABEL_HOURS = 72
MIN_PRICE_ETH = 0.01
RECOMMENDER_FEATURES = ["unexplained_share", "low_rank_share", "burst", "farm_share"]
BASELINE_FEATURES = ["log_age_days", "log_accounts_before", "log_accounts_last_hour", "buy_share"]


def active_candidates(task: pc.Task, users: np.ndarray) -> pd.Series:
    """Per account: tokens active in the 14 days before t (at least one account) that it has not traded before t."""
    window = task.history(pc.REFERENCE_DAYS)
    active = np.asarray(window.sum(axis=0)).ravel() >= 1
    rows = np.searchsorted(task.users, users)
    traded_active = np.asarray(task.past[rows][:, active].sum(axis=1)).ravel()
    return pd.Series(active.sum() - traded_active, index=users)


def window_features(pairs: pd.DataFrame, legs: pd.DataFrame, first: pd.DataFrame, task: pc.Task, s: pd.Timestamp,
                    farm_members: set, min_new: int = MIN_NEW_ACCOUNTS) -> pd.DataFrame:  # fmt: skip
    """One row per scored token for the window [s, s + 24 h): the features of the module docstring.

    pairs: WP9 pairs at the task's point; first: each (user, token)'s first trade time
    (columns user, token, first_time) over all legs.
    """
    d = s + WINDOW
    new = first[(first["first_time"] >= s) & (first["first_time"] < d) & first["token"].isin(task.tokens)]
    counts = new.groupby("token").size()
    tokens = counts[counts >= min_new].index
    new = new[new["token"].isin(tokens)]
    out = pd.DataFrame(index=pd.Index(tokens, name="token"))
    out["new_accounts"] = counts.loc[tokens]
    last_hour = new[new["first_time"] >= d - pd.Timedelta(hours=1)].groupby("token").size()
    out["burst"] = last_hour.reindex(tokens).fillna(0) / out["new_accounts"]
    out["farm_share"] = new.assign(farm=new["user"].isin(farm_members)).groupby("token")["farm"].mean()

    ranked = pairs[~pairs["cold"] & (pairs["first_trade_time"] >= s) & (pairs["first_trade_time"] < d) & pairs["token"].isin(tokens)]
    ranked = ranked.assign(candidates=ranked["user"].map(active_candidates(task, ranked["user"].unique())))
    ranked = ranked.assign(
        active_percentile=(ranked["rank"] / (ranked["candidates"] - 1).clip(lower=1)).clip(upper=1),
        unexplained=(ranked["max_similarity"] == 0) & ~ranked["in_head"].astype(bool),
    )
    grouped = ranked.groupby("token")
    out["ranked_accounts"] = grouped.size().reindex(tokens).fillna(0).astype(int)
    out["unexplained_share"] = grouped["unexplained"].mean()
    out["low_rank_share"] = grouped["active_percentile"].agg(lambda p: (p > 0.9).mean())

    before = legs[(legs["timestamp"] < d) & legs["token"].isin(tokens)]
    born = first.groupby("token")["first_time"].min()
    out["age_days"] = (d - born.reindex(tokens)).dt.total_seconds() / 86400
    out["accounts_before"] = before.groupby("token")["user"].nunique().reindex(tokens).fillna(0)
    hour = before[before["timestamp"] >= d - pd.Timedelta(hours=1)]
    out["accounts_last_hour"] = hour.groupby("token")["user"].nunique().reindex(tokens).fillna(0)
    in_window = before[before["timestamp"] >= s]
    out["buy_share"] = (in_window["side"] == "buy").groupby(in_window["token"]).mean().reindex(tokens)
    out["log_age_days"] = np.log1p(out["age_days"])
    out["log_accounts_before"] = np.log1p(out["accounts_before"])
    out["log_accounts_last_hour"] = np.log1p(out["accounts_last_hour"])
    out = out.reset_index()
    out.insert(0, "decision", d)
    out.insert(0, "window_start", s)
    out.insert(0, "point", task.t)
    return out


def collapse_label(swaps: pd.DataFrame, d: pd.Timestamp, late_trades: int | None = None, hours: int = LABEL_HOURS, min_eth: float = MIN_PRICE_ETH) -> bool:
    """The label of 13.2 for one token: its trading stops, and its price fell more than 90% below its peak
    (or it has no real priced swap left to show a price at all).

    swaps: the token's priced swaps (timestamp, pool, price_eth, volume_eth).
    late_trades: the token's trades in any pool in the last 24 hours of the label window
    (from the legs). The price tables hold only swaps against WETH, USDC or USDT,
    so counting them alone would call a token dead that trades in other pools
    (corrected 24 Nov 2026, project_direction.md). If None, the priced swaps are counted.
    """
    end = d + pd.Timedelta(hours=hours)
    window = swaps[(swaps["timestamp"] >= d) & (swaps["timestamp"] < end)]
    if late_trades is None:
        late_trades = int((window["timestamp"] >= end - pd.Timedelta(hours=24)).sum())
    if late_trades >= 5:
        return False  # still trading
    priced = window[window["volume_eth"] >= min_eth]
    if priced.empty:
        return True  # no real trade left to price it, and trading has stopped: dead
    pool = priced.groupby("pool")["volume_eth"].sum().idxmax()
    prices = priced[priced["pool"] == pool].sort_values("timestamp")["price_eth"].to_numpy()
    peak = np.maximum.accumulate(prices)
    return bool((prices < 0.1 * peak).any())


def fit_score(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Logistic regression (standardized features) fitted on train; collapse probabilities for test."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[features].fillna(train[features].median()), train["collapsed"])
    return model.predict_proba(test[features].fillna(train[features].median()))[:, 1]


def auc_difference_by_day(scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray, days: np.ndarray,
                          n_boot: int = 2000, seed: int = 0) -> dict:  # fmt: skip
    """AUC(b) - AUC(a) with a percentile bootstrap that resamples days (all tokens of a drawn day come along)."""
    scores_a, scores_b, labels, days = map(np.asarray, (scores_a, scores_b, labels, days))
    unique = np.unique(days)
    by_day = {day: np.flatnonzero(days == day) for day in unique}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        rows = np.concatenate([by_day[day] for day in rng.choice(unique, len(unique), replace=True)])
        if labels[rows].min() == labels[rows].max():
            continue
        diffs.append(roc_auc_score(labels[rows], scores_b[rows]) - roc_auc_score(labels[rows], scores_a[rows]))
    point = roc_auc_score(labels, scores_b) - roc_auc_score(labels, scores_a)
    low, high = np.quantile(diffs, [0.025, 0.975])
    return {"auc_a": roc_auc_score(labels, scores_a), "auc_b": roc_auc_score(labels, scores_b), "difference": point,
            "low": float(low), "high": float(high), "days": len(unique), "tokens": len(labels), "collapsed": int(labels.sum())}  # fmt: skip


def precision_at(scores: np.ndarray, labels: np.ndarray, share: float = 0.01) -> float:
    """Share of collapsed tokens among the top `share` of scores (at least one token)."""
    n = max(1, int(round(len(scores) * share)))
    return float(np.asarray(labels)[np.argsort(-np.asarray(scores), kind="stable")[:n]].mean())
