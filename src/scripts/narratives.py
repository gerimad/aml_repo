"""Narratives in the recommender's lists (work package 20, implementation_plan_applications.md section 20).

A narrative is a group of tokens traded by the same accounts: a Louvain
cluster of the item-neighbour similarity fitted on the whole history (the
clustering is descriptive; the same clustering is used for the model and for
every baseline). At each evaluation point a narrative's

  list share       share of all top-50 list slots of the frozen fixed blend (WP17's list_count)
  outcome share    share of the new buyers in the next 48 hours (WP17's n_known)
  popularity share share of 14-day popularity (accounts per token before t)
  recent share     share of the new buyers in the 48 hours before t (WP17's persistence)

The check: are list shares closer to outcome shares than popularity shares are
(mean absolute difference over narratives, per point)?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import pool_recsys as pr
import protocol as pc
import token_map as tm

MIN_ACCOUNTS = 20
RESOLUTIONS = [0.5, 1.0, 2.0, 4.0]
MIN_SHARE = 0.02
SHARES = ["list_share", "outcome_share", "popularity_share", "recent_share"]
COLUMNS = {"list_share": "list_count", "outcome_share": "n_known", "popularity_share": "popularity", "recent_share": "persistence"}


def whole_history_graph(task: pc.Task, k: int = 400, shrink: float = 200.0, min_accounts: int = MIN_ACCOUNTS) -> tm.Graph:
    """Symmetric item-neighbour similarity on the whole history before the task's point, tokens with >= min_accounts."""
    history = task.history(None)
    similarity = pr.ItemKNN(k, shrink).fit(history).similarity
    accounts = np.asarray(history.sum(axis=0)).ravel()
    keep = np.flatnonzero(accounts >= min_accounts)
    return tm.Graph(task.tokens[keep], tm.symmetric(similarity)[keep][:, keep].tocsr(), accounts[keep])


def popularity_14d(legs: pd.DataFrame, t: pd.Timestamp) -> pd.Series:
    """Accounts per token in the 14 days before t."""
    window = legs[(legs["timestamp"] < t) & (legs["timestamp"] >= t - pd.Timedelta(days=pc.REFERENCE_DAYS))]
    return window.groupby("token")["user"].nunique()


def shares(table: pd.DataFrame, narrative_of: pd.Series) -> pd.DataFrame:
    """Per (point, narrative): the four shares of the docstring. table: point, token and the COLUMNS values."""
    table = table.assign(narrative=table["token"].map(narrative_of).fillna("other"))
    sums = table.groupby(["point", "narrative"])[list(COLUMNS.values())].sum()
    totals = sums.groupby("point").transform("sum")
    out = (sums / totals).rename(columns={v: k for k, v in COLUMNS.items()})
    return out.reset_index()


def choose_resolution(table: pd.DataFrame, clusterings: dict[float, pd.Series]) -> float:
    """The smallest resolution whose largest cluster holds under half of all list slots."""
    for resolution in sorted(clusterings):
        slots = table.groupby(table["token"].map(clusterings[resolution]))["list_count"].sum()
        if slots.max() / table["list_count"].sum() < 0.5:
            return resolution
    return max(clusterings)


def group_small(table: pd.DataFrame, cluster_of: pd.Series, min_share: float = MIN_SHARE) -> pd.Series:
    """Cluster labels with every cluster under min_share of list slots at every point set to 'other'."""
    labelled = table.assign(cluster=table["token"].map(cluster_of))
    share = labelled.groupby(["point", "cluster"])["list_count"].sum() / labelled.groupby("point")["list_count"].sum()
    big = set(share[share >= min_share].index.get_level_values("cluster"))
    return cluster_of.where(cluster_of.isin(big), other=-1)


def mean_abs_gap(frame: pd.DataFrame, share: str) -> pd.DataFrame:
    """Per point: mean over narratives of |share - outcome share|."""
    gap = (frame[share] - frame["outcome_share"]).abs()
    return gap.groupby(frame["point"]).mean().rename("gap").reset_index()
