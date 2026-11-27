"""Safer recommendations (work package 16, implementation_plan_applications.md section 16).

How often does the recommender list tokens that collapse soon after, and what
does it cost in recall to list fewer of them?

At an evaluation point t, the frozen fixed blend's first DEPTH tokens per
evaluated account (past tokens removed, as in the protocol) are re-ordered by

    s_ui - lam * r_i(t)

with s_ui the blend score (each part scaled to a maximum of 1 per account, so
s_ui is at most 1) and r_i(t) the token's collapse risk at t: the WP13
baseline model (token age, accounts so far, accounts in the last hour, buy
share), fitted on WP13's Dev A windows and computed from trades before t only.
The first 10 after re-ordering are the list.

Outcomes per list entry: the token collapses in the 72 hours after t (WP13's
label, scam_push.collapse_label); a harmful hit is a hit (listed and then
picked up) on a token that collapses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import expectedness as ex
import pool_recsys as pr
import protocol as pc

DEPTH = 50
LAMBDAS = [0, 0.1, 0.2, 0.5, 1, 2]


def candidate_lists(task: pc.Task, depth: int = DEPTH, settings: dict = ex.FROZEN_BLEND) -> tuple[np.ndarray, np.ndarray]:
    """(columns, scores): the blend's first `depth` token columns per evaluated account, best first, and their scores."""
    scorer, _, _ = ex.blend_model(task, settings)
    rows_all, past = task.eval_rows, task.past
    batch = max(64, 25_000_000 // max(len(task.tokens), 1))
    columns, scores = [], []
    for start in range(0, len(rows_all), batch):
        rows = rows_all[start : start + batch]
        values = scorer(rows).astype(np.float32)
        excluded_rows, excluded_cols = past[rows].nonzero()
        values[excluded_rows, excluded_cols] = -np.inf  # past tokens stay last, as in protocol.evaluate_point
        top = pr.top_k(values, depth)
        columns.append(top)
        scores.append(np.take_along_axis(values, top, axis=1))
    return np.vstack(columns), np.vstack(scores)


def rerank(columns: np.ndarray, scores: np.ndarray, risk: np.ndarray, lam: float, k: int = pc.K) -> np.ndarray:
    """The first k columns per row after ordering by score - lam * risk (risk per token column). Ties keep the blend's order."""
    adjusted = scores - lam * risk[columns]
    order = np.argsort(-adjusted, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(columns, order, axis=1)


def list_outcomes(lists: np.ndarray, task: pc.Task, collapsed: np.ndarray) -> dict:
    """For top-k lists of the evaluated accounts: mean recall, the share of list entries whose token collapses,
    and the share of hits that are on collapsing tokens. collapsed: bool per token column (unknown = False)."""
    rows = task.eval_rows
    metrics = pr.ranking_metrics(lists, task.new[rows], task.cold_new[rows])
    hits = np.asarray(task.new[rows][np.arange(len(rows))[:, None], lists].todense()) > 0
    bad = collapsed[lists]
    return {
        "recall": float(metrics["recall"].mean()),
        "collapsed_share": float(bad.mean()),
        "harmful_hit_share": float((hits & bad).sum() / max(hits.sum(), 1)),
        "accounts_with_a_collapsing_token": float(bad.any(axis=1).mean()),
    }


def tradeoff(task: pc.Task, columns: np.ndarray, scores: np.ndarray, risk: np.ndarray, collapsed: np.ndarray, lams=LAMBDAS) -> pd.DataFrame:
    """list_outcomes for each lam at one point."""
    return pd.DataFrame([{"point": task.t, "lam": lam, **list_outcomes(rerank(columns, scores, risk, lam), task, collapsed)} for lam in lams])
