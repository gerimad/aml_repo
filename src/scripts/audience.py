"""Likely buyers of a token (work package 15, implementation_plan_applications.md section 15).

The project's recommender read by column: for a token, which accounts are
most likely to pick it up in the two days after an evaluation point t?

  audience   the accounts with at least one token in their 14-day window
             that have never traded the token before t
  buyers     the audience accounts that trade the token in [t, t + 2 days)
  tokens     the universe tokens with at least MIN_BUYERS buyers

The model is the frozen fixed blend. Its score is
    weight * popularity_i / max popularity + (1 - weight) * knn_ui / max_j knn_uj,
and the popularity part is the same for every account, so for one token the
accounts are ordered by the neighbour part alone: knn_ui / max_j knn_uj, with
knn_u = (the account's 14-day window) @ (the token similarity matrix). This is
computed with sparse products, so the dense accounts x tokens matrix is
never formed. Accounts with no co-trading path to the token score 0 and are
ordered by the activity baseline.

Baselines:
  activity   accounts ordered by the number of tokens in their 14-day window,
             the same order for every token
  random     a random order

Variants (added after the plan's score lost to activity at a first look, chosen on
Dev A only; project_direction.md, 24 Nov 2026):
  raw neighbours        knn_ui without the per-account scaling (it sums the
                        similarities from all the account's tokens, so an active
                        account with a matching history scores high)
  model x activity      the plan's score times the account's window size
  model x log activity  the plan's score times log(1 + window size)

Ties: every method's ties go to activity; activity ties to a fixed random order
(seed 0), never to the account id.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

import expectedness as ex
import protocol as pc

MIN_BUYERS = 10
K_RECALL = 1000
K_PRECISION = 100


def audience_rows(task: pc.Task) -> np.ndarray:
    """Rows of the accounts with at least one token in their 14-day window."""
    return np.flatnonzero(task.window_size >= 1)


def scored_tokens(task: pc.Task, rows: np.ndarray, min_buyers: int = MIN_BUYERS) -> np.ndarray:
    """Columns of the tokens with at least min_buyers buyers among the given rows."""
    buyers = np.asarray(task.new[rows].sum(axis=0)).ravel()
    return np.flatnonzero(buyers >= min_buyers)


def neighbour_columns(window: sparse.csr_matrix, similarity: sparse.csr_matrix, rows: np.ndarray, cols: np.ndarray, batch: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """(raw, row_max): knn_ui for the given rows (accounts) and cols (tokens), rows x cols float32, and each
    row's maximum knn_uj over all tokens (as pool_recsys.Blend scales it; 0 for a row with no neighbour score).
    """
    raw = np.zeros((len(rows), len(cols)), dtype=np.float32)
    row_max = np.zeros(len(rows))
    for start in range(0, len(rows), batch):
        part = rows[start : start + batch]
        product = (window[part] @ similarity).tocsr()
        row_max[start : start + len(part)] = product.max(axis=1).toarray().ravel()
        raw[start : start + len(part)] = product[:, cols].toarray()
    return raw, row_max


def model_columns(task: pc.Task, rows: np.ndarray, cols: np.ndarray, settings: dict = ex.FROZEN_BLEND) -> tuple[np.ndarray, np.ndarray]:
    """The fixed blend's neighbour part for the given accounts and tokens: (raw knn scores, each account's maximum)."""
    _, similarity, window = ex.blend_model(task, settings)
    return neighbour_columns(window, similarity, rows, cols)


def tiebreak_order(task: pc.Task, rows: np.ndarray, seed: int = 0) -> np.ndarray:
    """A key per row for the activity baseline: higher = earlier (window size, then a fixed random order)."""
    random_part = np.random.default_rng(seed).random(len(rows))
    return task.window_size[rows] + random_part  # window sizes are integers, so the random part only breaks ties


def rank_accounts(scores: np.ndarray, tiebreak: np.ndarray, k: int) -> np.ndarray:
    """For each column of scores (rows x tokens): the first k rows, best first; ties by tiebreak (higher first)."""
    k = min(k, scores.shape[0])
    order = np.empty((scores.shape[1], k), dtype=np.int64)
    for j in range(scores.shape[1]):
        # lexsort sorts by the last key first: score, then the tie-break
        ranked = np.lexsort((-tiebreak, -scores[:, j].astype(np.float64)))
        order[j] = ranked[:k]
    return order


def audience_metrics(order: np.ndarray, buyers: np.ndarray, k_recall: int = K_RECALL, k_precision: int = K_PRECISION) -> pd.DataFrame:
    """recall@k_recall and precision@k_precision per token.

    order: tokens x k (row positions, best first); buyers: rows x tokens, 0/1 (dense or sparse).
    """
    buyers = sparse.csc_matrix(buyers)
    rows = []
    for j in range(order.shape[0]):
        is_buyer = np.zeros(buyers.shape[0], dtype=bool)
        is_buyer[buyers.indices[buyers.indptr[j] : buyers.indptr[j + 1]]] = True
        n = int(is_buyer.sum())
        rows.append({
            "buyers": n,
            f"recall_{k_recall}": is_buyer[order[j, :k_recall]].sum() / n if n else np.nan,
            f"precision_{k_precision}": is_buyer[order[j, :k_precision]].mean(),
        })  # fmt: skip
    return pd.DataFrame(rows)


def point_table(task: pc.Task, exclude_users=(), settings: dict = ex.FROZEN_BLEND, seed: int = 0) -> pd.DataFrame:
    """One row per (scored token, method) at the task's point: token, method, buyers, recall@1000, precision@100,
    in_head, buyers_with_path (share of buyers with a nonzero neighbour score), audience (accounts).

    exclude_users: account ids left out of the audience (and so of the buyers), e.g. farm members.
    """
    rows = audience_rows(task)
    rows = rows[~np.isin(task.users[rows], np.asarray(list(exclude_users)))]
    cols = scored_tokens(task, rows)
    if len(cols) == 0:
        return pd.DataFrame()
    # Past pairs are never buyers (task.new excludes them) and are removed from each token's audience below.
    past = task.past[rows][:, cols].toarray().astype(bool)
    buyers = task.new[rows][:, cols]
    tiebreak = tiebreak_order(task, rows, seed)
    head = np.isin(cols, pc.head_tokens(task.history(pc.REFERENCE_DAYS)))

    raw, row_max = model_columns(task, rows, cols, settings)
    scaled = raw / np.maximum(row_max, 1e-12)[:, None]
    activity = task.window_size[rows].astype(np.float32)[:, None]
    methods = {
        "model": scaled,  # the plan's score: the blend's column
        "raw neighbours": raw,  # variants of section 15 (project_direction.md, 24 Nov 2026), chosen on Dev A
        "model x activity": scaled * activity,
        "model x log activity": scaled * np.log1p(activity),
        "activity": np.broadcast_to(tiebreak[:, None], scaled.shape),
        "random": np.random.default_rng(seed + 1).random(scaled.shape, dtype=np.float32),
    }
    tables = []
    for method, scores in methods.items():
        scores = np.where(past, -np.inf, scores)  # an account that traded the token before t is not in its audience
        metrics = audience_metrics(rank_accounts(scores, tiebreak, K_RECALL), buyers)
        metrics.insert(0, "method", method)
        metrics.insert(0, "token", task.tokens[cols])
        metrics["in_head"] = head
        tables.append(metrics)
    table = pd.concat(tables, ignore_index=True)
    with_path = np.asarray(buyers.multiply(raw > 0).sum(axis=0)).ravel() / np.maximum(np.asarray(buyers.sum(axis=0)).ravel(), 1)
    table["buyers_with_path"] = np.tile(with_path, len(methods))
    table["audience"] = len(rows)
    table.insert(0, "point", task.t)
    return table
