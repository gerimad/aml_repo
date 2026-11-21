"""Grid search for the fixed combination of popularity and nearest neighbours (WP3, WP4).

The model being tuned is the fixed combination of the project plan (section 6.1):

    score(u, i) = w * p~_i + (1 - w) * r~_ui

where p is popularity, r the item-neighbour score, and ~ scales a score to a
maximum of 1 per user (exactly what pool_recsys.Blend does). Its settings are
the neighbour list length k, the shrinkage lambda and the weight w; in work
package 4 also the half-life and the history length, which change the input
matrices, not this code.

Evaluating every setting with protocol.run would compute the neighbour scores
again for every weight w. blend_grid computes them once per (k, lambda) and
batch of users and reuses them for all weights. The lists are the same as
those of pool_recsys.Blend; a test checks this.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import pool_recsys as pr
import protocol as pc


def scale_rows(scores: np.ndarray) -> np.ndarray:
    """Divide each row by its maximum (as Blend does), so that the best token of every user scores 1."""
    return scores / np.maximum(scores.max(axis=1, keepdims=True), 1e-12)


def blend_grid(
    task: pc.Task,
    popularity_matrix,
    neighbour_matrix,
    ks: list[int],
    shrinks: list[float],
    weights: list[float],
) -> pd.DataFrame:
    """Recall@10 of every (k, shrink, weight) combination at one evaluation point.

    popularity_matrix   users x tokens matrix whose column sums are the popularity
                        (the 14-day window for the baseline's popularity)
    neighbour_matrix    users x tokens matrix used both to compute the token
                        similarities and as each user's profile

    Returns one row per setting with point, k, shrink, weight, users, recall
    (mean over evaluated users) and recall_1, recall_2-4, recall_5+ (by history group).
    """
    rows_all = task.eval_rows
    past = task.past
    groups = pc.history_group(task.window_size[rows_all])
    popularity = pr.Popularity().fit(popularity_matrix).item_scores
    popularity_scaled = popularity / max(popularity.max(), 1e-12)  # the same for every user

    recall = {}  # (k, shrink, weight) -> recall of every evaluated user
    batch = max(64, 25_000_000 // max(len(task.tokens), 1))
    for shrink in shrinks:
        widest = pr.ItemKNN(k=max(ks), shrink=shrink).fit(neighbour_matrix)
        for k in ks:
            # The k best neighbours are among the max(ks) best, so truncating is exact.
            similarity = widest.similarity if k == max(ks) else pr._keep_top_k_per_row(widest.similarity, k)
            per_weight = {w: [] for w in weights}
            for start in range(0, len(rows_all), batch):
                rows = rows_all[start : start + batch]
                neighbours = scale_rows(np.asarray((neighbour_matrix[rows] @ similarity).todense(), dtype=np.float32))
                for w in weights:
                    scores = w * popularity_scaled[None, :] + (1 - w) * neighbours
                    recommended = pr.top_k(scores, pc.K, exclude=past[rows])
                    metrics = pr.ranking_metrics(recommended, task.new[rows], task.cold_new[rows])
                    per_weight[w].append(metrics["recall"].to_numpy())
            for w in weights:
                recall[(k, shrink, w)] = np.concatenate(per_weight[w])

    table = []
    for (k, shrink, w), values in recall.items():
        row = {"point": task.t, "k": k, "shrink": shrink, "weight": w, "users": len(values), "recall": values.mean()}
        for group in ("1", "2-4", "5+"):
            row[f"recall_{group}"] = values[groups == group].mean()
        table.append(row)
    return pd.DataFrame(table)


def best_setting(grid: pd.DataFrame, settings: list[str], prefer: dict[str, str], tolerance: float = 0.002) -> pd.Series:
    """The setting with the highest mean recall over points, preferring simpler ones among near ties.

    `grid` holds blend_grid rows of several points. Settings whose mean recall
    is within `tolerance` of the best are near ties; among them the first is
    taken after sorting by `prefer`, which maps each setting to "small" or
    "large" (e.g. {"k": "small", "shrink": "large", "weight": "large"}: fewer
    neighbours, more shrinkage, more weight on popularity). The order of
    `prefer` is the order of the tie-breaks.
    """
    mean = grid.groupby(settings, as_index=False)["recall"].mean()
    near = mean[mean["recall"] >= mean["recall"].max() - tolerance]
    order = list(prefer)
    ascending = [prefer[s] == "small" for s in order]
    return near.sort_values(order, ascending=ascending).iloc[0]


def grid_settings(**values) -> list[dict]:
    """All combinations of the given setting values, as a list of dicts."""
    names = list(values)
    return [dict(zip(names, combo)) for combo in itertools.product(*values.values())]
