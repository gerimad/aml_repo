"""Recommend pools or tokens to traders from their on-chain swap histories.

Building blocks for the "next-pool recommendation" experiments. The pipeline:

    swaps -> (user, item) interactions per period -> sparse matrix
          -> recommender -> top-k lists -> ranking metrics

From a notebook in src/notebooks:

    import sys; sys.path.append("../scripts")
    import dex_swaps as ds
    import pool_recsys as pr

    swaps = ds.load_swaps()
    legs = pr.trade_legs(swaps, ds.load_pool_tokens(swaps["pool"].unique()))   # tokens bought / sold
    history = pr.interactions(legs[legs["block"] < split], "user", "token")
    future = pr.interactions(legs[legs["block"] >= split], "user", "token")
    data = pr.build_task(history, future)             # matrices and index for one evaluation
    model = pr.ItemKNN().fit(data["history"])
    result = pr.evaluate(model, data, k=10)           # one row per evaluated user
    result[["recall", "ndcg"]].mean()

Every recommender has `fit(matrix)` and `scores(user_rows)`; `matrix` is a
users x items sparse matrix of the training period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds

# --------------------------------------------------------------------------
# From swaps to interactions
# --------------------------------------------------------------------------


def trade_legs(swaps: pd.DataFrame, pool_tokens: pd.DataFrame) -> pd.DataFrame:
    """Reduce every transaction to the tokens its sender ended up buying and selling.

    A routed trade passes through several pools: ETH -> USDC -> TOKEN shows
    as two swaps, but the trader only chose to sell ETH and buy TOKEN. Within
    a transaction, a token that is both received and paid out is an
    intermediate hop and is dropped. Returns one row per (transaction, token)
    with block, tx_hash, user (the signer), token and side ("buy" or "sell").
    Swaps on pools without known tokens are ignored.
    """
    tokens = pool_tokens.dropna().set_index("pool")
    known = swaps[swaps["pool"].isin(tokens.index)]
    token0 = known["pool"].map(tokens["token0"]).to_numpy()
    token1 = known["pool"].map(tokens["token1"]).to_numpy()
    pool_received_0 = (known["amount0"] > 0).to_numpy()  # the trader paid in token0
    base = known[["block", "tx_hash", "tx_from"]].rename(columns={"tx_from": "user"})
    sold = base.assign(token=np.where(pool_received_0, token0, token1), side="sell")
    bought = base.assign(token=np.where(pool_received_0, token1, token0), side="buy")
    legs = pd.concat([sold, bought], ignore_index=True).drop_duplicates(["tx_hash", "token", "side"])
    # A token on both sides of one transaction was only passed through.
    both_sides = legs.duplicated(["tx_hash", "token"], keep=False)
    return legs[~both_sides].sort_values(["block", "tx_hash"], ignore_index=True)


def interactions(events: pd.DataFrame, user: str, item: str) -> pd.DataFrame:
    """Count how often each user touched each item: columns user, item, count, first_block, last_block."""
    grouped = events.groupby([user, item], sort=False)["block"].agg(["size", "min", "max"])
    grouped = grouped.reset_index()
    grouped.columns = ["user", "item", "count", "first_block", "last_block"]
    return grouped


def to_matrix(pairs: pd.DataFrame, users: pd.Index, items: pd.Index, binary: bool = True):
    """Sparse users x items matrix from an interactions frame; unknown users or items are dropped."""
    rows, cols = users.get_indexer(pairs["user"]), items.get_indexer(pairs["item"])
    keep = (rows >= 0) & (cols >= 0)
    values = np.ones(keep.sum()) if binary else pairs["count"].to_numpy(float)[keep]
    shape = (len(users), len(items))
    return sparse.csr_matrix((values, (rows[keep], cols[keep])), shape=shape, dtype=np.float32)


def build_task(history: pd.DataFrame, future: pd.DataFrame, min_history: int = 1) -> dict:
    """Set up one evaluation: who is recommended to, what is known, what counts as a hit.

    `history` and `future` are interaction frames of two consecutive periods.
    Users and items are indexed from the history only, so nothing about the
    future enters a model. Returns a dict with:
      users, items       the index of the history matrix
      history            users x items matrix of the history (binary)
      new, repeat        users x items matrices of the future, split into
                         items the user had not used before and items it had
      cold_new           per user, the number of future items that did not
                         exist in the history at all (they cannot be recommended)
      eval_users         row numbers of users with at least `min_history`
                         history items and at least one new future item
    """
    users, items = pd.Index(history["user"].unique()), pd.Index(history["item"].unique())
    past = to_matrix(history, users, items)
    known_future = future[future["user"].isin(users)]
    ahead = to_matrix(known_future, users, items)
    repeat = ahead.multiply(past).tocsr()
    new = (ahead - repeat).tocsr()
    new.eliminate_zeros()

    cold = known_future[~known_future["item"].isin(items)].groupby("user").size()
    cold_new = np.zeros(len(users))
    cold_new[users.get_indexer(cold.index)] = cold.to_numpy()

    n_history = np.asarray(past.sum(axis=1)).ravel()
    n_new = np.asarray(new.sum(axis=1)).ravel() + cold_new
    eval_users = np.flatnonzero((n_history >= min_history) & (n_new > 0))
    return {
        "users": users, "items": items, "history": past, "new": new, "repeat": repeat,
        "cold_new": cold_new, "eval_users": eval_users,
    }  # fmt: skip


# --------------------------------------------------------------------------
# Recommenders
# --------------------------------------------------------------------------


class Popularity:
    """Recommend the items with the most users. The same list for everyone.

    Fit it on a matrix of the most recent days only to get "recent popularity".
    """

    def fit(self, matrix) -> Popularity:
        self.item_scores = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float32)
        return self

    def scores(self, user_rows) -> np.ndarray:
        return np.tile(self.item_scores, (user_rows.shape[0], 1))


class OwnHistory:
    """Recommend what the user already used, most used first; ties and the rest by popularity.

    Only meaningful when repeats count as hits (`target="all"`). Fit it on a
    matrix of interaction counts, not a binary one, to rank by frequency.
    """

    def fit(self, matrix) -> OwnHistory:
        popularity = np.asarray((matrix > 0).sum(axis=0)).ravel().astype(np.float32)
        self.tie_break = popularity / (popularity.max() + 1) * 0.5  # always below one use
        return self

    def scores(self, user_rows) -> np.ndarray:
        return np.asarray(user_rows.todense(), dtype=np.float32) + self.tie_break


class ItemKNN:
    """Item-based nearest neighbours: recommend items co-used with the user's items.

    Similarity between two items is the cosine of their user columns, with
    `shrink` added to the denominator to distrust items with few users. Only
    the `k` most similar items of each item are kept.
    """

    def __init__(self, k: int = 50, shrink: float = 10.0):
        self.k, self.shrink = k, shrink

    def fit(self, matrix) -> ItemKNN:
        matrix = matrix.tocsc()
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel())
        co_use = (matrix.T @ matrix).tocoo()
        off_diagonal = co_use.row != co_use.col
        rows, cols = co_use.row[off_diagonal], co_use.col[off_diagonal]
        values = co_use.data[off_diagonal] / (norms[rows] * norms[cols] + self.shrink)
        similarity = sparse.csr_matrix((values, (rows, cols)), shape=co_use.shape, dtype=np.float32)
        self.similarity = _keep_top_k_per_row(similarity, self.k)
        return self

    def scores(self, user_rows) -> np.ndarray:
        return np.asarray((user_rows @ self.similarity).todense(), dtype=np.float32)


class PureSVD:
    """Truncated SVD of the interaction matrix; a user's scores are its row in the low-rank copy."""

    def __init__(self, rank: int = 64, seed: int = 0):
        self.rank, self.seed = rank, seed

    def fit(self, matrix) -> PureSVD:
        rank = min(self.rank, min(matrix.shape) - 1)
        start = np.random.default_rng(self.seed).normal(size=min(matrix.shape))
        _, _, vt = svds(matrix.astype(np.float64), k=rank, v0=start)
        self.item_factors = vt.T.astype(np.float32)  # items x rank
        return self

    def scores(self, user_rows) -> np.ndarray:
        return np.asarray((user_rows @ self.item_factors) @ self.item_factors.T, dtype=np.float32)


class Blend:
    """Weighted sum of two recommenders' scores, each scaled to a maximum of 1 per user."""

    def __init__(self, first, second, weight: float = 0.5):
        self.first, self.second, self.weight = first, second, weight

    def fit(self, matrix) -> Blend:  # the parts are fitted by the caller
        return self

    def scores(self, user_rows) -> np.ndarray:
        def scaled(model):
            values = model.scores(user_rows)
            return values / np.maximum(values.max(axis=1, keepdims=True), 1e-12)

        return self.weight * scaled(self.first) + (1 - self.weight) * scaled(self.second)


def _keep_top_k_per_row(matrix, k: int):
    """Zero all but the k largest entries of each row of a sparse matrix."""
    matrix = matrix.tocsr()
    data, indices, indptr = [], [], [0]
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        values, columns = matrix.data[start:end], matrix.indices[start:end]
        if len(values) > k:
            top = np.argpartition(values, -k)[-k:]
            values, columns = values[top], columns[top]
        data.append(values)
        indices.append(columns)
        indptr.append(indptr[-1] + len(values))
    return sparse.csr_matrix(
        (np.concatenate(data), np.concatenate(indices), indptr), shape=matrix.shape, dtype=np.float32
    )


# --------------------------------------------------------------------------
# Ranking and metrics
# --------------------------------------------------------------------------


def top_k(scores: np.ndarray, k: int, exclude=None) -> np.ndarray:
    """The k best item numbers per row, best first; `exclude` is a sparse mask of items to skip."""
    scores = scores.astype(np.float32, copy=True)
    if exclude is not None:
        rows, cols = exclude.nonzero()
        scores[rows, cols] = -np.inf
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def ranking_metrics(recommended: np.ndarray, relevant, extra_relevant=None) -> pd.DataFrame:
    """Recall, NDCG and hit per user for top-k lists against the truly relevant items.

    `relevant` is a sparse users x items matrix aligned with the rows of
    `recommended`. `extra_relevant` counts relevant items per user that could
    not be recommended at all (for example items that did not exist yet); they
    lower recall and NDCG, as they should.
    """
    n_users, k = recommended.shape
    hits = np.asarray(relevant[np.arange(n_users)[:, None], recommended].todense()) > 0
    n_relevant = np.asarray(relevant.sum(axis=1)).ravel()
    if extra_relevant is not None:
        n_relevant = n_relevant + np.asarray(extra_relevant)
    discounts = 1 / np.log2(np.arange(2, k + 2))
    dcg = (hits * discounts).sum(axis=1)
    ideal = np.array([discounts[: int(min(n, k))].sum() for n in n_relevant])
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(n_relevant > 0, hits.sum(axis=1) / n_relevant, np.nan)
        ndcg = np.where(ideal > 0, dcg / ideal, np.nan)
    return pd.DataFrame({"recall": recall, "ndcg": ndcg, "hit": hits.any(axis=1).astype(float)})


def evaluate(model, task: dict, k: int = 10, target: str = "new", batch: int = 2000) -> pd.DataFrame:
    """Recommend to every evaluation user of a task and score the lists.

    `target="new"` ranks only items the user has not used and counts items new
    to the user as hits. `target="all"` ranks every item and counts any item
    used in the future. Returns one row per evaluated user with recall, ndcg,
    hit, the user's number of history items, and the recommended items in the
    attribute `.attrs["recommended"]` (for coverage).
    """
    history, users = task["history"], task["eval_users"]
    if target == "new":
        relevant, extra = task["new"], task["cold_new"]
    elif target == "all":
        relevant, extra = (task["new"] + task["repeat"]).tocsr(), task["cold_new"]
    else:
        raise ValueError(f"unknown target {target!r}")
    parts, lists = [], []
    for start in range(0, len(users), batch):
        rows = users[start : start + batch]
        exclude = history[rows] if target == "new" else None
        recommended = top_k(model.scores(history[rows]), k, exclude)
        parts.append(ranking_metrics(recommended, relevant[rows], extra[rows]))
        lists.append(recommended)
    result = pd.concat(parts, ignore_index=True)
    result["n_history"] = np.asarray(history[users].sum(axis=1)).ravel()
    result.index = task["users"][users]
    result.attrs["recommended"] = np.vstack(lists)
    return result


def coverage(recommended: np.ndarray, n_items: int) -> float:
    """Share of all items that appear in at least one recommendation list."""
    return len(np.unique(recommended)) / n_items


def bootstrap_mean(values, n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean of per-user values (NaNs ignored)."""
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(low), float(high)
