"""Token similarity map (work package 12, implementation_plan_applications.md section 12).

Tokens are similar when the same accounts trade them. The similarity is the
frozen neighbour model's (item-based cosine on the 14-day window, k = 400,
shrink = 200), made symmetric: s_ij = (S_ij + S_ji) / 2, since the model keeps
only each token's k most similar tokens and so is not symmetric.

  similarity_graph   the symmetric similarity among tokens with at least
                     min_accounts accounts in the window (sparse, token ids)
  clusters           Louvain communities of that graph (networkx, weighted)
  layout             2-D t-SNE of the similarity rows of the n largest tokens
  related            a token's n most similar tokens
  copycats           pairs of tokens in one cluster whose symbols are equal or
                     one edit apart (case ignored)
  next_token_hits    the check of 12.2: is an account's next new token among
                     the related tokens of the new token before it?
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

import expectedness as ex
import protocol as pc

MIN_ACCOUNTS = 20
RESOLUTIONS = [0.5, 1.0, 2.0, 4.0]
N_RELATED = 10


@dataclass
class Graph:
    """A symmetric token similarity matrix and the token id of each row/column."""

    tokens: np.ndarray
    matrix: sparse.csr_matrix
    accounts: np.ndarray  # accounts per token in the 14-day window


def symmetric(similarity: sparse.csr_matrix) -> sparse.csr_matrix:
    """(S + S^T) / 2 without the diagonal."""
    matrix = ((similarity + similarity.T) / 2).tocsr()
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    return matrix


def similarity_graph(task: pc.Task, settings: dict = ex.FROZEN_BLEND, min_accounts: int = 0) -> Graph:
    """The frozen neighbour similarity at the task's point, symmetric, over tokens with >= min_accounts accounts."""
    _, similarity, window = ex.blend_model(task, settings)
    accounts = np.asarray(window.sum(axis=0)).ravel()
    keep = np.flatnonzero(accounts >= min_accounts)
    matrix = symmetric(similarity)[keep][:, keep].tocsr()
    return Graph(task.tokens[keep], matrix, accounts[keep])


def clusters(graph: Graph, resolution: float = 1.0, seed: int = 0) -> pd.Series:
    """Louvain cluster of every token with at least one link (index: token id), numbered from the largest."""
    coo = sparse.triu(graph.matrix, k=1).tocoo()
    g = nx.Graph()
    g.add_weighted_edges_from(zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()))
    communities = nx.community.louvain_communities(g, weight="weight", resolution=resolution, seed=seed)
    communities = sorted(communities, key=len, reverse=True)
    label = {node: c for c, members in enumerate(communities) for node in members}
    nodes = np.array(sorted(label))
    return pd.Series([label[n] for n in nodes], index=pd.Index(graph.tokens[nodes], name="token"), name="cluster")


def stability(first: pd.Series, second: pd.Series) -> float:
    """Adjusted Rand index of two clusterings on the tokens they share (nan if fewer than 2)."""
    from sklearn.metrics import adjusted_rand_score

    common = first.index.intersection(second.index)
    if len(common) < 2:
        return np.nan
    return float(adjusted_rand_score(first.loc[common], second.loc[common]))


def layout(graph: Graph, n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """2-D t-SNE of the similarity rows of the n tokens with the most accounts: token, x, y, accounts."""
    from sklearn.manifold import TSNE

    top = np.argsort(-graph.accounts, kind="stable")[:n]
    rows = graph.matrix[top][:, top].toarray()
    rows[np.arange(len(top)), np.arange(len(top))] = rows.max(axis=1).clip(min=1e-6)  # each token closest to itself
    points = TSNE(n_components=2, metric="cosine", init="random", perplexity=min(30, max(2, len(top) // 4)), random_state=seed).fit_transform(rows + 1e-9)
    return pd.DataFrame({"token": graph.tokens[top], "x": points[:, 0], "y": points[:, 1], "accounts": graph.accounts[top]})


def related(graph: Graph, token, n: int = N_RELATED, skip=()) -> list:
    """The n tokens most similar to `token` (ids), best first; never the token itself nor any in `skip`."""
    where = np.searchsorted(graph.tokens, token)
    if where >= len(graph.tokens) or graph.tokens[where] != token:
        return []
    start, end = graph.matrix.indptr[where], graph.matrix.indptr[where + 1]
    columns, values = graph.matrix.indices[start:end], graph.matrix.data[start:end]
    order = columns[np.lexsort((columns, -values))]
    out, skip = [], set(skip) | {token}
    for c in order:
        candidate = graph.tokens[c]
        if candidate not in skip:
            out.append(candidate)
            if len(out) == n:
                break
    return out


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two symbols, ignoring case."""
    a, b = a.lower(), b.lower()
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def copycats(cluster_of: pd.Series, symbols: pd.Series, max_edit: int = 1) -> pd.DataFrame:
    """Pairs of tokens in one cluster whose symbols are at most max_edit edits apart: token_a, token_b, cluster, distance.

    symbols: symbol per token id; tokens without a readable symbol are skipped.
    """
    rows = []
    named = cluster_of[cluster_of.index.isin(symbols.dropna().index)]
    for c, members in named.groupby(named):
        ids = members.index.to_numpy()
        names = [str(symbols.loc[i]) for i in ids]
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                if abs(len(names[x]) - len(names[y])) > max_edit:
                    continue
                d = edit_distance(names[x], names[y])
                if d <= max_edit:
                    rows.append({"token_a": ids[x], "token_b": ids[y], "cluster": int(c), "distance": d})
    return pd.DataFrame(rows, columns=["token_a", "token_b", "cluster", "distance"])


# --------------------------------------------------------------------------
# The check of 12.2: related tokens and the next new token
# --------------------------------------------------------------------------


def next_new_pairs(legs: pd.DataFrame, task: pc.Task) -> pd.DataFrame:
    """For each account with a past before t: its first new universe token in [t, t + 2 days) and the next one
    strictly after it (user, token_i, token_j). New: never traded before t. Cold tokens are left out."""
    t = task.t
    target = legs[(legs["timestamp"] >= t) & (legs["timestamp"] < t + pc.HORIZON)]
    first = target.groupby(["user", "token"], observed=True)["timestamp"].min().reset_index()
    first = first[first["user"].isin(task.users) & first["token"].isin(task.tokens)]
    rows, cols = np.searchsorted(task.users, first["user"]), np.searchsorted(task.tokens, first["token"])
    past = np.asarray(task.past[rows, cols]).ravel() > 0
    first = first[~past].sort_values(["user", "timestamp", "token"])
    pairs = []
    for user, part in first.groupby("user"):
        times, tokens = part["timestamp"].to_numpy(), part["token"].to_numpy()
        later = np.flatnonzero(times > times[0])
        if len(later):
            pairs.append((user, tokens[0], tokens[later[0]]))
    return pd.DataFrame(pairs, columns=["user", "token_i", "token_j"])


def next_token_hits(pairs: pd.DataFrame, graph: Graph, task: pc.Task, n: int = N_RELATED) -> pd.DataFrame:
    """For each pair: is token_j among token_i's n related tokens, and among the n most popular tokens?

    Both lists leave out the tokens the account traded before t and token_i
    itself, so neither wastes places on tokens the account cannot pick up.
    Popularity: accounts in the 14-day window.
    """
    window = task.history(pc.REFERENCE_DAYS)
    popularity = np.asarray(window.sum(axis=0)).ravel()
    by_popularity = task.tokens[np.argsort(-popularity, kind="stable")]
    past = task.past
    out = []
    for user, token_i, token_j in pairs.itertuples(index=False):
        row = np.searchsorted(task.users, user)
        own = set(task.tokens[past.indices[past.indptr[row] : past.indptr[row + 1]]].tolist()) | {token_i}
        related_list = related(graph, token_i, n, skip=own)
        popular_list = []
        for candidate in by_popularity:
            if candidate not in own:
                popular_list.append(candidate)
                if len(popular_list) == n:
                    break
        out.append({"user": user, "token_i": token_i, "token_j": token_j, "related_hit": token_j in related_list,
                    "popular_hit": token_j in popular_list, "has_related": len(related_list) > 0})  # fmt: skip
    return pd.DataFrame(out)
