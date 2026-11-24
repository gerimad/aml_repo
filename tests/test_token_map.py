"""Tests for src/scripts/token_map.py (WP12), on small hand-made markets."""

import numpy as np
import pandas as pd

import protocol as pc
import token_map as tm

T = pd.Timestamp("2026-07-10", tz="UTC")
DAY = pd.Timedelta(days=1)
HOUR = pd.Timedelta(hours=1)
SETTINGS = {"k": 10, "shrink": 0.0, "weight": 0.5}


def legs(rows):
    """Legs from (time, user, token) triples."""
    return pd.DataFrame([{"timestamp": when, "user": u, "token": i} for when, u, i in rows])


# Accounts 0-9 trade tokens 10-13 together; accounts 20-29 trade tokens 20-23 together.
TWO_GROUPS = legs([(T - DAY, u, tok) for u in range(10) for tok in range(10, 14)] + [(T - DAY, u, tok) for u in range(20, 30) for tok in range(20, 24)])


def test_two_disjoint_groups_of_co_traded_tokens_form_two_clusters():
    graph = tm.similarity_graph(pc.make_task(TWO_GROUPS, T), SETTINGS)
    cluster = tm.clusters(graph)
    assert cluster.loc[[10, 11, 12, 13]].nunique() == 1 and cluster.loc[[20, 21, 22, 23]].nunique() == 1
    assert cluster.loc[10] != cluster.loc[20]


def test_the_graph_is_symmetric_and_has_no_self_links():
    graph = tm.similarity_graph(pc.make_task(TWO_GROUPS, T), SETTINGS)
    assert abs(graph.matrix - graph.matrix.T).max() == 0
    assert graph.matrix.diagonal().sum() == 0


def test_related_never_returns_the_token_itself_or_skipped_tokens():
    graph = tm.similarity_graph(pc.make_task(TWO_GROUPS, T), SETTINGS)
    assert 10 not in tm.related(graph, 10)
    assert set(tm.related(graph, 10)) == {11, 12, 13}
    assert tm.related(graph, 10, skip={11}) and 11 not in tm.related(graph, 10, skip={11})
    assert tm.related(graph, 999) == []


def test_edit_distance_ignores_case():
    assert tm.edit_distance("PEPE", "pepe") == 0
    assert tm.edit_distance("PEPE", "PEPE2") == 1
    assert tm.edit_distance("DOGE", "PEPE") == 3


def test_copycats_are_pairs_in_one_cluster_with_close_symbols():
    cluster_of = pd.Series({1: 0, 2: 0, 3: 0, 4: 1})
    symbols = pd.Series({1: "PEPE", 2: "Pepe2", 3: "DOGE", 4: "PEPE"})
    pairs = tm.copycats(cluster_of, symbols)
    assert pairs[["token_a", "token_b"]].values.tolist() == [[1, 2]], "4 is in another cluster; DOGE is far"


def test_next_token_pairs_and_hits():
    # Account 0 traded 10 before t; in the target it picks up 11 first, then 12. Token 13 is the most popular.
    rows = [(T - DAY, u, tok) for u in range(1, 10) for tok in (11, 12)] + [(T - DAY, u, 13) for u in range(30, 60)]
    rows += [(T - DAY, 0, 10), (T + HOUR, 0, 11), (T + 2 * HOUR, 0, 12)]
    task = pc.make_task(legs(rows), T)
    pairs = tm.next_new_pairs(legs(rows), task)
    assert pairs.values.tolist() == [[0, 11, 12]]
    hits = tm.next_token_hits(pairs, tm.similarity_graph(task, SETTINGS), task, n=1)
    assert hits["related_hit"].tolist() == [True] and hits["popular_hit"].tolist() == [False]
