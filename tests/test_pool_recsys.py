"""Tests for src/scripts/pool_recsys.py."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import pool_recsys as pr


def swap(block, tx, pool, amount0, signer):
    return {"block": block, "tx_hash": tx, "pool": pool, "amount0": amount0, "tx_from": signer}


POOLS = pd.DataFrame({
    "pool": ["eth_usdc", "usdc_tok", "eth_tok", "broken"],
    "token0": ["ETH", "USDC", "ETH", None],
    "token1": ["USDC", "TOK", "TOK", None],
})  # fmt: skip


def test_trade_legs_single_swap():
    # The pool received token0 (ETH): the trader sold ETH and bought TOK.
    legs = pr.trade_legs(pd.DataFrame([swap(1, "a", "eth_tok", 5.0, "u")]), POOLS)
    assert set(zip(legs["token"], legs["side"])) == {("ETH", "sell"), ("TOK", "buy")}
    assert legs["user"].unique().tolist() == ["u"]


def test_trade_legs_drops_the_intermediate_token_of_a_route():
    # ETH -> USDC -> TOK in one transaction: USDC is only passed through.
    swaps = pd.DataFrame([swap(1, "a", "eth_usdc", 5.0, "u"), swap(1, "a", "usdc_tok", 9.0, "u")])
    legs = pr.trade_legs(swaps, POOLS)
    assert set(zip(legs["token"], legs["side"])) == {("ETH", "sell"), ("TOK", "buy")}


def test_trade_legs_keeps_separate_transactions_apart_and_skips_unknown_pools():
    swaps = pd.DataFrame([
        swap(1, "a", "eth_usdc", 5.0, "u"),     # sells ETH, buys USDC
        swap(2, "b", "usdc_tok", 9.0, "u"),     # sells USDC, buys TOK, in another transaction
        swap(3, "c", "broken", 1.0, "u"),
    ])  # fmt: skip
    legs = pr.trade_legs(swaps, POOLS)
    assert len(legs) == 4 and "c" not in set(legs["tx_hash"])
    assert set(legs.loc[legs["tx_hash"] == "b", "token"]) == {"USDC", "TOK"}


def test_trade_legs_arbitrage_cycle_leaves_nothing():
    # ETH -> USDC -> TOK -> ETH: every token is on both sides.
    swaps = pd.DataFrame([
        swap(1, "a", "eth_usdc", 5.0, "bot"), swap(1, "a", "usdc_tok", 9.0, "bot"), swap(1, "a", "eth_tok", -3.0, "bot"),
    ])  # fmt: skip
    assert pr.trade_legs(swaps, POOLS).empty


def test_interactions():
    events = pd.DataFrame({"user": ["u", "u", "v"], "pool": ["p", "p", "q"], "block": [5, 9, 7]})
    out = pr.interactions(events, "user", "pool").set_index(["user", "item"])
    assert out.loc[("u", "p")].tolist() == [2, 5, 9]
    assert out.loc[("v", "q")].tolist() == [1, 7, 7]


def test_to_matrix():
    pairs = pd.DataFrame({"user": ["u", "v", "w"], "item": ["a", "b", "a"], "count": [3, 1, 2]})
    users, items = pd.Index(["u", "v"]), pd.Index(["a", "b"])
    assert pr.to_matrix(pairs, users, items).toarray().tolist() == [[1, 0], [0, 1]]  # w is unknown
    assert pr.to_matrix(pairs, users, items, binary=False).toarray().tolist() == [[3, 0], [0, 1]]


def frame(pairs):
    return pd.DataFrame([{"user": u, "item": i, "count": 1, "first_block": 0, "last_block": 0} for u, i in pairs])


def test_build_task_separates_new_repeat_and_cold_items():
    history = frame([("u", "a"), ("u", "b"), ("v", "a"), ("w", "c")])
    future = frame([("u", "a"), ("u", "c"), ("u", "z"), ("v", "a"), ("x", "a")])   # z is a new item, x a new user
    task = pr.build_task(history, future)
    users, items = task["users"], task["items"]
    u = users.get_loc("u")
    assert task["repeat"][u, items.get_loc("a")] == 1 and task["new"][u, items.get_loc("a")] == 0
    assert task["new"][u, items.get_loc("c")] == 1
    assert task["cold_new"][u] == 1
    # Only u has something new; v only repeats, w is absent from the future, x has no history.
    assert task["users"][task["eval_users"]].tolist() == ["u"]
    assert "z" not in items and "x" not in users


def test_build_task_min_history():
    history = frame([("u", "a"), ("v", "a"), ("v", "b")])
    future = frame([("u", "b"), ("v", "c"), ("w", "c")])
    # v's only future item, c, did not exist in the history: it still counts as something new.
    task = pr.build_task(history, future, min_history=2)
    assert task["users"][task["eval_users"]].tolist() == ["v"]
    assert task["cold_new"][task["users"].get_loc("v")] == 1
    assert pr.build_task(history, future, min_history=3)["eval_users"].tolist() == []
    assert set(pr.build_task(history, future)["users"][pr.build_task(history, future)["eval_users"]]) == {"u", "v"}


MATRIX = sparse.csr_matrix(np.array([
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 1],
], dtype=np.float32))  # fmt: skip


def test_popularity():
    model = pr.Popularity().fit(MATRIX)
    assert model.item_scores.tolist() == [3, 2, 3, 2]
    assert model.scores(MATRIX[:2]).shape == (2, 4)


def test_own_history_ranks_used_items_first_then_popular_ones():
    counts = sparse.csr_matrix(np.array([[0, 5, 1, 0], [3, 0, 0, 0], [2, 0, 0, 0]], dtype=np.float32))
    model = pr.OwnHistory().fit(counts)
    # User 0 used item 1 five times and item 2 once; item 0 is the most popular of the rest.
    assert pr.top_k(model.scores(counts[:1]), 3).tolist() == [[1, 2, 0]]


def test_top_k_orders_and_excludes():
    scores = np.array([[0.1, 0.9, 0.5, 0.3], [0.4, 0.2, 0.8, 0.6]])
    assert pr.top_k(scores, 2).tolist() == [[1, 2], [2, 3]]
    exclude = sparse.csr_matrix(np.array([[0, 1, 0, 0], [0, 0, 0, 0]]))
    assert pr.top_k(scores, 2, exclude).tolist() == [[2, 3], [2, 3]]
    assert pr.top_k(scores, 10).shape == (2, 4)


def test_item_knn_recommends_co_used_items():
    model = pr.ItemKNN(k=2, shrink=0.0).fit(MATRIX)
    # A user who only has item 0 should get item 1 (co-used twice) before items 2 and 3.
    user = sparse.csr_matrix(np.array([[1, 0, 0, 0]], dtype=np.float32))
    assert pr.top_k(model.scores(user), 1, exclude=user).tolist() == [[1]]
    # A user with item 3 should get item 2.
    user = sparse.csr_matrix(np.array([[0, 0, 0, 1]], dtype=np.float32))
    assert pr.top_k(model.scores(user), 1, exclude=user).tolist() == [[2]]
    assert model.similarity.diagonal().sum() == 0
    assert (np.diff(model.similarity.indptr) <= 2).all()


def test_pure_svd_recovers_block_structure():
    model = pr.PureSVD(rank=2).fit(MATRIX)
    user = sparse.csr_matrix(np.array([[0, 0, 0, 1]], dtype=np.float32))
    assert pr.top_k(model.scores(user), 1, exclude=user).tolist() == [[2]]
    assert model.item_factors.shape == (4, 2)


def test_blend():
    first, second = pr.Popularity().fit(MATRIX), pr.ItemKNN(k=2, shrink=0.0).fit(MATRIX)
    user = sparse.csr_matrix(np.array([[0, 0, 0, 1]], dtype=np.float32))
    only_first = pr.Blend(first, second, weight=1.0).fit(MATRIX).scores(user)
    assert only_first[0] == pytest.approx(first.scores(user)[0] / 3)
    only_second = pr.Blend(first, second, weight=0.0).scores(user)
    assert pr.top_k(only_second, 1, exclude=user).tolist() == [[2]]


def test_ranking_metrics():
    recommended = np.array([[0, 1, 2], [3, 2, 1]])
    relevant = sparse.csr_matrix(np.array([[1, 0, 1, 0], [0, 0, 0, 0]]))
    out = pr.ranking_metrics(recommended, relevant)
    assert out.loc[0, "recall"] == pytest.approx(1.0) and out.loc[0, "hit"] == 1.0
    discounts = 1 / np.log2([2, 3, 4])
    assert out.loc[0, "ndcg"] == pytest.approx((discounts[0] + discounts[2]) / (discounts[0] + discounts[1]))
    assert np.isnan(out.loc[1, "recall"]) and out.loc[1, "hit"] == 0.0


def test_ranking_metrics_counts_items_that_could_not_be_recommended():
    recommended = np.array([[0, 1]])
    relevant = sparse.csr_matrix(np.array([[1, 0, 0]]))
    out = pr.ranking_metrics(recommended, relevant, extra_relevant=np.array([1.0]))
    assert out.loc[0, "recall"] == pytest.approx(0.5)


def test_evaluate_new_and_all_targets():
    history = frame([("u", "a"), ("v", "a"), ("v", "b"), ("w", "a"), ("w", "b")])
    future = frame([("u", "a"), ("u", "b")])
    task = pr.build_task(history, future)
    model = pr.ItemKNN(k=5, shrink=0.0).fit(task["history"])
    new = pr.evaluate(model, task, k=1, target="new")
    assert new.index.tolist() == ["u"] and new.loc["u", "recall"] == 1.0 and new.loc["u", "n_history"] == 1
    assert new.attrs["recommended"].tolist() == [[task["items"].get_loc("b")]]
    # With repeats allowed and one slot, popularity puts the already used item first.
    everything = pr.evaluate(pr.Popularity().fit(task["history"]), task, k=1, target="all")
    assert everything.loc["u", "recall"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        pr.evaluate(model, task, target="other")


def test_coverage_and_bootstrap_mean():
    assert pr.coverage(np.array([[0, 1], [1, 2]]), n_items=6) == pytest.approx(0.5)
    values = np.r_[np.random.default_rng(0).normal(0.3, 0.1, 500), np.nan]
    low, high = pr.bootstrap_mean(values, n_boot=300)
    assert low < 0.3 < high and high - low < 0.05
