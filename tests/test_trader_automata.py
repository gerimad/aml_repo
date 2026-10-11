"""Tests for src/scripts/trader_automata.py."""

import numpy as np
import pandas as pd
import pytest

import trader_automata as ta


def swap(block, tx, pool, amount0, sender, signer, log=0):
    return {"block": block, "tx_index": 0, "log_index": log, "tx_hash": tx, "pool": pool,
            "amount0": amount0, "sender": sender, "tx_from": signer}  # fmt: skip


def toy_swaps():
    rows = []
    # A public router used by 60 accounts, one swap each.
    for i in range(60):
        rows.append(swap(100 + i, f"r{i}", "poolA", 1.0, "router", f"user{i}"))
    # A contract driven by a fleet of 55 accounts, 30 transactions each: not a router.
    for i in range(55 * 30):
        rows.append(swap(400 + i, f"f{i}", "poolA", 1.0, "fleet", f"fleet{i % 55}"))
    # A private contract used by one account.
    for i in range(6):
        rows.append(swap(200 + i, f"b{i}", "poolB", (-1.0) ** i, "botcontract", "operator"))
    # A caller used by five accounts: neither.
    for i in range(5):
        rows.append(swap(300 + i, f"m{i}", "poolA", 1.0, "middling", f"m{i}"))
    return pd.DataFrame(rows)


def test_classify_callers():
    callers = ta.classify_callers(toy_swaps())
    assert callers.loc["router", "kind"] == "public" and callers.loc["router", "n_signers"] == 60
    assert callers.loc["fleet", "kind"] == "unclear" and callers.loc["fleet", "txs_per_signer"] == 30
    assert callers.loc["botcontract", "kind"] == "private" and callers.loc["botcontract", "n_swaps"] == 6
    assert callers.loc["middling", "kind"] == "unclear"


def test_assign_traders():
    swaps = toy_swaps()
    events = ta.assign_traders(swaps, ta.classify_callers(swaps))
    assert len(events) == 66  # the "unclear" swaps are dropped
    bots = events[events["trader_kind"] == "bot"]
    assert set(bots["trader"]) == {"botcontract"}
    users = events[events["trader_kind"] == "router_user"]
    assert users["trader"].nunique() == 60 and "router" not in set(users["trader"])
    assert events["block"].is_monotonic_increasing


def test_add_symbols_gap_and_relation():
    events = pd.DataFrame([
        {"trader": "t", "block": 10, "tx_hash": "a", "pool": "p1", "amount0": 5.0},
        {"trader": "t", "block": 10, "tx_hash": "a", "pool": "p2", "amount0": 5.0},     # same tx, new pool
        {"trader": "t", "block": 10, "tx_hash": "b", "pool": "p2", "amount0": -5.0},    # same block, reversed
        {"trader": "t", "block": 13, "tx_hash": "c", "pool": "p2", "amount0": -1.0},    # within 5, same direction
        {"trader": "t", "block": 200, "tx_hash": "d", "pool": "p1", "amount0": 1.0},    # within 300, known pool
        {"trader": "t", "block": 900, "tx_hash": "e", "pool": "p3", "amount0": 1.0},    # later, new pool
    ])  # fmt: skip
    out = ta.add_symbols(events)
    assert out["gap"].tolist() == [-1, 0, 1, 2, 3, 4]
    assert out["relation"].tolist() == [-1, 3, 1, 0, 2, 3]
    assert out["symbol"].tolist() == [-1, 3, 5, 8, 14, 19]
    assert ta.symbol_name(5) == "same block / same pool, reversed"


def test_add_symbols_keeps_traders_apart_and_is_causal():
    events = pd.DataFrame([
        {"trader": "x", "block": 1, "tx_hash": "a", "pool": "p", "amount0": 1.0},
        {"trader": "y", "block": 2, "tx_hash": "b", "pool": "p", "amount0": 1.0},
        {"trader": "x", "block": 3, "tx_hash": "c", "pool": "p", "amount0": 1.0},
        {"trader": "y", "block": 4, "tx_hash": "d", "pool": "q", "amount0": 1.0},
    ])  # fmt: skip
    out = ta.add_symbols(events)
    assert out["symbol"].tolist() == [-1, -1, 2 * 4 + 0, 2 * 4 + 3]
    # Appending a later event must not change earlier symbols.
    more = pd.concat([events, pd.DataFrame([{"trader": "x", "block": 9, "tx_hash": "e", "pool": "q", "amount0": 1.0}])], ignore_index=True)
    assert ta.add_symbols(more)["symbol"].tolist()[:4] == out["symbol"].tolist()


def test_add_symbols_size_terciles_are_within_pool():
    events = pd.DataFrame({
        "trader": ["t"] * 6, "block": range(6), "tx_hash": list("abcdef"),
        "pool": ["small"] * 3 + ["big"] * 3, "amount0": [1.0, -2.0, 3.0, 1000.0, -2000.0, 3000.0],
    })  # fmt: skip
    assert ta.add_symbols(events)["size"].tolist() == [1, 2, 2, 1, 2, 2]


def test_symbol_column_and_sequences_by_trader():
    events = pd.DataFrame({
        "trader": ["x", "y", "x", "x"], "gap": [-1, -1, 2, 4], "relation": [-1, -1, 0, 3],
        "symbol": [-1, -1, 8, 19], "size": [0, 1, 2, 0],
    })  # fmt: skip
    assert ta.symbol_column(events, "gap").tolist() == [-1, -1, 2, 4]
    assert ta.symbol_column(events, "gap_relation_size").tolist() == [-1, -1, 26, 57]
    with pytest.raises(ValueError):
        ta.symbol_column(events, "other")
    sequences = ta.sequences_by_trader(events)
    assert list(sequences) == ["x"] and sequences["x"].tolist() == [8, 19]
    assert ta.sequences_by_trader(events, "relation")["x"].tolist() == [0, 3]
    assert max(ta.ALPHABETS.values()) == 60


def two_state_trace(n, seed):
    """A process with hidden memory: symbol 1 behaves differently after 0 than after 2."""
    rng = np.random.default_rng(seed)
    state, out = 0, []
    for _ in range(n):
        symbol = rng.choice(3, p=[0.7, 0.2, 0.1] if state == 0 else [0.1, 0.2, 0.7])
        out.append(symbol)
        state = 1 if symbol == 2 else (0 if symbol == 0 else state)
    return np.array(out)


def test_ngram_model_probabilities():
    model = ta.NGramModel(order=1, n_symbols=2, alpha=0.0).fit([0, 1, 0, 1, 0, 1])
    # After 0 always 1, after 1 always 0.
    assert np.exp(model.log_probs[0, 1]) == pytest.approx(1.0)
    assert np.exp(model.log_probs[1, 0]) == pytest.approx(1.0)
    order0 = ta.NGramModel(order=0, n_symbols=2, alpha=0.0).fit([0, 0, 0, 1])
    assert order0.mean_log_likelihood([0, 0]) == pytest.approx(np.log(0.75))


def test_fit_many_does_not_count_across_sequence_boundaries():
    model = ta.NGramModel(order=1, n_symbols=2, alpha=0.0).fit_many([[0, 0, 0], [1, 1, 1]])
    # Within the sequences 0 follows 0 and 1 follows 1; the join 0 -> 1 is not a transition.
    assert np.exp(model.log_probs[0, 0]) == pytest.approx(1.0)
    assert np.exp(model.log_probs[1, 1]) == pytest.approx(1.0)
    joined = ta.NGramModel(order=1, n_symbols=2, alpha=0.0).fit([0, 0, 0, 1, 1, 1])
    assert np.exp(joined.log_probs[0, 1]) == pytest.approx(1 / 3)

    pooled = ta.PdfaModel(window=3, n_symbols=3).fit_many([two_state_trace(3000, 0), two_state_trace(3000, 1)])
    assert pooled.n_states >= 3 and np.isfinite(pooled.mean_log_likelihood(two_state_trace(200, 2)))


def test_ngram_higher_order_fits_structured_data_better():
    train, test = two_state_trace(20000, 0), two_state_trace(5000, 1)
    scores = [ta.NGramModel(order, n_symbols=3).fit(train).mean_log_likelihood(test) for order in (0, 1, 2)]
    assert scores[0] < scores[1] < scores[2]


def test_ngram_smoothing_handles_unseen_symbols():
    model = ta.NGramModel(order=1, n_symbols=4, alpha=0.5).fit([0, 1, 0, 1])
    assert np.isfinite(model.mean_log_likelihood([3, 3, 2]))


def test_pdfa_model_recovers_hidden_structure():
    train, test = two_state_trace(20000, 0), two_state_trace(5000, 1)
    pdfa = ta.PdfaModel(window=5, n_symbols=3).fit(train)
    bigram = ta.NGramModel(order=1, n_symbols=3).fit(train)
    # A handful of states is enough, far fewer than the 3**4 contexts of a window.
    assert 4 <= pdfa.n_states <= 12
    # It captures the hidden memory that a bigram cannot.
    assert pdfa.mean_log_likelihood(test) > bigram.mean_log_likelihood(test)


def test_pdfa_describe_lists_states_and_likely_moves():
    pdfa = ta.PdfaModel(window=5, n_symbols=3).fit(two_state_trace(20000, 0))
    table = pdfa.describe(top=2)
    assert len(table) == pdfa.n_states
    assert table.loc[pdfa.initial, "entered_by"] == ta.PdfaModel.START
    for moves in table["next"]:
        assert len(moves) <= 2
        assert all(0 < prob <= 1 and 0 <= target < pdfa.n_states for _, prob, target in moves)
    # After a 0 the process most likely emits another 0 (probability 0.7).
    after_zero = table[table["entered_by"] == 0].iloc[0]["next"][0]
    assert after_zero[0] == 0 and after_zero[1] == pytest.approx(0.7, abs=0.03)


def test_pdfa_model_handles_unseen_symbols_and_short_input():
    pdfa = ta.PdfaModel(window=4, n_symbols=5).fit([0, 1, 0, 1, 0, 1, 0, 1])
    assert np.isfinite(pdfa.mean_log_likelihood([4, 4, 0, 1, 3]))
    assert ta.PdfaModel(window=4, n_symbols=5).fit([0, 1]).n_states >= 2
    assert pdfa.mean_log_likelihood([]) == 0.0


def test_pool_profile():
    profile = ta.PoolProfile().fit(["a", "a", "b"])
    assert profile.mean_log_likelihood(["a", "a", "b"]) == pytest.approx(1.0)
    assert profile.mean_log_likelihood(["c"]) == pytest.approx(0.0)
    assert 0 < profile.mean_log_likelihood(["a", "c"]) < 1


def test_identification_scores_and_metrics():
    # Three "traders" with clearly different habits.
    rng = np.random.default_rng(0)
    habits = {"a": [0.8, 0.1, 0.1], "b": [0.1, 0.8, 0.1], "c": [0.1, 0.1, 0.8]}
    enrolled = {k: rng.choice(3, 500, p=p) for k, p in habits.items()}
    queries = {k: rng.choice(3, 200, p=p) for k, p in habits.items()}
    scores = ta.identification_scores(enrolled, queries, lambda: ta.NGramModel(0, n_symbols=3))
    assert scores.shape == (3, 3)
    metrics = ta.identification_metrics(scores)
    assert metrics["top1"] == 1.0 and metrics["mrr"] == 1.0
    assert metrics["n_candidates"] == 3 and metrics["chance_top1"] == pytest.approx(1 / 3)


def test_identification_metrics_ranks_and_ties():
    scores = pd.DataFrame(
        [[0.9, 0.1, 0.0], [0.5, 0.2, 0.9], [0.3, 0.3, 0.3]],
        index=["a", "b", "c"], columns=["a", "b", "c"],
    )  # fmt: skip
    metrics = ta.identification_metrics(scores)
    # a is ranked 1st, b 3rd, c tied with everything and therefore 3rd.
    assert metrics["top1"] == pytest.approx(1 / 3)
    assert metrics["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 3) / 3)
    assert metrics["top5"] == 1.0


def test_identification_ranks():
    scores = pd.DataFrame([[0.9, 0.1], [0.8, 0.2]], index=["a", "b"], columns=["a", "b"])
    ranks = ta.identification_ranks(scores)
    assert ranks.to_dict() == {"a": 1, "b": 2}


def test_assign_groups_is_stable_and_roughly_proportional():
    keys = [f"0x{i:040x}" for i in range(3000)]
    groups = ta.assign_groups(keys)
    assert groups.equals(ta.assign_groups(keys[::-1]).reindex(groups.index))
    shares = groups.value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.6, abs=0.03)
    assert shares["dev"] == pytest.approx(0.2, abs=0.03)
    assert ta.assign_groups(["a", "a", "b"]).index.tolist() == ["a", "b"]
