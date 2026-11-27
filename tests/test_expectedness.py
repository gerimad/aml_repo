"""Tests for src/scripts/expectedness.py (WP9), on small hand-made markets."""

import numpy as np
import pandas as pd
import pytest

import expectedness as ex
import protocol as pc

T = pd.Timestamp("2026-07-02", tz="UTC")
DAY = pd.Timedelta(days=1)


def legs(rows):
    """Legs from (time, user, token) triples."""
    return pd.DataFrame([{"timestamp": when, "user": u, "token": i} for when, u, i in rows])


def fixed_scorer(task, by_token):
    """A scorer giving every user the same score per token id (missing tokens score 0)."""
    values = np.array([by_token.get(int(tok), 0.0) for tok in task.tokens], dtype=np.float32)
    return lambda rows: np.tile(values, (len(rows), 1))


# User 1 traded token 10 before T; tokens 10-15 exist before T (traded by user 2).
MARKET = legs([(T - DAY, 1, 10)] + [(T - 2 * DAY, 2, tok) for tok in range(10, 16)] + [(T + DAY, 1, 12), (T + DAY, 1, 15)])


def test_the_first_choice_has_percentile_0_and_the_last_1():
    task = pc.make_task(MARKET, T)
    scorer = fixed_scorer(task, {11: 5, 12: 4, 13: 3, 14: 2, 15: 1})
    row = np.searchsorted(task.users, 1)
    cols = np.searchsorted(task.tokens, [11, 15])
    assert ex.rank_percentiles(task, scorer, np.array([row, row]), cols).tolist() == [0.0, 1.0]


def test_past_tokens_never_count_in_the_denominator():
    # Token 10 is user 1's past token with the highest score; it must not push 12 down.
    task = pc.make_task(MARKET, T)
    scorer = fixed_scorer(task, {10: 99, 11: 5, 12: 4, 13: 3, 14: 2, 15: 1})
    row = np.searchsorted(task.users, 1)
    percentile = ex.rank_percentiles(task, scorer, np.array([row]), np.searchsorted(task.tokens, [12]))
    assert percentile[0] == pytest.approx(1 / 4), "candidates are 11-15 (5 tokens); 12 is second: 1 / (5 - 1)"


def test_a_pair_is_a_hit_exactly_when_its_rank_is_below_10():
    # 30 tokens with distinct scores; user 1 picks up tokens ranked 3rd, 10th, 11th and 25th.
    rows = [(T - 2 * DAY, 2, tok) for tok in range(100, 130)] + [(T - DAY, 1, 100)]
    picks = [103, 110, 111, 125]
    rows += [(T + DAY, 1, tok) for tok in picks]
    task = pc.make_task(legs(rows), T)
    scorer = fixed_scorer(task, {tok: 1000 - tok for tok in range(100, 130)})  # 101 best, then 102, ...
    per_user, lists = pc.evaluate_point(scorer, task)
    hits_from_protocol = set(task.tokens[lists[per_user.index[per_user["user"] == 1][0]]]) & set(picks)
    row = np.searchsorted(task.users, 1)
    cols = np.searchsorted(task.tokens, picks)
    percentiles = ex.rank_percentiles(task, scorer, np.full(len(picks), row), cols)
    n_candidates = len(task.tokens) - 1
    ranks = np.round(percentiles * (n_candidates - 1)).astype(int)
    hits_from_ranks = {tok for tok, rank in zip(picks, ranks) if rank < 10}
    assert hits_from_ranks == hits_from_protocol == {103, 110}


def test_the_pair_table_has_every_positive_pair_and_marks_cold_tokens():
    market = pd.concat([MARKET, legs([(T + DAY, 1, 99)])], ignore_index=True)  # token 99 is born after T
    task = pc.make_task(market, T)
    tokens = pd.DataFrame({"token_id": [10, 11, 12, 13, 14, 15, 99],
                           "first_timestamp": [T - 2 * DAY] * 6 + [T + DAY]})  # fmt: skip
    table = ex.pair_table(market, task, tokens).set_index("token")
    assert sorted(table.index) == [12, 15, 99]
    assert table.loc[99, "cold"] and np.isnan(table.loc[99, "rank_percentile"])
    assert not table.loc[12, "cold"] and 0 <= table.loc[12, "rank_percentile"] <= 1
    assert table.loc[12, "token_age_days"] == pytest.approx(2.0)
    assert table.loc[12, "first_trade_time"] == T + DAY


def test_max_similarity_is_zero_without_a_co_trading_path():
    # Tokens 10 and 11 are co-traded by users 2 and 3; token 12 only by user 4.
    rows = [(T - DAY, u, tok) for u in (2, 3) for tok in (10, 11)] + [(T - DAY, 4, 12), (T - DAY, 1, 10)]
    rows += [(T + DAY, 1, 11), (T + DAY, 1, 12)]
    market = legs(rows)
    task = pc.make_task(market, T)
    _, similarity, window = ex.blend_model(task, {"k": 10, "shrink": 0.0, "weight": 0.5})
    row = np.searchsorted(task.users, 1)
    values = ex.max_similarity(similarity, window, np.array([row, row]), np.searchsorted(task.tokens, [11, 12]))
    assert values[0] > 0 and values[1] == 0


def test_trades_after_the_target_change_nothing():
    later = pd.concat([MARKET, legs([(T + 3 * DAY, 1, 13), (T + 5 * DAY, 3, 11)])], ignore_index=True)
    tokens = pd.DataFrame({"token_id": list(range(10, 16)), "first_timestamp": [T - 2 * DAY] * 6})
    a = ex.pair_table(MARKET, pc.make_task(MARKET, T), tokens)
    b = ex.pair_table(later, pc.make_task(later, T), tokens)
    pd.testing.assert_frame_equal(a, b)


def test_the_frozen_blend_scorer_equals_the_tuning_scorer():
    import tuning

    task = pc.make_task(MARKET, T)
    scorer, _, _ = ex.blend_model(task)
    rows = task.eval_rows
    np.testing.assert_allclose(scorer(rows), tuning.fixed_blend_scorer(task, ex.FROZEN_BLEND)(rows))
