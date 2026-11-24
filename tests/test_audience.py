"""Tests for src/scripts/audience.py (WP15), on small hand-made markets."""

import numpy as np
import pandas as pd

import audience as au
import expectedness as ex
import protocol as pc

T = pd.Timestamp("2026-07-02", tz="UTC")
DAY = pd.Timedelta(days=1)


def legs(rows):
    """Legs from (time, user, token) triples."""
    return pd.DataFrame([{"timestamp": when, "user": u, "token": i} for when, u, i in rows])


def random_market(seed=0, users=60, tokens=12, trades=400):
    rng = np.random.default_rng(seed)
    rows = [(T - int(rng.integers(1, 13)) * DAY, int(u), int(i)) for u, i in zip(rng.integers(0, users, trades), rng.integers(0, tokens, trades))]
    rows += [(T + DAY, int(u), int(i)) for u, i in zip(rng.integers(0, users, 80), rng.integers(0, tokens, 80))]
    return legs(rows)


SETTINGS = {"k": 5, "shrink": 1.0, "weight": 0.5}


def test_a_tokens_column_equals_the_blend_scores_minus_the_popularity_constant():
    task = pc.make_task(random_market(), T)
    scorer, _, _ = ex.blend_model(task, SETTINGS)
    rows = au.audience_rows(task)
    cols = np.arange(len(task.tokens))
    full = scorer(rows)
    raw, row_max = au.model_columns(task, rows, cols, SETTINGS)
    columns = raw / np.maximum(row_max, 1e-12)[:, None]
    constant = full - (1 - SETTINGS["weight"]) * columns
    np.testing.assert_allclose(constant, np.tile(constant[0], (len(rows), 1)), atol=1e-5)


def test_an_account_that_traded_the_token_before_is_never_in_its_audience():
    # User 1 traded token 10 before t and is the most active account; token 10 gets 10 buyers.
    rows = [(T - DAY, 1, tok) for tok in range(10, 30)] + [(T - DAY, u, 11) for u in range(100, 120)]
    rows += [(T + DAY, u, 10) for u in range(100, 110)]
    task = pc.make_task(legs(rows), T)
    table = au.point_table(task)
    assert set(table["token"]) == {10}
    # 20 accounts can be in token 10's audience; user 1 cannot, so it must never take a buyer's place:
    # with every one of the 19 others ranked before it, recall@1000 is 1 for every method.
    assert (table["recall_1000"] == 1).all()


def test_a_perfect_score_puts_every_buyer_first():
    scores = np.zeros((50, 2), dtype=np.float32)
    buyers = np.zeros((50, 2))
    buyers[[3, 7, 9], 0] = 1
    buyers[[1, 2], 1] = 1
    scores[buyers == 1] = 1.0
    order = au.rank_accounts(scores, np.zeros(50), k=5)
    metrics = au.audience_metrics(order, buyers, k_recall=5, k_precision=3)
    assert metrics["recall_5"].tolist() == [1.0, 1.0]
    assert metrics["precision_3"].tolist() == [1.0, 2 / 3]


def test_ties_are_broken_by_activity_not_by_account_id():
    scores = np.zeros((4, 1), dtype=np.float32)
    activity = np.array([1.0, 5.0, 3.0, 2.0])
    assert au.rank_accounts(scores, activity, k=4)[0].tolist() == [1, 2, 3, 0]


def test_excluded_accounts_leave_the_audience_and_the_buyers():
    rows = [(T - DAY, u, 11) for u in range(100, 130)] + [(T - DAY, 999, 10)] + [(T + DAY, u, 10) for u in range(100, 115)]
    task = pc.make_task(legs(rows), T)
    full = au.point_table(task)
    fewer = au.point_table(task, exclude_users=range(100, 104))
    assert full["buyers"].iloc[0] == 15 and fewer["buyers"].iloc[0] == 11
    assert fewer["audience"].iloc[0] == full["audience"].iloc[0] - 4
