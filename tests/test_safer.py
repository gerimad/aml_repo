"""Tests for src/scripts/safer.py (WP16)."""

import numpy as np
import pandas as pd

import expectedness as ex
import protocol as pc
import safer as sf

T = pd.Timestamp("2026-07-10", tz="UTC")
DAY = pd.Timedelta(days=1)
SETTINGS = {"k": 5, "shrink": 1.0, "weight": 0.5}


def market(seed=0, users=50, tokens=30, trades=600):
    rng = np.random.default_rng(seed)
    rows = [(T - DAY * float(rng.uniform(0.1, 13)), int(u), int(i)) for u, i in zip(rng.integers(0, users, trades), rng.integers(0, tokens, trades))]
    rows += [(T + DAY * float(rng.uniform(0, 1.9)), int(u), int(i)) for u, i in zip(rng.integers(0, users, 150), rng.integers(0, tokens, 150))]
    return pd.DataFrame(rows, columns=["timestamp", "user", "token"])


def test_lambda_0_gives_the_frozen_lists_exactly():
    task = pc.make_task(market(), T)
    columns, scores = sf.candidate_lists(task, depth=20, settings=SETTINGS)
    scorer, _, _ = ex.blend_model(task, SETTINGS)
    _, frozen = pc.evaluate_point(scorer, task)
    np.testing.assert_array_equal(sf.rerank(columns, scores, np.zeros(len(task.tokens)), 0.0), frozen)


def test_a_risky_token_with_a_large_lambda_leaves_every_list():
    task = pc.make_task(market(), T)
    columns, scores = sf.candidate_lists(task, depth=20, settings=SETTINGS)
    risky = columns[0, 0]
    risk = np.zeros(len(task.tokens))
    risk[risky] = 1.0
    assert (columns[:, :10] == risky).any()
    assert not (sf.rerank(columns, scores, risk, lam=10.0) == risky).any()


def test_reranking_only_reorders_the_first_depth_tokens():
    task = pc.make_task(market(), T)
    columns, scores = sf.candidate_lists(task, depth=15, settings=SETTINGS)
    risk = np.random.default_rng(1).random(len(task.tokens))
    lists = sf.rerank(columns, scores, risk, lam=5.0)
    for row in range(len(lists)):
        assert set(lists[row]) <= set(columns[row])


def test_outcomes_count_collapsed_entries_and_harmful_hits():
    task = pc.make_task(market(), T)
    columns, scores = sf.candidate_lists(task, depth=10, settings=SETTINGS)
    none = sf.list_outcomes(columns, task, np.zeros(len(task.tokens), dtype=bool))
    every = sf.list_outcomes(columns, task, np.ones(len(task.tokens), dtype=bool))
    assert none["collapsed_share"] == 0 and none["harmful_hit_share"] == 0
    assert every["collapsed_share"] == 1 and (every["harmful_hit_share"] == 1 or none["recall"] == 0)
