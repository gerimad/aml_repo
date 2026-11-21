"""Tests for src/scripts/tuning.py and the time weighting in pool_recsys."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import pool_recsys as pr
import protocol as pc
import tuning

T = pd.Timestamp("2026-07-02", tz="UTC")
DAY = pd.Timedelta(days=1)


def random_market(seed=0, n_users=60, n_tokens=25):
    """Random trades in the 20 days before T and the 2 days after it."""
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(n_users):
        favourite = rng.integers(0, 5)  # gives the market some co-usage structure
        for _ in range(rng.integers(1, 6)):
            token = int((favourite * 5 + rng.integers(0, 6)) % n_tokens)
            rows.append({"timestamp": T - rng.uniform(0.01, 20) * DAY, "user": user, "token": token})
        for _ in range(rng.integers(0, 3)):
            token = int((favourite * 5 + rng.integers(0, 8)) % n_tokens)
            rows.append({"timestamp": T + rng.uniform(0, 1.99) * DAY, "user": user, "token": token})
    return pd.DataFrame(rows)


def test_blend_grid_gives_the_same_recall_as_blend_with_protocol():
    task = pc.make_task(random_market(), T)
    window = task.history(14)
    grid = tuning.blend_grid(task, window, window, ks=[3, 8], shrinks=[0.0, 5.0], weights=[0.0, 0.3, 1.0])
    for k, shrink, w in [(3, 0.0, 0.3), (8, 5.0, 0.0), (3, 5.0, 1.0)]:
        popularity = pr.Popularity().fit(window)
        neighbours = pr.ItemKNN(k=k, shrink=shrink).fit(window)
        blend = pr.Blend(popularity, neighbours, weight=w)
        per_user, _ = pc.evaluate_point(lambda rows: blend.scores(window[rows]), task)
        row = grid[(grid["k"] == k) & (grid["shrink"] == shrink) & (grid["weight"] == w)].iloc[0]
        assert row["recall"] == pytest.approx(per_user["recall"].mean()), f"setting k={k}, shrink={shrink}, w={w}"


def test_blend_with_weight_one_is_the_popularity_baseline():
    task = pc.make_task(random_market(seed=3), T)
    window = task.history(14)
    grid = tuning.blend_grid(task, window, window, ks=[5], shrinks=[1.0], weights=[1.0])
    baseline, _ = pc.evaluate_point(pc.popularity_scorer(task), task)
    assert grid["recall"].iloc[0] == pytest.approx(baseline["recall"].mean())


def test_best_setting_prefers_the_simpler_setting_among_near_ties():
    grid = pd.DataFrame({
        "k": [10, 50, 10, 50], "weight": [0.5, 0.5, 0.5, 0.5], "point": [1, 1, 2, 2],
        "recall": [0.200, 0.201, 0.200, 0.201],
    })  # fmt: skip
    best = tuning.best_setting(grid, ["k", "weight"], {"k": "small", "weight": "large"})
    assert best["k"] == 10, "k=50 is better by only 0.001, within the tolerance"
    clear = tuning.best_setting(grid.assign(recall=[0.2, 0.25, 0.2, 0.25]), ["k", "weight"], {"k": "small"})
    assert clear["k"] == 50


def test_grid_settings():
    assert tuning.grid_settings(a=[1, 2], b=["x"]) == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]


def test_time_weights_halve_at_the_half_life():
    ages = sparse.csr_matrix(np.array([[0.0, 7.0], [14.0, 0.0]]))
    ages[0, 0] = 1e-9  # an age of (almost) zero, stored explicitly
    weights = pr.time_weights(ages, half_life_days=7).toarray()
    assert weights[0, 0] == pytest.approx(1.0) and weights[0, 1] == pytest.approx(0.5)
    assert weights[1, 0] == pytest.approx(0.25) and weights[1, 1] == 0, "missing pairs stay missing"


def test_infinite_half_life_gives_the_binary_history():
    task = pc.make_task(random_market(seed=5), T)
    weights = pr.time_weights(task.history_ages(14), half_life_days=None)
    assert (weights != task.history(14)).nnz == 0
