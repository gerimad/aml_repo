"""Tests for src/scripts/ranker.py, on a random market with a known structure."""

import numpy as np
import pandas as pd
import pytest

import protocol as pc
import ranker as rk
from test_tuning import random_market

T = pd.Timestamp("2026-07-02", tz="UTC")
SETTINGS = {"k": 5, "shrink": 1.0, "weight": 0.5, "half_life": 3.0, "history_days": None, "candidates": 4}
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"


def market_with_sides(seed=0):
    legs = random_market(seed)
    legs["side"] = np.where(np.arange(len(legs)) % 3 == 0, "sell", "buy")
    return legs


def token_table(legs):
    """A token table as token_data writes it; every token launched against WETH (id 1000)."""
    first = legs.groupby("token")["timestamp"].min()
    traded = pd.DataFrame({
        "token_id": first.index, "address": [f"0x{i:040x}" for i in first.index],
        "first_timestamp": first.to_numpy(), "censored": False, "launch_base_id": 1000,
    })  # fmt: skip
    weth = pd.DataFrame({"token_id": [1000], "address": [WETH], "first_timestamp": pd.Series([pd.NaT], dtype=legs["timestamp"].dtype), "censored": [False], "launch_base_id": [-1]})
    return pd.concat([traded, weth], ignore_index=True)


def setup(seed=0):
    legs = market_with_sides(seed)
    task = pc.make_task(legs, T)
    stage = rk.FirstStage(task, SETTINGS)
    return legs, task, stage


def test_candidates_never_include_the_users_past_and_are_at_most_two_lists_long():
    legs, task, stage = setup()
    pairs = rk.candidates(stage, task.eval_rows)
    assert task.past[pairs["row"].to_numpy(), pairs["col"].to_numpy()].sum() == 0
    assert pairs.groupby("row").size().max() <= 2 * SETTINGS["candidates"]
    assert 0 <= rk.candidate_recall(task, pairs) <= 1


def test_features_have_every_feature_and_a_label():
    legs, task, stage = setup()
    table = rk.features(stage, rk.candidates(stage, task.eval_rows), legs, token_table(legs))
    assert set(rk.ALL_FEATURES) <= set(table.columns)
    assert table["label"].isin([0, 1]).all()
    assert (table["launch_weth"] == 1).all(), "every token in this market launched against WETH"


def test_no_feature_changes_when_the_future_is_deleted():
    legs, task, stage = setup()
    pairs = rk.candidates(stage, task.eval_rows)
    tokens = token_table(legs)
    with_future = rk.features(stage, pairs, legs, tokens)
    without_future = rk.features(stage, pairs, legs[legs["timestamp"] < T], tokens)
    pd.testing.assert_frame_equal(with_future[rk.ALL_FEATURES], without_future[rk.ALL_FEATURES])


def test_sample_negatives_keeps_every_positive():
    rows = pd.DataFrame({"point": [1] * 6, "user": [7] * 6, "label": [1, 0, 0, 0, 0, 1]})
    sampled = rk.sample_negatives(rows, per_user=2)
    assert sampled["label"].sum() == 2 and len(sampled) == 4


def test_point_folds_are_consecutive_pairs():
    assert rk.point_folds(list(range(10))) == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]


def test_recall_at_k_counts_missing_positives_as_misses():
    rows = pd.DataFrame({"point": [1, 1, 1], "user": [5, 5, 5], "label": [1, 0, 0]})
    positives = pd.Series([2], index=pd.MultiIndex.from_tuples([(1, 5)], names=["point", "user"]))
    # The positive candidate ranks first; the user's second positive is not a candidate.
    assert rk.recall_at_k(rows, np.array([0.9, 0.5, 0.1]), positives, k=1) == 0.5


@pytest.mark.parametrize("kind, params", [
    ("logistic", {"C": 1.0}),
    ("gbdt", {"num_leaves": 3, "learning_rate": 0.1, "min_data_in_leaf": 2}),
    ("lambdamart", {"num_leaves": 3, "learning_rate": 0.1, "min_data_in_leaf": 2}),
])  # fmt: skip
def test_every_model_kind_fits_predicts_and_plugs_into_the_protocol(kind, params):
    legs, task, stage = setup()
    rows = rk.features(stage, rk.candidates(stage, task.eval_rows), legs, token_table(legs))
    rows = rows.sort_values(["point", "user"], kind="stable").reset_index(drop=True)
    model = rk.fit_model(kind, rows, rk.ALL_FEATURES, params, n_trees=5)
    scorer = rk.ranker_scorer(model, rk.ALL_FEATURES, task, rows)
    per_user, lists = pc.evaluate_point(scorer, task)
    candidate_set = set(zip(rows["row"], rows["col"]))
    for row, listed in zip(task.eval_rows, lists):
        n_candidates = sum(1 for r, _ in candidate_set if r == row)
        assert all((row, c) in candidate_set for c in listed[: min(n_candidates, pc.K)]), "candidates come first"
    assert per_user["recall"].between(0, 1).all()


def test_cv_over_points_never_trains_on_a_held_out_point():
    tasks, all_rows = {}, []
    for i, t in enumerate(pc.POINTS["dev_a"][:4]):
        legs = market_with_sides(seed=i)
        legs["timestamp"] += t - T
        tasks[t] = pc.make_task(legs, t)
        stage = rk.FirstStage(tasks[t], SETTINGS)
        all_rows.append(rk.features(stage, rk.candidates(stage, tasks[t].eval_rows), legs, token_table(legs)))
    rows = pd.concat(all_rows, ignore_index=True)
    score = rk.cv_over_points("logistic", {"C": 1.0}, rows, rows, rk.positives_per_user(tasks), rk.ALL_FEATURES)
    assert 0 <= score <= 1


@pytest.mark.parametrize("kind, params", [
    ("logistic", {"C": 1.0}),
    ("gbdt", {"num_leaves": 3, "learning_rate": 0.1, "min_data_in_leaf": 2}),
])  # fmt: skip
def test_a_saved_model_predicts_the_same_after_loading(tmp_path, kind, params):
    legs, task, stage = setup()
    rows = rk.features(stage, rk.candidates(stage, task.eval_rows), legs, token_table(legs))
    model = rk.fit_model(kind, rows, rk.ALL_FEATURES, params, n_trees=5)
    rk.save_model(model, tmp_path / "model")
    again = rk.load_model(tmp_path / "model", kind)
    assert np.allclose(rk.predict(model, rows, rk.ALL_FEATURES), rk.predict(again, rows, rk.ALL_FEATURES))


def test_point_rows_cache_refuses_other_settings(tmp_path):
    legs, task, stage = setup()
    tokens = token_table(legs)
    first = rk.point_rows(legs, tokens, task, SETTINGS, cache_dir=tmp_path)
    again = rk.point_rows(legs, tokens, task, SETTINGS, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(first, again)
    with pytest.raises(ValueError, match="delete it to rebuild"):
        rk.point_rows(legs, tokens, task, {**SETTINGS, "k": 99}, cache_dir=tmp_path)
