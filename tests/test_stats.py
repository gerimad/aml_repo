"""Tests for src/scripts/stats.py."""

import numpy as np
import pandas as pd
import pytest

import stats


def per_point(values, points=None):
    points = points if points is not None else range(len(values))
    return pd.DataFrame({"point": list(points), "recall": values})


def test_paired_differences_are_taken_point_by_point():
    model, baseline = per_point([0.3, 0.2, 0.4]), per_point([0.1, 0.25, 0.4])
    assert stats.paired_differences(model, baseline).round(3).tolist() == [0.2, -0.05, 0.0]


def test_paired_differences_refuse_different_points():
    with pytest.raises(ValueError):
        stats.paired_differences(per_point([0.1, 0.2]), per_point([0.1, 0.2], points=[0, 5]))


def test_bootstrap_of_constant_differences_has_zero_width():
    assert stats.bootstrap_over_points([0.02] * 14) == pytest.approx((0.02, 0.02))


def test_bootstrap_interval_contains_the_mean():
    d = np.random.default_rng(1).normal(0.03, 0.02, 14)
    low, high = stats.bootstrap_over_points(d)
    assert low < d.mean() < high


def test_sign_test_needs_12_wins_of_14():
    assert stats.sign_test([1] * 12 + [-1] * 2)["p_value"] < 0.05
    assert stats.sign_test([1] * 11 + [-1] * 3)["p_value"] > 0.05


def test_sign_test_leaves_ties_out():
    result = stats.sign_test([1, 1, 0, -1])
    assert (result["wins"], result["losses"], result["ties"]) == (2, 1, 1)


def test_compare_puts_the_baseline_first_and_counts_wins():
    results = {"blend": per_point([0.3, 0.3, 0.1]), "popularity": per_point([0.2, 0.2, 0.2])}
    table = stats.compare(results, baseline="popularity").set_index("model")
    assert table.index.tolist() == ["popularity", "blend"]
    assert table.loc["blend", "wins"] == "2 of 3"
    assert table.loc["blend", "mean_difference"] == pytest.approx(0.1 / 3)


def test_success_needs_an_interval_above_zero():
    baseline = per_point([0.2] * 14)
    assert stats.success(per_point([0.23] * 14), baseline)["success"]
    mixed = per_point([0.3, 0.1] * 7)  # mean difference 0, interval around zero
    assert not stats.success(mixed, baseline)["success"]
