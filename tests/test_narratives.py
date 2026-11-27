"""Tests for src/scripts/narratives.py (WP20)."""

import numpy as np
import pandas as pd

import narratives as nr

T1, T2 = pd.Timestamp("2026-07-10", tz="UTC"), pd.Timestamp("2026-07-12", tz="UTC")


def table():
    # two points, three tokens: tokens 1, 2 in narrative "a", token 3 in "b"
    return pd.DataFrame({
        "point": [T1, T1, T1, T2, T2, T2], "token": [1, 2, 3, 1, 2, 3],
        "list_count": [30, 10, 60, 50, 30, 20], "n_known": [1, 1, 2, 3, 1, 0],
        "popularity": [10, 10, 80, 10, 10, 80], "persistence": [0, 2, 2, 2, 2, 0],
    })  # fmt: skip


def test_shares_sum_to_one_per_point_and_follow_the_narratives():
    out = nr.shares(table(), pd.Series({1: "a", 2: "a", 3: "b"}))
    assert np.allclose(out.groupby("point")[nr.SHARES].sum(), 1)
    first = out[(out["point"] == T1) & (out["narrative"] == "a")].iloc[0]
    assert first["list_share"] == 0.4 and first["outcome_share"] == 0.5 and first["popularity_share"] == 0.2


def test_tokens_without_a_narrative_go_to_other():
    out = nr.shares(table(), pd.Series({1: "a"}))
    assert set(out["narrative"]) == {"a", "other"}


def test_the_gap_is_the_mean_absolute_difference_over_narratives():
    out = nr.shares(table(), pd.Series({1: "a", 2: "a", 3: "b"}))
    gap = nr.mean_abs_gap(out, "list_share").set_index("point")["gap"]
    assert np.isclose(gap[T1], 0.1)  # |0.4 - 0.5| and |0.6 - 0.5|


def test_resolution_rule_takes_the_first_without_a_dominant_cluster():
    clusterings = {0.5: pd.Series({1: 0, 2: 0, 3: 0}), 1.0: pd.Series({1: 0, 2: 0, 3: 1}), 2.0: pd.Series({1: 0, 2: 1, 3: 2})}
    # at 1.0 tokens 1 and 2 hold 120 of 200 slots (60%), at 2.0 the largest holds 40%
    assert nr.choose_resolution(table(), clusterings) == 2.0


def test_small_clusters_are_grouped():
    grouped = nr.group_small(table(), pd.Series({1: 0, 2: 1, 3: 2}), min_share=0.35)
    assert grouped.to_dict() == {1: 0, 2: -1, 3: 2}
