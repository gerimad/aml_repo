"""Tests for src/scripts/dev_backtest.py: grids, the decision cut-off and the choice rule."""

import pandas as pd

import dev_backtest as db


def test_the_tail_grid_has_every_planned_setting():
    grid = db.tail_grid()
    # 3 candidate versions x K 3 x H 3 x stop 2 x weighting 2, and 4 benchmarks x K 3 x H 3 x stop 2.
    assert grid["candidate"].sum() == 3 * 3 * 3 * 2 * 2
    assert (~grid["candidate"]).sum() == 4 * 3 * 3 * 2
    assert grid.apply(db.setting_id, axis=1).is_unique


def test_the_launch_grid_has_every_planned_setting():
    grid = db.launch_grid()
    # momentum: N 2 x b 2 x K 2 x H 3 x ladder 3; random: the same without b.
    assert grid["candidate"].sum() == 2 * 2 * 2 * 3 * 3
    assert (~grid["candidate"]).sum() == 2 * 2 * 3 * 3
    assert grid.apply(db.setting_id, axis=1).is_unique


def test_the_last_decision_leaves_room_for_the_longest_hold_and_the_grace():
    end = pd.Timestamp("2026-09-10", tz="UTC")
    assert db.last_decision(48, end) == pd.Timestamp("2026-09-07", tz="UTC")
    assert db.last_decision(72, end) == pd.Timestamp("2026-09-06", tz="UTC")


def test_a_missing_day_counts_as_no_change_in_the_paired_difference():
    days = pd.date_range("2026-07-02", periods=3, freq="D", tz="UTC")
    a = pd.Series([0.1, 0.2, 0.3], index=days)
    b = pd.Series([0.05, 0.05], index=days[:2])
    assert db.paired_difference(a, b).round(10).tolist() == [0.05, 0.15, 0.3]


def test_the_choice_goes_to_the_simpler_setting_within_one_standard_error():
    table = pd.DataFrame({
        "k": [10, 3, 5], "score": [0.010, 0.008, 0.009], "se": [0.003, 0.003, 0.003],
    })  # fmt: skip
    chosen = db.choose(table, "score", "se", [("k", True)])
    assert chosen["k"] == 3, "K = 3 is within one standard error of the best (K = 10) and simpler"


def test_the_choice_keeps_the_best_when_no_other_is_close():
    table = pd.DataFrame({"k": [10, 3], "score": [0.010, 0.001], "se": [0.002, 0.002]})
    assert db.choose(table, "score", "se", [("k", True)])["k"] == 10
