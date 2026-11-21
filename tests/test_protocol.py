"""Tests for src/scripts/protocol.py, on small hand-made trade logs."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import protocol as pc

T = pd.Timestamp("2026-07-02", tz="UTC")  # the first Dev A point
SECOND = pd.Timedelta(seconds=1)
DAY = pd.Timedelta(days=1)


def legs(rows):
    """Legs from (time, user, token) triples."""
    return pd.DataFrame([{"timestamp": when, "user": u, "token": i} for when, u, i in rows])


# A small market around T. Users 1-4, tokens 10-14.
LOG = legs([
    (T - 20 * DAY, 1, 10),        # user 1 traded token 10 long ago (outside the 14-day window)
    (T - 2 * DAY, 1, 11),         # ... and token 11 recently
    (T - SECOND, 2, 11),          # one second before T: still the past
    (T - 3 * DAY, 3, 12),
    (T - 30 * DAY, 4, 10),        # user 4 has nothing in the window: never evaluated
    (T, 1, 12),                   # at T exactly: the target. Token 12 is new to user 1.
    (T + DAY, 1, 10),             # user 1 repeats token 10: not a positive pair
    (T + DAY, 2, 99),             # token 99 never traded before T: a cold pair for user 2
    (T + DAY, 3, 12),             # user 3 repeats: user 3 has no positive pair
    (T + DAY, 4, 11),             # user 4 has a positive pair but is not evaluated
    (T + 2 * DAY, 1, 13),         # exactly 2 days after T: outside the target
    (T + DAY, 5, 10),             # user 5 has no past at all: unknown, ignored
])  # fmt: skip


def test_the_plan_has_10_11_and_14_points_two_days_apart():
    assert [len(pc.POINTS[p]) for p in ("dev_a", "dev_b", "test")] == [10, 11, 14]
    assert pc.POINTS["dev_a"][0] == T
    assert pc.POINTS["test"][-1] == pd.Timestamp("2026-09-08", tz="UTC")
    every_point = sum(pc.POINTS.values(), [])
    assert all(b - a == 2 * DAY for a, b in zip(every_point, every_point[1:]))


def test_part_of():
    assert pc.part_of(T) == "dev_a"
    assert pc.part_of(pd.Timestamp("2026-08-13", tz="UTC")) == "test"
    with pytest.raises(ValueError):
        pc.part_of(T + DAY)


def test_task_indexes_users_and_tokens_from_the_past_only():
    task = pc.make_task(LOG, T)
    assert task.users.tolist() == [1, 2, 3, 4], "user 5 only appears after T"
    assert task.tokens.tolist() == [10, 11, 12], "tokens 13 and 99 only appear after T"


def test_positive_pairs_are_new_tokens_in_the_two_days_after_t():
    task = pc.make_task(LOG, T)
    new = pd.DataFrame(task.new.toarray(), index=task.users, columns=task.tokens)
    assert new.loc[1].tolist() == [0, 0, 1], "user 1: token 12 is new, token 10 is a repeat"
    assert new.loc[3].sum() == 0, "user 3 only repeats"
    assert task.cold_new.tolist() == [0, 1, 0, 0], "user 2's token 99 did not exist before T"


def test_evaluated_users_need_a_token_in_the_window_and_a_positive_pair():
    task = pc.make_task(LOG, T)
    assert task.users[task.eval_rows].tolist() == [1, 2]


def test_history_keeps_the_age_of_the_last_trade_and_can_be_cut_by_days():
    task = pc.make_task(LOG, T)
    ages = pd.DataFrame(task.history_ages().toarray(), index=task.users, columns=task.tokens)
    assert ages.loc[1, 10] == pytest.approx(20) and ages.loc[1, 11] == pytest.approx(2)
    assert ages.loc[2, 11] == pytest.approx(1 / 86400)
    assert task.history(14)[0].toarray().tolist() == [[0, 1, 0]], "token 10 is older than 14 days"
    assert task.past[0].toarray().tolist() == [[1, 1, 0]]


def test_describe_counts_the_task():
    sizes = pc.make_task(LOG, T).describe()
    assert sizes["users_evaluated"] == 2 and sizes["positive_pairs"] == 2 and sizes["cold_share"] == 0.5


def test_head_tokens_cover_the_requested_share():
    # Token 0 holds 6 of 10 interactions, token 1 holds 3, token 2 holds 1.
    window = sparse.csr_matrix(np.array([[1, 1, 1]] + [[1, 1, 0]] * 2 + [[1, 0, 0]] * 3))
    assert pc.head_tokens(window, share=1 / 3).tolist() == [0]
    assert pc.head_tokens(window, share=0.8).tolist() == [0, 1]


def test_history_group():
    assert pc.history_group(np.array([1, 2, 4, 5, 30])).tolist() == ["1", "2-4", "2-4", "5+", "5+"]


def test_evaluate_point_removes_the_past_and_counts_cold_pairs_as_misses():
    task = pc.make_task(LOG, T)

    def prefers_token_10(rows):  # token 10 first, then 12, then 11
        return np.tile(np.array([3.0, 1.0, 2.0]), (len(rows), 1))

    per_user, lists = pc.evaluate_point(prefers_token_10, task, k=2)
    # User 1 has traded 10 and 11, so its list is [12] plus nothing better: 12 is a hit.
    assert lists[0][0] == 2, "user 1's own tokens are skipped; token 12 (column 2) comes first"
    assert per_user.set_index("user").loc[1, "recall"] == 1.0
    # User 2's only positive is the cold token 99: recall 0 whatever the list.
    assert per_user.set_index("user").loc[2, "recall"] == 0.0
    # Both users have exactly one token in the 14-day window (user 1's token 10 is older).
    assert per_user["group"].tolist() == ["1", "1"]


def test_popularity_scorer_counts_users_in_the_window():
    task = pc.make_task(LOG, T)
    scores = pc.popularity_scorer(task, days=14)(np.array([0]))
    assert scores.tolist() == [[0, 2, 1]], "in the last 14 days: token 11 by users 1 and 2, token 12 by user 3"


def test_run_refits_at_each_point_and_summarizes(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "load_task", lambda legs, t: pc.make_task(legs, t))
    later = pc.POINTS["dev_a"][1]
    log = pd.concat([LOG, legs([(later - DAY, 1, 11), (later, 1, 14)])], ignore_index=True)
    result = pc.run("popularity", pc.popularity_scorer, log, [T, later], verbose=False)
    assert result.per_point["point"].tolist() == [T, later]
    assert set(result.per_user["point"]) == {T, later}
    assert {"recall", "recall_low", "coverage", "recall_no_head", "recall_1"} <= set(result.per_point.columns)
