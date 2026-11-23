"""Tests for src/scripts/error_analysis.py, on small hand-made trade logs."""

import numpy as np
import pandas as pd

import error_analysis as ea
import protocol as pc

T = pd.Timestamp("2026-07-02", tz="UTC")
HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)


def legs(rows):
    return pd.DataFrame([{"timestamp": when, "user": u, "token": i} for when, u, i in rows])


def token_table(log):
    first = log.groupby("token")["timestamp"].min()
    return pd.DataFrame({"token_id": first.index, "first_timestamp": first.to_numpy()})


def test_pair_outcomes_marks_hits_cold_pairs_and_token_ages():
    # User 1 traded token 10 before T; after T it buys token 11 (traded by user 2 three days
    # before T, so 3 days old) and token 50, which launches after T (cold).
    log = legs([
        (T - 5 * DAY, 1, 10), (T - 3 * DAY, 2, 11), (T - 2 * DAY, 2, 10),
        (T + HOUR, 1, 11), (T + 2 * HOUR, 1, 50),
    ])  # fmt: skip
    task = pc.make_task(log, T)
    row = list(task.eval_rows).index(int(np.searchsorted(task.users, 1)))
    lists = np.zeros((len(task.eval_rows), 1), dtype=int)
    lists[row, 0] = list(task.tokens).index(11)  # user 1's list holds token 11
    pairs = ea.pair_outcomes(log, task, lists, token_table(log)).set_index(["user", "token"])
    assert bool(pairs.loc[(1, 11), "hit"]) and not bool(pairs.loc[(1, 11), "cold"])
    assert pairs.loc[(1, 11), "token_age"] == "1 to 7 days"
    assert pairs.loc[(1, 11), "window_users"] == 1
    assert not bool(pairs.loc[(1, 50), "hit"]) and bool(pairs.loc[(1, 50), "cold"])
    assert pairs.loc[(1, 50), "token_age"] == "did not exist yet"


def test_returning_share_counts_pairs_traded_only_before_the_window():
    # User 1 traded token 10 twenty days before T (before the 14-day window) and trades it
    # again after T: a returning pair. Its new token 11 is a positive pair.
    log = legs([
        (T - 20 * DAY, 1, 10), (T - 2 * DAY, 1, 12), (T - 3 * DAY, 2, 11),
        (T + HOUR, 1, 10), (T + HOUR, 1, 11),
    ])  # fmt: skip
    share = ea.returning_share(log, pc.make_task(log, T))
    assert share["positives"] == 1 and share["returning"] == 1
    assert share["returning_share"] == 0.5


def test_outcome_of_pairs_separates_the_reasons_for_a_miss():
    log = legs([
        (T - 5 * DAY, 1, 10), (T - 3 * DAY, 2, 11), (T - 3 * DAY, 2, 12), (T - 2 * DAY, 2, 10),
        (T + HOUR, 1, 11), (T + HOUR, 1, 12), (T + 2 * HOUR, 1, 13), (T + 3 * HOUR, 1, 50),
    ])  # fmt: skip
    log.loc[log["token"] == 13, "timestamp"] = T + 2 * HOUR  # token 13 traded only after T: cold
    log = pd.concat([log, legs([(T - 4 * DAY, 3, 14), (T + HOUR, 1, 14)])], ignore_index=True)
    task = pc.make_task(log, T)
    column = {int(token): c for c, token in enumerate(task.tokens)}
    row = list(task.eval_rows).index(int(np.searchsorted(task.users, 1)))
    lists = np.zeros((len(task.eval_rows), 1), dtype=int)
    lists[row, 0] = column[11]  # found
    candidates = pd.DataFrame({"user": [1, 1], "col": [column[11], column[12]]})  # 12: candidate, not listed
    pairs = ea.pair_outcomes(log, task, lists, token_table(log))
    pairs["outcome"] = ea.outcome_of_pairs(pairs, candidates, task)
    by_token = pairs[pairs["user"] == 1].set_index("token")["outcome"]
    assert by_token[11] == "found"
    assert by_token[12] == "candidate, ranked below 10"
    assert by_token[14] == "not a candidate"
    assert by_token[50] == "did not exist yet" and by_token[13] == "did not exist yet"
