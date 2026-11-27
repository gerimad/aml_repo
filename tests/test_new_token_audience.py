"""Tests for src/scripts/new_token_audience.py (WP19)."""

import numpy as np
import pandas as pd
from scipy import sparse

import new_token_audience as nt

T = pd.Timestamp("2026-07-10", tz="UTC")
MIN = pd.Timedelta(minutes=1)
HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)


def legs(rows):
    return pd.DataFrame([{"timestamp": when, "user": u, "token": i, "side": "buy"} for when, u, i in rows])


def test_seeds_are_first_hour_buyers_and_buyers_come_in_the_next_24_hours():
    launch = T + 2 * HOUR
    rows = [(launch + m * MIN, u, 99) for m, u in [(0, 1), (10, 2), (50, 3)]]  # three seeds
    rows += [(launch + HOUR + (u - 9) * HOUR, u, 99) for u in range(10, 20)]  # ten buyers, 1-10 hours after launch
    rows += [(launch + 26 * HOUR, 20, 99)]  # too late
    rows += [(launch + 3 * HOUR, 30, 99)]  # not in the audience
    table = legs(rows)
    audience = pd.Series(range(21), index=range(21))
    found = nt.new_tokens(table, T, nt.launch_times(table), audience)
    assert len(found) == 1
    row = found.iloc[0]
    assert sorted(row.seed_rows) == [1, 2, 3]
    assert sorted(row.buyer_rows) == list(range(10, 20)), "the late buyer and the non-audience account are left out"


def test_a_token_with_too_few_seeds_is_not_scored():
    launch = T + HOUR
    rows = [(launch, 1, 99), (launch + 5 * MIN, 2, 99)] + [(launch + 2 * HOUR, u, 99) for u in range(10, 30)]
    table = legs(rows)
    found = nt.new_tokens(table, T, nt.launch_times(table), pd.Series(range(30), index=range(30)))
    assert found.empty


def test_a_token_launched_before_t_is_not_new():
    rows = [(T - HOUR, 5, 99)] + [(T + m * MIN, u, 99) for m, u in [(1, 1), (2, 2), (3, 3)]]
    rows += [(T + 2 * HOUR, u, 99) for u in range(10, 25)]
    table = legs(rows)
    found = nt.new_tokens(table, T, nt.launch_times(table), pd.Series(range(25), index=range(25)))
    assert found.empty


def test_seed_profile_scores_accounts_that_share_the_seeds_tokens():
    # accounts 0, 1 are seeds holding tokens 0 and 1; account 2 holds token 0; account 3 holds token 2 only
    window = sparse.csr_matrix(np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32))
    similarity = sparse.csr_matrix(np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32))
    scores = nt.method_scores(window, similarity, np.array([0, 1]))
    assert scores["seed profile"][2] == 2 and scores["seed profile"][3] == 0
    # through the similarity matrix, token 1 of the seeds points to token 2: account 3 gets a score
    assert scores["recommender"][3] > 0 and scores["recommender"][2] == 0


def test_launch_hunting_counts_tokens_bought_within_a_day_of_launch():
    rows = [(T - 5 * DAY, 9, 1), (T - 5 * DAY + HOUR, 1, 1),  # token 1 launched by user 9; user 1 an hour later
            (T - 4 * DAY, 9, 2), (T - 2 * DAY, 1, 2),  # token 2: user 1 two days after launch
            (T - 30 * DAY, 9, 3), (T - 3 * DAY, 2, 3)]  # fmt: skip
    table = legs(rows)
    hunting = nt.launch_hunting(table, T, nt.launch_times(table))
    assert hunting.get(1) == 1 and 2 not in hunting.index
