"""Tests for src/scripts/demand_forecast.py (WP17), on small hand-made markets."""

import numpy as np
import pandas as pd

import demand_forecast as dfc
import protocol as pc

T = pd.Timestamp("2026-07-10", tz="UTC")
HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)


def legs(rows):
    """Legs from (time, user, token) triples."""
    return pd.DataFrame([{"timestamp": when, "user": u, "token": i} for when, u, i in rows])


def tokens_table(market):
    born = market.groupby("token")["timestamp"].min()
    return pd.DataFrame({"token_id": born.index, "first_timestamp": born.to_numpy()})


# Token 10: user 1 traded it 5 days ago (and again in the target), users 2 and 3 first trade it 10 h before t,
# users 4 and 5 for the first time in the target. Token 11 keeps the users active; token 12 was last traded 20 days ago.
MARKET = legs(
    [(T - 5 * DAY, 1, 10), (T - 10 * HOUR, 2, 10), (T - 10 * HOUR, 3, 10)]
    + [(T - 3 * DAY, u, 11) for u in (1, 2, 3, 4, 5)]
    + [(T - 20 * DAY, 6, 12)]
    + [(T + 5 * HOUR, 1, 10), (T + 5 * HOUR, 4, 10), (T + 30 * HOUR, 5, 10), (T + 3 * DAY, 6, 10)]
)


def table():
    return dfc.point_table(MARKET, dfc.first_ever(MARKET), pc.make_task(MARKET, T), tokens_table(MARKET)).set_index("token")


def test_a_buyer_who_traded_the_token_before_t_is_not_counted():
    row = table().loc[10]
    assert row["n_known"] == 2, "users 4 and 5; user 1 traded token 10 before t, user 6 trades it after 48 h"
    assert row["n_all"] == 2


def test_persistence_uses_only_the_48_hours_before_t():
    row = table().loc[10]
    assert row["persistence"] == 2, "users 2 and 3; user 1's first trade was 5 days before t"
    assert row["users_24h"] == 2


def test_tokens_without_a_trade_in_the_14_days_are_not_scored():
    assert 12 not in table().index


def test_a_list_counts_each_account_at_most_once_per_token():
    task = pc.make_task(MARKET, T)
    counts = dfc.list_counts(task, depth=3)
    users_with_window = int((task.window_size >= 1).sum())
    assert counts.max() <= users_with_window
    assert counts.sum() <= users_with_window * 3


def test_fit_and_score_reports_zero_error_for_an_exact_linear_target():
    rng = np.random.default_rng(0)
    rows = pd.DataFrame({"point": np.repeat([T, T + DAY], 50), "x": rng.random(100), "in_head": rng.random(100) < 0.3})
    rows["y"] = 2 * rows["x"] + 1
    scored = dfc.fit_and_score(rows, rows, ["x"], target="y")
    assert np.allclose(scored["mae"], 0) and np.allclose(scored["spearman"], 1)
