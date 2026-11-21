"""Tests for src/scripts/cold_start.py, on a hand-made market with a token launched inside the target."""

import numpy as np
import pandas as pd
import pytest

import cold_start as cs
import protocol as pc

T = pd.Timestamp("2026-07-02", tz="UTC")
HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)
NEW_TOKEN = 50


def market():
    """Users 0-9 trade old tokens before T. Token 50 launches at T + 2h; users 3-9 buy it
    between T + 2h and T + 4h, user 1 buys it at T + 10h. User 0 buys old token 11 at T + 1h."""
    rows = [(T - 3 * DAY, u, 10) for u in range(10)] + [(T - 2 * DAY, u, 11) for u in range(1, 6)]
    rows += [(T + 2 * HOUR + u * 10 * pd.Timedelta(minutes=1), u, NEW_TOKEN) for u in range(3, 10)]
    rows += [(T + 10 * HOUR, 1, NEW_TOKEN), (T + 1 * HOUR, 0, 11)]
    return pd.DataFrame([{"timestamp": when, "user": u, "token": i} for when, u, i in rows])


def token_table(legs):
    first = legs.groupby("token")["timestamp"].min()
    return pd.DataFrame({"token_id": first.index, "first_timestamp": first.to_numpy()})


def test_refresh_times():
    assert len(cs.refresh_times(T, 6)) == 8
    assert cs.refresh_times(T, 48) == [T]
    assert cs.refresh_times(T, 5)[-1] == T + 45 * HOUR


def test_first_trades_are_exactly_the_positive_pairs_with_their_times():
    legs = market()
    pairs = cs.first_trades_in_target(legs, pc.make_task(legs, T)).set_index(["user", "token"])
    assert pairs.loc[(1, NEW_TOKEN), "first_time"] == T + 10 * HOUR
    assert bool(pairs.loc[(1, NEW_TOKEN), "cold"]) and not bool(pairs.loc[(0, 11), "cold"])


def test_refreshing_every_48_hours_is_the_main_protocol():
    legs = market()
    task = pc.make_task(legs, T)
    main, _ = pc.evaluate_point(pc.popularity_scorer(task), task)
    refreshed = cs.refreshed_evaluate(cs.popularity_factory, legs, task, token_table(legs), hours=48)
    assert refreshed["recall"].mean() == pytest.approx(main["recall"].mean())


def test_a_token_launched_in_the_target_becomes_reachable_with_refreshed_lists():
    legs = market()
    task = pc.make_task(legs, T)
    tokens = token_table(legs)
    once = cs.refreshed_evaluate(cs.newest_popular_factory(24), legs, task, tokens, hours=48)
    hourly = cs.refreshed_evaluate(cs.newest_popular_factory(24), legs, task, tokens, hours=1)
    user1_once = once.set_index("user").loc[1]
    user1_hourly = hourly.set_index("user").loc[1]
    assert user1_once["cold_hits"] == 0, "at T the token does not exist"
    assert user1_hourly["cold_hits"] == 1, "at T + 10h the token is young and popular"


def test_newest_popular_ignores_old_tokens():
    legs = market()
    state = pc.make_task(legs, T + 5 * HOUR)
    scores = cs.newest_popular_scores(state, token_table(legs), max_age_hours=24)
    by_token = dict(zip(state.tokens, scores))
    assert by_token[10] == 0 and by_token[11] == 0, "tokens older than a day score 0"
    assert by_token[NEW_TOKEN] == 7, "seven users bought the new token before T + 5h"


def test_mixed_list_puts_young_tokens_first_and_keeps_the_rest_from_the_base_model():
    legs = market()
    state = pc.make_task(legs, T + 5 * HOUR)
    factory = cs.mixed_factory(cs.popularity_factory, max_age_hours=24, n_new=1)
    scores = factory(state, token_table(legs))(np.searchsorted(state.users, [1]))
    order = state.tokens[np.argsort(-scores[0])]
    assert order[0] == NEW_TOKEN
