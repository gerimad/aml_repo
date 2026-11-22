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
    """Users 0-9 trade old tokens before T. Token 50 launches at T + 2h30; users 3-9 buy it
    between T + 2h30 and T + 3h30, user 1 buys it at T + 10h. User 0 buys old token 11 at T + 1h."""
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
    refreshed = cs.refreshed_evaluate({"popularity": cs.popularity_factory}, legs, task, token_table(legs), hours=48)
    assert refreshed["popularity"]["recall"].mean() == pytest.approx(main["recall"].mean())


def test_a_token_launched_in_the_target_becomes_reachable_with_refreshed_lists():
    legs = market()
    task = pc.make_task(legs, T)
    tokens = token_table(legs)
    models = {"newest": cs.newest_popular_factory(24), "baseline": cs.popularity_factory}
    once = cs.refreshed_evaluate(models, legs, task, tokens, hours=48)
    hourly = cs.refreshed_evaluate(models, legs, task, tokens, hours=1)
    user1_once = once["newest"].set_index("user").loc[1]
    user1_hourly = hourly["newest"].set_index("user").loc[1]
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


def test_a_shared_factory_is_fitted_once_per_refresh_time():
    legs = market()
    fits = []

    def counting_factory(state, tokens):
        fits.append(state.t)
        return cs.popularity_factory(state, tokens)

    shared = cs.shared_across_models(counting_factory)
    models = {f"mixed {n}": cs.mixed_factory(shared, max_age_hours=24, n_new=n) for n in (1, 2, 3)}
    task = pc.make_task(legs, T)
    cs.refreshed_evaluate(models, legs, task, token_table(legs), hours=12)
    assert len(fits) == len(set(fits)), "the base model is fitted at most once per refresh time"


def token_table_with_launch(legs):
    """The toy token table with the columns the feature score needs: token 0 is WETH, everything launches against it."""
    table = token_table(legs)
    weth = pd.DataFrame({"token_id": [0], "first_timestamp": [T - 30 * DAY]})
    table = pd.concat([weth, table], ignore_index=True)
    table["is_base"] = table["token_id"] == 0
    table["symbol"] = np.where(table["is_base"], "WETH", "X")
    table["launch_base_id"] = np.where(table["is_base"], np.nan, 0.0)
    return table


def test_young_token_features_count_early_users_only_before_the_refresh_time():
    legs = market()
    tokens = token_table_with_launch(legs)
    early = cs.EarlyTrades(legs, tokens)
    # Token 50 launches at T + 2h30 (user 3). At T + 3h05, users 3-6 have bought it
    # (at +2h30, +2h40, +2h50, +3h00); users 7-9 come later.
    state = pc.make_task(legs, T + 3 * HOUR + 5 * pd.Timedelta(minutes=1))
    features = cs.young_token_features(state, tokens, early, max_age_hours=24)
    assert list(state.tokens[features.index]) == [NEW_TOKEN], "only token 50 is younger than a day"
    row = features.iloc[0]
    assert row["age_hours"] == pytest.approx(35 / 60)
    assert row["log_users_first_hour"] == pytest.approx(np.log1p(4))
    assert row["log_users_first_day"] == pytest.approx(np.log1p(4))
    assert row["launch_other"] == 0 and row["launch_usdc"] == 0, "launched against WETH, the reference"


def test_feature_training_rows_count_pick_ups_until_the_next_refresh():
    legs = market()
    tokens = token_table_with_launch(legs)
    task = pc.make_task(legs, T)
    rows = cs.feature_training_rows(legs, task, tokens, cs.EarlyTrades(legs, tokens), hours=6, max_age_hours=24)
    new = rows[rows["token"] == NEW_TOKEN].set_index("refresh")
    # At T + 6h users 3-9 have bought it; user 1 buys it at T + 10h, inside [T + 6h, T + 12h).
    assert new.loc[T + 6 * HOUR, "picked"] == 1
    assert new.loc[T + 12 * HOUR, "picked"] == 0
    evaluated = len(task.eval_rows)
    assert new.loc[T + 6 * HOUR, "not_picked"] == evaluated - 7 - 1, "7 evaluated accounts already traded it"


def test_feature_mixed_list_puts_the_young_token_first():
    legs = market()
    tokens = token_table_with_launch(legs)
    early = cs.EarlyTrades(legs, tokens)
    task = pc.make_task(legs, T)
    training = cs.feature_training_rows(legs, task, tokens, early, hours=3, max_age_hours=24)
    model = cs.fit_feature_score(training)
    state = pc.make_task(legs, T + 5 * HOUR)
    factory = cs.feature_mixed_factory(cs.popularity_factory, early, model, max_age_hours=24, n_new=1)
    scores = factory(state, tokens)(np.searchsorted(state.users, [1]))
    assert state.tokens[np.argmax(scores[0])] == NEW_TOKEN
