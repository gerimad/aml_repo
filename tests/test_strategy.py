"""Tests for src/scripts/strategy.py: the selection rules, on hand-made signal tables."""

import numpy as np
import pandas as pd

import strategy as st

S = pd.Timestamp("2026-08-13 06:00", tz="UTC")


def signal(rows):
    """A signal table from (token, predicted accounts, buyers 24h, sellers 24h, age hours, has pool)."""
    table = pd.DataFrame([{
        "decision": S, "token": t, "predicted_1": p, "predicted_2-4": 0, "predicted_5+": 0,
        "predicted_eth_1": p * 0.1, "predicted_eth_2-4": 0.0, "predicted_eth_5+": 0.0,
        "buyers_24h": b, "sellers_24h": sl, "buy_eth_24h": b * 0.2, "age_hours": age, "pool": "p" if pool else None,
    } for t, p, b, sl, age, pool in rows])  # fmt: skip
    return table


def test_universe_needs_a_pool_a_day_of_age_and_twenty_sellers():
    table = signal([(1, 5, 50, 30, 48, True), (2, 5, 50, 30, 10, True), (3, 5, 50, 5, 48, True), (4, 5, 50, 30, 48, False)])
    assert st.universe(table)["token"].tolist() == [1]


def test_surprise_prefers_predicted_demand_that_is_not_yet_visible():
    table = signal([(1, 100, 99, 30, 48, True), (2, 50, 4, 30, 48, True)])
    assert st.select(table, "surprise", k=1)["token"].tolist() == [2]


def test_filtered_visible_ranks_only_the_most_visible_tokens_by_prediction():
    rows = [(t, p, b, 30, 48, True) for t, p, b in [(1, 1, 90), (2, 9, 80), (3, 5, 70), (4, 100, 1)]]
    # k=1: the shortlist is the 3 most visible tokens (1, 2, 3); token 4 has the most predicted demand but is not on it.
    assert st.select(signal(rows), "filtered_visible", k=1)["token"].tolist() == [2]


def test_tail_popularity_ignores_the_recommender():
    rows = [(1, 100, 10, 30, 48, True), (2, 0, 60, 30, 48, True)]
    assert st.select(signal(rows), "tail_popularity", k=1)["token"].tolist() == [2]


def test_random_is_reproducible_for_a_decision_time():
    rows = [(t, 1, 30, 30, 48, True) for t in range(20)]
    a = st.select(signal(rows), "random", k=5)["token"].tolist()
    b = st.select(signal(rows), "random", k=5)["token"].tolist()
    assert a == b and len(set(a)) == 5


def test_hit_rates_weight_the_history_groups():
    table = signal([(1, 10, 30, 30, 48, True)])
    table["predicted_5+"] = 10
    assert st.predicted(table, "accounts", {"1": 0.5, "2-4": 1.0, "5+": 0.1}).iloc[0] == 10 * 0.5 + 10 * 0.1


def test_decision_times_are_six_hours_apart_and_exclude_the_end():
    times = st.decision_times("2026-08-13", "2026-08-14")
    assert len(times) == 4 and times[1] - times[0] == pd.Timedelta(hours=6)


def toy_market(seed=0, n_users=80, n_tokens=30):
    """Legs and priced swaps over 20 days before and 2 days after S, with some co-trading structure."""
    rng = np.random.default_rng(seed)
    legs, prices = [], []
    for user in range(n_users):
        favourite = rng.integers(0, 6)
        for _ in range(rng.integers(2, 8)):
            token = int((favourite * 5 + rng.integers(0, 7)) % n_tokens)
            when = S + pd.Timedelta(days=float(rng.uniform(-20, 2)))
            side = "buy" if rng.random() < 0.6 else "sell"
            legs.append({"block": 0, "timestamp": when, "user": user, "token": token, "side": side, "pool": token})
            prices.append({"timestamp": when, "tx_from": f"0x{user:040x}", "side": side, "volume_eth": float(rng.uniform(0.01, 2)),
                           "token": f"0xt{token}", "pool": f"pool{token}"})  # fmt: skip
    legs = pd.DataFrame(legs).sort_values("timestamp", ignore_index=True)
    first = legs.groupby("token")["timestamp"].min()
    tokens = pd.DataFrame({"token_id": first.index, "address": [f"0xt{t}" for t in first.index], "first_timestamp": first.to_numpy()})
    users = pd.DataFrame({"user_id": range(n_users), "address": [f"0x{u:040x}" for u in range(n_users)]})
    return legs, pd.DataFrame(prices), tokens, users


def test_the_signal_does_not_change_when_everything_after_the_decision_time_is_deleted():
    legs, prices, tokens, users = toy_market()
    settings = {"k": 5, "shrink": 1.0, "weight": 0.5}
    full = st.signal_table(legs, prices, tokens, users, S, settings)
    past_legs = legs[legs["timestamp"] < S]
    past_prices = prices[prices["timestamp"] < S]
    past_tokens = tokens[tokens["first_timestamp"] < S]
    cut = st.signal_table(past_legs, past_prices, past_tokens, users, S, settings)
    assert len(full) > 0
    pd.testing.assert_frame_equal(full.reset_index(drop=True), cut.reset_index(drop=True))


def test_unexplained_flow_prefers_visible_demand_the_model_did_not_predict():
    rows = [(1, 50, 60, 30, 48, True), (2, 0, 40, 30, 48, True)]
    # token 1: 60 / 51; token 2: 40 / 1 -> token 2
    assert st.select(signal(rows), "unexplained_flow", k=1)["token"].tolist() == [2]
