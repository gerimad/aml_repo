"""Tests for src/scripts/alerts.py (WP14)."""

import numpy as np
import pandas as pd

import alerts as al

T = pd.Timestamp("2026-07-10", tz="UTC")
HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)
ACCOUNT, OLD_FRIEND, FRESH = "0xaa", "0xbb", "0xcc"


def legs(rows):
    return pd.DataFrame([{"timestamp": when, "user": u, "token": i, "side": s} for when, u, i, s in rows])


def transfers(rows):
    return pd.DataFrame([{"timestamp": when, "sender": s, "recipient": r, "amount_usd": a} for when, s, r, a in rows])


# Account 1 bought tokens 10-14 in the last 14 days and sells 4 of them 2 hours after t.
MARKET = legs([(T - 3 * DAY, 1, tok, "buy") for tok in range(10, 15)] + [(T + 2 * HOUR, 1, tok, "sell") for tok in range(10, 14)])
ADDRESSES = pd.Series({1: ACCOUNT})


def sales():
    return al.liquidation(MARKET, T, [1])


def test_liquidation_is_the_share_of_window_tokens_sold_in_the_block():
    row = sales().iloc[0]
    assert row["block"] == T and row["liquidation"] == 4 / 5


def test_selling_and_sending_to_a_fresh_address_within_an_hour_is_an_exit():
    sent = transfers([(T - 5 * DAY, ACCOUNT, OLD_FRIEND, 10), (T + 2.5 * HOUR, ACCOUNT, FRESH, 900)])
    assert al.exits(sales(), sent, ADDRESSES, pools=set()).iloc[0] == 900


def test_keeping_the_proceeds_or_paying_a_known_address_is_not():
    to_friend = transfers([(T - 5 * DAY, ACCOUNT, OLD_FRIEND, 10), (T + 2.5 * HOUR, ACCOUNT, OLD_FRIEND, 900)])
    assert al.exits(sales(), to_friend, ADDRESSES, pools=set()).iloc[0] == 0
    too_late = transfers([(T + 5 * HOUR, ACCOUNT, FRESH, 900)])
    assert al.exits(sales(), too_late, ADDRESSES, pools=set()).iloc[0] == 0
    to_pool = transfers([(T + 2.5 * HOUR, ACCOUNT, FRESH, 900)])
    assert al.exits(sales(), to_pool, ADDRESSES, pools={FRESH}).iloc[0] == 0


def test_new_address_means_never_a_recipient_before_the_first_sale():
    sent = transfers([(T + 2.1 * HOUR, ACCOUNT, FRESH, 5), (T + 2.5 * HOUR, ACCOUNT, FRESH, 900)])
    assert al.exits(sales(), sent, ADDRESSES, pools=set()).iloc[0] == 905, "both are after the first sale, so FRESH is new"


def test_all_three_signals_fire_together_only():
    table = pd.DataFrame({"deviation": [2.0, 2.0, 0.1], "liquidation": [0.9, 0.9, 0.9], "exit_usd": [500.0, 0.0, 500.0]})
    assert al.fire(table, deviation_min=1.0, liquidation_min=0.5).tolist() == [True, False, False]


def test_deviation_compares_with_the_accounts_own_earlier_ranks():
    earlier = pd.DataFrame({"point": T - 2 * DAY, "user": 1, "token": [1, 2, 3], "cold": False, "rank": [9, 9, 9], "first_trade_time": T - 2 * DAY})
    now = pd.DataFrame({"point": T, "user": 1, "token": [20, 21], "cold": False, "rank": [999, 999], "first_trade_time": [T + HOUR, T + 2 * HOUR]})
    result = al.deviation(now, earlier).iloc[0]
    assert result["block"] == T and np.isclose(result["deviation"], np.log1p(999) - np.log1p(9))
