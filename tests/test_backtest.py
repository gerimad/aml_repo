"""Tests for src/scripts/backtest.py, on one toy pool with hand-written swaps."""

import numpy as np
import pandas as pd
import pytest

import backtest as bt

T0 = pd.Timestamp("2026-08-13 06:00", tz="UTC")
MIN = pd.Timedelta(minutes=1)


def priced(rows):
    """Priced swaps of pool "p" from (minutes after T0, block, side, price in ETH, volume in ETH)."""
    return pd.DataFrame([{"block": b, "tx_index": 0, "log_index": i, "timestamp": T0 + m * MIN, "pool": "p",
                          "side": side, "price_eth": price, "volume_eth": vol}
                         for i, (m, b, side, price, vol) in enumerate(rows)])  # fmt: skip


def selection(buy_eth=100.0):
    return pd.DataFrame([{"decision": T0, "token": 1, "pool": "p", "buy_eth_24h": buy_eth}])


NO_COSTS = bt.Settings(holding_hours=1, gas_gwei=0, position_eth=0.5)


def test_entry_is_a_later_block_than_the_decision_block_and_a_buy():
    swaps = priced([(0, 10, "buy", 1.0, 1), (1, 11, "sell", 1.1, 1), (2, 12, "buy", 1.2, 1), (70, 20, "sell", 1.5, 1)])
    trades = bt.run(selection(), bt.PoolPrices(swaps), NO_COSTS)
    assert trades["entry_price"].iloc[0] == 1.2, "block 10 is the decision block; the first later buy is at 1.2"
    assert trades["exit_price"].iloc[0] == 1.5 and trades["reason"].iloc[0] == "time"


def test_the_stop_exits_at_the_next_sell_after_the_fall_not_at_the_stop_level():
    swaps = priced([(0, 10, "buy", 1.0, 1), (1, 11, "buy", 1.0, 1), (5, 12, "sell", 0.4, 1), (6, 13, "sell", 0.3, 1), (70, 20, "sell", 2.0, 1)])
    settings = bt.Settings(holding_hours=1, gas_gwei=0, stop=0.5)
    trades = bt.run(selection(), bt.PoolPrices(swaps), settings)
    assert trades["reason"].iloc[0] == "stop" and trades["exit_price"].iloc[0] == 0.3


def test_a_position_without_a_sell_swap_is_worth_zero():
    swaps = priced([(0, 10, "buy", 1.0, 1), (1, 11, "buy", 1.0, 1), (3000, 30, "buy", 5.0, 1)])
    trades = bt.run(selection(), bt.PoolPrices(swaps), NO_COSTS)
    assert trades["reason"].iloc[0] == "no exit" and trades["gross_return"].iloc[0] == -1.0


def test_round_trip_costs_without_a_price_change_are_impact_and_gas():
    swaps = priced([(0, 10, "buy", 1.0, 1), (1, 11, "buy", 1.0, 1), (70, 20, "sell", 1.0, 1)])
    settings = bt.Settings(holding_hours=1, gas_gwei=10, gas_units=100_000, position_eth=0.5)
    trades = bt.run(selection(), bt.PoolPrices(swaps), settings)
    impact = trades["impact"].iloc[0]
    gas_share = 2 * 10 * 100_000 * 1e-9 / 0.5
    assert trades["gross_return"].iloc[0] == 0.0
    assert trades["net_return"].iloc[0] == pytest.approx((1 - impact) / (1 + impact) - 1 - gas_share)


def test_size_is_capped_by_the_pools_recent_buy_volume():
    swaps = priced([(0, 10, "buy", 1.0, 1), (1, 11, "buy", 1.0, 1), (70, 20, "sell", 1.0, 1)])
    trades = bt.run(selection(buy_eth=10.0), bt.PoolPrices(swaps), NO_COSTS)
    assert trades["size_eth"].iloc[0] == pytest.approx(0.1)


def test_impact_is_the_median_move_across_swaps_of_similar_size():
    rows = []
    for i in range(12):
        rows.append((-600 + i, i, "buy", 1.0 if i % 2 == 0 else 1.02, 0.5))
    pool = bt.PoolPrices(priced(rows)).pool("p")
    estimate = bt.impact_estimate(pool, np.datetime64(T0.tz_localize(None)), size_eth=0.5)
    assert estimate == pytest.approx(0.0, abs=1e-12), "prices two swaps apart are equal here"


def test_daily_returns_count_a_decision_without_positions_as_zero():
    trades = pd.DataFrame([
        {"decision": T0, "reason": "time", "net_return": 0.10},
        {"decision": T0, "reason": "time", "net_return": -0.02},
        {"decision": T0 + pd.Timedelta(hours=6), "reason": "not entered", "net_return": np.nan},
    ])  # fmt: skip
    daily = bt.daily_returns(trades)
    assert daily.iloc[0] == pytest.approx((0.04 + 0.0) / 2)
