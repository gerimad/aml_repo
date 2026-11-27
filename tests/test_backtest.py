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
    # The decision lies between the swaps of blocks 10 and 11: block 11 is the decision block.
    return pd.DataFrame([{"decision": T0 + MIN / 2, "token": 1, "pool": "p", "buy_eth_24h": buy_eth}])


NO_COSTS = bt.Settings(holding_hours=1, gas_gwei=0, position_eth=0.5)


def test_entry_is_at_the_last_buy_before_the_decision_block():
    swaps = priced([(-2, 9, "buy", 0.9, 1), (0, 10, "sell", 0.8, 1), (1, 11, "buy", 1.2, 1), (70, 20, "sell", 1.5, 1)])
    trades = bt.run(selection(), bt.PoolPrices(swaps), NO_COSTS)
    assert trades["entry_price"].iloc[0] == 0.9, "block 11 is the decision block; the last earlier buy is at 0.9"
    assert trades["entry_time"].iloc[0] == np.datetime64((T0 + MIN / 2).tz_localize(None))
    assert trades["exit_price"].iloc[0] == 1.5 and trades["reason"].iloc[0] == "time"


def test_a_sell_before_the_decision_is_not_an_exit():
    swaps = priced([(-5, 8, "buy", 1.0, 1), (0, 10, "sell", 0.4, 1), (1, 11, "buy", 1.0, 1), (70, 20, "sell", 1.0, 1)])
    settings = bt.Settings(holding_hours=1, gas_gwei=0, stop=0.5)
    trades = bt.run(selection(), bt.PoolPrices(swaps), settings)
    assert trades["reason"].iloc[0] == "time", "the fall to 0.4 happened before s, so the stop is not triggered"


def test_a_token_never_traded_after_the_decision_is_a_total_loss_not_skipped():
    # A rug pull: buyers until just before s, then nothing in this pool (the decision block comes from another pool).
    swaps = priced([(-3, 8, "buy", 1.0, 1), (0, 10, "buy", 1.1, 1)])
    other = priced([(1, 11, "buy", 5.0, 1)]).assign(pool="q")
    trades = bt.run(selection(), bt.PoolPrices(pd.concat([swaps, other], ignore_index=True)), NO_COSTS)
    assert trades["reason"].iloc[0] == "no exit" and trades["gross_return"].iloc[0] == -1.0


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
    gas = 10 * 100_000 * 1e-9  # one swap: 0.001 ETH
    assert trades["gross_return"].iloc[0] == 0.0
    # Paid: the stake 0.5 and the buying gas. Got back: the stake after impact, minus the selling gas.
    assert trades["net_return"].iloc[0] == pytest.approx((0.5 * (1 - impact) / (1 + impact) - gas) / (0.5 + gas) - 1)


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


LADDER = bt.Settings(holding_hours=1, gas_gwei=0, take_profit=(5, 20), min_trigger_eth=0.05)


def test_the_ladder_sells_half_at_each_level_at_the_next_sell():
    swaps = priced([(0, 10, "buy", 1.0, 1), (1, 11, "buy", 1.0, 1),
                    (5, 12, "sell", 6.0, 1), (6, 13, "sell", 5.5, 1),      # 5x reached; next sell at 5.5
                    (20, 14, "sell", 25.0, 1), (21, 15, "sell", 22.0, 1)])  # 20x reached; next sell at 22  # fmt: skip
    trades = bt.run(selection(), bt.PoolPrices(swaps), LADDER)
    assert trades["exit_price"].iloc[0] == pytest.approx((5.5 + 22.0) / 2)


def test_a_tiny_swap_at_an_absurd_price_does_not_trigger_the_ladder():
    swaps = priced([(0, 10, "buy", 1.0, 1), (1, 11, "buy", 1.0, 1),
                    (5, 12, "sell", 100.0, 0.001), (6, 13, "sell", 1.0, 1),
                    (70, 20, "sell", 2.0, 1)])  # fmt: skip
    trades = bt.run(selection(), bt.PoolPrices(swaps), LADDER)
    assert trades["exit_price"].iloc[0] == pytest.approx(2.0), "no trigger: both halves leave at the time limit"


def test_a_level_reached_without_a_later_buyer_is_worth_zero_for_that_half():
    swaps = priced([(0, 10, "buy", 1.0, 1), (1, 11, "buy", 1.0, 1), (5, 12, "sell", 6.0, 1)])
    trades = bt.run(selection(), bt.PoolPrices(swaps), LADDER)
    assert trades["exit_price"].iloc[0] == pytest.approx(0.0)


def test_a_dust_sell_at_an_absurd_price_is_not_an_exit():
    # After a rug pull: a sell of 1e-10 ETH at a price a billion times higher, then nothing that counts.
    swaps = priced([(-2, 9, "buy", 1.0, 1), (1, 11, "buy", 1.0, 1), (70, 20, "sell", 1e9, 1e-10)])
    trades = bt.run(selection(), bt.PoolPrices(swaps), NO_COSTS)
    assert trades["reason"].iloc[0] == "no exit" and trades["gross_return"].iloc[0] == -1.0


def test_a_total_loss_with_gas_is_still_minus_one():
    swaps = priced([(-2, 9, "buy", 1.0, 1), (1, 11, "buy", 1.0, 1)])
    settings = bt.Settings(holding_hours=1, gas_gwei=50, position_eth=0.05)
    trades = bt.run(selection(), bt.PoolPrices(swaps), settings)
    assert trades["net_return"].iloc[0] == pytest.approx(-1.0)


def test_a_position_too_small_for_its_gas_is_not_opened():
    swaps = priced([(-2, 9, "buy", 1.0, 1), (1, 11, "buy", 1.0, 1), (70, 20, "sell", 2.0, 1)])
    trades = bt.run(selection(buy_eth=0.5), bt.PoolPrices(swaps), NO_COSTS)  # 1% of 0.5 ETH = 0.005 ETH
    assert trades["reason"].iloc[0] == "not entered" and trades["skip"].iloc[0] == "too small"
