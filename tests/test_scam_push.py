"""Tests for src/scripts/scam_push.py (WP13)."""

import numpy as np
import pandas as pd

import scam_push as sp

D = pd.Timestamp("2026-07-10", tz="UTC")
HOUR = pd.Timedelta(hours=1)


def swaps(prices, hours, pool="p", volume=1.0):
    return pd.DataFrame({"timestamp": [D + h * HOUR for h in hours], "pool": pool, "price_eth": prices, "volume_eth": volume})


def test_a_95_percent_fall_that_stops_trading_is_a_collapse():
    s = swaps([1.0, 2.0, 0.05, 0.04], [1, 5, 10, 60])  # peak 2 at hour 5, then 0.05; one swap in the last 24 h
    assert sp.collapse_label(s, D)


def test_a_95_percent_fall_that_recovers_and_keeps_trading_is_not():
    s = swaps([1.0, 2.0, 0.05] + [1.5] * 10, [1, 5, 10] + list(range(49, 59)))
    assert not sp.collapse_label(s, D)


def test_a_token_with_no_real_trade_after_d_is_collapsed_and_dust_is_ignored():
    dust = swaps([5.0, 6.0], [2, 3], volume=0.001)
    assert sp.collapse_label(dust, D)
    before = swaps([1.0], [-2])
    assert sp.collapse_label(before, D), "a swap before d is not used"


def test_a_token_still_trading_in_other_pools_is_not_dead():
    dust = swaps([5.0, 6.0], [2, 3], volume=0.001)
    assert not sp.collapse_label(dust, D, late_trades=40), "no priced swap, but 40 trades in the last 24 h"
    assert sp.collapse_label(dust, D, late_trades=0)


def test_labels_use_only_swaps_after_d():
    steady = swaps([1.0] * 12, list(range(1, 72, 6)))
    crash_before = pd.concat([swaps([100.0, 0.01], [-10, -5]), steady])
    assert sp.collapse_label(steady, D) == sp.collapse_label(crash_before, D) is False


def test_the_bootstrap_resamples_days_not_tokens():
    rng = np.random.default_rng(0)
    days = np.repeat(np.arange(10), 30)
    labels = rng.random(300) < 0.3
    a = rng.random(300)
    b = a + labels * 0.5
    result = sp.auc_difference_by_day(a, b, labels, days, n_boot=200)
    assert result["days"] == 10 and result["tokens"] == 300
    assert result["low"] > 0
    # Each day's rows travel together: a single-day dataset gives a degenerate (zero-width) interval.
    one = sp.auc_difference_by_day(a[:30], b[:30], labels[:30], days[:30], n_boot=50)
    assert one["low"] == one["high"]


def test_features_use_only_data_before_the_end_of_the_window():
    import expectedness as ex
    import protocol as pc

    T = pd.Timestamp("2026-07-02", tz="UTC")
    day = pd.Timedelta(days=1)
    rng = np.random.default_rng(0)
    rows = [(T - day * float(rng.uniform(0.1, 10)), int(u), int(i), "buy") for u, i in zip(rng.integers(0, 60, 600), rng.integers(0, 8, 600))]
    rows += [(T + HOUR * float(rng.uniform(0, 23)), int(u), int(i), "buy") for u, i in zip(rng.integers(0, 60, 200), rng.integers(0, 8, 200))]
    market = pd.DataFrame(rows, columns=["timestamp", "user", "token", "side"])
    later = pd.concat([market, pd.DataFrame([(T + day + HOUR * 3, u, 3, "sell") for u in range(60)], columns=market.columns)], ignore_index=True)
    tokens = pd.DataFrame({"token_id": range(8), "first_timestamp": T - 11 * day})

    def features(m):
        task = pc.make_task(m, T)
        pairs = ex.pair_table(m, task, tokens, {"k": 5, "shrink": 1.0, "weight": 0.5})
        first = m.groupby(["user", "token"])["timestamp"].min().rename("first_time").reset_index()
        return sp.window_features(pairs, m, first, task, T, farm_members={1, 2}, min_new=2)

    a, b = features(market), features(later)
    assert len(a) > 0
    pd.testing.assert_frame_equal(a, b)
