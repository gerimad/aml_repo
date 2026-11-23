"""Tests for src/scripts/performance.py."""

import math

import numpy as np
import pandas as pd
import pytest

import performance as pf

T0 = pd.Timestamp("2026-08-13 00:00", tz="UTC")
H = pd.Timedelta(hours=1)


def position(decision_hours, exit_hours, ret, reason="time"):
    d = T0 + decision_hours * H
    return {"decision": d, "exit_time": (T0 + exit_hours * H) if exit_hours is not None else None, "reason": reason,
            "net_return": ret, "size_eth": 0.5}  # fmt: skip


def test_one_sleeve_out_of_five_gains_ten_percent():
    trades = pd.DataFrame([position(0, 25, 0.10)])
    curve = pf.equity_curve(trades, holding_hours=24)
    assert curve.iloc[0] == pytest.approx(1.0)
    assert curve.iloc[-1] == pytest.approx(1 + 0.10 / 5)


def test_a_busy_sleeve_skips_the_decision():
    # 6-hour holds: 2 sleeves. Decisions at 0, 6, 12 h; the first exits only at 13 h, so the decision at 12 h
    # (sleeve 0 again) is skipped.
    trades = pd.DataFrame([position(0, 13, 0.1), position(6, 12.5, 0.1), position(12, 19, 0.5)])
    curve = pf.equity_curve(trades, holding_hours=6)
    assert curve.attrs["skipped"] == 1
    assert curve.iloc[-1] == pytest.approx(0.5 * 1.1 + 0.5 * 1.1)


def test_a_position_without_exit_loses_its_share():
    trades = pd.DataFrame([position(0, None, -1.0, reason="no exit")])
    curve = pf.equity_curve(trades, holding_hours=24)
    assert curve.iloc[-1] == pytest.approx(1 - 1 / 5)


def test_max_drawdown_and_its_length():
    equity = pd.Series([1.0, 1.2, 0.9, 1.0, 1.3], index=pd.date_range("2026-01-01", periods=5))
    drawdown, days = pf.max_drawdown(equity)
    assert drawdown == pytest.approx(0.25) and days == 2


def test_sharpe_sortino_and_kelly_on_known_numbers():
    daily = pd.Series([0.01, -0.01, 0.02, 0.0])
    mean, sd = daily.mean(), daily.std(ddof=1)
    assert pf.sharpe(daily) == pytest.approx(mean / sd * math.sqrt(365))
    assert pf.sortino(daily) == pytest.approx(mean / math.sqrt(0.0001 / 4) * math.sqrt(365))
    assert pf.kelly_continuous(daily) == pytest.approx(mean / daily.var(ddof=1))
    # 3 wins of +0.2 and 1 loss of -0.1: p = 0.75, b = 2, Kelly = 0.75 - 0.25 / 2.
    assert pf.kelly_discrete(pd.Series([0.2, 0.2, 0.2, -0.1])) == pytest.approx(0.625)


def test_deflated_sharpe_falls_as_more_settings_are_tried():
    rng = np.random.default_rng(0)
    daily = pd.Series(rng.normal(0.002, 0.02, 200))
    one = pf.deflated_sharpe(daily, trials=1, trial_sharpes_sd=0.0)
    many = pf.deflated_sharpe(daily, trials=100, trial_sharpes_sd=0.05)
    assert 0 <= many < one <= 1
