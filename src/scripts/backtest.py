"""The backtest engine (work package 8, implementation plan section 12.3).

A position opened at decision time s in token i:

  entry   the first BUY swap of i in its chosen pool in a block after the
          decision block (the first block at or after s). Its price is what
          a real buyer paid, so it already contains the pool's fee and that
          buyer's own price impact.
  exit    the first SELL swap at least H hours after entry, at the price a
          real seller got (again fee included). With a stop, the first sell
          swap after a sell price at or below (1 - stop) x entry price.
          Without a sell swap within H + 24 hours the position is worth zero.
  costs   on top of the observed prices: the extra price impact of our own
          trade (impact_estimate) on entry and on exit, and gas for two swaps.
  size    min(position_eth, cap x the pool's buy volume in the 24 hours
          before s); the sum of these over all positions is the capacity.

Returns are in ETH: a token priced in USDC is converted at each swap's ETH
price (prices.py), so "hold ETH" earns zero by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAY = pd.Timedelta(hours=24)


@dataclass
class Settings:
    """The backtest's settings; those tuned on development data are K, holding_hours, stop and the signal version."""

    holding_hours: float = 24
    stop: float | None = None  # e.g. 0.5 = exit after a 50% fall
    position_eth: float = 0.5
    cap: float = 0.01  # share of the pool's 24-hour buy volume
    gas_gwei: float = 5.0
    gas_units: int = 150_000


class PoolPrices:
    """The priced swaps of each pool as sorted arrays, for fast lookups by time."""

    def __init__(self, prices: pd.DataFrame):
        ordered = prices.sort_values(["block", "tx_index", "log_index"], kind="stable").copy()
        # Times as plain UTC datetime64, so that numpy can compare and search them.
        ordered["timestamp"] = ordered["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
        self.by_pool = {
            pool: {
                "block": part["block"].to_numpy(),
                "time": part["timestamp"].to_numpy(),
                "price": part["price_eth"].to_numpy(),
                "buy": (part["side"] == "buy").to_numpy(),
                "volume": part["volume_eth"].to_numpy(),
            }
            for pool, part in ordered.groupby("pool", sort=False)
        }
        self.times = ordered["timestamp"].to_numpy()  # UTC, without time zone
        self.blocks = ordered["block"].to_numpy()

    def pool(self, pool: str) -> dict | None:
        """The arrays of one pool, or None if it has no priced swap."""
        return self.by_pool.get(pool)

    def first_block_at(self, when: np.datetime64) -> int | None:
        """The block of the first priced swap at or after `when` (the decision block), or None after the data ends."""
        i = np.searchsorted(self.times, when, side="left")
        return int(self.blocks[i]) if i < len(self.times) else None


def impact_estimate(pool: dict, before: np.datetime64, size_eth: float, days: int = 7) -> float:
    """The extra relative price move a trade of size_eth would cause in a pool, from swaps in the days before.

    For each earlier swap: the absolute log price change from the swap before
    it to the swap after it. The estimate is the median of these over swaps
    whose volume lies within a factor 2 of size_eth. With fewer than 5 such
    swaps, the median move of the pool's swaps is scaled by
    size / their median volume. Capped at 50%.
    """
    start = np.searchsorted(pool["time"], before - np.timedelta64(days, "D"))
    end = np.searchsorted(pool["time"], before)
    price, volume = pool["price"][start:end], pool["volume"][start:end]
    if len(price) < 3:
        return 0.5
    moves = np.abs(np.log(price[2:]) - np.log(price[:-2]))
    middle_volume = volume[1:-1]
    near = (middle_volume >= size_eth / 2) & (middle_volume <= size_eth * 2)
    if near.sum() >= 5:
        estimate = float(np.median(moves[near]))
    else:
        estimate = float(np.median(moves) * size_eth / max(np.median(middle_volume), 1e-12))
    return min(estimate, 0.5)


def open_and_close(pool: dict, decision_block: int, entry_after: np.datetime64, settings: Settings) -> dict:
    """Entry and exit of one position in one pool (module docstring). Returns times, prices and the exit reason."""
    blocks, times, price, buy = pool["block"], pool["time"], pool["price"], pool["buy"]
    after = np.flatnonzero((blocks > decision_block) & buy)
    if len(after) == 0:
        return {"entered": False}
    i = after[0]
    entry_time, entry_price = times[i], price[i]
    horizon = entry_time + np.timedelta64(int(settings.holding_hours * 3600), "s")
    give_up = horizon + np.timedelta64(24 * 3600, "s")
    sells = np.flatnonzero(~buy & (np.arange(len(buy)) > i))
    if settings.stop is not None:
        fallen = sells[(price[sells] <= (1 - settings.stop) * entry_price) & (times[sells] < horizon)]
        if len(fallen):
            later = sells[sells > fallen[0]]
            if len(later) and times[later[0]] < give_up:
                return {"entered": True, "entry_time": entry_time, "entry_price": entry_price,
                        "exit_time": times[later[0]], "exit_price": price[later[0]], "reason": "stop"}  # fmt: skip
            return {"entered": True, "entry_time": entry_time, "entry_price": entry_price, "exit_time": None, "exit_price": 0.0, "reason": "no exit"}
    due = sells[times[sells] >= horizon]
    if len(due) and times[due[0]] < give_up:
        return {"entered": True, "entry_time": entry_time, "entry_price": entry_price,
                "exit_time": times[due[0]], "exit_price": price[due[0]], "reason": "time"}  # fmt: skip
    return {"entered": True, "entry_time": entry_time, "entry_price": entry_price, "exit_time": None, "exit_price": 0.0, "reason": "no exit"}


def run(selections: pd.DataFrame, pool_prices: PoolPrices, settings: Settings) -> pd.DataFrame:
    """One row per position of the selections (columns decision, token, pool, buy_eth_24h).

    Columns of the result: decision, token, pool, size_eth, entry_time,
    entry_price, exit_time, exit_price, reason, impact, gas_share,
    gross_return (on observed prices), net_return (after our impact and gas).
    Selections whose pool has no buy swap after the decision are listed with
    reason "not entered" and no return.
    """
    gas_eth = 2 * settings.gas_gwei * settings.gas_units * 1e-9
    rows = []
    for item in selections.itertuples(index=False):
        pool = pool_prices.pool(item.pool)
        decision = np.datetime64(pd.Timestamp(item.decision).tz_convert("UTC").tz_localize(None))
        base = {"decision": item.decision, "token": item.token, "pool": item.pool}
        decision_block = pool_prices.first_block_at(decision)
        if pool is None or decision_block is None:
            rows.append({**base, "reason": "not entered"})
            continue
        size = min(settings.position_eth, settings.cap * float(item.buy_eth_24h))
        trade = open_and_close(pool, decision_block, decision, settings)
        if not trade["entered"] or size <= 0:
            rows.append({**base, "reason": "not entered"})
            continue
        impact = impact_estimate(pool, decision, size)
        gross = trade["exit_price"] / trade["entry_price"] - 1
        net = trade["exit_price"] * (1 - impact) / (trade["entry_price"] * (1 + impact)) - 1 - gas_eth / size
        rows.append({**base, "size_eth": size, **{k: trade[k] for k in ("entry_time", "entry_price", "exit_time", "exit_price", "reason")},
                     "impact": impact, "gas_share": gas_eth / size, "gross_return": gross, "net_return": net})  # fmt: skip
    return pd.DataFrame(rows)


def daily_returns(trades: pd.DataFrame, column: str = "net_return") -> pd.Series:
    """Mean return per decision (over its entered positions), then the mean of the decisions of each day.

    A decision with no entered position counts as 0 (the capital stays in ETH).
    """
    entered = trades[trades["reason"] != "not entered"]
    per_decision = entered.groupby("decision")[column].mean().reindex(trades["decision"].unique()).fillna(0.0)
    days = pd.to_datetime(per_decision.index).floor("D")
    return per_decision.groupby(days).mean()
