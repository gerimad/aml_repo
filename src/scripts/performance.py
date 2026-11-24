"""Performance indicators of a backtested strategy (work package 8).

The backtest gives one row per position. A portfolio needs to know how the
capital is shared between overlapping positions, so the equity curve is
built from sleeves:

  - the capital is split into n = ceil(holding hours / decision interval) + 1
    equal sleeves (24-hour holds with a decision every 6 hours give 5: four
    overlapping positions and one spare, because exits come a little after
    the holding time);
  - decision j uses sleeve j mod n, and puts it into its entered positions
    in equal parts (a decision with no entered position leaves the sleeve
    in ETH);
  - a sleeve's value changes when its positions close (realized, not marked
    to market in between).

The daily return is the change of the total value from one day's end to the
next. Every indicator below is computed from those daily returns or from the
positions. Returns are in ETH (holding ETH earns 0), and crypto trades every
day, so annual figures use 365 days.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sps

DAYS_PER_YEAR = 365


def equity_curve(trades: pd.DataFrame, holding_hours: float, interval_hours: float = 6, column: str = "net_return") -> pd.Series:
    """Total value at the end of each day, starting at 1, from the positions of the backtest (module docstring).

    There is one sleeve more than holding hours / interval, because an exit
    comes a little after the holding time. A decision whose sleeve is still
    invested (its earlier positions have not all closed) is skipped and
    counted as not invested; their number is in the attribute "skipped".
    """
    n_sleeves = math.ceil(holding_hours / interval_hours) + 1
    value = np.full(n_sleeves, 1.0 / n_sleeves)
    busy_until = [pd.Timestamp.min.tz_localize("UTC")] * n_sleeves
    changes = []  # (time, sleeve, value after the change)
    skipped = 0
    decisions = sorted(pd.to_datetime(trades["decision"].unique(), utc=True))
    for j, decision in enumerate(decisions):
        sleeve = j % n_sleeves
        entered = trades[(pd.to_datetime(trades["decision"], utc=True) == decision) & (trades["reason"] != "not entered")]
        if entered.empty:
            continue
        if decision < busy_until[sleeve]:
            skipped += 1
            continue
        # Positions without an exit are worth nothing; they close at the give-up time.
        exits = pd.to_datetime(entered["exit_time"], utc=True).fillna(decision + pd.Timedelta(hours=holding_hours + 24))
        value[sleeve] *= float((1 + entered[column].to_numpy()).mean())
        busy_until[sleeve] = exits.max()
        changes.append((exits.max(), sleeve, value[sleeve]))

    first_day = decisions[0].floor("D")
    last_day = max([decisions[-1]] + [c[0] for c in changes]).floor("D")
    days = pd.date_range(first_day, last_day, freq="D")
    current = np.full(n_sleeves, 1.0 / n_sleeves)
    changes.sort(key=lambda c: c[0])
    k, totals = 0, []
    for day in days:
        end = day + pd.Timedelta(days=1)
        while k < len(changes) and changes[k][0] < end:
            current[changes[k][1]] = changes[k][2]
            k += 1
        totals.append(current.sum())
    curve = pd.Series(totals, index=days, name="value")
    curve.attrs["skipped"] = skipped
    return curve


def daily_returns_from_equity(equity: pd.Series) -> pd.Series:
    """Day-over-day relative change of the equity curve (the first day against the starting value 1)."""
    return equity / equity.shift(1, fill_value=1.0) - 1


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """The largest fall from a previous peak (as a positive fraction) and the longest time under a peak, in days."""
    peak = np.maximum.accumulate(np.r_[1.0, equity.to_numpy()])[1:]
    drawdown = 1 - equity.to_numpy() / peak
    under = drawdown > 0
    longest, current = 0, 0
    for flag in under:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return float(drawdown.max()), longest


def sharpe(daily: pd.Series) -> float:
    """Annualized Sharpe ratio: mean / standard deviation of daily returns, times sqrt(365). The risk-free rate in ETH is 0."""
    sd = daily.std(ddof=1)
    return float(daily.mean() / sd * math.sqrt(DAYS_PER_YEAR)) if sd > 0 else float("nan")


def sortino(daily: pd.Series) -> float:
    """Annualized Sortino ratio: mean / downside deviation (root mean square of the negative daily returns)."""
    downside = math.sqrt(float((np.minimum(daily, 0) ** 2).mean()))
    return float(daily.mean() / downside * math.sqrt(DAYS_PER_YEAR)) if downside > 0 else float("nan")


def kelly_continuous(daily: pd.Series) -> float:
    """Kelly fraction for a continuous return: mean / variance of the daily returns."""
    var = daily.var(ddof=1)
    return float(daily.mean() / var) if var > 0 else float("nan")


def kelly_discrete(position_returns: pd.Series) -> float:
    """Kelly fraction from the positions: p - (1 - p) / b, with p the hit rate and b the average win over the average loss."""
    wins, losses = position_returns[position_returns > 0], position_returns[position_returns < 0]
    if wins.empty or losses.empty:
        return float("nan")
    p, b = len(wins) / len(position_returns), wins.mean() / -losses.mean()
    return float(p - (1 - p) / b)


def sharpe_interval(daily: pd.Series, resamples: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of the annualized Sharpe ratio, resampling days."""
    rng = np.random.default_rng(seed)
    values = daily.to_numpy()
    draws = rng.choice(values, size=(resamples, len(values)), replace=True)
    sd = draws.std(axis=1, ddof=1)
    ratios = np.where(sd > 0, draws.mean(axis=1) / np.where(sd > 0, sd, 1) * math.sqrt(DAYS_PER_YEAR), np.nan)
    low, high = np.nanpercentile(ratios, [2.5, 97.5])
    return float(low), float(high)


def deflated_sharpe(daily: pd.Series, trials: int, trial_sharpes_sd: float) -> float:
    """Probability that the true Sharpe ratio is above zero after trying `trials` settings (Bailey and López de Prado, 2014).

    The best of `trials` noisy Sharpe ratios is expected to exceed zero by
    chance; the benchmark SR0 is that expectation, from the spread of the
    trials' (daily, not annualized) Sharpe ratios. The result corrects the
    observed daily Sharpe ratio for the number of days, skewness and
    kurtosis of the returns. Values near 1 mean the result is unlikely to be
    a product of the search alone.
    """
    n = len(daily)
    sr = daily.mean() / daily.std(ddof=1)
    gamma = 0.5772156649  # Euler-Mascheroni constant
    if trials > 1:
        sr0 = trial_sharpes_sd * ((1 - gamma) * sps.norm.ppf(1 - 1 / trials) + gamma * sps.norm.ppf(1 - 1 / (trials * math.e)))
    else:
        sr0 = 0.0
    skew, kurt = sps.skew(daily), sps.kurtosis(daily, fisher=False)
    denominator = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    return float(sps.norm.cdf((sr - sr0) * math.sqrt(n - 1) / denominator))


def report(
    trades: pd.DataFrame, holding_hours: float, column: str = "net_return", trials: int = 1, trial_sharpes_sd: float = 0.0, interval_hours: float = 6
) -> dict:
    """All indicators of one backtest, as one row. interval_hours is the time between decisions (1 for launch momentum)."""
    entered = trades[trades["reason"] != "not entered"]
    equity = equity_curve(trades, holding_hours, interval_hours=interval_hours, column=column)
    daily = daily_returns_from_equity(equity)
    drawdown, drawdown_days = max_drawdown(equity)
    years = len(daily) / DAYS_PER_YEAR
    total = float(equity.iloc[-1] - 1)
    annual = float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else float("nan")
    returns = entered[column]
    wins, losses = returns[returns > 0], returns[returns < 0]
    low, high = sharpe_interval(daily)
    return {
        "days": len(daily),
        "positions": len(entered),
        "total_return": total,
        "annual_return": annual,
        "daily_mean": float(daily.mean()),
        "daily_volatility": float(daily.std(ddof=1)),
        "sharpe": sharpe(daily),
        "sharpe_low": low,
        "sharpe_high": high,
        "deflated_sharpe": deflated_sharpe(daily, trials, trial_sharpes_sd),
        "sortino": sortino(daily),
        "max_drawdown": drawdown,
        "drawdown_days": drawdown_days,
        "calmar": annual / drawdown if drawdown > 0 else float("nan"),
        "hit_rate": float((returns > 0).mean()) if len(returns) else float("nan"),
        "average_win": float(wins.mean()) if len(wins) else float("nan"),
        "average_loss": float(losses.mean()) if len(losses) else float("nan"),
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("nan"),
        "kelly_continuous": kelly_continuous(daily),
        "kelly_discrete": kelly_discrete(returns),
        "no_exit_share": float((entered["reason"] == "no exit").mean()) if len(entered) else float("nan"),
        "capacity_eth": float(entered["size_eth"].sum() / max(trades["decision"].nunique(), 1)),
        "skipped_decisions": int(equity.attrs["skipped"]),
    }


# --------------------------------------------------------------------------
# Indicators for a strategy built on rare large gains (implementation plan 12.8)
# --------------------------------------------------------------------------


def log_growth(daily: pd.Series, resamples: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Mean daily ln(1 + r), the rate at which the portfolio compounds, with its 95% bootstrap interval over days."""
    logs = np.log1p(daily.to_numpy())
    rng = np.random.default_rng(seed)
    means = rng.choice(logs, size=(resamples, len(logs)), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(logs.mean()), float(low), float(high)


def probability_of_loss(daily: pd.Series, resamples: int = 10_000, seed: int = 0) -> float:
    """Share of resampled periods (same number of days, days drawn with replacement) that end below the start."""
    rng = np.random.default_rng(seed)
    logs = np.log1p(daily.to_numpy())
    totals = rng.choice(logs, size=(resamples, len(logs)), replace=True).sum(axis=1)
    return float((totals < 0).mean())


def top_share(returns: pd.Series, share: float) -> float:
    """Share of the total profit (sum of position returns) that comes from the best `share` of positions; NaN if the total is not positive."""
    total = returns.sum()
    if total <= 0 or returns.empty:
        return float("nan")
    n = max(1, int(math.ceil(share * len(returns))))
    return float(returns.nlargest(n).sum() / total)


def longest_losing_run(returns: pd.Series) -> int:
    """The longest run of consecutive losing positions, in the order given (entry time)."""
    longest, current = 0, 0
    for r in returns:
        current = current + 1 if r < 0 else 0
        longest = max(longest, current)
    return longest


def effective_bets_per_day(trades: pd.DataFrame, column: str = "net_return") -> tuple[float, float, float]:
    """Positions per day, their intraclass correlation within a day, and the effective number of independent bets per day.

    Positions opened on the same day share the market they live in; if they
    move together they count as fewer bets. With ICC the intraclass
    correlation of returns grouped by entry day and m the mean positions per
    day, the effective number is m / (1 + (m - 1) ICC). ICC is the one-way
    ANOVA estimate, floored at 0.
    """
    entered = trades[trades["reason"] != "not entered"].copy()
    entered["day"] = pd.to_datetime(entered["entry_time"]).dt.floor("D")
    groups = [g[column].to_numpy() for _, g in entered.groupby("day") if len(g) >= 2]
    if len(groups) < 2:
        return float("nan"), float("nan"), float("nan")
    sizes = np.array([len(g) for g in groups])
    grand = np.concatenate(groups).mean()
    between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (len(groups) - 1)
    within = sum(((g - g.mean()) ** 2).sum() for g in groups) / (sizes.sum() - len(groups))
    m0 = (sizes.sum() - (sizes**2).sum() / sizes.sum()) / (len(groups) - 1)
    icc = max(0.0, (between - within) / (between + (m0 - 1) * within)) if between + (m0 - 1) * within > 0 else 0.0
    m = sizes.mean()
    return float(m), float(icc), float(m / (1 + (m - 1) * icc))


def fat_tail_report(trades: pd.DataFrame, holding_hours: float, interval_hours: float = 1, column: str = "net_return") -> dict:
    """The indicators of implementation plan 12.8, as one row."""
    entered = trades[trades["reason"] != "not entered"].sort_values("entry_time")
    returns = entered[column]
    daily = daily_returns_from_equity(equity_curve(trades, holding_hours, interval_hours, column))
    growth, growth_low, growth_high = log_growth(daily)
    per_day, icc, effective = effective_bets_per_day(trades, column)
    return {
        "log_growth": growth,
        "log_growth_low": growth_low,
        "log_growth_high": growth_high,
        "probability_of_loss": probability_of_loss(daily),
        "mean_position": float(returns.mean()),
        "median_position": float(returns.median()),
        "skewness": float(sps.skew(returns)) if len(returns) > 2 else float("nan"),
        "top_1pct_share": top_share(returns, 0.01),
        "top_5pct_share": top_share(returns, 0.05),
        "longest_losing_run": longest_losing_run(returns),
        "positions_per_day": per_day,
        "same_day_correlation": icc,
        "effective_bets_per_day": effective,
    }
