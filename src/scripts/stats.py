"""Comparing a model with the baseline over evaluation points (work package 2).

The unit of comparison is the evaluation point, not the user: the market
situation of a point moves recall far more than the sampling of users does
(project plan, section 5.4). So every comparison is paired by point:

    d_p = recall@10(model at point p) - recall@10(baseline at point p)

and the statistics are about the list of d_p.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def paired_differences(model: pd.DataFrame, baseline: pd.DataFrame, metric: str = "recall") -> pd.Series:
    """d_p for each point, indexed by point.

    Both arguments are Result.per_point tables. Raises if they do not cover
    exactly the same points, because then the comparison would not be paired.
    """
    a = model.set_index("point")[metric]
    b = baseline.set_index("point")[metric]
    if set(a.index) != set(b.index):
        raise ValueError("the two results must cover the same evaluation points")
    return (a - b.loc[a.index]).rename("difference")


def bootstrap_over_points(d, n_boot: int = 10_000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean of the paired differences.

    Points are resampled with replacement. With 14 points this interval is
    rough, so the individual d_p are always reported next to it.
    """
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    low, high = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(low), float(high)


def sign_test(d) -> dict:
    """Exact two-sided sign test of the paired differences.

    Returns wins (d > 0), losses (d < 0), ties (d == 0, left out of the test)
    and the p-value. With 14 points and no ties, 12 wins are needed for p < 0.05.
    """
    d = np.asarray(d, float)
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    p_value = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0
    return {"wins": wins, "losses": losses, "ties": len(d) - wins - losses, "p_value": float(p_value)}


def compare(results: dict[str, pd.DataFrame], baseline: str, metric: str = "recall") -> pd.DataFrame:
    """The standard comparison table: one row per model, the baseline first.

    `results` maps a model name to its Result.per_point table. Columns:
      mean, lowest, highest      the metric over points
      mean_difference            mean of d_p against the baseline
      interval_low/high          95% bootstrap interval of that mean, over points
      wins                       "w of n" points where the model is better
      sign_test_p                two-sided exact sign test
    """
    reference = results[baseline]
    rows = []
    for name, table in results.items():
        row = {
            "model": name,
            "points": len(table),
            "mean": table[metric].mean(),
            "lowest": table[metric].min(),
            "highest": table[metric].max(),
        }
        if name != baseline:
            d = paired_differences(table, reference, metric)
            low, high = bootstrap_over_points(d)
            signs = sign_test(d)
            row.update({
                "mean_difference": d.mean(),
                "interval_low": low,
                "interval_high": high,
                "wins": f"{signs['wins']} of {len(d)}",
                "sign_test_p": signs["p_value"],
            })  # fmt: skip
        rows.append(row)
    table = pd.DataFrame(rows)
    order = [baseline] + [n for n in results if n != baseline]
    return table.set_index("model").loc[order].reset_index()


def success(model: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    """The success criterion of the project plan, checked on the given per-point tables.

    Success: the mean paired difference in recall@10 is positive and its 95%
    bootstrap interval over points excludes zero.
    """
    d = paired_differences(model, baseline)
    low, high = bootstrap_over_points(d)
    return {"mean_difference": float(d.mean()), "interval": (low, high), "success": bool(d.mean() > 0 and low > 0)}
