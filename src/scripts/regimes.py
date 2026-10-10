"""Building blocks for the market regime experiments (HMM vs. MDL segmentation).

Everything here works on plain NumPy arrays or pandas objects, so it can be
used from notebooks and tested in isolation. From a notebook in src/notebooks:

    import sys; sys.path.append("../scripts")
    import regimes as rg

    daily = rg.daily_bars(hourly)                      # returns, realized variance
    parts = rg.chrono_split(daily, "2022-12-31", "2024-12-31")
    breaks, bits = rg.mdl_segment(parts["train"]["ret"].to_numpy())
    model = rg.fit_gaussian_hmm(parts["train"]["ret"].to_numpy(), n_states=3)
    out = rg.hmm_filter(daily["ret"].to_numpy(), **rg.hmm_params(model))
    out["loglik"]                                      # log p(return_t | past)

All "forecast" functions are causal: the value at position t uses only
observations before t, so they can be scored out of sample without leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp

LOG_2PI = np.log(2 * np.pi)


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------


def daily_bars(hourly: pd.DataFrame, min_hours: int = 24) -> pd.DataFrame:
    """Aggregate hourly candles to UTC days.

    Returns one row per day with:
      close      last hourly close of the day
      ret        close-to-close log return
      rv         realized variance, the sum of squared hourly log returns
      volume     traded volume in quote currency (USDT)
      n_hours    number of hourly bars present
      gap_days   calendar days since the previous row (1 when no day is missing)
      complete   True if the day has at least `min_hours` bars and follows
                 directly on the previous day, so `ret` and `rv` cover one day

    The first day is dropped because it has no previous close. Days without
    any hourly bar do not appear at all; the next day then has gap_days > 1.
    """
    hourly_ret = np.log(hourly["close"]).diff()
    day = hourly.index.floor("D")
    daily = pd.DataFrame(
        {
            "close": hourly["close"].groupby(day).last(),
            "rv": (hourly_ret**2).groupby(day).sum(),
            "volume": hourly["quote_volume"].groupby(day).sum(),
            "n_hours": hourly["close"].groupby(day).size(),
        }
    )
    daily["ret"] = np.log(daily["close"]).diff()
    daily["gap_days"] = daily.index.to_series().diff().dt.days
    daily["complete"] = (daily["n_hours"] >= min_hours) & (daily["gap_days"] == 1)
    daily.index.name = "date"
    daily = daily.iloc[1:].astype({"gap_days": int})
    return daily[["close", "ret", "rv", "volume", "n_hours", "gap_days", "complete"]]


def chrono_split(df: pd.DataFrame, train_end: str, dev_end: str) -> dict[str, pd.DataFrame]:
    """Split a time-indexed frame into train / dev / test by date, in order.

    `train_end` and `dev_end` are the last days (inclusive) of train and dev.
    The parts never overlap and never shuffle, so the future cannot leak back.
    """
    train_end, dev_end = pd.Timestamp(train_end, tz="UTC"), pd.Timestamp(dev_end, tz="UTC")
    if not train_end < dev_end:
        raise ValueError("train_end must be before dev_end")
    index = df.index
    return {
        "train": df[index <= train_end],
        "dev": df[(index > train_end) & (index <= dev_end)],
        "test": df[index > dev_end],
    }


# --------------------------------------------------------------------------
# Densities and scores
# --------------------------------------------------------------------------


def gaussian_logpdf(x, mean, var):
    """Log density of N(mean, var) at x, elementwise."""
    x, mean, var = np.asarray(x, float), np.asarray(mean, float), np.asarray(var, float)
    return -0.5 * (LOG_2PI + np.log(var) + (x - mean) ** 2 / var)


def student_t_logpdf(x, mean, var, df: float):
    """Log density of a Student-t with the given mean and variance (df > 2)."""
    scale = np.sqrt(np.asarray(var, float) * (df - 2) / df)
    return stats.t.logpdf(np.asarray(x, float), df, loc=mean, scale=scale)


def qlike(realized_var, forecast_var) -> float:
    """QLIKE loss of a variance forecast: mean(rv/f - log(rv/f) - 1).

    Zero for a perfect forecast, and robust to noise in the realized variance
    proxy (Patton, 2011). Lower is better.
    """
    ratio = np.asarray(realized_var, float) / np.asarray(forecast_var, float)
    return float(np.mean(ratio - np.log(ratio) - 1))


def mean_diff_se(a, b, lags: int = 10) -> tuple[float, float]:
    """Mean of a - b and its standard error, robust to autocorrelation.

    Use it to judge whether one model's daily scores beat another's by more
    than noise (a Diebold-Mariano style comparison). The standard error is
    the Newey-West estimate with a Bartlett kernel over `lags` lags.
    """
    diff = np.asarray(a, float) - np.asarray(b, float)
    n = len(diff)
    centered = diff - diff.mean()
    long_run_var = centered @ centered / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1 - lag / (lags + 1)
        long_run_var += 2 * weight * (centered[lag:] @ centered[:-lag]) / n
    return float(diff.mean()), float(np.sqrt(max(long_run_var, 0.0) / n))


def run_lengths(labels) -> np.ndarray:
    """Lengths of the runs of equal consecutive values, in order."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return np.array([], dtype=int)
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    return np.diff(np.concatenate([[0], edges, [len(labels)]]))


def bic(loglik: float, n_params: int, n_obs: int) -> float:
    """Bayesian information criterion, lower is better."""
    return -2 * loglik + n_params * np.log(n_obs)


# --------------------------------------------------------------------------
# Baseline variance forecasts
# --------------------------------------------------------------------------


def ewma_variance(returns, lam: float = 0.94, init_var: float | None = None) -> np.ndarray:
    """One-step-ahead EWMA variance forecasts (RiskMetrics).

    forecast[t] = lam * forecast[t-1] + (1 - lam) * returns[t-1]**2, so
    forecast[t] only uses returns before t. forecast[0] is `init_var`
    (default: the squared first return).
    """
    returns = np.asarray(returns, float)
    forecast = np.empty(len(returns))
    forecast[0] = returns[0] ** 2 if init_var is None else init_var
    for t in range(1, len(returns)):
        forecast[t] = lam * forecast[t - 1] + (1 - lam) * returns[t - 1] ** 2
    return forecast


def garch_variance(returns, omega: float, alpha: float, beta: float, mu: float, init_var: float):
    """One-step-ahead GARCH(1,1) variance forecasts for fixed parameters.

    forecast[t] = omega + alpha * (returns[t-1] - mu)**2 + beta * forecast[t-1],
    with forecast[0] = init_var.
    """
    returns = np.asarray(returns, float)
    forecast = np.empty(len(returns))
    forecast[0] = init_var
    for t in range(1, len(returns)):
        forecast[t] = omega + alpha * (returns[t - 1] - mu) ** 2 + beta * forecast[t - 1]
    return forecast


# --------------------------------------------------------------------------
# MDL segmentation
# --------------------------------------------------------------------------


def mdl_segment(x, min_len: int = 10) -> tuple[list[int], float]:
    """Segment a series into Gaussian pieces of minimum description length.

    Each segment is coded as i.i.d. N(mean_j, var_j). The total code length
    (two-part MDL, in nats, up to a constant) is the sum over segments of

        n_j / 2 * (log(2 pi var_j) + 1)    data given the segment's parameters
        + log(n_j)                         two parameters at 1/2 log(n_j) each
        + log(N)                           position of the segment boundary

    The exact minimum over all segmentations is found by dynamic programming
    in O(N^2) time. Returns (breakpoints, code_length), where breakpoints are
    the end indices (exclusive) of the segments, the last one being N.
    """
    x = np.asarray(x, float)
    n = len(x)
    if n < min_len:
        raise ValueError("series is shorter than min_len")
    csum = np.concatenate([[0.0], np.cumsum(x)])
    csum2 = np.concatenate([[0.0], np.cumsum(x**2)])
    var_floor = 1e-12 * max(np.var(x), 1e-300)

    best = np.full(n + 1, np.inf)  # best[t]: minimal code length of x[:t]
    best[0] = 0.0
    prev = np.zeros(n + 1, dtype=int)
    for end in range(min_len, n + 1):
        starts = np.arange(0, end - min_len + 1)
        length = end - starts
        total = csum[end] - csum[starts]
        var = (csum2[end] - csum2[starts]) / length - (total / length) ** 2
        var = np.maximum(var, var_floor)
        cost = 0.5 * length * (LOG_2PI + np.log(var) + 1) + np.log(length) + np.log(n)
        candidates = best[starts] + cost
        i = int(np.argmin(candidates))
        best[end], prev[end] = candidates[i], starts[i]

    breaks, end = [], n
    while end > 0:
        breaks.append(end)
        end = int(prev[end])
    return breaks[::-1], float(best[n])


def segment_labels(n: int, breaks: list[int]) -> np.ndarray:
    """Turn breakpoints (segment end indices) into one segment id per observation."""
    labels = np.empty(n, dtype=int)
    start = 0
    for i, end in enumerate(breaks):
        labels[start:end] = i
        start = end
    return labels


# --------------------------------------------------------------------------
# Gaussian hidden Markov model
# --------------------------------------------------------------------------


def fit_gaussian_hmm(x, n_states: int, n_init: int = 10, seed: int = 0, n_iter: int = 500):
    """Fit a Gaussian HMM by EM, keeping the best of `n_init` random starts.

    `x` is one series of shape (n,), or several observed together with shape
    (n, d); dimensions are modelled as independent within a state (diagonal
    covariance). EM only finds a local optimum, so several starts are needed.
    The states of the returned hmmlearn model are sorted by increasing
    variance of the first dimension.
    """
    import logging

    from hmmlearn.hmm import GaussianHMM

    # hmmlearn logs a message whenever EM stops on a tiny likelihood decrease.
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)

    obs = np.asarray(x, float)
    obs = obs.reshape(-1, 1) if obs.ndim == 1 else obs
    best_model, best_score = None, -np.inf
    for i in range(n_init):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,
            tol=1e-6,
            random_state=seed + i,
        )
        try:
            model.fit(obs)
            score = model.score(obs)
        except ValueError:
            continue  # a degenerate start (for example an empty state)
        if np.isfinite(score) and score > best_score:
            best_model, best_score = model, score
    if best_model is None:
        raise RuntimeError("no HMM fit converged")

    variances = best_model._covars_.copy()  # shape (n_states, d) for "diag"
    order = np.argsort(variances[:, 0])
    best_model.startprob_ = best_model.startprob_[order]
    best_model.transmat_ = best_model.transmat_[np.ix_(order, order)]
    best_model.means_ = best_model.means_[order]
    best_model.covars_ = variances[order]
    return best_model


def hmm_params(model) -> dict[str, np.ndarray]:
    """Extract the parameters of a fitted hmmlearn Gaussian HMM for `hmm_filter`.

    Means and variances have shape (n_states,) for a one-dimensional model
    and (n_states, d) otherwise.
    """
    means, variances = model.means_, model._covars_
    if means.shape[1] == 1:
        means, variances = means[:, 0], variances[:, 0]
    return {
        "startprob": model.startprob_.copy(),
        "transmat": model.transmat_.copy(),
        "means": means.copy(),
        "variances": variances.copy(),
    }


def hmm_n_params(n_states: int, n_dims: int = 1) -> int:
    """Free parameters of a diagonal Gaussian HMM: transitions, start, means, variances."""
    return n_states * (n_states - 1) + (n_states - 1) + 2 * n_states * n_dims


def forward_filter(log_emission, startprob, transmat) -> dict[str, np.ndarray]:
    """Forward recursion of an HMM for any emission model.

    `log_emission[t, j]` is log p(x[t] | state j). Returns, for each t:
      prior[t]     P(state_t | x[:t]), using only earlier observations
      filtered[t]  P(state_t | x[:t+1])
      loglik[t]    log p(x[t] | x[:t])
    """
    log_emission = np.asarray(log_emission, float)
    n, k = log_emission.shape
    log_trans = np.log(np.asarray(transmat, float) + 1e-300)
    log_prior = np.log(np.asarray(startprob, float) + 1e-300)

    prior, filtered, loglik = np.empty((n, k)), np.empty((n, k)), np.empty(n)
    for t in range(n):
        log_prior = log_prior - logsumexp(log_prior)
        prior[t] = np.exp(log_prior)
        joint = log_prior + log_emission[t]
        loglik[t] = logsumexp(joint)
        log_post = joint - loglik[t]
        filtered[t] = np.exp(log_post)
        log_prior = logsumexp(log_post[:, None] + log_trans, axis=0)
    return {"prior": prior, "filtered": filtered, "loglik": loglik}


def hmm_filter(x, startprob, transmat, means, variances) -> dict[str, np.ndarray]:
    """Filter a Gaussian HMM and return causal one-step-ahead quantities.

    `x` has shape (n,) or (n, d), matching `means` and `variances` from
    `hmm_params`. For each t, using only observations before t:
      prior[t]            P(state_t | past), the predicted state distribution
      loglik[t]           log p(x[t] | past), all dimensions jointly
      loglik_marginal[t]  log p(x[t, i] | past) for each dimension i alone
      mean[t], var[t]     predictive mean and variance of each dimension
    and, using observations up to and including t:
      filtered[t]         P(state_t | x[:t+1])
    For a one-dimensional `x` the per-dimension outputs have shape (n,).
    """
    x = np.asarray(x, float)
    one_dim = x.ndim == 1
    obs = x.reshape(len(x), -1)
    means = np.asarray(means, float).reshape(-1, obs.shape[1])
    variances = np.asarray(variances, float).reshape(-1, obs.shape[1])

    # per_dim[t, j, i] = log N(x[t, i]; means[j, i], variances[j, i])
    per_dim = gaussian_logpdf(obs[:, None, :], means[None], variances[None])
    out = forward_filter(per_dim.sum(axis=2), startprob, transmat)

    prior = out["prior"]
    mean = prior @ means
    var = prior @ (variances + means**2) - mean**2
    marginal = logsumexp(np.log(prior + 1e-300)[:, :, None] + per_dim, axis=1)
    if one_dim:
        mean, var, marginal = mean[:, 0], var[:, 0], marginal[:, 0]
    return {**out, "loglik_marginal": marginal, "mean": mean, "var": var}
