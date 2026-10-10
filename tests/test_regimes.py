"""Tests for src/scripts/regimes.py."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import regimes as rg


def make_hourly(n_hours=72, start="2024-01-01"):
    index = pd.date_range(start, periods=n_hours, freq="h", tz="UTC")
    close = 100 * np.exp(np.cumsum(np.full(n_hours, 0.001)))
    return pd.DataFrame({"close": close, "quote_volume": 10.0}, index=index)


def test_daily_bars_returns_and_realized_variance():
    daily = rg.daily_bars(make_hourly())
    # 3 days of data, the first is dropped (no previous close).
    assert len(daily) == 2
    # Every hourly log return is 0.001, so a day returns 24 * 0.001.
    assert daily["ret"].to_numpy() == pytest.approx([0.024, 0.024])
    assert daily["rv"].to_numpy() == pytest.approx([24 * 0.001**2] * 2)
    assert daily["volume"].to_numpy() == pytest.approx([240.0, 240.0])
    assert daily["complete"].all()


def test_daily_bars_flags_incomplete_days():
    hourly = make_hourly().drop(pd.Timestamp("2024-01-02 05:00", tz="UTC"))
    daily = rg.daily_bars(hourly)
    assert daily["n_hours"].tolist() == [23, 24]
    assert daily["complete"].tolist() == [False, True]
    # The price change over the gap is still counted, so returns are unchanged.
    assert daily["ret"].to_numpy() == pytest.approx([0.024, 0.024])


def test_daily_bars_flags_day_after_a_missing_day():
    hourly = make_hourly(96)
    hourly = hourly[hourly.index.floor("D") != pd.Timestamp("2024-01-03", tz="UTC")]
    daily = rg.daily_bars(hourly)
    assert daily["gap_days"].tolist() == [1, 2]
    assert daily["complete"].tolist() == [True, False]
    # The return after the gap spans two days.
    assert daily["ret"].to_numpy() == pytest.approx([0.024, 0.048])


def test_chrono_split_is_ordered_and_disjoint():
    index = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame({"v": range(10)}, index=index)
    parts = rg.chrono_split(df, "2024-01-04", "2024-01-07")
    assert [len(parts[k]) for k in ("train", "dev", "test")] == [4, 3, 3]
    assert parts["train"].index.max() < parts["dev"].index.min()
    assert parts["dev"].index.max() < parts["test"].index.min()
    assert sum(len(p) for p in parts.values()) == len(df)


def test_chrono_split_rejects_wrong_order():
    df = pd.DataFrame({"v": [1]}, index=pd.date_range("2024-01-01", periods=1, tz="UTC"))
    with pytest.raises(ValueError):
        rg.chrono_split(df, "2024-02-01", "2024-01-01")


def test_gaussian_logpdf_matches_scipy():
    x = np.array([-1.0, 0.0, 2.5])
    expected = stats.norm.logpdf(x, loc=0.5, scale=2.0)
    assert rg.gaussian_logpdf(x, 0.5, 4.0) == pytest.approx(expected)


def test_student_t_logpdf_has_requested_variance_and_integrates_to_one():
    grid = np.linspace(-60, 60, 200001)
    density = np.exp(rg.student_t_logpdf(grid, 1.0, 4.0, df=5))
    step = grid[1] - grid[0]
    assert density.sum() * step == pytest.approx(1.0, abs=1e-4)
    assert (density * (grid - 1.0) ** 2).sum() * step == pytest.approx(4.0, rel=2e-2)


def test_qlike_is_zero_for_perfect_forecast_and_positive_otherwise():
    rv = np.array([1.0, 2.0, 3.0])
    assert rg.qlike(rv, rv) == pytest.approx(0.0)
    assert rg.qlike(rv, 2 * rv) > 0
    assert rg.qlike(rv, 0.5 * rv) > 0


def test_mean_diff_se_matches_plain_standard_error_without_lags():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0.3, 1, 400), rng.normal(0.0, 1, 400)
    mean, se = rg.mean_diff_se(a, b, lags=0)
    diff = a - b
    assert mean == pytest.approx(diff.mean())
    assert se == pytest.approx(diff.std() / np.sqrt(400))


def test_mean_diff_se_grows_with_positive_autocorrelation():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, 3000)
    smooth = np.convolve(noise, np.ones(10) / 10, mode="same")  # autocorrelated
    _, se_plain = rg.mean_diff_se(smooth, np.zeros(3000), lags=0)
    _, se_robust = rg.mean_diff_se(smooth, np.zeros(3000), lags=20)
    assert se_robust > 2 * se_plain


def test_run_lengths():
    assert rg.run_lengths([0, 0, 1, 1, 1, 0]).tolist() == [2, 3, 1]
    assert rg.run_lengths([5]).tolist() == [1]
    assert rg.run_lengths([]).tolist() == []


def test_bic():
    assert rg.bic(loglik=-10.0, n_params=3, n_obs=100) == pytest.approx(20 + 3 * np.log(100))


def test_ewma_variance_is_causal():
    returns = np.array([0.1, 0.2, -0.3, 0.05])
    forecast = rg.ewma_variance(returns, lam=0.9, init_var=0.04)
    assert forecast[0] == pytest.approx(0.04)
    assert forecast[1] == pytest.approx(0.9 * 0.04 + 0.1 * 0.1**2)
    assert forecast[2] == pytest.approx(0.9 * forecast[1] + 0.1 * 0.2**2)
    # Changing the last return must not change any forecast.
    changed = rg.ewma_variance(np.array([0.1, 0.2, -0.3, 9.0]), lam=0.9, init_var=0.04)
    assert changed == pytest.approx(forecast)


def test_ewma_variance_default_init():
    assert rg.ewma_variance(np.array([0.5, 0.1]))[0] == pytest.approx(0.25)


def test_garch_variance_recursion():
    returns = np.array([0.1, -0.2, 0.3])
    forecast = rg.garch_variance(returns, omega=0.01, alpha=0.1, beta=0.8, mu=0.0, init_var=0.05)
    assert forecast[0] == pytest.approx(0.05)
    assert forecast[1] == pytest.approx(0.01 + 0.1 * 0.01 + 0.8 * 0.05)
    assert forecast[2] == pytest.approx(0.01 + 0.1 * 0.04 + 0.8 * forecast[1])


def test_mdl_segment_finds_variance_change():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 300), rng.normal(0, 5, 300), rng.normal(0, 1, 300)])
    breaks, code_length = rg.mdl_segment(x)
    assert len(breaks) == 3 and breaks[-1] == 900
    assert abs(breaks[0] - 300) <= 15 and abs(breaks[1] - 600) <= 15
    # Segmenting must compress better than one Gaussian for the whole series.
    one_segment = 0.5 * 900 * (rg.LOG_2PI + np.log(x.var()) + 1) + 2 * np.log(900)
    assert code_length < one_segment


def test_mdl_segment_keeps_iid_noise_in_one_segment():
    x = np.random.default_rng(1).normal(0, 1, 500)
    breaks, _ = rg.mdl_segment(x)
    assert breaks == [500]


def test_mdl_segment_rejects_short_series():
    with pytest.raises(ValueError):
        rg.mdl_segment(np.zeros(3), min_len=10)


def test_segment_labels():
    assert rg.segment_labels(6, [2, 5, 6]).tolist() == [0, 0, 1, 1, 1, 2]


def simulate_hmm(n, transmat, means, sds, seed=0):
    rng = np.random.default_rng(seed)
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(len(means), p=transmat[states[t - 1]])
    return states, rng.normal(np.asarray(means)[states], np.asarray(sds)[states])


def test_fit_gaussian_hmm_recovers_two_volatility_states():
    transmat = np.array([[0.98, 0.02], [0.03, 0.97]])
    _, x = simulate_hmm(4000, transmat, [0.0, 0.0], [1.0, 4.0])
    model = rg.fit_gaussian_hmm(x, n_states=2, n_init=3)
    params = rg.hmm_params(model)
    # States come back sorted by variance.
    assert params["variances"][0] < params["variances"][1]
    assert np.sqrt(params["variances"]) == pytest.approx([1.0, 4.0], rel=0.15)
    assert np.diag(params["transmat"]) == pytest.approx([0.98, 0.97], abs=0.03)


def test_hmm_n_params():
    assert rg.hmm_n_params(1) == 2
    assert rg.hmm_n_params(3) == 6 + 2 + 6
    assert rg.hmm_n_params(3, n_dims=2) == 6 + 2 + 12


def test_hmm_params_shapes():
    _, x = simulate_hmm(600, np.array([[0.95, 0.05], [0.05, 0.95]]), [0.0, 0.0], [1.0, 4.0])
    one = rg.hmm_params(rg.fit_gaussian_hmm(x, n_states=2, n_init=1))
    assert one["means"].shape == (2,) and one["variances"].shape == (2,)
    assert one["transmat"].shape == (2, 2) and one["startprob"].shape == (2,)
    two = rg.hmm_params(rg.fit_gaussian_hmm(np.column_stack([x, x**2]), n_states=2, n_init=1))
    assert two["means"].shape == (2, 2) and two["variances"].shape == (2, 2)


def test_forward_filter_on_a_hand_computed_example():
    # Two states, emissions given directly as likelihoods.
    emission = np.array([[0.9, 0.2], [0.1, 0.8]])
    transmat = np.array([[0.7, 0.3], [0.4, 0.6]])
    out = rg.forward_filter(np.log(emission), [0.5, 0.5], transmat)
    lik0 = 0.5 * 0.9 + 0.5 * 0.2
    post0 = np.array([0.45, 0.10]) / lik0
    prior1 = post0 @ transmat
    lik1 = prior1 @ emission[1]
    assert out["prior"][0] == pytest.approx([0.5, 0.5])
    assert out["filtered"][0] == pytest.approx(post0)
    assert out["prior"][1] == pytest.approx(prior1)
    assert np.exp(out["loglik"]) == pytest.approx([lik0, lik1])


def test_hmm_filter_two_dimensions():
    transmat = np.array([[0.97, 0.03], [0.05, 0.95]])
    states, x = simulate_hmm(2000, transmat, [0.0, 0.0], [1.0, 3.0], seed=3)
    # A second, more informative series: its mean depends on the state.
    y = np.random.default_rng(4).normal(np.array([-1.0, 2.0])[states], 0.5)
    obs = np.column_stack([x, y])
    model = rg.fit_gaussian_hmm(obs, n_states=2, n_init=2)
    out = rg.hmm_filter(obs, **rg.hmm_params(model))

    assert out["loglik"].sum() == pytest.approx(model.score(obs), rel=1e-6)
    assert out["mean"].shape == (2000, 2) and out["var"].shape == (2000, 2)
    assert out["loglik_marginal"].shape == (2000, 2)
    assert (out["filtered"].argmax(axis=1) == states).mean() > 0.97
    # Predictive variance of x is a mix of the two state variances.
    assert out["var"][:, 0].min() > 0.8 and out["var"][:, 0].max() < 10.5


def test_hmm_filter_matches_hmmlearn_likelihood_and_is_causal():
    transmat = np.array([[0.95, 0.05], [0.10, 0.90]])
    states, x = simulate_hmm(1500, transmat, [0.0, 0.0], [1.0, 3.0], seed=2)
    model = rg.fit_gaussian_hmm(x, n_states=2, n_init=2)
    params = rg.hmm_params(model)
    out = rg.hmm_filter(x, **params)

    assert out["loglik"].sum() == pytest.approx(model.score(x.reshape(-1, 1)), rel=1e-6)
    assert out["loglik_marginal"] == pytest.approx(out["loglik"])
    assert out["prior"].sum(axis=1) == pytest.approx(np.ones(len(x)))
    assert out["filtered"].sum(axis=1) == pytest.approx(np.ones(len(x)))
    assert (out["var"] > 0).all()
    # The filtered state should track the true state most of the time.
    assert (out["filtered"].argmax(axis=1) == states).mean() > 0.85

    # Causality: changing the future leaves earlier outputs untouched.
    x2 = x.copy()
    x2[1000:] = 50.0
    out2 = rg.hmm_filter(x2, **params)
    assert out2["prior"][:1001] == pytest.approx(out["prior"][:1001])
    assert out2["loglik"][:1000] == pytest.approx(out["loglik"][:1000])
