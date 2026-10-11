"""Tests for src/scripts/complexity.py."""

import numpy as np
import pytest

import complexity as cx


def bits(text):
    return np.array([int(c) for c in text], dtype=np.uint8)


def test_sign_symbols_drops_zeros_and_nans():
    returns = [0.1, -0.2, 0.0, np.nan, 0.3]
    assert cx.sign_symbols(returns).tolist() == [1, 0, 1]
    assert cx.sign_symbols(returns, drop_zeros=False).tolist() == [1, 0, 0, 1]


def test_ternary_symbols():
    returns = [-0.02, -0.001, 0.0, 0.001, 0.03]
    assert cx.ternary_symbols(returns).tolist() == [0, 0, 1, 2, 2]
    assert cx.ternary_symbols(returns, band=0.005).tolist() == [0, 1, 1, 1, 2]


def test_quantile_symbols_are_balanced_even_with_ties():
    returns = np.array([0.0] * 6 + [1.0, 2.0])
    symbols = cx.quantile_symbols(returns, n_bins=4)
    assert np.bincount(symbols).tolist() == [2, 2, 2, 2]
    assert symbols[-1] == 3  # the largest return is in the top bin


def test_alphabet_size():
    assert cx.alphabet_size([0, 2, 1]) == 3
    assert cx.alphabet_size([0, 0]) == 2
    assert cx.alphabet_size([]) == 2


def test_lz76_phrases_on_textbook_examples():
    # Examples from Kaspar and Schuster (1987), as quoted by Giglio et al. (2008).
    assert cx.lz76_phrases(bits("0000000000")) == 2
    assert cx.lz76_phrases(bits("0101010101")) == 3
    assert cx.lz76_phrases(bits("0110001001")) == 5


def test_lz76_normalized_separates_random_from_regular():
    rng = np.random.default_rng(0)
    random = rng.integers(0, 2, 5000)
    periodic = np.tile([0, 1, 1, 0], 1250)
    assert cx.lz76_normalized(random) > 0.9
    assert cx.lz76_normalized(periodic) < 0.05


def test_compressed_bits_per_symbol():
    rng = np.random.default_rng(0)
    random = rng.integers(0, 2, 20000)
    constant = np.zeros(20000, dtype=np.uint8)
    for codec in ["zlib", "bz2", "lzma"]:
        assert 1.0 <= cx.compressed_bits_per_symbol(random, codec) < 1.4
        assert cx.compressed_bits_per_symbol(constant, codec) < 0.1
    with pytest.raises(ValueError):
        cx.compressed_bits_per_symbol(random, "zip")


def test_block_counts():
    assert cx.block_counts(bits("0110"), 1).tolist() == [2, 2]
    # Blocks of two: 01, 11, 10 -> codes 1, 3, 2.
    assert cx.block_counts(bits("0110"), 2).tolist() == [0, 1, 1, 1]
    assert cx.block_counts(bits("01"), 3).tolist() == [0] * 8


def test_entropy_bits():
    assert cx.entropy_bits([5, 5]) == pytest.approx(1.0)
    assert cx.entropy_bits([10, 0]) == pytest.approx(0.0)
    assert cx.entropy_bits([1, 1, 1, 1]) == pytest.approx(2.0)


def test_conditional_entropy_matches_theory_for_a_markov_chain():
    p = 0.7
    chain = cx.simulate_markov_chain(200_000, p_repeat=p, seed=1)
    expected = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    assert cx.conditional_entropy(chain, order=0) == pytest.approx(1.0, abs=1e-3)
    assert cx.conditional_entropy(chain, order=1) == pytest.approx(expected, abs=5e-3)
    assert cx.conditional_entropy(chain, order=3) == pytest.approx(expected, abs=5e-3)


def test_surrogate_test_detects_structure_and_accepts_noise():
    def statistic(s):
        return cx.conditional_entropy(s, order=1)

    structured = cx.surrogate_test(cx.simulate_markov_chain(4000, 0.6, seed=2), statistic, 99)
    assert structured["deficit"] > 0 and structured["z"] > 3
    assert structured["p_value"] == pytest.approx(0.01)

    noise = cx.surrogate_test(cx.simulate_markov_chain(4000, 0.5, seed=3), statistic, 99)
    assert abs(noise["z"]) < 3 and noise["p_value"] > 0.01
    assert noise["value"] == pytest.approx(noise["surrogate_mean"] - noise["deficit"])


def test_surrogate_test_is_reproducible():
    chain = cx.simulate_markov_chain(1000, 0.55, seed=4)
    first = cx.surrogate_test(chain, cx.lz76_phrases, 20, seed=7)
    second = cx.surrogate_test(chain, cx.lz76_phrases, 20, seed=7)
    assert first == second


def test_complexity_profile():
    chain = cx.simulate_markov_chain(3000, 0.65, seed=8)
    profile = cx.complexity_profile(chain, n_surrogates=30, n_slow_surrogates=5, seed=1)
    assert profile["n"] == 3000
    for name in cx.ESTIMATORS:
        assert {f"{name}_value", f"{name}_deficit", f"{name}_z", f"{name}_p_value"} <= set(profile)
    # A 65/35 chain is clearly regular for the entropy estimators.
    assert profile["H1_z"] > 5 and profile["H1_p_value"] == pytest.approx(1 / 31)
    assert profile["H1_deficit"] == pytest.approx(1 - 0.934, abs=0.02)
    # Same seed and estimator give the same result as the single test.
    single = cx.surrogate_test(chain, cx.ESTIMATORS["H2"], 30, seed=1)
    assert profile["H2_deficit"] == pytest.approx(single["deficit"])
    only = cx.complexity_profile(chain, n_surrogates=10, names=["H1"])
    assert set(only) == {"n", "H1_value", "H1_deficit", "H1_z", "H1_p_value"}


def test_benjamini_hochberg():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.2, 0.9])
    # Thresholds at q = 0.05: 0.0083, 0.0167, 0.025, 0.033, ... -> the first two pass.
    assert cx.benjamini_hochberg(p, 0.05).tolist() == [True, True, False, False, False, False]
    assert not cx.benjamini_hochberg(np.array([0.5, 0.6]), 0.05).any()
    assert cx.benjamini_hochberg(np.array([0.01, 0.02, 0.03]), 0.05).all()


def test_markov_fit_probabilities():
    model = cx.markov_fit(bits("0101010101"), order=1, alpha=0.5)
    # After 0 always 1 (5 times), after 1 always 0 (4 times), plus smoothing.
    assert model["probs"][0] == pytest.approx([0.5 / 6, 5.5 / 6])
    assert model["probs"][1] == pytest.approx([4.5 / 5, 0.5 / 5])
    order0 = cx.markov_fit(bits("0001"), order=0, alpha=0.0)
    assert order0["probs"].tolist() == [[0.75, 0.25]]


def test_markov_score_out_of_sample():
    train = cx.simulate_markov_chain(50_000, 0.6, seed=5)
    test = cx.simulate_markov_chain(50_000, 0.6, seed=6)
    memoryless = cx.markov_score(test, cx.markov_fit(train, order=0))
    order1 = cx.markov_score(test, cx.markov_fit(train, order=1))
    assert memoryless["bits"] == pytest.approx(1.0, abs=5e-3)
    assert memoryless["accuracy"] == pytest.approx(0.5, abs=0.02)
    assert order1["accuracy"] == pytest.approx(0.6, abs=0.01)
    assert order1["bits"] == pytest.approx(0.971, abs=5e-3)
    assert order1["n"] == 49_999


def test_markov_score_on_too_short_sequence():
    model = cx.markov_fit(bits("0101"), order=3)
    assert cx.markov_score(bits("01"), model)["n"] == 0


def test_markov_predictions():
    model = cx.markov_fit(bits("0101010101"), order=1)
    assert cx.markov_predictions(bits("0011"), model).tolist() == [1, 1, 0]


def test_simulate_markov_chain():
    always = cx.simulate_markov_chain(100, 1.0, seed=0)
    assert len(set(always.tolist())) == 1
    never = cx.simulate_markov_chain(100, 0.0, seed=0)
    assert (never[1:] != never[:-1]).all()
    chain = cx.simulate_markov_chain(100_000, 0.55, seed=1)
    assert (chain[1:] == chain[:-1]).mean() == pytest.approx(0.55, abs=0.005)
