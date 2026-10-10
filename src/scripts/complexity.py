"""Complexity and predictability measures for symbol sequences.

Building blocks for the "algorithmic complexity as a market efficiency
measure" experiments. The pipeline is: turn returns into a sequence of a few
symbols, measure how compressible the sequence is, and compare with shuffled
copies of itself, which have the same symbol frequencies but no order.

From a notebook in src/notebooks:

    import sys; sys.path.append("../scripts")
    import complexity as cx

    symbols = cx.sign_symbols(returns)                  # 0 = down, 1 = up
    cx.conditional_entropy(symbols, order=1)            # bits per symbol
    cx.compressed_bits_per_symbol(symbols, "lzma")      # bits per symbol
    result = cx.surrogate_test(symbols, cx.lz76_phrases, n_surrogates=200, seed=0)
    result["deficit"], result["p_value"]

    model = cx.markov_fit(train_symbols, order=2)       # fit on the past
    score = cx.markov_score(test_symbols, model)        # score on the future
    score["bits"], score["accuracy"]

Symbols are small non-negative integers. All entropies are in bits.
"""

from __future__ import annotations

import bz2
import lzma
import zlib

import numpy as np

# --------------------------------------------------------------------------
# From returns to symbols
# --------------------------------------------------------------------------


def sign_symbols(returns, drop_zeros: bool = True) -> np.ndarray:
    """Binary coding of returns: 1 for up, 0 for down.

    Zero returns carry no direction. With `drop_zeros` they are removed, so
    the sequence gets shorter; otherwise they are coded as 0.
    """
    returns = np.asarray(returns, float)
    returns = returns[np.isfinite(returns)]
    if drop_zeros:
        returns = returns[returns != 0]
    return (returns > 0).astype(np.uint8)


def ternary_symbols(returns, band: float = 0.0) -> np.ndarray:
    """Ternary coding: 0 for down, 1 for flat (|return| <= band), 2 for up."""
    returns = np.asarray(returns, float)
    returns = returns[np.isfinite(returns)]
    symbols = np.ones(len(returns), dtype=np.uint8)
    symbols[returns < -band] = 0
    symbols[returns > band] = 2
    return symbols


def quantile_symbols(returns, n_bins: int = 4) -> np.ndarray:
    """Code returns by equal-frequency bins, so every symbol is equally common.

    Ties are broken by position, which keeps the bins the same size even when
    many returns are identical (for example zero).
    """
    returns = np.asarray(returns, float)
    returns = returns[np.isfinite(returns)]
    ranks = np.argsort(np.argsort(returns, kind="stable"), kind="stable")
    return (ranks * n_bins // len(returns)).astype(np.uint8)


def alphabet_size(symbols) -> int:
    """Number of symbols in the alphabet, taken as the largest symbol plus one (at least 2)."""
    symbols = np.asarray(symbols)
    return max(int(symbols.max()) + 1, 2) if len(symbols) else 2


# --------------------------------------------------------------------------
# Complexity estimators
# --------------------------------------------------------------------------


def lz76_phrases(symbols) -> int:
    """Lempel-Ziv (1976) complexity: the number of phrases in the exhaustive parsing.

    The sequence is read left to right and cut into phrases, each being the
    shortest piece that has not appeared before (overlap with the piece itself
    allowed). Regular sequences need few phrases, random ones many. This is
    the measure computed by the Kaspar-Schuster algorithm.
    """
    data = np.asarray(symbols, dtype=np.uint8).tobytes()
    n = len(data)
    position = phrases = 0
    while position < n:
        length = 1
        # Grow the phrase while it can still be copied from earlier text.
        while position + length <= n and data.find(
            data[position : position + length], 0, position + length - 1
        ) != -1:
            length += 1
        phrases += 1
        position += length
    return phrases


def lz76_normalized(symbols) -> float:
    """LZ76 phrase count divided by its asymptotic value for a random sequence, n / log_k(n).

    Close to 1 for random sequences, lower for regular ones. On finite
    sequences it is biased upwards, so compare with shuffled copies, not with 1.
    """
    symbols = np.asarray(symbols)
    n, k = len(symbols), alphabet_size(symbols)
    return lz76_phrases(symbols) * (np.log(n) / np.log(k)) / n


def compressed_bits_per_symbol(symbols, codec: str = "lzma") -> float:
    """Size of the compressed sequence in bits per symbol, with a general-purpose compressor.

    One symbol is stored per byte before compressing. `codec` is "zlib"
    (the algorithm of gzip), "bz2" or "lzma". The value includes the
    compressor's fixed overhead, so compare with shuffled copies.
    """
    data = np.asarray(symbols, dtype=np.uint8).tobytes()
    if codec == "zlib":
        packed = zlib.compress(data, 9)
    elif codec == "bz2":
        packed = bz2.compress(data, 9)
    elif codec == "lzma":
        packed = lzma.compress(data, preset=9)
    else:
        raise ValueError(f"unknown codec {codec!r}")
    return 8 * len(packed) / len(data)


def block_counts(symbols, length: int, k: int | None = None) -> np.ndarray:
    """Counts of every block of `length` consecutive symbols, as an array of size k**length."""
    symbols = np.asarray(symbols, dtype=np.int64)
    k = k or alphabet_size(symbols)
    n_blocks = len(symbols) - length + 1
    if n_blocks <= 0:
        return np.zeros(k**length, dtype=np.int64)
    codes = np.zeros(n_blocks, dtype=np.int64)
    for offset in range(length):
        codes = codes * k + symbols[offset : offset + n_blocks]
    return np.bincount(codes, minlength=k**length)


def entropy_bits(counts) -> float:
    """Plug-in Shannon entropy, in bits, of the distribution given by `counts`."""
    counts = np.asarray(counts, float)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def conditional_entropy(symbols, order: int = 1, k: int | None = None) -> float:
    """Plug-in entropy of a symbol given the `order` symbols before it, in bits.

    Order 0 is the entropy of the symbol frequencies. Order 1 measures the
    same dependence as the lag-1 autocorrelation. Higher orders see longer
    patterns but are biased downwards when the sequence is short relative to
    k**(order + 1), so compare with shuffled copies.
    """
    k = k or alphabet_size(symbols)
    joint = entropy_bits(block_counts(symbols, order + 1, k))
    context = entropy_bits(block_counts(symbols, order, k)) if order > 0 else 0.0
    return joint - context


# --------------------------------------------------------------------------
# Surrogate tests
# --------------------------------------------------------------------------


def surrogate_test(symbols, statistic, n_surrogates: int = 200, seed: int = 0) -> dict[str, float]:
    """Compare a complexity statistic with its values on shuffled copies.

    Shuffling keeps the symbol frequencies and the length and destroys all
    order, so the copies are what an efficient market with the same symbol
    frequencies would produce, including the estimator's finite-sample bias.

    Returns:
      value            the statistic on the sequence
      surrogate_mean   its mean over the shuffled copies
      surrogate_sd     its standard deviation over the shuffled copies
      deficit          surrogate_mean - value (positive = more regular than random)
      z                deficit / surrogate_sd
      p_value          share of copies at least as regular as the sequence
                       (one-sided, with the +1 correction so it is never 0)
    """
    symbols = np.asarray(symbols)
    rng = np.random.default_rng(seed)
    value = float(statistic(symbols))
    surrogates = np.array([statistic(rng.permutation(symbols)) for _ in range(n_surrogates)], float)
    mean, sd = surrogates.mean(), surrogates.std(ddof=1)
    return {
        "value": value,
        "surrogate_mean": float(mean),
        "surrogate_sd": float(sd),
        "deficit": float(mean - value),
        "z": float((mean - value) / sd) if sd > 0 else 0.0,
        "p_value": float((1 + (surrogates <= value).sum()) / (n_surrogates + 1)),
    }


# The estimators compared in the experiments. All return "lower = more regular".
ESTIMATORS = {
    "H1": lambda s: conditional_entropy(s, 1),
    "H2": lambda s: conditional_entropy(s, 2),
    "H3": lambda s: conditional_entropy(s, 3),
    "H5": lambda s: conditional_entropy(s, 5),
    "H8": lambda s: conditional_entropy(s, 8),
    "zlib": lambda s: compressed_bits_per_symbol(s, "zlib"),
    "bz2": lambda s: compressed_bits_per_symbol(s, "bz2"),
    "lzma": lambda s: compressed_bits_per_symbol(s, "lzma"),
    "lz76": lz76_normalized,
}
SLOW_ESTIMATORS = {"lz76"}


def complexity_profile(
    symbols, n_surrogates: int = 200, n_slow_surrogates: int = 50, seed: int = 0, names=None
) -> dict[str, float]:
    """Run the surrogate test for every estimator in ESTIMATORS on one sequence.

    Returns a flat dict with keys like "H1_value", "H1_deficit", "H1_z" and
    "H1_p_value", plus "n" (sequence length). All estimators see the same
    shuffled copies. The slow estimators use `n_slow_surrogates` copies.
    """
    symbols = np.asarray(symbols)
    out: dict[str, float] = {"n": len(symbols)}
    for name in names or ESTIMATORS:
        n = n_slow_surrogates if name in SLOW_ESTIMATORS else n_surrogates
        result = surrogate_test(symbols, ESTIMATORS[name], n, seed)
        for key in ("value", "deficit", "z", "p_value"):
            out[f"{name}_{key}"] = result[key]
    return out


def benjamini_hochberg(p_values, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg procedure: which hypotheses are rejected at false discovery rate q."""
    p_values = np.asarray(p_values, float)
    n = len(p_values)
    order = np.argsort(p_values)
    passed = p_values[order] <= q * np.arange(1, n + 1) / n
    rejected = np.zeros(n, dtype=bool)
    if passed.any():
        rejected[order[: np.flatnonzero(passed).max() + 1]] = True
    return rejected


# --------------------------------------------------------------------------
# Markov predictor: the out-of-sample check
# --------------------------------------------------------------------------


def markov_fit(symbols, order: int = 1, k: int | None = None, alpha: float = 0.5) -> dict:
    """Fit an order-`order` Markov model: P(next symbol | previous `order` symbols).

    Probabilities are smoothed by adding `alpha` to every count (0.5 is the
    Krichevsky-Trofimov estimator), so unseen contexts get a uniform forecast.
    Order 0 is the model without memory: the symbol frequencies.
    """
    k = k or alphabet_size(symbols)
    counts = block_counts(symbols, order + 1, k).reshape(k**order, k) + alpha
    return {"order": order, "k": k, "probs": counts / counts.sum(axis=1, keepdims=True)}


def markov_score(symbols, model: dict) -> dict[str, float]:
    """Score a fitted Markov model on a sequence it was not fitted on.

    Returns:
      bits       mean code length, -log2 P(symbol | context), in bits per symbol
      accuracy   share of symbols equal to the model's most likely symbol
      n          number of symbols scored (the first `order` have no context)
    """
    symbols = np.asarray(symbols, dtype=np.int64)
    order, k, probs = model["order"], model["k"], model["probs"]
    n = len(symbols) - order
    if n <= 0:
        return {"bits": float("nan"), "accuracy": float("nan"), "n": 0}
    context = np.zeros(n, dtype=np.int64)
    for offset in range(order):
        context = context * k + symbols[offset : offset + n]
    target = symbols[order:]
    return {
        "bits": float(-np.log2(probs[context, target]).mean()),
        "accuracy": float((probs[context].argmax(axis=1) == target).mean()),
        "n": int(n),
    }


def markov_predictions(symbols, model: dict) -> np.ndarray:
    """The model's most likely next symbol at each position from `order` onwards."""
    symbols = np.asarray(symbols, dtype=np.int64)
    order, k = model["order"], model["k"]
    n = len(symbols) - order
    context = np.zeros(max(n, 0), dtype=np.int64)
    for offset in range(order):
        context = context * k + symbols[offset : offset + n]
    return model["probs"][context].argmax(axis=1)


def simulate_markov_chain(n: int, p_repeat: float, seed: int = 0) -> np.ndarray:
    """Binary sequence in which each symbol repeats the previous one with probability `p_repeat`.

    0.5 gives a fair coin (an efficient market), above 0.5 momentum, below
    0.5 reversal. Used to measure how small a dependence an estimator detects.
    """
    rng = np.random.default_rng(seed)
    repeat = rng.random(n) < p_repeat
    symbols = np.empty(n, dtype=np.uint8)
    symbols[0] = rng.integers(2)
    # A symbol flips whenever it does not repeat: a cumulative XOR of the flips.
    flips = np.cumsum(~repeat[1:]) % 2
    symbols[1:] = symbols[0] ^ flips
    return symbols
