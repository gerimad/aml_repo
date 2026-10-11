# Idea: market regimes with HMM vs. MDL segmentation

Segment crypto price series into regimes (for example trending, ranging,
crash) in two principled ways and compare them: a hidden Markov model, and a
change point method that picks the segmentation of minimum description
length.

## Summary

- **Data:** OHLCV candles for major pairs from Binance public data, free and
  years deep.
- **Topics covered:** the learner survey (hidden Markov models, Bayesian
  methods); algorithmic information theory (minimum description length);
  unsupervised learning.
- **Why high quality:** two principled methods answer the same question, and
  the number of regimes is chosen by a criterion instead of by hand.
- **What you learn:** EM / Baum-Welch, Viterbi decoding, model selection by
  MDL vs. BIC, and how to evaluate an unsupervised result without labels.
- **Novelty:** medium. HMM regimes are common; the MDL comparison and a
  rigorous out-of-sample test are less so.

## Data

- Candles (klines) from <https://data.binance.vision>, downloadable in bulk
  without an API key. Hourly or daily bars for BTC, ETH and a handful of other
  liquid pairs.
- Derived series: log returns, realized volatility, volume.
- No labels exist. Known events (exchange collapses, halvings, major crashes)
  serve as a sanity check, not as ground truth.

## Research questions

1. Do the HMM and the MDL segmentation agree on where regimes change?
2. How many regimes does the data support, and do MDL, BIC and held-out
   likelihood give the same answer?
3. Are the regimes stable when the sample period, the bar size or the coin
   changes?
4. Are the regimes useful out of sample, for example for volatility
   forecasting, compared with a model that ignores regimes?

## Approach

1. **Gaussian HMM** on returns (and volatility), fitted with Baum-Welch,
   decoded with Viterbi, for 2 to 6 states.
2. **MDL segmentation.** Encode the series as piecewise models (for example
   constant mean and variance per segment), with total cost equal to model
   bits plus data bits, minimized with dynamic programming or PELT.
3. **Model selection.** Compare the number of regimes chosen by MDL, BIC and
   held-out likelihood.
4. **Extensions if time allows:** Student-t emissions for heavy tails, a
   Bayesian HMM, or a sticky HMM to stop rapid state switching.
5. **Baselines:** rolling-volatility thresholds and a GARCH model.

## Evaluation

- Agreement between methods: change point distance and adjusted Rand index
  between the two labelings.
- Out-of-sample log likelihood on later data, with walk-forward refitting.
- Downstream value: volatility forecast error with and without regime
  information.
- Stability: refit on bootstrap or shifted windows and measure how much the
  regimes move.

## Risks

- **No ground truth.** The evaluation has to rest on held-out likelihood,
  stability and downstream use, so these need to be designed up front.
- **Non-stationarity.** Parameters fitted on 2019 data may not describe 2025.
- **Lookahead.** Viterbi and smoothing use the whole series; anything claimed
  as predictive must use filtered (past-only) state estimates.

## Relation to other ideas

Shares data and tooling with
[algorithmic complexity as a market efficiency measure](idea_market_complexity_efficiency.md).
Together they make one coherent project on information-theoretic views of
crypto markets.
