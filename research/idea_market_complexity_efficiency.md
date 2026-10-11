# Idea: algorithmic complexity as a market efficiency measure

An efficient market produces returns that look random, and a random sequence
cannot be compressed. So the compressibility of a coin's return series is a
measure of how inefficient its market is. Estimate it for many coins and test
whether lower complexity (more structure) goes with lower liquidity, younger
markets or predictability.

## Summary

- **Data:** Binance public klines or trades for a broad cross-section of
  coins, plus market cap and volume from CoinGecko.
- **Topics covered:** algorithmic information theory (Kolmogorov complexity,
  compression estimators, entropy rate); information theory and descriptive
  statistics.
- **Why high quality:** a sharp, falsifiable hypothesis with surrogate-data
  controls and a cross-section large enough for statistics.
- **What you learn:** Lempel-Ziv and compressor-based complexity estimates,
  their finite-sample bias, and how discretization choices drive the result.
- **Novelty:** medium. The idea exists for equities and Bitcoin; a careful
  cross-sectional study across hundreds of coins is less explored.

## Data

- Candles from <https://data.binance.vision> for as many spot pairs as have
  enough history, at one or more bar sizes (for example 1 minute, 1 hour,
  1 day).
- Per-coin covariates: market cap, traded volume, listing age, bid-ask spread
  if available.
- Every series must be long enough for the estimators to be meaningful, which
  will exclude recently listed coins.

## Research questions

1. Are crypto return series measurably more compressible than shuffled
   versions of themselves?
2. Does complexity rise with liquidity and market age, as the efficient
   market view predicts?
3. Has complexity increased over the years as the market matured?
4. Does low complexity mean real predictability, that is, can a simple model
   beat chance on the low-complexity coins?
5. How much do the answers depend on the estimator and the discretization?

## Approach

1. **Discretize** returns into symbols: binary (up / down), ternary (up /
   flat / down) and quantile bins. Report every result for several choices.
2. **Estimate complexity** with several methods:
   - Lempel-Ziv complexity (LZ76),
   - compression ratio with general compressors (gzip, bz2, lzma),
   - entropy rate from block entropies,
   - optionally the block decomposition method for short sequences.
3. **Control with surrogates.** Compare each series with shuffled copies
   (destroys all time structure) and with volatility-preserving surrogates, so
   that volatility clustering is not mistaken for return predictability.
4. **Cross-section.** Regress complexity on liquidity, age and market cap.
5. **Over time.** Rolling-window complexity per coin, linked to known events.
6. **Predictability check.** Fit simple predictors on the least and most
   complex coins and compare out-of-sample accuracy.

## Evaluation

- Significance of each coin's complexity against its surrogate distribution,
  with multiple-testing correction across coins.
- Agreement between estimators (rank correlation across coins).
- Robustness to bar size, alphabet and sample length.
- Out-of-sample prediction accuracy as the external check on the measure.

## Risks

- **Finite-sample bias.** Compressors have overhead and estimators converge
  slowly; short series look falsely complex or falsely simple.
- **Discretization artifacts.** Tick size and zero returns create structure
  that has nothing to do with inefficiency, especially at small bar sizes.
- **Volatility clustering** makes series compressible without making returns
  predictable; the surrogate design has to separate the two.
- **A null result is likely for the liquid coins.** That is still a finding,
  but the write-up should be planned around it.

## References

- Giglio, Matsushita, Figueiredo and Da Silva (2008), "Algorithmic complexity
  theory and the relative efficiency of financial markets".
- Brandouy, Delahaye, Ma and Zenil (2014), "Algorithmic complexity of
  financial motions".

Citations are from memory; verify them before use.

## Relation to other ideas

Shares data and tooling with
[market regimes with HMM vs. MDL segmentation](idea_market_regimes_hmm_mdl.md).
Together they make one coherent project on information-theoretic views of
crypto markets.
