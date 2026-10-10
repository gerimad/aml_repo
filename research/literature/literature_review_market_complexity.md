# Literature review: algorithmic complexity as a market efficiency measure

Review for the idea in
[`../idea_market_complexity_efficiency.md`](../idea_market_complexity_efficiency.md):
measure how compressible the return series of many coins are, and test whether
low complexity goes with low liquidity, young markets or predictability.

## 1. Scope and method

The review covers four questions:

1. How has algorithmic complexity been used to measure market efficiency?
2. What do entropy-based studies of efficiency teach about method?
3. How reliable are complexity and entropy estimators on finite sequences?
4. What is known about the efficiency of crypto markets?

Sixteen open-access papers were downloaded to `papers/` (PDFs are ignored by
git, the links below are the source of record). The "Read" column says how much
of each was read for this review.

| # | Paper | File in `papers/` | Read |
| --- | --- | --- | --- |
| 1 | Giglio, Matsushita, Figueiredo, Gleria, Da Silva (2008), [Algorithmic complexity theory and the relative efficiency of financial markets](https://mpra.ub.uni-muenchen.de/11150/), EPL 84, 48005 | `giglio2008_complexity_efficiency_updated.pdf` (earlier version: `..._v1.pdf`) | Full (updated version) |
| 2 | Giglio, Da Silva (2009), [Ranking the stocks listed on Bovespa according to their relative efficiency](https://mpra.ub.uni-muenchen.de/22720/) | `giglio2010_bovespa_ranking.pdf` | Abstract, introduction |
| 3 | Da Silva (2015), [Financial market efficiency should be gauged in relative rather than absolute terms](https://mpra.ub.uni-muenchen.de/64497/) | `dasilva2015_relative_efficiency_review.pdf` | Abstract |
| 4 | Brandouy, Delahaye, Ma (2015), [Estimating the algorithmic complexity of stock markets](https://arxiv.org/abs/1504.04296) | `brandouy2015_complexity_stock_markets.pdf` | Abstract, introduction, conclusion |
| 5 | Zenil, Delahaye (2010), [An algorithmic information-theoretic approach to the behaviour of financial markets](https://arxiv.org/abs/1008.1846) | `zenil2010_algorithmic_markets.pdf` | Abstract |
| 6 | Calcagnile, Corsi, Marmi (2016), [Entropy and efficiency of the ETF market](https://arxiv.org/abs/1609.04199) | `calcagnile2016_entropy_etf.pdf` | Abstract, introduction, conclusions, start of the estimation appendix |
| 7 | Fiedor (2013), [Frequency effects on predictability of stock returns](https://arxiv.org/abs/1310.5540) | `fiedor2013_frequency_predictability.pdf` | Abstract, method, results |
| 8 | Fiedor (2014), [Maximum entropy production principle for stock returns](https://arxiv.org/abs/1408.3728) | `fiedor2014_max_entropy_production.pdf` | Abstract |
| 9 | Gao, Kontoyiannis, Bienenstock (2008), [Estimating the entropy of binary time series: methodology, some theory and a simulation study](https://arxiv.org/abs/0802.4363) | `gao2008_entropy_estimation_binary.pdf` | Abstract |
| 10 | Cilibrasi, Vitányi (2005), [Clustering by compression](https://arxiv.org/abs/cs/0312044) | `cilibrasi2005_clustering_by_compression.pdf` | Abstract |
| 11 | Zenil et al. (2018), [A decomposition method for global evaluation of Shannon entropy and local estimations of algorithmic complexity](https://arxiv.org/abs/1609.00110) | `zenil2018_block_decomposition.pdf` | Abstract |
| 12 | Sigaki, Perc, Ribeiro (2019), [Clustering patterns in efficiency and the coming-of-age of the cryptocurrency market](https://arxiv.org/abs/1901.04967) | `sigaki2019_crypto_permutation_entropy.pdf` | Abstract, introduction, discussion |
| 13 | Bariviera (2017), [The inefficiency of Bitcoin revisited: a dynamic approach](https://arxiv.org/abs/1709.08090) | `bariviera2017_inefficiency_bitcoin_dynamic.pdf` | Abstract, conclusions |
| 14 | Bariviera, Zunino, Rosso (2018), [An analysis of high-frequency cryptocurrencies prices dynamics using permutation-information-theory quantifiers](https://arxiv.org/abs/1808.01926) | `bariviera2018_hf_crypto_permutation.pdf` | Abstract, conclusions |
| 15 | Tartakovsky, Plesovskikh, Sarmakeeva, Bibik (2020), [Autocorrelation of returns in major cryptocurrency markets](https://arxiv.org/abs/2003.13517) | `tartakovsky2020_crypto_autocorrelation.pdf` | Abstract |

Fifteen papers in sixteen files (paper 1 in two versions). Most were read only
in part, so the summaries below should not be taken as complete accounts.
Relevant papers that could not be downloaded are listed in section 5.

## 2. Summaries

### 2.1 Algorithmic complexity as an efficiency measure

**[1] Giglio et al. (2008): the founding paper of the approach.**
The argument: efficiency should be measured in relative terms, as the distance
of a market from the ideal one, and algorithmic complexity gives such a
measure, because an efficient price series is one that cannot be compressed.

- **Data:** daily returns of 36 stock indices and 20 exchange rates, July 2000
  to July 2007, 2,000 observations each.
- **Coding:** ternary, with a "stability basin" *b* = 0.25%: down if the
  return is below -*b*, up if above +*b*, flat in between. The authors report
  that rankings "did not alter too much" with *b*, and that more research on
  the choice is needed.
- **Measure:** the Lempel-Ziv (1976) complexity computed with the
  Kaspar-Schuster algorithm, normalized by its limit for a random sequence,
  n / log n. The index is not bounded by 1.
- **Procedure:** sliding windows of 1,000 observations moved one day at a
  time. A market's efficiency is the share of windows whose index exceeds 1.
- **Results:** a pseudo-random series is "98.8% efficient". The Dow Jones is
  95.4% efficient, Shanghai 49.5%, Karachi 23.7%, Colombo 10.5%. Among
  currencies the pound and the euro are above 99%, the Chinese yuan 17.9%
  (it was pegged for part of the sample). Developed markets rank on top.
- **Limits visible in the paper:** the threshold of 1 is a convention; the
  average index of the random series (1.018) and of the Dow Jones (1.020)
  are almost the same; windows that overlap in 999 of 1,000 points are not
  independent; no significance test is made. A comparison with a second
  method (detrended fluctuation analysis) gives rankings that match only
  partly.

**[2] Giglio and Da Silva (2009)** apply the same method to the stocks of the
São Paulo exchange and give eight alternative top-ten lists, which itself
shows how much the ranking depends on settings. **[3] Da Silva (2015)** is a
short restatement of the case for relative efficiency.

**[4] Brandouy, Delahaye, Ma (2015): compression as an efficiency test.**
Uses lossless compressors on return series, inside an iterative "regularity
erasing procedure": remove a known regularity, compress again, see what is
left.

- On simulated data, compression reveals some structures that standard
  statistical tests miss.
- On daily Dow Jones returns the result is "an extremely high Kolmogorov
  complexity", read as support for the impossibility of beating the market,
  although after removing known stylized facts some unexplained structure
  seems to remain.
- The authors name the main limit themselves: "currently available lossless
  compression tools, initially developed for text files, are obviously not
  particularly designed for financial data".

**[5] Zenil and Delahaye (2010)** take a different route: they compare the
frequencies of short binary patterns in price series with those predicted by
algorithmic probability. It is a conceptual precursor more than a method to
reuse. **[11] Zenil et al. (2018)** describe the block decomposition method,
which estimates the complexity of short sequences from a table of small
programs instead of from a compressor. The abstract notes that it "performs
like Shannon entropy when it loses accuracy". **[10] Cilibrasi and Vitányi
(2005)** define the normalized compression distance, the standard way to turn
a real compressor into an approximation of an uncomputable quantity.

### 2.2 Entropy-based efficiency studies

**[6] Calcagnile, Corsi, Marmi (2016): the methodological benchmark.**
Shannon entropy of 2-symbol and 3-symbol codings of 1-minute and 5-minute
returns of 55 exchange traded funds.

- The paper's subject is how to **separate real inefficiency from other
  sources of regularity**: the intraday pattern of volatility, volatility
  clustering, and microstructure effects.
- With the **binary** coding (up / down), volatility and the intraday pattern
  have no effect, because they only rescale returns. Only microstructure
  remains, which produces linear autocorrelation. They remove it by fitting
  ARMA models and measuring the entropy of the residuals, or, for simple
  cases, by comparing with the theoretical entropy of an AR(1) or MA(1)
  process.
- With the **ternary** coding, volatility accounts for 62% of the measured
  regularity on average, the intraday pattern for 18% and microstructure for
  20%. Their conclusion: "when studying the randomness of a three-symbol
  discretised time series, the volatility must be filtered out", otherwise
  the result tells "more on the predictable character of the volatility than
  on, for example, market efficiency".
- After all filtering, efficiency is still rejected for the large majority of
  the funds.
- They find "a strong relationship between low entropy and high relative tick
  size": a coarse price grid makes price changes more predictable.
- The naive plug-in entropy estimator is "strongly biased"; they use the
  Grassberger correction.

**[7] Fiedor (2013): predictability and sampling frequency.**
Entropy rate of NYSE 100 stocks, daily and intraday, estimated with a
Lempel-Ziv estimator and with context tree weighting (CTW). Returns are coded
into four quartiles, so every symbol is equally frequent.

- Intraday price changes are more predictable than daily ones.
- The average estimated entropy rate of daily changes is 2.04 bits with CTW
  and 1.92 with Lempel-Ziv, "out of the theoretical maximum of 2". An
  estimate above the maximum, and two estimators 0.12 bits apart, show the
  size of estimator bias relative to the effect being measured.

**[8] Fiedor (2014)** reports that standard trading algorithms "do not use
the structure inherent in the stock returns to any significant degree":
measured predictability and realized profitability are different things.

### 2.3 Estimating entropy on finite sequences

**[9] Gao, Kontoyiannis, Bienenstock (2008).** A systematic comparison of
entropy estimators for binary sequences: plug-in, four Lempel-Ziv based ones,
context tree weighting and others. Conclusions stated in the abstract:

- "For all estimators considered, the main source of error is the bias."
- "The CTW method is repeatedly and consistently seen to provide the most
  accurate results."
- Lempel-Ziv estimators are often comparable to the plug-in method.
- The plug-in method "with small word-lengths fails to detect longer-range
  structure", and with longer ones "the empirical distribution is severely
  undersampled, leading to large biases".

### 2.4 Efficiency of crypto markets

**[12] Sigaki, Perc, Ribeiro (2019): the closest study to our idea.**
Daily log returns of 437 cryptocurrencies, sliding windows, permutation
entropy and statistical complexity.

- **A coin is called efficient in a window when both measures are
  indistinguishable from their values on shuffled data** (inside the 95%
  band of the shuffles).
- 37% of the coins are efficient more than 80% of the time; 20% are efficient
  less than 20% of the time.
- **"The efficiency is not correlated with the market capitalization of the
  cryptocurrencies."**
- Coins fall into four groups by how their efficiency evolves; the authors
  speak of a market "coming of age".

**[13] Bariviera (2017).** Hurst exponent of daily Bitcoin returns, 2011 to
2017, in sliding windows. Returns were persistent until 2014 and compatible
with white noise afterwards, while volatility had long memory throughout.
Detrended fluctuation analysis is preferred to the rescaled-range method,
which "is prone to detect long memory".

**[14] Bariviera, Zunino, Rosso (2018).** Intraday prices of twelve
cryptocurrencies during the 2017-18 boom and bust, with permutation entropy.
Most coins behave alike, a few differ (ETC, ETH, DASH, XEM).

**[15] Tartakovsky et al. (2020).** Autocorrelation tests on BTC and ETH
markets at 5-minute, 1-hour, 1-day and 1-week bars. "Statistically significant
autocorrelation is persistently present on the 5m and 1H time frames on all
markets", while tests disagree at daily and weekly bars.

A further paper found but not downloaded: Soloviev, Semerikov and Solovieva
(2020) apply the Lempel-Ziv measure to daily Bitcoin returns as a possible
early warning of crises.

## 3. Insights gathered

1. **Relative efficiency is the accepted framing.** No market is perfectly
   efficient, so the useful question is how far each is from the ideal
   [1, 3, 6].
2. **Complexity-based studies find markets very close to random.** Index
   values near 1 [1], "extremely high" complexity [4]. They have trouble
   telling a developed market from a random number generator [1].
3. **Entropy-based studies on high-frequency data reject efficiency almost
   everywhere** [6, 7, 15]. The two families of method reach opposite
   headline conclusions, which calls for an explanation in terms of the
   methods, not the markets.
4. **Estimator bias is the main source of error, and it is as large as the
   effects of interest** [7, 9]. Comparing a measure with a theoretical
   constant (1 for the Lempel-Ziv index, 2 bits for four symbols) is unsafe.
5. **Shuffled copies of the same series are the right reference** [12]. They
   carry the same bias and the same symbol frequencies.
6. **The coding decides what is measured.** With more than two symbols,
   volatility clustering dominates [6]. The basin width of a ternary coding
   is a free parameter [1]. Equal-frequency bins remove the dependence on
   symbol frequencies [7] but not on volatility.
7. **The binary sign coding is immune to volatility and intraday patterns**
   and leaves only microstructure to deal with [6].
8. **Price-grid effects create measured inefficiency** [6]. Coarse grids are
   common among low-priced coins.
9. **Predictability rises with sampling frequency** [7, 15]. Daily data shows
   little, intraday data shows much.
10. **Predictable is not the same as profitable** [8].
11. **General-purpose compressors are not built for this** [4]. Context
    models (CTW) are the best entropy estimators for binary sequences [9].
12. **In crypto, efficiency has increased over time** [12, 13], and **it does
    not follow market capitalization** [12], against the intuition behind our
    hypothesis.
13. **Window design matters.** Heavily overlapping windows [1] give dependent
    observations; rankings change with settings [1, 2].

## 4. What this means for our idea

**The idea is less novel than first rated.** A shuffle-controlled efficiency
study across hundreds of coins exists [12]. What it does not have: intraday
data, compression or Lempel-Ziv measures, any link to liquidity (it used
market capitalization), and an out-of-sample check that measured inefficiency
means real predictability. Those are the gaps to aim at.

**The central risk is statistical power.** Insights 2 to 4 suggest that
complexity measures report "random" simply because they are too blunt at the
sample sizes used (1,000 to 2,000 daily observations). The first experiments
in
[`../../src/notebooks/02_market_complexity_data_and_baseline.ipynb`](../../src/notebooks/02_market_complexity_data_and_baseline.ipynb)
confirm this: on a year of hourly data, Lempel-Ziv complexity and general
compressors cannot detect a dependence that a first-order entropy detects
almost always, and on real coins they find nothing where entropy measures
find structure in over 90% of coin-years.

**Changes to the idea that follow from the literature:**

| Original plan | Change | Source |
| --- | --- | --- |
| Compression ratio and Lempel-Ziv complexity as the measures | Keep them as the method under examination, but add entropy estimators (plug-in with shuffle control, and CTW) as the sensitive reference. Make "which estimator can see what" a research question of its own. | [4, 7, 9] |
| Binary, ternary and quantile codings on equal footing | Binary sign coding as the main one. Any coding with more symbols only after removing volatility and the intraday pattern. | [6] |
| Compare with shuffled series | Keep, and make it the only reference: no thresholds such as "index above 1". | [1, 12] |
| "Volatility-preserving surrogates" | Not needed for the binary coding. Needed for richer codings. | [6] |
| Daily and intraday bars | Intraday is where the effects are. Daily series are too short for the estimators. | [7, 15] |
| Rolling windows per coin | Non-overlapping windows, so that observations are independent in time. | [1] |
| Hypothesis: low liquidity means low efficiency | Keep as a test, but expect a weak or absent relation, and control for the price grid. | [6, 12] |
| "Does low complexity mean real predictability?" | Promote from a side check to the main validation: fit a predictor on the past, score it out of sample in bits. | [7, 8] |
| Predictability as the end point | Add an economic check against trading costs. | [8] |
| Microstructure | With candle data it cannot be removed properly. State it as a limitation, check robustness at coarser bars, and plan for ARMA filtering or bid / ask data. | [6] |

**A sharper research question.** The literature contains an unresolved
contradiction: complexity measures say markets are nearly random, entropy
measures say they are not. The project can resolve it on one data set:
**how large must an inefficiency be for each family of estimator to detect it
at realistic sample sizes, and where do real crypto markets fall on that
scale?** This keeps the algorithmic information theory content of the idea,
turns the weakness of compressors from a problem into a result, and gives the
efficiency ranking of coins a measured error bar.

## 5. Papers to obtain

Not open access, so not downloaded.

| Paper | Why it matters |
| --- | --- |
| Lempel, Ziv (1976), "On the complexity of finite sequences", IEEE Transactions on Information Theory | Definition of the complexity measure. |
| Kaspar, Schuster (1987), "Easily calculable measure for the complexity of spatiotemporal patterns", Physical Review A | The algorithm used in [1] and implemented here. Our implementation reproduces the three worked examples quoted in [1]; it should be checked against the original. |
| Shmilovici, Alon-Brimer, Hauser (2003), "Using a stochastic complexity measure to check the efficient market hypothesis", Computational Economics | The first compression-based efficiency test, using a context tree model, which is close to what our results point to. |
| Wei (2018), "Liquidity and market efficiency in cryptocurrencies", Economics Letters | Reported (secondhand) to find, across 456 coins, that return predictability diminishes as liquidity rises. It is the direct prior test of our hypothesis and seems to disagree with [12]. Must be read. |
| Kristoufek, Vosvrda (2019), "Cryptocurrencies market efficiency ranking: not so straightforward", Physica A | An efficiency index from long-range dependence, fractal dimension and entropy; reports that rankings depend on the denomination currency. |
| Urquhart (2016), "The inefficiency of Bitcoin", Economics Letters | The starting point of the crypto efficiency literature. |
| Willems, Shtarkov, Tjalkens (1995), "The context-tree weighting method: basic properties", IEEE Transactions on Information Theory | The estimator recommended by [9] and the natural next model. |
