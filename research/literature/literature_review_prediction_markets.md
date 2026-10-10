# Literature review: are short-horizon crypto prediction markets calibrated?

Review for the idea in
[`../idea_prediction_market_calibration.md`](../idea_prediction_market_calibration.md):
test whether the prices of Polymarket's five-minute "Bitcoin Up or Down"
markets are calibrated probabilities, and whether a model built on spot
prices forecasts the outcome better than the market.

## 1. Scope and method

The review covers four questions:

1. Can a prediction market price be read as a probability at all?
2. What biases are known in such prices?
3. What is known about these particular markets?
4. How are probability forecasts scored and compared?

Eleven open-access papers were downloaded to `papers/` (PDFs are ignored by
git, the links below are the source of record). The "Read" column says how much
of each was read for this review.

| # | Paper | File in `papers/` | Read |
| --- | --- | --- | --- |
| 1 | Wolfers, Zitzewitz (2004), [Prediction markets](https://www.nber.org/papers/w10504), NBER working paper 10504 | `wolfers2004_prediction_markets.pdf` | Abstract |
| 2 | Manski (2004), [Interpreting the predictions of prediction markets](https://www.nber.org/papers/w10359), NBER working paper 10359 | `manski2004_interpreting_prediction_markets.pdf` | Abstract |
| 3 | Wolfers, Zitzewitz (2006), [Interpreting prediction market prices as probabilities](https://www.nber.org/papers/w12200), NBER working paper 12200 | `wolfers2006_prices_as_probabilities.pdf` | Abstract |
| 4 | Snowberg, Wolfers (2010), [Explaining the favorite-longshot bias: is it risk-love or misperceptions?](https://www.nber.org/papers/w15923), NBER working paper 15923 | `snowberg2010_favorite_longshot_bias.pdf` | Abstract |
| 5 | Le (2026), [Decomposing crowd wisdom: domain-specific calibration dynamics in prediction markets](https://arxiv.org/abs/2602.19520) | `le2026_decomposing_crowd_wisdom.pdf` | Abstract, start of introduction |
| 6 | Fabi, Marfè, Ruffo, Schönleber (2026), [Are decentralized prediction markets efficient? Evidence from Bitcoin options](https://fc26.ifca.ai/defi/papers/market-efficiency-prediction-markets.pdf) (talk abstract) | `schonleber2026_market_efficiency_prediction_markets.pdf` | First page of three |
| 7 | Dai, Jia, Yu (2026), [Settlement manipulation in prediction markets](https://arxiv.org/abs/2606.31675) | `dai2026_settlement_manipulation.pdf` | Abstract, parts of the introduction (134 pages in all) |
| 8 | Gneiting, Raftery (2007), [Strictly proper scoring rules, prediction, and estimation](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf), JASA | `gneiting2007_proper_scoring_rules.pdf` | Abstract |
| 9 | Guo et al. (2017), [On calibration of modern neural networks](https://arxiv.org/abs/1706.04599) | `guo2017_calibration_neural_networks.pdf` | Abstract |
| 10 | Diebold, Mariano (1994), [Comparing predictive accuracy](https://www.nber.org/papers/t0169), NBER technical working paper 169 | `diebold1994_comparing_predictive_accuracy.pdf` | Not read (scanned, no text layer) |
| 11 | [Price as focal point: prediction markets, conditional reflexivity, and the politics of common knowledge](https://arxiv.org/abs/2604.24147) (2026) | `arxiv2604_24147_price_as_focal_point.pdf` | Not read |

**This review is at abstract level,** with two exceptions: the introduction
of [7], which is the paper about exactly these markets, and the first page
of [6]. Papers [5], [6], [7] and [11] are from 2026 and not peer reviewed as
far as I could see; their claims are reported as claims.

Two non-academic sources came up in the search and are relevant enough to
name, without having been read: a blog analysis titled "Does Polymarket know
which way BTC will move?", described as a calibration study of 23,700
short-horizon markets, and a profile of traders in the fifteen-minute
markets by Telonex. Papers that could not be downloaded are in section 5.

## 2. Summaries

### 2.1 Is a market price a probability?

**[1] Wolfers and Zitzewitz (2004).** The standard survey. Across many
settings "market-generated forecasts are typically fairly accurate, and ...
outperform most moderately sophisticated benchmarks". Contracts can be
designed to reveal probabilities, means or medians.

**[2] Manski (2004).** A warning. With risk-neutral traders who hold
different beliefs, the price of an all-or-nothing contract "reveals nothing
about the dispersion of traders' beliefs and partially identifies the central
tendency". The mean belief lies in an interval around the price, not at it.

**[3] Wolfers and Zitzewitz (2006).** The reply. Under stated conditions
prices equal mean beliefs, and "for a broad class of models prediction market
prices are usually close to the mean beliefs of traders", though "sometimes
biased". What matters is traders' risk aversion and how their beliefs are
spread.

### 2.2 Known biases

**[4] Snowberg and Wolfers (2010).** The favourite-longshot bias: "longshots
are overbet, while favorites are underbet", a long-standing regularity of
betting odds. Using bets on horse races, including compound bets, the paper
finds "evidence in favor of the view that misperceptions of probability drive
the favorite-longshot bias", not a taste for risk.

**[5] Le (2026).** Calibration on Kalshi and Polymarket from 353 million
trades in 429,000 binary contracts.

- "The most robust pattern is persistent underconfidence in political
  markets, where prices compress toward 50%; it replicates on Polymarket."
- Calibration is "conditional: a price's meaning depends on what, when and
  how much is traded".
- A measurement warning: under conservative standard errors "roughly half
  of the raw slope variation reflects estimation noise".
- Calibration is summarized by the slope of a logistic recalibration of the
  price, the same quantity our experiments fit.

**[6] Fabi et al. (2026).** Compares Polymarket prices of Bitcoin
price-level bets with probabilities implied by options.

- Prices "broadly track option-implied benchmarks".
- Simple bets on the final price are well aligned; path-dependent bets
  deviate.
- "Mispricing is most pronounced at contract inception and near expiration."
- "Low-probability tail events are persistently overpriced."

These are bets over days to months, not minutes.

### 2.3 The five-minute Bitcoin markets

**[7] Dai, Jia, Yu (2026).** The one paper found on exactly these markets.

- **What the contract is:** it "pays $1 if Bitcoin is higher at the close of
  a five-minute window than at its open". It launched on 12 February 2026.
- **How it resolves:** "not on Polymarket but against a Chainlink oracle
  that averages the price across major spot exchanges". The Binance mid
  price "sits about two and a half basis points from the oracle" and
  finishes "on the same side of the strike as the resolution about 85% of
  the time".
- **The finding:** because a trader can move the underlying, "settlement-time
  spot order flow spikes, causing large price reversals after settlement".
  Manipulators profit, "mostly from retail".
- **The remedy:** "manipulation is largely absent in the fifteen-minute
  contracts".

### 2.4 Scoring and calibrating probability forecasts

**[8] Gneiting and Raftery (2007).** A scoring rule is **proper** if a
forecaster does best, in expectation, by stating what it really believes.
The logarithmic and the quadratic (Brier) scores are proper. Proper scores
"relate to information measures, entropy functions, and Bregman
divergences".

**[9] Guo et al. (2017).** Modern neural networks are poorly calibrated, and
"temperature scaling, a single-parameter variant of Platt scaling, is
surprisingly effective" at fixing it. A one-parameter stretch of the log
odds is the natural first recalibration to try.

**[10] Diebold and Mariano (1994)** is the standard test for whether two
forecasts differ in accuracy, allowing for autocorrelated errors. The
downloaded copy is a scan without text and was not read; the experiments use
the same idea (a mean loss difference with an autocorrelation-robust
standard error).

## 3. Insights gathered

1. **Market prices are usually good forecasts and hard to beat** [1].
2. **A price is not guaranteed to be a probability.** Theory allows a gap
   [2], usually small [3].
3. **The typical bias is compression towards 50%:** favourites underpriced,
   longshots overpriced [4, 5, 6]. It is found in horse racing, political
   markets and Bitcoin bets alike.
4. **Calibration depends on domain, horizon and time to expiry** [5, 6]. A
   result for one kind of market does not carry to another.
5. **Mispricing concentrates at the start and near the end of a contract**
   [6].
6. **Much apparent variation in calibration is noise** [5]. Standard errors
   must respect dependence between observations.
7. **These markets resolve on an oracle price, not on Binance** [7]. Binance
   agrees with the resolution about 85% of the time overall.
8. **The last seconds of these markets are not clean.** Order flow in the
   underlying spikes at settlement, with price reversals afterwards [7].
9. **Score with proper rules** [8]; **recalibrate with one parameter first**
   [9]; **compare forecasts by paired loss differences with robust errors**
   [10].
10. **Calibration is summarized by a recalibration slope** [5, 9]: above 1
    means prices are compressed.

## 4. What this means for our idea

**The question is topical and only partly answered.** The five-minute markets
are months old. One academic paper studies them, for manipulation, not
calibration [7]. A blog analysis reportedly covers calibration. A careful,
out-of-sample study of calibration and of whether models beat the price would
still be new, as far as this search shows.

**Three things in the literature change the design.**

- **Labels (insights 7 and 8).** The idea file planned to check outcomes
  against Binance. [7] says in advance what that check finds: Binance and the
  resolution disagree in about 15% of markets, concentrated in close calls.
  The first experiments in
  [`../../src/notebooks/06_prediction_market_data_and_baseline.ipynb`](../../src/notebooks/06_prediction_market_data_and_baseline.ipynb)
  confirm it on our data (84% agreement for moves under one basis point) and
  find something worse: the dataset's outcomes are themselves inferred from
  the last quote, and missing for exactly those close calls.
- **What "beating the market" must mean (insights 1 and 6).** The default
  expectation is that the market wins. A model must beat it out of sample,
  by more than a robust standard error, at a decision time fixed in advance.
- **What bias to look for (insights 3, 5, 10).** Compression towards 50%,
  measured by a recalibration slope, especially early in the contract.

**What the first experiments found, against these expectations:**

- The market is well calibrated, and no model beats it on the dev period:
  in line with [1].
- There is a mild compression towards 50% one minute into the market, with a
  recalibration slope above 1: the bias of [4, 5, 6]. It does not survive as
  an out-of-sample gain.
- The spot-only model loses clearly, so the market knows more than the
  Binance price path. Given [7], part of that is the oracle price itself,
  which is not the Binance price.

**Changes to the idea that follow from the literature:**

| Original plan | Change | Source |
| --- | --- | --- |
| "Exact labels: the resolutions" | The labels in the available data are inferred and incomplete. Use soft labels, test sensitivity, and obtain true resolutions before any final claim. | [7]; data |
| Check outcomes against Binance spot | Keep as a diagnostic only. Binance is a feature, not a label. | [7] |
| Features from Binance spot | Add the oracle's own price feed if it can be obtained; it is what the contract settles on. | [7] |
| Calibration by reliability diagram | Keep, and add the recalibration slope as the summary number. | [5, 9] |
| Recalibrate with a monotone correction | One-parameter stretch first; the flexible version overfits. | [9]; data |
| Decision times 0, 30, 60, 120, 240 seconds | Keep several, fix one as primary. Expect mispricing, if any, near the open and near the end. | [6] |
| Economic check against the spread | Keep. Near the end, manipulation of the settlement price is a risk a backtest does not show. | [7] |
| Cross-check on other assets and on 15-minute markets | More interesting than planned: [7] reports that the 15-minute contracts behave differently at settlement. | [7] |

**A sharper research question.** *One minute into a five-minute market, is
the price compressed towards 50%, and is the compression stable enough to
correct out of sample?* It is specific, it has a known counterpart in other
betting markets, the first data shows a hint of it, and it needs only true
resolutions and more weeks of data to settle.

## 5. Papers to obtain

| Paper | Why it matters |
| --- | --- |
| Page, Clemen (2013), "Do prediction markets produce well-calibrated probability forecasts?", Economic Journal | The classical calibration study of prediction markets, including the dependence on time to expiry. |
| Niculescu-Mizil, Caruana (2005), "Predicting good probabilities with supervised learning", ICML | Platt scaling against isotonic regression: when the flexible method overfits. The download failed. |
| Dimitriadis, Gneiting, Jordan (2021), "Stable reliability diagrams for probabilistic classifiers", PNAS | A principled way to draw reliability diagrams without arbitrary bins. A wrong file was downloaded under this name and removed. |
| Murphy (1973), "A new vector partition of the probability score" | The decomposition of the Brier score used in the experiments. |
| The full text of [7] | Read in part. Its data section and its treatment of the oracle price are directly relevant to our labels. |
| The blog analysis "Does Polymarket know which way BTC will move?" and its code | The closest prior calibration analysis. Not academic, but it should be read before claiming novelty. |
