# Write-up: algorithmic complexity as a market efficiency measure

Feasibility study for the idea in
[`idea_market_complexity_efficiency.md`](idea_market_complexity_efficiency.md).
It covers the literature, the data, the split, a baseline and a first test of
the proposed method.

- Literature review: [`literature/literature_review_market_complexity.md`](literature/literature_review_market_complexity.md)
- Notebook with all numbers and charts: [`../src/notebooks/02_market_complexity_data_and_baseline.ipynb`](../src/notebooks/02_market_complexity_data_and_baseline.ipynb)
- Code: [`../src/scripts/binance_data.py`](../src/scripts/binance_data.py), [`../src/scripts/complexity.py`](../src/scripts/complexity.py), tests in [`../tests/`](../tests/)

## Verdict

**The method as proposed does not work, and that is the most useful result
of the study.** Compression-based measures (Lempel-Ziv complexity, zlib,
lzma) cannot detect inefficiencies of the size that exists in crypto markets:
on a year of hourly data they find nothing in any of 717 coin-years. A
first-order entropy measure, on the same data, finds structure in over 90%
of them, and the structure is real: it predicts return signs out of sample.

**The hypothesis is not supported either.** Less liquid coins are not
consistently less efficient; the relation is weak and changes sign between
periods.

**There is still a good project here, with a different question:** how large
must an inefficiency be for each kind of estimator to see it, and where do
real markets fall on that scale. The data and tooling for it are in place.

Candidate measures were computed on train and dev only. The test period was
used only for the fixed baseline.

## 1. Data

| | |
| --- | --- |
| Source | Binance spot candles from [data.binance.vision](https://data.binance.vision), free, no API key |
| Universe | 166 coins quoted in USDT: listed by December 2020, still trading in December 2022, excluding stablecoins, leveraged and pegged tokens |
| Content | Hourly candles from each coin's listing to 2026-10-09 (or to its delisting) |
| Size | 8.9 million hourly bars, 420 MB |
| Unit of analysis | The coin-year: one coin, one calendar year, about 8,700 hourly up / down symbols |
| Coin-years | 836 with a full year of data: 427 train, 290 dev, 119 test |

**The universe was fixed by a rule on listing dates**, before any measure was
computed, and it includes coins that were later delisted: 62 of the 166
stopped trading before the end of the data.

**Quality.** No duplicates, no impossible prices, no missing values. Missing
hours are rare (median 30 per coin), except for two coins with gaps of
months. Four coins have a single hourly price change by a factor of 10 to
1,000, consistent with token redenominations; this is harmless for an
analysis of signs.

**Is it enough and relevant?** At hourly bars, yes. At daily bars, no: about
1,000 daily observations per coin are too few to confirm anything out of
sample. Most of the earlier literature uses daily data.

**Features and labels.**

- **Observation:** the sequence of hourly up / down symbols of a coin.
- **Features per coin-year:** complexity measures of that sequence, each
  expressed as a *deficit* against shuffled copies of the same sequence, in
  bits per symbol. Covariates: median daily traded value (liquidity), share
  of zero returns (coarseness of the price grid), volatility.
- **Labels:** none for the measure. For the out-of-sample check the label is
  the next symbol, which is free.

Market capitalization, named in the idea, was not collected. Traded value
from the candles is used as the liquidity measure.

**What the descriptive statistics say.**

- Ups and downs are balanced (48.4 to 51.8% up).
- Zero returns are common: 3.4% of hours in the median coin, above 10% in 46
  coin-years, and more frequent in less liquid coins (rank correlation
  -0.56).
- Liquidity spans three orders of magnitude across coins, which is what a
  test of the liquidity hypothesis needs.

## 2. Preprocessing and split

| Issue | Decision |
| --- | --- |
| Volatility clustering and intraday patterns | Use the binary sign coding, which they cannot affect. |
| Zero returns | Dropped from the sequence. Their share is kept as a covariate. |
| Estimator bias on finite sequences | Every measure is compared with shuffled copies (same length, same symbol frequencies), never with a theoretical constant. |
| Partial years, suspended coins | Coin-years with fewer than 8,500 hourly bars are excluded. |
| Overlapping windows | Not used. Coin-years do not overlap. |

**Split, by date:** train before 2023, dev 2023 and 2024, test from 2025 to
2026-10-09.

The measures have no parameters, so the split works differently than in a
prediction project. Train is where the measurement was designed, dev checks
that findings repeat in a later period, test is held back. For the prediction
check the split is used in the usual way: fit on train, score later.

**Leakage and bias.** The universe rule looks only at the train period.
Predictors are fitted on train only. Shuffling is done within a coin-year.
One limit remains: coins move together, so coin-years of the same year are
not independent, and cross-sectional results are reported per year instead
of pooled. Dev and test contain fewer coins than train because of
delistings.

## 3. Baseline

**Task.** From the past signs of a coin's hourly returns, give a probability
for the next sign. In an efficient market nothing beats 50 / 50.

**Metrics, and why.**

| Metric | Reason |
| --- | --- |
| **Code length**: mean of -log2 P(symbol), bits per symbol, out of sample | It is the compression rate a model actually achieves on new data, so it measures the idea's quantity directly and honestly. An efficient market gives 1 bit. |
| **Accuracy** of the predicted sign | Easy to read; 50% for an efficient market. |
| **Gross return per trade** in basis points | Separates statistical from economic significance. |

**Protocol.** Fit on train, score on dev and test, no refitting. A coin is
scored if it has at least 90% of the period's hours.

**The fixed baseline** is a Markov model of order 1: the forecast depends
only on the previous sign. It is the classical lag-1 autocorrelation test of
efficiency, expressed in bits. The reference below it is the memoryless
model.

| | Dev (140 coins) | Test (106 coins) |
| --- | --- | --- |
| Memoryless: bits per symbol | 1.0000 | 1.0001 |
| Memoryless: accuracy | 50.4% | 50.1% |
| **Order 1: bits per symbol** | **0.9983** | **0.9988** |
| **Order 1: accuracy** | **52.4%** | **52.2%** |
| Order 1: range of accuracy across coins | 50.5 to 54.7% | 49.2 to 55.4% |
| Coins for which order 1 saves code length | 94% | 83% |
| Coins with accuracy more than 3 standard errors above 50% | 95% | 86% |
| Gross return per hour, median coin | 1.4 bps | 1.0 bps |
| Mean absolute hourly move, median coin | 70 bps | 70 bps |

**How the baseline performs.**

- **It beats the efficient-market reference out of sample for almost every
  coin.** The typical coin is about six standard errors above 50%.
- **The effect is a reversal.** For every coin scored, an hourly return is
  less likely than not to have the sign of the previous one (median
  probability 0.475 on train). It holds for BTC (53.7% accuracy on dev,
  51.7% on test).
- **It is small:** the sign sequence is compressible by about 0.15%.
- **It is persistent:** coins with strong reversal on train have it on dev
  (rank correlation 0.39), and it shows equally at 4-hour bars.
- **It is not exploitable as it stands.** One to 1.4 basis points per hour
  gross is far below the cost of a round trip at ordinary exchange fees (a
  fee of 0.1% per trade would mean 20 basis points). The market is
  inefficient in the statistical sense and efficient after costs.

The baseline sets the bar for the proposed method: **a complexity measure
must at least see this known one-step structure, and to justify itself it
must find more.**

## 4. Does the proposed method fit the task and data?

**No, not at this sample size.** Two independent pieces of evidence.

**Simulation.** Sequences of 8,760 binary symbols in which each symbol
repeats the previous one with a given probability. Share of sequences
declared non-random at the 5% level:

| Repeat probability | True deficit (bits) | Order-1 entropy | Lempel-Ziv | zlib | bz2 | lzma |
| --- | --- | --- | --- | --- | --- | --- |
| 0.50 (fair coin) | 0 | 5% | 6% | 4% | 3% | 4% |
| 0.48 | 0.0012 | 94% | 11% | 8% | 15% | 5% |
| 0.47 | 0.0026 | 100% | 22% | 11% | 33% | 10% |
| 0.45 | 0.0072 | 100% | 48% | 27% | 76% | 27% |
| 0.40 | 0.0290 | 100% | 100% | 97% | 100% | 99% |

Real coins have repeat probabilities between 0.45 and 0.495 (median 0.475).
In that range the compressors are close to blind. They respond reliably only
to an effect of 0.40, a market predictable 60% of the time.

**Real coin-years** (train and dev, 717 in total):

| Measure | Coin-years non-random after false discovery control, train / dev | Typical z-score | Year-to-year rank correlation of a coin's deficit |
| --- | --- | --- | --- |
| Entropy, order 1 | 91% / 92% | 13 | 0.37 |
| Entropy, order 2 | 97% / 92% | 9 to 12 | 0.41 |
| Entropy, order 3 | 96% / 90% | 6 to 8 | 0.38 |
| Entropy, order 8 | 32% / 14% | 1.3 to 1.5 | 0.20 |
| bz2 | 32% / 14% | 1.4 to 1.8 | 0.11 |
| lzma | 0% / 0% | 0.5 to 0.6 | 0.09 |
| zlib | 0% / 0% | 0.4 to 0.5 | -0.01 |
| Lempel-Ziv complexity | 0% / 0% | 0.5 to 0.6 | 0.11 |

- The compressors' deficits are mostly noise: not significant, and not
  stable from one year to the next.
- The normalized Lempel-Ziv index is above 1 in every coin-year. By the
  criterion of Giglio et al. (2008) all 717 would count as efficient.
- This explains a contradiction in the literature, where complexity studies
  find markets nearly random and entropy studies reject efficiency: the
  difference is in the sensitivity of the tools.

**Why.** At a repeat probability of 0.48 the whole year of hourly data is
compressible by about 10 bits. A general compressor's output differs by far
more than that between two random sequences of the same length.

**The hypothesis.** Rank correlation, across coins within a year, between
the order-1 deficit and traded value:

| 2020 | 2021 | 2022 | 2023 | 2024 |
| --- | --- | --- | --- | --- |
| -0.25 | -0.07 | -0.19 | +0.14 | +0.25 |

- Negative (less liquid, less efficient) in the train years, positive in the
  dev years, never strong. The hypothesis fails its first out-of-period
  check.
- BTC, the most liquid coin, has a larger deficit than the median coin in
  every year except 2021.
- **The clear pattern is over time:** the median deficit fell from about 5
  millibits per symbol in 2018 and 2019 to 1 to 2 since 2021. The early
  years rest on few coins (5 and 17).

**The coding matters, modestly here.** Keeping zero returns as a third symbol
raises the measured deficit by about 18% and ties it more closely to the
share of zero returns (0.39 against 0.22). Wider bands or more symbols, which
the literature warns let volatility in, were not tested.

## 5. What to expect from more complex models

| Extension | Expected effect | Basis |
| --- | --- | --- |
| Higher-order Markov models | Almost nothing. Order 3 saves 1.8 millibits on dev against 1.6 for order 1; order 8 is worse than no model. | Measured. |
| Context tree weighting (CTW) | The best candidate: it mixes all orders, is a universal code like the compressors, and has the sensitivity of the entropy estimators. Expected to match or slightly beat order 3. It would give the project a compression method that works. | Gao et al. (2008); our order 1 to 3 results. |
| Longer sequences (several years per coin, or 5-minute bars) | More power for every estimator. At about 100,000 symbols the compressors may start to see the effect. Worth testing in simulation first. | The power table. |
| Richer codings (quantile bins, magnitude) | Larger measured deficits, but mostly volatility unless it is filtered first. | Calcagnile et al. (2016). |
| Neural sequence models | Unlikely to help. The structure is short-range and tiny, and 20,000 symbols per coin is little data. | Order 5 and 8 already overfit. |
| Removing microstructure (ARMA residuals, or mid prices from bid / ask data) | The decisive test of whether the reversal is a genuine inefficiency or a trading-mechanics effect. May remove most of the measured deficit. | Calcagnile et al. (2016). |
| Pooling coins (one model for all) | More data per parameter; could support higher orders. | Coins share the same reversal. |

**A caution on expectations.** The total inefficiency on offer is about
0.002 bits per symbol. No model will find a large effect, because there is
none to find at this horizon. More complex models can only refine a
measurement of something small.

## 6. Conclusions

**Dataset.** Good, and reusable: a rule-based universe with delisted coins,
clean hourly data, a wide range of liquidity. Hourly is the right frequency;
daily is too short. Two properties need care: zero returns, which are tied
to liquidity, and co-movement of coins, which limits the effective sample in
any cross-sectional test.

**Baseline.** Informative beyond expectation. A two-parameter model finds a
reversal in hourly return signs that is present in every coin, confirmed out
of sample on dev and test, stable over time, and too small to trade. It
defines precisely what "inefficiency" means in this data and how big it is.

**Method.** Compression-based complexity, the core of the idea, is the wrong
tool at these sample sizes. It is not slightly worse than the entropy
measures: it sees nothing. Entropy measures of order 1 to 3 with shuffle
control are the tool that works.

**Is there potential?**

- **For the idea as written: no.** Ranking coins by compressibility would
  rank noise, and the liquidity hypothesis does not hold up.
- **For a reframed project: yes.** The finding that the standard
  complexity-based efficiency measure has no power in its usual setting,
  shown both in simulation and on 717 coin-years, is concrete, checkable and
  explains a disagreement in the literature. A project built on "what can
  each estimator detect, and how inefficient are crypto markets on that
  scale" keeps the algorithmic information theory content, adds CTW as the
  constructive part, and can report efficiency over time with honest error
  bars.
- **As a trading idea: no.** The effect is far below costs.

**Recommended next steps.**

1. Implement CTW and add it to the power study and to the coin-year table.
2. Extend the power study to longer sequences, to find the length at which
   compressors become usable.
3. Test whether the reversal survives the removal of microstructure effects.
4. Read Wei (2018) and Shmilovici et al. (2003) before writing up: the first
   is the direct prior test of the liquidity hypothesis, the second the
   closest prior method.
5. Run the final coin-year analysis once on the test period.

## Limitations of this study

- **Microstructure is not removed.** The reversal may partly reflect trading
  mechanics. That it is equally present at 4-hour bars and in BTC speaks
  against a simple bid-ask bounce, but this is not a proper test.
- **Coins are not independent.** No significance is claimed for
  cross-sectional correlations; they are reported per year to show their
  instability.
- **Survivorship in dev and test.** Coins delisted before a period ends drop
  out of it.
- **Simulated power covers one kind of structure** (first-order dependence).
  A limited check with a dependence at lag 3, during exploration, gave the
  expected picture: the order-1 entropy cannot see it, an entropy of order 4
  detects it well before the compressors do. It is not in the notebook.
- **Trading costs are an assumption** (0.1% per trade), not measured.
- **Only one coding was studied in depth.**
- **The literature review is partial.** Most papers were read in part (see
  the table in the review), and the two most relevant paywalled papers were
  not read.
