# Write-up: market regimes with HMM vs. MDL segmentation

Feasibility study for the idea in
[`idea_market_regimes_hmm_mdl.md`](idea_market_regimes_hmm_mdl.md). It covers
the literature, the data, the split, a baseline and a first fit of the two
proposed methods.

- Literature review: [`literature/literature_review.md`](literature/literature_review.md)
- Notebook with all numbers and charts: [`../src/notebooks/01_market_regimes_data_and_baseline.ipynb`](../src/notebooks/01_market_regimes_data_and_baseline.ipynb)
- Code: [`../src/scripts/binance_data.py`](../src/scripts/binance_data.py), [`../src/scripts/regimes.py`](../src/scripts/regimes.py), tests in [`../tests/`](../tests/)

## Verdict

**The idea is worth doing, in a changed form.** The data is clean and
sufficient, the evaluation can be made objective, and the baseline leaves
room. But the method as first proposed (a Gaussian HMM on daily returns) does
not find regimes: its states last one to three days. Both methods work, and
agree with each other, once they observe daily realized variance instead of
the raw return. The project should be built on that, and its question should
shift from "find the regimes" to "do regimes recur (HMM) or is each one new
(segmentation)".

All results below are for BTC/USDT unless another coin is named. Candidate
models were scored on the dev period only, so nothing here is a final result.

## 1. Data

| | |
| --- | --- |
| Source | Binance spot candles from [data.binance.vision](https://data.binance.vision), free, no API key |
| Content | Hourly candles for BTC, ETH, BNB, SOL, XRP against USDT, from listing to 2026-10-09 |
| Size | 54,000 to 80,000 hourly bars per coin, 22 MB in total |
| Daily observations | 3,340 for BTC and ETH, 3,259 for BNB, 3,080 for XRP, 2,250 for SOL |

**Quality.** Under 0.2% of hours are missing for every coin. The gaps are
exchange outages, almost all before 2022. The largest, about 32 hours in
February 2018, left 43 bars that do not start on the hour; their prices are
valid. There are no duplicates and no impossible prices. BTC was cross-checked
against a second copy of the same market (one-minute candles): over 59,345
common hours, prices differ by more than 0.01% in at most 2.

**Is it enough and relevant?** Yes for the models in question, which have at
most a few dozen parameters. Nine years contain several full volatility cycles. It is
too little for data-hungry models at daily frequency; hourly data is there if
needed.

**Features and labels.**

- **Features:** the daily log return, and the log of the daily realized
  variance (sum of the 24 squared hourly returns).
- **Labels:** none. The true regime of a day is unknown. Models are judged by
  out-of-sample predictive scores, by the persistence of their regimes, and by
  agreement between the two methods.

**What the descriptive statistics say.**

- Returns are heavy-tailed: excess kurtosis of 8 to 22 across coins, worst
  days of -50 to -59%.
- Returns have no autocorrelation, so direction is not predictable from the
  series. The regimes available in this data are **volatility regimes**, not
  "trending vs. ranging" as the idea first phrased it.
- Volatility has memory lasting months, which is what a regime model can use.
- Log realized variance is a far cleaner signal than the return: lag-1
  autocorrelation 0.64 against 0.18 for the absolute return, and a nearly
  Gaussian distribution.
- Volatility on weekends is only 69 to 78% of the weekday level, for every
  coin.

## 2. Preprocessing and split

**Preprocessing:** log returns, log realized variance, outage days kept for
returns but excluded from variance scoring, no removal of extreme days.
One step is identified but not applied yet: removing the day-of-week effect
from realized variance, so that weekends are not mistaken for calm regimes.

**Split, by date:**

| Part | Period | BTC days | BTC volatility per year |
| --- | --- | --- | --- |
| Train | 2017-08-18 to 2022-12-31 | 1,962 | 79% |
| Dev | 2023-01-01 to 2024-12-31 | 731 | 48% |
| Test | 2025-01-01 to 2026-10-09 | 647 | 43% |

**The main property of this split is a distribution shift.** Every coin is
much calmer after 2022 than in the train period. That is how the market
developed, not a flaw of the split, and it decides which models do well: those
that adapt beat those that remember.

**Leakage.** Splits are by date only. Every forecast for day *t* uses data
before *t*, which is tested in code by changing future values and checking
that earlier outputs do not move. Viterbi paths and hindsight segmentations
are used only to describe the train period. Model choices use dev only. The
full checklist is in the notebook, section 3.3.

## 3. Baseline

**Task.** At the end of each day, give a probability distribution for the next
day's return.

**Metrics, and why.**

| Metric | Reason |
| --- | --- |
| **Log score**: mean log predictive density, in nats per day, higher is better | A proper scoring rule that judges the whole distribution. Its negative is the code length of the data under the model, so it puts the HMM and the MDL method on one scale. |
| **QLIKE** of the variance forecast against realized variance, lower is better | Isolates the volatility forecast, and ranks forecasts correctly even though realized variance is a noisy proxy. |
| **Persistence**: number of switches, run length | A property the regimes must have to be regimes. |

**Protocol.** Fit on train, run forward through dev and test without
refitting, score one-step-ahead forecasts. Differences between models are
reported with a standard error that allows for autocorrelation.

**The fixed baseline** is a Gaussian with zero mean and an exponentially
weighted variance with decay 0.94 (the RiskMetrics rule). It has no fitted
parameter and no notion of regimes.

| BTC | Dev log score | Test log score | Dev QLIKE | Test QLIKE |
| --- | --- | --- | --- | --- |
| Constant Gaussian | 2.08 | 2.12 | 0.94 | 1.00 |
| Constant Student-t | 2.25 | 2.33 | 1.33 | 1.42 |
| Rolling 30-day Gaussian | 2.20 | 2.35 | 0.59 | 0.49 |
| **EWMA Gaussian (fixed baseline)** | **2.23** | **2.38** | **0.53** | **0.44** |
| GARCH(1,1) Gaussian | 2.21 | 2.30 | 0.65 | 0.64 |
| GARCH(1,1) Student-t | 2.34 | 2.46 | 0.55 | 0.48 |

**How the baseline performs.**

- It is a strong variance forecaster: the best QLIKE of all baselines on BTC,
  ahead of GARCH, because it has no long-run level learned from the turbulent
  train years.
- Its weakness is its Gaussian tail. A constant Student-t, which knows nothing
  about current volatility, matches it on the dev log score. GARCH with
  Student-t errors beats it by 0.11 nats per day on dev (3.5 standard errors)
  and 0.08 on test (2.7).
- On the other four coins the ranking is mostly the same. The exception shows
  the weakness clearly: on XRP dev the baseline scores below the constant
  Gaussian, because of one day (13 July 2023, +55% after a court ruling) that
  was a 27-sigma event under its forecast.

So there are two bars for a regime model: **beat the EWMA baseline to be
useful, match GARCH with Student-t errors to be competitive.**

## 4. Does the proposed method fit the task and data?

**As proposed, no.**

| On BTC train (1,962 days) | Switches or breaks | Typical duration |
| --- | --- | --- |
| Gaussian HMM on returns, 3 states | 347 | median run 2 days |
| MDL segmentation of returns | 14 | median segment 67 days |
| Agreement (adjusted Rand index) | 0.16 | |

- The HMM uses its states to explain fat tails: calm days go to one state and
  wild days to another, largely independent of the day before. It is a good
  density model (BIC prefers 3 states to 1 by a wide margin) and a poor regime
  detector. This is what the literature reports for crypto (Koki et al., 2020).
- The MDL segmentation does give regimes on the intended time scale, stable
  under its one setting (minimum segment length). But it compresses the train
  returns worse than the HMM (-3,596 against -3,710 nats), again because of
  fat tails inside each Gaussian segment.
- The two therefore describe different things, and agree at chance level.

**With log realized variance as the observation, yes.**

| On BTC train | Switches or breaks | Agreement with MDL (adjusted Rand index) |
| --- | --- | --- |
| HMM on log realized variance, 2 states | 35 (median run 26 days) | 0.63 |
| HMM on log realized variance, 3 states | 73 (median run 14 days) | 0.44 |
| MDL segmentation of log realized variance | 24 | |

The HMM becomes persistent and two independent methods find largely the same
structure. Without labels, that agreement is the best available evidence that
the regimes are in the data and not produced by a model.

**Out of sample (BTC dev), against the baseline:**

| Model | Log score | QLIKE | Log score minus EWMA (std. error) |
| --- | --- | --- | --- |
| EWMA Gaussian (fixed baseline) | 2.23 | 0.53 | |
| GARCH(1,1) Student-t | 2.34 | 0.55 | +0.11 (0.03) |
| HMM on returns, 3 states | 2.30 | 0.70 | +0.06 (0.04) |
| HMM on returns, 5 states | 2.31 | 0.62 | +0.08 (0.03) |
| HMM on returns and log realized variance, 4 states | 2.32 | 0.53 | +0.09 (0.03) |
| MDL forecaster (variance of the latest segment) | 2.20 | 0.62 | -0.03 (0.02) |

- The HMMs beat the baseline on the log score, but the evidence is weak: the
  best are about two standard errors ahead, the 2-state model is not
  distinguishable from the baseline.
- The HMM on returns has a clearly worse variance forecast. Its forecast is
  almost flat at the train-period level, because non-persistent states carry
  little information from one day to the next.
- Adding realized variance improves both scores and brings QLIKE level with
  the baseline. Its states are still short-lived, though (median run 2 to 5
  days on train).
- The MDL segmentation, used as a forecaster, is slightly below the baseline.
  It needs 20 days of evidence before it opens a new segment.
- No regime model reaches GARCH with Student-t errors yet.

## 5. What to expect from more complex models

Ordered by how much the evidence above supports them.

| Extension | Expected effect | Basis |
| --- | --- | --- |
| Refit as time moves forward (walk-forward) | Large gain in QLIKE for every fitted model. The static fit is the main handicap under the volatility decline. | Split statistics; EWMA beating GARCH on QLIKE. |
| Student-t emissions in the HMM, Student-t segments in MDL | Higher log score, fewer states needed, states freed to track volatility level instead of tails. Should close much of the gap to GARCH Student-t. | Constant Student-t matching EWMA; Ardia et al. (2019). |
| Persistence built in: sticky transitions, a jump penalty, or a semi-Markov model | Regimes of weeks instead of days in the predictive HMM. Effect on scores uncertain. | Shu et al. (2024), Suda and Spiteri (2019). |
| Day-of-week adjustment of realized variance | Fewer spurious switches. Small but nearly free. | Weekend volatility is 22 to 31% lower. |
| Online segmentation (Bayesian online change point detection, or MDL with a faster rule for the newest segment) | Removes the 20-day lag of the MDL forecaster. | Adams and MacKay (2007). |
| Several coins jointly | Regimes in correlations; more data per regime. | Hallac et al. (2019). |

**A caution on expectations.** The gains available are small in absolute
terms. The whole distance from the fixed baseline to the strong reference is
0.11 nats per day, and one standard error on a 731-day period is about 0.03.
Differences between good models will often be inside the noise on one coin
and one period. The project needs all five coins, both dev and test, and
honest standard errors to say anything firm. It should also be expected that
a well-built regime model matches GARCH Student-t rather than beats it. The
case for regime models is then interpretability at equal accuracy.

## 6. Conclusions

**Dataset.** Good. Free, clean, long enough, validated against a second
source, and already split. Its defining feature is non-stationarity: a
declining volatility level that punishes static models. Two findings were not
anticipated by the idea: direction is unpredictable, so only volatility
regimes are available, and there is a weekly cycle to remove.

**Baseline.** The EWMA rule is a hard baseline for variance and a soft one
for the density. That is a useful combination: it shows precisely where a
regime model can add value (tails, and structure in the level) and where it
cannot easily (tracking day-to-day variance).

**Method.** The original pairing, Gaussian HMM on returns against MDL
segmentation of returns, should be dropped: the outcome is known (they
disagree, for a known reason). The pairing that works is on realized
variance, with fat-tailed distributions and walk-forward fitting.

**Is there potential?** Yes, with a realistic view of where.

- **As a forecasting project: limited.** HMM regimes on crypto are well
  covered in the literature, and beating GARCH Student-t by a margin that
  survives standard errors is unlikely.
- **As a study of the structure of crypto volatility: good.** The question
  "do volatility regimes recur, or is each one new" is concrete, can be
  measured in code length and predictive score, was not answered in the
  papers reviewed, and fits the course topics (HMMs, MDL, unsupervised
  learning) closely. The first evidence is already informative in both
  directions: two methods agree on the regimes (adjusted Rand index 0.63),
  which favours real structure, while the typical level of volatility after
  2022 sits well below the train-period average, so that states learned on
  train fit later years poorly, which favours "each one is new".

**Recommended next steps.**

1. Walk-forward refitting for all models, with the day-of-week adjustment.
2. Student-t HMM and Student-t MDL segments on realized variance.
3. A persistence mechanism in the HMM, chosen on dev.
4. Run the final comparison once on test, for all five coins.
5. Before claiming novelty, read the papers listed in section 5 of the
   literature review, in particular Thies and Molnár (2018) and Davis et al.
   (2006).

## Limitations of this study

- Candidate models were compared on one coin (BTC) and one period (dev). The
  baselines were run on all five coins.
- Models were fitted once on train. This understates every fitted model
  relative to the EWMA baseline, which adapts by construction.
- The MDL code is a simple two-part code. Refined MDL may segment differently.
- HMMs were fitted by EM from 10 to 20 random starts; a better optimum may
  exist for the larger models.
- During exploration the HMM on returns and the MDL forecaster were also
  scored once on the test period. Those numbers were not used for any choice
  and are not reported.
- The literature search was a handful of queries, and several papers were read
  only in part (see the table in the literature review).
