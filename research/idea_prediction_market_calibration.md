# Idea: are short-horizon crypto prediction markets calibrated?

Polymarket runs markets on whether Bitcoin will be up or down over the next
five minutes. The price of the "Up" share is the market's probability of an
up move. Test whether those probabilities are calibrated, and whether a
model built from spot price data can forecast the outcome better than the
market does.

## Summary

- **Data:** already in `data/`. Second-by-second quotes for 15,682 five-minute
  BTC markets over 56 days, with outcomes, plus one-minute BTC spot candles
  for the same period.
- **Topics covered:** classification with probability outputs; the learner
  survey (logistic / maximum entropy, Bayesian updating, boosting); information
  theory and statistics (proper scoring rules, calibration).
- **Why high quality:** exact labels, thousands of non-overlapping events, and
  the strongest baseline available anywhere: a liquid market's own price.
- **What you learn:** probabilistic forecasting, calibration, proper scoring,
  and how to test a claim of market inefficiency without fooling yourself.
- **Novelty:** medium to high. These markets are recent. This is my estimate,
  not the result of a literature search.

## Data

Two Polymarket datasets are in the repository. I inspected their structure
and coverage; I did not analyse prices or outcomes.

**Main dataset: `data/polymarket_5m_kachoio/`**

| | |
| --- | --- |
| Markets | 15,682 five-minute "Bitcoin Up or Down" markets |
| Period | 24 March to 18 May 2026, 56 days, 288 markets on a full day |
| Outcomes | Up 7,177, Down 7,049; about 1,450 markets without a recorded outcome |
| Quotes | 4.7 million rows: one per second, 300 per market |
| Per second | best bid and ask of the Up share and of the Down share, sizes at the best prices, and depth |
| Token labels | each market names its Up token and its Down token |

**Second dataset: `data/polymarket_crypto_5m_15m/`**

| | |
| --- | --- |
| Markets | BTC, ETH, SOL and XRP, five-minute and fifteen-minute |
| Dense period | about four and a half days, 1 to 6 March 2026, 1,536 markets a day |
| Trades | 9 million, with price, size and side |
| Order books | snapshots, about 160 per market, with full bid and ask levels |
| Outcomes | 7,399 resolutions; about 770 of them are fifteen-minute BTC markets from 2 to 9 January 2026 without trade data |
| Token labels | not given; the Up token has to be inferred (the winning token trades near 1 at the end) |

**Spot prices:** `data/btcusdt_spot_1m/` has one-minute Binance BTC candles
from 2020 to October 2026, covering both periods.

Three properties shape the project:

- **The main dataset is the right size and shape.** About 14,000 consecutive,
  non-overlapping events with outcomes. One standard error of an observed
  frequency near 50% is about 0.4 percentage points, so a miscalibration of
  one to two points is detectable.
- **It has quotes, not trades.** The forecast to beat is the mid price, and
  the bid-ask spread says directly what a forecast would have to gain to be
  worth acting on.
- **The second dataset is short but broad.** Four assets and two horizons
  over a few days: useful to check that a finding is not specific to BTC,
  not to establish one. Markets on different assets in the same five
  minutes are far from independent.

A correction to what I said when first describing this idea: the trades do
not span January to March. They cover about four and a half days in March.
The 56-day dataset is the one to build on.

> **Update after the feasibility study.** The outcomes in the main dataset
> are not exact: they were inferred from each market's last quote and are
> missing for about 9% of markets, the close calls. The summary above calls
> the labels "exact"; that is wrong. See
> [`prediction_market_writeup.md`](prediction_market_writeup.md).

## Research questions

1. **Calibration.** When the market prices Up at 60%, does Up happen 60% of
   the time? How does that depend on how far into the five minutes the price
   is read?
2. **At the open.** Before the interval starts there is little to go on. Is
   the opening price 50%, and should it be? The complexity study found that
   the direction of an hourly return tends to reverse (the next sign was
   predictable 52.4% of the time on dev). Does the same hold at five minutes,
   and does the market price it in?
3. **During the interval.** Once the price has moved, the chance of finishing
   up follows from the size of the move, the time left and the volatility.
   Does the market price agree with that simple formula, or does it
   under-react or over-react?
4. **Can a model beat the market?** Does adding spot-based features to the
   market price improve the forecast out of sample?
5. **Is any edge worth anything?** Is the improvement larger than the
   bid-ask spread?

## Approach

1. **Fix the decision times.** Read the market at set moments, for example
   0, 30, 60, 120 and 240 seconds into the interval. Each moment is its own
   forecasting problem.
2. **Baseline: the market.** The mid price of the Up share at the decision
   time.
3. **Reference: no information.** Always 50%, and the historical share of Up.
4. **A structural model.** The probability of finishing up as a function of
   the move since the open, divided by volatility times the square root of
   the time left. This has one or two parameters and a clear meaning.
5. **Learned models.** Logistic regression on: the market price, the move so
   far, recent volatility, the direction of the previous intervals, time of
   day, and order book imbalance. Then gradient boosting on the same
   features.
6. **Recalibration.** The simplest "model" of all: a monotone correction of
   the market price fitted on the training period. If the market is
   miscalibrated, this alone beats it.
7. **Cross-check** the main findings on the four-asset dataset.

## Evaluation

- **Split by time:** for example the first 36 days to fit, the next 10 to
  tune, the last 10 to test. No shuffling: neighbouring markets share
  volatility and news.
- **Main metric: log loss** of the forecast probability (equivalently, its
  code length in bits per market). It is a proper scoring rule, and the
  market price can be scored with it exactly like a model. Also the Brier
  score, which is less sensitive to forecasts near 0 or 1.
- **Calibration:** reliability diagrams with intervals, and the
  decomposition of the Brier score into calibration and sharpness.
- **Significance:** paired differences in per-market log loss between a
  model and the market, with standard errors that allow for dependence
  between neighbouring markets.
- **Economic check:** trade only when the model disagrees with the market by
  more than the spread; report the profit after paying the spread, on the
  test period.
- **One decision time is primary** and is chosen before looking at results,
  so that the best of several is not reported as the finding.

## Risks

- **The market may simply be right.** That is the most likely outcome, and it
  is still a result: a clean, well-powered test that finds no inefficiency.
  The write-up should be planned for it.
- **Timing leakage.** A forecast for decision time *t* may use nothing after
  *t*. Spot candles are stamped at the start of the minute and are complete
  only at its end; using the candle that contains *t* would leak. This is the
  main trap.
- **The reference price.** The markets resolve against a specific price
  feed, which I have not confirmed and which may not be Binance spot. If it
  differs, features built from Binance are slightly off, and "up" in our spot
  data will not always match the recorded outcome. Check the agreement
  first.
- **Missing outcomes.** About 9% of the markets in the main dataset have no
  recorded outcome. If they are missing for a reason related to the
  outcome, calibration estimates are biased. Find out why, or reconstruct
  them from the final quotes.
- **Stale quotes.** A quoted mid price in a thin moment is not a price
  someone would trade at. Use the sizes at the best prices to filter.
- **Many comparisons.** Decision times, feature sets and models multiply
  quickly. Fix the primary analysis in advance.
- **Dependence.** Five-minute intervals are consecutive, and volatility
  persists, so 14,000 markets are fewer than 14,000 independent draws.
- **One venue, eight weeks.** Nothing here says how these markets behave in
  another period.

## Relation to other ideas

- Tests, on a different market, the hourly reversal found in
  [algorithmic complexity as a market efficiency measure](idea_market_complexity_efficiency.md):
  there it was statistically clear and too small to trade after costs.
- Uses the volatility forecasts of
  [market regimes with HMM vs. MDL segmentation](idea_market_regimes_hmm_mdl.md)
  as an input to the structural model.
- Listed as idea 9 in [`project_ideas_round2.md`](project_ideas_round2.md).

## References

To be verified before use; cited from memory.

- Brier (1950), "Verification of forecasts expressed in terms of
  probability".
- Gneiting and Raftery (2007), "Strictly proper scoring rules, prediction,
  and estimation".
- Murphy (1973), "A new vector partition of the probability score".
- Wolfers and Zitzewitz (2004), "Prediction markets".
- Niculescu-Mizil and Caruana (2005), "Predicting good probabilities with
  supervised learning".
