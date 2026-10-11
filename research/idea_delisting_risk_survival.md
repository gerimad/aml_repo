# Idea: delisting risk, a survival analysis of exchange listings

A third of the trading pairs Binance has ever listed against USDT are no
longer traded. Model how long a pair survives from the quality of its
market, and turn that into a risk score that says, months ahead, which
listed tokens are heading for removal.

## Summary

- **Data:** Binance public candles for every USDT pair, delisted ones
  included. Hourly candles for 166 pairs are already in
  `data/binance_spot_1h/`; the list of all 756 pairs with their first and
  last month is cached in the repo.
- **Topics covered:** classification and regression in the form of survival
  analysis; statistics of censored data and feature engineering; the learner
  survey (Cox regression, boosting, trees, Bayesian models).
- **Why high quality:** exact labels recorded by the exchange itself, a
  workable number of events, features already built and tested in an earlier
  study, and a method that fits the question better than a plain classifier.
- **What you learn:** survival analysis with censoring and time-varying
  covariates, and how to evaluate a risk score that is about *when*, not
  only *whether*.
- **Who benefits:** holders of a token, who are often left with something
  hard to sell after a delisting; exchanges, which monitor exactly this.
- **Novelty:** medium. The price effect of delistings is studied; predicting
  delisting from market quality is less so. This is my estimate, not the
  result of a literature search.

## Data

From the listing of Binance's public data bucket, cached in
`data/binance_spot_1h/_listing.json` (first and last month per pair):

| | |
| --- | --- |
| USDT pairs ever published | 756 |
| No longer published (last month before September 2026) | 249 |
| Of those: ordinary tokens | 188 |
| Of those: leveraged tokens (removed as a product) | 50 |
| Of those: stablecoins, fiat, pegged tokens | 11 |
| Median lifetime of a delisted pair | 36 months (range 1 to 85) |
| Pairs with gaps in their monthly files | 9 |

Delistings by year of the last published month:

| 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 3 | 18 | 20 | 38 | 24 | 46 | 51 | 47 |

Three properties shape the project:

- **About 190 usable events, not 249.** The leveraged tokens were withdrawn
  together as a product decision and the stablecoin pairs for other reasons.
  They say nothing about market quality and must be set aside.
- **Some "delistings" are renames.** The complexity study met several:
  tickers that changed after a rebrand or token swap, where the old pair
  ends and a new one begins. They have to be linked or removed, by hand or
  by rule.
- **The remaining 500 or so pairs are censored:** still alive when the data
  ends. Survival methods use them; a classifier that drops them wastes most
  of the data, and one that labels them "safe" is wrong.

## Research questions

1. Which properties of a market predict that a pair will be delisted, and
   how far ahead?
2. Is traded value alone enough, or do measures of market quality (share of
   hours without a price change, volatility, the inefficiency measure of the
   complexity study) add to it?
3. Does the hazard depend on how long a pair has been listed, and on the
   year it was listed in?
4. How early is a warning possible: six months, three, one?
5. Is a predicted risk calibrated: of pairs given a 20% chance of delisting
   within a year, do about 20% go?

## Approach

1. **Clean the events.** Separate product withdrawals and renames from
   delistings for cause. Record the reason where the exchange published one.
2. **Build monthly covariates per pair,** each from past data only: traded
   value and its trend, share of zero-return hours, volatility, drawdown
   from the peak, number of trades, price relative to the listing price,
   performance relative to BTC, age.
3. **Describe.** Kaplan-Meier survival curves by listing year and by
   liquidity group.
4. **Baselines.** Traded value alone; a logistic model for "delisted within
   12 months" on the latest month's features.
5. **Survival models.** Cox regression with time-varying covariates; a
   discrete-time hazard model (one row per pair and month, fitted by
   logistic regression or boosting); a random survival forest.
6. **Read the model.** Which covariates carry the risk, and how the
   predicted hazard of delisted pairs evolved in their last year.

## Evaluation

- **Split by calendar time:** fit on pair-months up to a cut-off date,
  predict delistings after it. For example fit to the end of 2023, tune on
  2024, test on 2025 and 2026, which contain 46, 51 and 47 delistings.
  Never split pairs at random: a model must not learn from the future of the
  market.
- **Concordance index** (does the model rank pairs that are delisted sooner
  as riskier?), computed for fixed horizons of 3, 6 and 12 months.
- **Time-dependent AUC and Brier score** for "delisted within the horizon",
  with the censoring handled properly.
- **Calibration** of the predicted probabilities, by groups of predicted
  risk.
- **A practical reading:** of the 20 pairs the model ranks riskiest at the
  start of a year, how many are gone by its end?

## Risks

- **Announcements.** Exchanges announce a delisting days or weeks before
  trading stops, and the price collapses on the announcement. Features from
  that window make prediction trivial and useless. The last month or two
  before removal must be excluded, or the announcement dates collected.
- **Few events per test period.** About 50 a year, fewer after cleaning.
  Differences between good models will be inside the noise; the comparison
  that can be made reliably is "model against traded value alone".
- **Delisting is a decision, not a law of nature.** Exchange policy changes
  over the years (the yearly numbers above roughly double from 2020-21 to
  2024-26), and criteria include
  things not in market data: team activity, regulation, security incidents.
- **Selection into listing.** Every pair in the data was once good enough to
  be listed. The model describes decline after listing, not token quality in
  general.
- **One exchange.** A pair delisted on Binance often keeps trading
  elsewhere.
- **Renames and migrations** can masquerade as failures if not caught.

## Relation to other ideas

- Reuses the candle downloader and the universe rules of
  [algorithmic complexity as a market efficiency measure](idea_market_complexity_efficiency.md),
  and its features. That study found 62 of 166 older coins had stopped
  trading by the end of the data.
- Listed as idea 2 in [`project_ideas_round3.md`](project_ideas_round3.md).

## References

To be verified before use; cited from memory.

- Cox (1972), "Regression models and life-tables".
- Kaplan and Meier (1958), "Nonparametric estimation from incomplete
  observations".
- Ishwaran et al. (2008), "Random survival forests".
- Harrell et al. (1982), "Evaluating the yield of medical tests" (the
  concordance index).
- Graf et al. (1999), "Assessment and comparison of prognostic
  classification schemes for survival data" (the Brier score under
  censoring).
