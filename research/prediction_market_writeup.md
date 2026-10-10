# Write-up: are short-horizon crypto prediction markets calibrated?

Feasibility study for the idea in
[`idea_prediction_market_calibration.md`](idea_prediction_market_calibration.md).
It covers the literature, the data, the split, a baseline and a first test of
whether models can beat the market.

- Literature review: [`literature/literature_review_prediction_markets.md`](literature/literature_review_prediction_markets.md)
- Notebook with all numbers and charts: [`../src/notebooks/06_prediction_market_data_and_baseline.ipynb`](../src/notebooks/06_prediction_market_data_and_baseline.ipynb)
- Code: [`../src/scripts/prediction_markets.py`](../src/scripts/prediction_markets.py), tests in [`../tests/test_prediction_markets.py`](../tests/test_prediction_markets.py)

## Verdict

**The market is well calibrated, and nothing tested beats it.** That is the
outcome the idea file called most likely, and the study is large and clean
enough for it to mean something: 15,682 markets, a strict timing protocol,
and a baseline that costs nothing to compute.

**Two things qualify the result.**

- **The labels are weaker than assumed.** The dataset's outcomes are
  inferred from the last quote of each market, and are missing for 9% of
  markets, which are the close calls. The true resolutions could not be
  obtained.
- **There is one lead.** One minute into a market, strong favourites win
  about three points more often than their price says, with the same sign in
  all three periods. It is not yet evidence.

**As a project it is feasible and well-posed, with a likely negative
headline.** Whether that is attractive depends on whether a careful "the
market is efficient" result is acceptable as the outcome.

Candidate models were scored on dev only. The market price, being the fixed
baseline, was also scored on test.

## 1. Data

| | |
| --- | --- |
| Market data | Public dataset `kachoio/polymarket-5-minute-crypto-up-down-markets` on Hugging Face (CC0); the BTC files were already in `data/` |
| Content | 15,682 five-minute "Bitcoin Up or Down" markets, 24 March to 18 May 2026, with the best bid and ask of both shares every second (4.7 million rows) |
| Spot data | Binance BTC/USDT one-second candles for the same period, downloaded for this study (48 MB) |

**Validation.**

- 99.97% of markets start five minutes after the previous one. The longest
  gap is 80 minutes, a documented recording outage.
- No missing quotes; 162 crossed quotes in 4.7 million; a typical spread of
  one cent; Up and Down mid prices add up to 1.
- Spot prices cover every second.

**The label problem.** I described the labels as "exact" in the idea file.
That was wrong.

- The dataset's `outcome` is **inferred from the last recorded quote**, one
  second before the end: Up if the Up share's bid is at least 0.91, Down if
  at most 0.08, empty otherwise.
- **1,456 markets (9.3%) have no outcome, and they are the close calls.**
  Where Bitcoin moved less than one basis point, 43% of markets are
  unlabeled; where it moved more than ten, under 1%.
- Dropping them would select on the outcome and distort calibration.
- **Binance cannot fill the gap.** The markets settle on a Chainlink oracle
  price, not on Binance. Recorded outcomes agree with the Binance direction
  in 98.6% of markets, but in only 84% of those with a move under one basis
  point.
- **The true resolutions could not be obtained.** Polymarket's API is not
  reachable from this machine, and an attempt to read the resolution events
  from the Polygon chain found none.

**Decision:** soft labels. A labeled market has target 1 or 0; an unlabeled
one has the Up share's last mid price as its target. Conclusions were
checked against two other treatments.

**Is it enough and relevant?** Enough to measure calibration to about half a
percentage point over the whole sample. Thin inside a ten-day period (about
one point). Directly relevant: it is the market itself, second by second.

**Features and labels.**

- **Forecast to beat:** the Up share's mid price at a fixed second.
- **Features for models:** the Binance move since the open, divided by
  recent volatility and the square root of the time left; the previous
  five-minute move; the last hour's move; the spread.
- **Decision times:** 0, 60, 120 and 240 seconds. **60 seconds is primary**
  and was fixed before any score was computed.

## 2. Preprocessing and split

| Issue | Decision |
| --- | --- |
| Outcomes inferred, missing for close calls | Soft labels, with sensitivity checks. |
| A spot candle is complete only at the end of its second | A forecast at time *t* uses spot prices up to *t* - 1 (tested by changing future prices). |
| Binance is not the settlement price | Used as a feature, never as a label. |
| Volatility varies | Spot moves are scaled by the volatility of the preceding hour. |
| Neighbouring markets are dependent | Standard errors allow for autocorrelation over one hour. |

**Split by date:** train 24 March to 28 April (10,083 markets), dev 29 April
to 8 May (2,880), test 9 to 18 May (2,719).

**An exploratory finding worth recording.** The idea asked whether the
reversal seen at hourly bars exists at five minutes and whether the market
prices it.

- On Binance, a five-minute move repeats the direction of the previous one
  49.0% of the time (standard error 0.4).
- After an Up market the next opens at 0.496 on average and goes Up 49.4% of
  the time. After a Down market it opens at 0.509 and goes Up 51.4% of the
  time.
- The opening price beats a flat 50% by 0.008 bits per market (standard
  error 0.001).

**The reversal is there, and the market already prices it.**

**Leakage.** Timing is strict and tested; the split is by date; the primary
decision time was fixed in advance; candidates were not run on test. One
thing cannot be fixed with this data: the label is the market's own price one
second before the end.

## 3. Baseline

**Task.** At a fixed second into each market, give the probability that it
resolves Up.

**Metrics, and why.**

| Metric | Reason |
| --- | --- |
| **Log loss in bits** | Proper scoring rule. A flat 50% costs 1 bit, so the saving is the information in the forecast. The market is scored like any model. |
| Brier score, with its reliability and resolution parts | Reliability is miscalibration itself, separated from how informative the forecast is. |
| Reliability table | Calibration bin by bin, with standard errors. |
| Paired loss differences with robust standard errors | Whether one forecast beats another by more than noise. |

**Protocol.** Fit on train, score dev, then test, no refitting.

**The fixed baseline is the market's mid price.** Nothing is fitted.

| Market mid price | Dev: log loss | Dev: saved vs. 50% | Dev: miscalibration | Test: log loss | Test: saved vs. 50% | Test: miscalibration |
| --- | --- | --- | --- | --- | --- | --- |
| At the open | 0.993 | 0.007 | 0.0006 | 0.988 | 0.012 | 0.0005 |
| **60 seconds in** | **0.903** | **0.097** | **0.0002** | **0.878** | **0.123** | **0.0008** |
| 120 seconds in | 0.780 | 0.220 | 0.0005 | 0.776 | 0.224 | 0.0005 |
| 240 seconds in | 0.533 | 0.467 | 0.0002 | 0.485 | 0.515 | 0.0001 |

Log loss and savings are in bits per market; miscalibration is the
reliability part of the Brier score, to be compared with Brier scores of
0.09 to 0.23.

**How the baseline performs.**

- **It is well calibrated.** Miscalibration is a fraction of a percent of
  the Brier score at every decision time, on dev and on test.
- **It is informative from the first second,** and gains about a tenth of a
  bit per minute early on.
- **One pattern: mild compression towards 50%.** Over the whole sample, 60
  seconds in, shares priced 0.8 to 0.9 win 88% of the time, and shares priced
  0.1 to 0.3 win two points less often than priced.

| Sides priced at 0.80 or more, 60 seconds in | Markets | Mean price | Won | Gap | Profit per share after the spread |
| --- | --- | --- | --- | --- | --- |
| Train | 843 | 0.853 | 0.884 | +3.1 points (s.e. 1.1) | 2.5 cents |
| Dev | 240 | 0.851 | 0.885 | +3.4 points (s.e. 2.0) | 2.8 cents |
| Test | 294 | 0.852 | 0.875 | +2.3 points (s.e. 1.9) | 1.6 cents |

The sign is the same in all three periods. It is significant only in train,
and it was singled out after looking at several price bins and two decision
times, so it should be treated as a hypothesis for new data. It matches the
favourite-longshot bias reported for other betting markets.

## 4. Does the proposed method fit the task and data?

**The method fits; the answer it gives is that the market wins.**

| Dev period, 60 seconds in | Log loss (bits) | Minus market | Standard error |
| --- | --- | --- | --- |
| **Market mid price (fixed baseline)** | **0.9027** | | |
| Always 50% | 1.0000 | +0.0973 | 0.0085 |
| Market, log-odds stretch | 0.9053 | +0.0026 | 0.0018 |
| Market, isotonic recalibration | 0.9116 | +0.0089 | 0.0032 |
| Spot only (structural) | 0.9286 | +0.0260 | 0.0051 |
| Market + spot, logistic | 0.9075 | +0.0049 | 0.0020 |
| Market + spot, gradient boosting | 0.9168 | +0.0141 | 0.0040 |

- **No candidate beats the market** at the open, at 60 seconds or at 240
  seconds.
- **Recalibration does not help.** The stretch fitted on train (about 20%)
  is not confirmed on dev. The flexible isotonic correction overfits.
- **A model without the market price is five standard errors worse.** The
  market knows more than the Binance price path, volatility and time left.
- **Adding spot features to the market price adds nothing;** boosting
  overfits.
- **The verdict does not depend on the label treatment.** With only labeled
  markets, the stretch is 0.001 bits ahead of the market, well inside one
  standard error. With unlabeled markets set to the Binance direction, no
  model is ahead.

**Why the models lose.** Three reasons, of which the first is certain:

1. **The market is fast and well informed.** Its price one minute in
   already contains the spot move.
2. **The spot features are from the wrong feed.** The contract settles on an
   oracle that averages several exchanges. Binance is close to it, not equal.
3. **The compression is small.** A three-point bias in 9% of markets is
   worth little in average log loss, and ten days do not pin it down.

## 5. What to expect from more complex models

| Extension | Expected effect | Basis |
| --- | --- | --- |
| More elaborate models on the same inputs (neural networks, more features) | None, or worse. Boosting already overfits and the market already contains the inputs. | Section 4. |
| The oracle's own price feed as a feature | The one input that could plausibly add information, especially late in a market. Not available here. | Dai et al. (2026). |
| Order book features (sizes, depth, imbalance), which the dataset has | Possibly a small gain very close to the decision time. Not tried. | The dataset's size and depth columns are unused. |
| A targeted test of the favourite underpricing on new data | The most promising step. If it holds on true resolutions and fresh weeks, it is a clean finding with a known counterpart in other markets. | The table in section 3. |
| The other six coins in the same dataset | Seven times the markets, though strongly correlated across coins. Would test whether calibration differs by asset. | The dataset includes ETH, SOL, XRP, DOGE, HYPE and BNB. |
| Fifteen-minute markets | A comparison of interest: settlement behaviour is reported to differ. | Dai et al. (2026); `data/polymarket_crypto_5m_15m/`. |

**A caution on expectations.** The honest forecast is that a full project
ends where this study did: the market is calibrated to within a point or
two and no model beats it. The one opening is narrow.

## 6. Conclusions

**Dataset.** Excellent in what it records (quotes, every second, eight
weeks) and flawed in what it does not: true resolutions. That flaw matters
most exactly where calibration is hardest to judge, in the close calls.

**Baseline.** The market price is calibrated, informative, free, and was not
beaten. It also prices in the short-term reversal found in the complexity
study.

**Method.** Scoring probability forecasts with proper rules, fixed decision
times and robust paired comparisons works as intended and gives a clear
answer. The models themselves add nothing over the market.

**Is there potential?**

- **For beating the market with a model: no,** on this evidence.
- **For a rigorous calibration study: yes.** It is well-posed, the data is
  rich, the markets are new, and the only academic paper found on them
  studies something else. The expected headline is "calibrated, with a mild
  favourite-longshot bias early in the market". The course topics it covers
  (probabilistic classification, proper scoring, calibration, logistic
  models) are not covered by the other studies.
- **The decisive missing piece is the labels.** Without true resolutions,
  every conclusion carries a footnote.

**Recommended next steps.**

1. Obtain the true resolutions (from a network where Polymarket's API is
   reachable, or from the chain), and redo the calibration tables.
2. Pre-register the favourite test: sides priced 0.8 or more at 60 seconds,
   on weeks not yet looked at.
3. Extend to the other six coins and to the fifteen-minute markets.
4. Read Dai et al. (2026) in full before interpreting anything near the end
   of a market.

## Limitations of this study

- **Labels are the market's own last quote.** A market labeled Up had an Up
  bid of at least 0.91 one second before the end; a small share of those
  will have resolved Down. The soft labels for unlabeled markets assume the
  last mid price is itself calibrated, which cannot be checked here. Both
  effects favour the market in a comparison with any model.
- **Settlement-time manipulation** is reported for these markets. The last
  second, on which the labels rest, is the least clean moment of a market.
- **The favourite pattern was found by looking.** Several bins and two
  decision times were examined, and the test period was among them.
- **One asset, eight weeks, one venue.**
- **Fees are not included** in the profit figures, and quoted prices are
  assumed to be tradable for one share.
- **Models were fitted on labeled markets only,** which overstates the
  recalibration stretch.
- **Order book sizes and depth were not used** as features.
- **The on-chain lookup of resolutions failed** for a reason I did not
  establish; it may be possible with the right contract address.
- **The literature review is at abstract level,** and two downloaded papers
  were not read.
