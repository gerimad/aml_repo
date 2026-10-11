# Literature review: market regimes with HMM vs. MDL segmentation

Review for the idea in
[`../idea_market_regimes_hmm_mdl.md`](../idea_market_regimes_hmm_mdl.md):
segment crypto price series into regimes with a hidden Markov model (HMM) and
with a change point method based on minimum description length (MDL), and
compare the two.

## 1. Scope and method

The review covers four questions:

1. What is known about regime switching in crypto markets?
2. What are the known weaknesses of HMMs as regime detectors?
3. How does change point detection work, and where does MDL fit in?
4. Which statistical properties of crypto returns matter for these models?

Twelve open-access papers were downloaded to `papers/` (PDFs are ignored by
git, the links below are the source of record). The "Read" column says how much
of each was read for this review, so that the summaries are not taken for more
than they are.

| # | Paper | File in `papers/` | Read |
| --- | --- | --- | --- |
| 1 | Ardia, Bluteau, Rüede (2019), [Regime changes in Bitcoin GARCH volatility dynamics](https://doi.org/10.1016/j.frl.2018.08.009), Finance Research Letters | `ardia2019_bitcoin_msgarch.pdf` | Full |
| 2 | Koki, Leonardos, Piliouras (2020), [Exploring the predictability of cryptocurrencies via Bayesian hidden Markov models](https://arxiv.org/abs/2011.03741) | `koki2020_bayesian_hmm_crypto.pdf` | Abstract, introduction, conclusion |
| 3 | Suda, Spiteri (2019), [Comparing market phase features for cryptocurrency and benchmark stock index using HMM and HSMM filtering](https://www.um.edu.mt/library/oar/handle/123456789/102459) | `malta_hsmm_bitcoin.pdf` | Full |
| 4 | Shu, Yu, Mulvey (2024), [Downside risk reduction using regime-switching signals: a statistical jump model approach](https://arxiv.org/abs/2402.05272) | `shu2024_jump_model_downside_risk.pdf` | Abstract, introduction, results table, conclusion |
| 5 | Shu, Mulvey (2024), [Dynamic factor allocation leveraging regime-switching signals](https://arxiv.org/abs/2410.14841) | `shu2024_dynamic_factor_jump.pdf` | Abstract only |
| 6 | Fox, Sudderth, Jordan, Willsky (2011), [A sticky HDP-HMM with application to speaker diarization](https://arxiv.org/abs/0905.2592), Annals of Applied Statistics | `fox2011_sticky_hdp_hmm.pdf` | Abstract only |
| 7 | Killick, Fearnhead, Eckley (2012), [Optimal detection of changepoints with a linear computational cost](https://arxiv.org/abs/1101.1438), JASA | `killick2012_pelt.pdf` | Abstract, introduction, both simulation sections |
| 8 | Truong, Oudre, Vayatis (2020), [Selective review of offline change point detection methods](https://arxiv.org/abs/1801.00718), Signal Processing | `truong2020_changepoint_review.pdf` | Abstract, introduction, evaluation metrics, penalties |
| 9 | Hallac, Nystrup, Boyd (2019), [Greedy Gaussian segmentation of multivariate time series](https://arxiv.org/abs/1610.07435) | `hallac2019_greedy_gaussian_segmentation.pdf` | Abstract, introduction, financial example |
| 10 | Adams, MacKay (2007), [Bayesian online changepoint detection](https://arxiv.org/abs/0710.3742) | `adams2007_bocpd.pdf` | Abstract, introduction, examples overview |
| 11 | Grünwald (2004), [A tutorial introduction to the minimum description length principle](https://arxiv.org/abs/math/0406077) | `grunwald2004_mdl_tutorial.pdf` | Chapter 1 overview, the passage on BIC |
| 12 | Bariviera, Basgall, Hasperué, Naiouf (2017), [Some stylized facts of the Bitcoin market](https://arxiv.org/abs/1708.04532), Physica A | `bariviera2017_bitcoin_stylized_facts.pdf` | Abstract, conclusions |

Relevant papers that could not be downloaded (paywalled) are listed in
section 5. They are cited from their abstracts or from how the papers above
describe them, and should be read before the project is written up.

## 2. Summaries

### 2.1 Regime switching in crypto markets

**[1] Ardia, Bluteau, Rüede (2019): Markov-switching GARCH for Bitcoin.**
Daily Bitcoin log returns from August 2011 to March 2018 (2,355 days). The
authors fit 18 models: GARCH and GJR variance, Normal / Student-t / skewed
Student-t errors, one to three regimes, estimated by Bayesian MCMC.

- Two regimes beat one for every specification (by the deviance information
  criterion). A third regime only helps when errors are Normal, where it is
  "required to capture the large unconditional kurtosis". With fat-tailed
  errors, two regimes are enough.
- The two regimes are a low and a high volatility regime (annualized
  unconditional volatility about 26% and 202%). Both are persistent, with
  probabilities of staying of 0.94 and 0.93.
- Out of sample (1,355 days, rolling window of 1,000, refit every 10 days),
  single-regime models fail Value-at-Risk backtests, regime-switching ones
  pass at the 1% level.
- Descriptive statistics of their sample: daily standard deviation 5.36%,
  kurtosis 26, worst day -66%.

**[2] Koki, Leonardos, Piliouras (2020): Bayesian HMMs for BTC, ETH, XRP.**
Eight HMMs with 2 to 5 states, with fixed or predictor-dependent transition
probabilities, against a random walk, an AR(5) and a linear model with all
predictors. Models are compared on out-of-sample one-step-ahead forecasts,
scored with the continuous ranked probability score and squared error.

- A 4-state HMM with predictor-dependent transitions forecasts best for all
  three coins.
- For Bitcoin the states read as one bear regime (negative returns, high
  volatility), two bull regimes and one calm regime.
- The key remark for us: "the hidden states for all cryptocurrencies are not
  persistent, but they present frequent alternations", unlike what is seen
  for conventional exchange rates.
- No predictor works in all states; only the US Treasury yield and the VIX
  matter for all three coins.

**[3] Suda, Spiteri (2019): HMM vs. hidden semi-Markov model (HSMM).**
Daily BTC/USD (2016 to January 2019, 1,124 days) and S&P 500 (2000 to 2019),
Gaussian states, 2 to 4 states. An HSMM replaces the HMM's geometric
distribution of the time spent in a state with a negative binomial one.

- A 4-state HSMM fits best for both assets on AIC, BIC and HQC. Five states
  could not be estimated for numerical reasons.
- For Bitcoin, two of the four states have weak persistence in both models,
  while the bull state of the HSMM has a modal run of 47 days. The states are
  "not as interpretable as in the case of S&P 500".
- Evaluation is by an expanding-window procedure: refit, apply Viterbi up to
  today, read off the current state, trade on it. The HSMM strategy beats
  buy-and-hold, the HMM not always, partly because the HSMM trades far less.
- The authors stress that the Bitcoin history is short and its behaviour may
  still change.

Other crypto regime papers were found in the search but not downloaded: a
multivariate HMM on the log returns of several coins (Pennoni et al., which
identifies regimes differing in both volatility and expected returns), and
two 2026 papers on HMM regime detection for Bitcoin. They confirm that this
is an active and already crowded line of work.

### 2.2 Weaknesses of HMMs and the persistence problem

**[4] Shu, Yu, Mulvey (2024): statistical jump models.**
The paper argues that HMMs are fragile regime detectors and tests an
alternative, the statistical jump model (JM) of Nystrup et al.: cluster a set
of features over time, and add a fixed penalty for every change of state.

- Stated weaknesses of HMMs, citing Nystrup et al. (2020): sensitivity to
  "limited sample sizes, unbalanced data, or high state persistence", and to
  misspecification from "non-normal and time-varying distributions,
  characteristics all typical and inherent to financial return series".
- The JM does not observe raw returns. It observes **features computed from
  the return series** (risk and return measures over trailing windows).
- The jump penalty is chosen by time-series cross-validation on the quantity
  that matters downstream (the Sharpe ratio of a simple in / out strategy),
  not by a statistical criterion.
- On the S&P 500, DAX and Nikkei, 1990 to 2023, with transaction costs and a
  one-day trading delay, the JM strategy beats the HMM strategy and
  buy-and-hold on volatility, drawdown and Sharpe ratio (S&P 500 Sharpe ratio
  0.68 against 0.54 and 0.48), and it degrades less when the trading delay
  grows.

**[5] Shu, Mulvey (2024): sparse jump models for factor allocation.**
Extends the approach with feature selection (sparse JM) and reports regimes
with "enhanced stability and interpretability compared to traditional
methods". Read at abstract level only.

**[6] Fox et al. (2011): sticky HDP-HMM.**
From a different field (who speaks when, in audio), but the same disease: the
nonparametric HMM "tends to over-segment the audio data, creating redundant
states and rapidly switching among them". The cure is an extra parameter that
favours self-transitions and gives "effective control over the switching
rate". Read at abstract level only.

### 2.3 Change point detection and MDL

**[7] Killick, Fearnhead, Eckley (2012): PELT.**
Change point detection as minimizing a sum of segment costs plus a penalty per
change point. The paper notes that this covers "penalised likelihood and
minimum description length" alike.

- Dynamic programming (optimal partitioning) finds the exact minimum in
  O(n²). PELT adds a pruning step and reaches O(n) when the number of change
  points grows with n, which is the natural assumption for a financial series
  observed for longer and longer.
- The exact method is more accurate than binary segmentation, the popular
  greedy alternative.
- Volatility changes in finance are a motivating example (the supplementary
  material analyses Dow Jones returns), using a Gaussian cost with changing
  variance.
- An MDL cost is treated explicitly, with piecewise autoregressive models,
  and compared with the genetic algorithm of Davis et al. (2006).

**[8] Truong, Oudre, Vayatis (2020): review of offline methods.**
Organizes every method by three choices: cost function, search method, and
constraint on the number of changes. Accompanies the `ruptures` Python
package.

- With an unknown number of changes, a penalty has to be calibrated. Options
  are model selection criteria (BIC-type penalties), cross-validation, the
  slope heuristic, or supervised tuning on annotated signals.
- Evaluation metrics for segmentations, when a reference exists: the
  difference in the number of change points, the Hausdorff distance (worst
  misplacement), the Rand index (agreement of two segmentations) and
  precision / recall within a tolerance.

**[9] Hallac, Nystrup, Boyd (2019): greedy Gaussian segmentation.**
Segments a multivariate series into pieces that are i.i.d. Gaussian, with a
regularized covariance, by a greedy search that is linear in the length.

- The number of segments and the regularization are chosen by
  cross-validation on held-out points, not by an information criterion.
- The financial example (stocks, oil, bonds) finds segments driven by changes
  in **correlations** as well as volatilities, and argues that a segmentation
  is more interpretable than a sliding window, because it has "no dependence
  on a (prespecified, fixed) window length".

**[10] Adams, MacKay (2007): Bayesian online change point detection.**
Computes, after each new observation, the probability distribution of the
"run length" (time since the last change), by a simple recursion. One of its
three examples is changes of variance in daily Dow Jones returns. It is the
standard reference for doing segmentation **online** instead of in hindsight.

**[11] Grünwald (2004): MDL tutorial.**
The principle: the best model of the data is the one that compresses it most,
counting the bits needed to describe the model itself.

- "Crude" two-part MDL (bits for the model plus bits for the data given the
  model) is the easy version; refined MDL uses a single universal code
  (normalized maximum likelihood) and handles small samples and models of
  equal size but different flexibility.
- MDL is **not** the same as BIC, although it is often said to be. The BIC
  approximation only holds when the number of parameters is fixed and the
  sample grows. When the number of parameters grows with the data, the two
  select "quite different models".

### 2.4 Statistical properties of crypto returns

**[12] Bariviera et al. (2017): stylized facts of Bitcoin.**
Daily and intraday Bitcoin prices, 2011 to 2017, with a sliding-window Hurst
exponent.

- Bitcoin's volatility is large but "reducing over time".
- Returns showed persistence (Hurst exponent above 0.5) until 2014 and then
  moved towards 0.5, the value of a memoryless series. The dynamics of the
  market changed as it matured.

## 3. Insights gathered

1. **Regimes in crypto volatility are well established.** Every study finds
   that more than one regime is needed [1, 2, 3], and a low / high volatility
   pair is the robust core [1].
2. **The number of regimes depends on the emission distribution.** With
   Gaussian states, extra states are spent on fat tails; with fat-tailed
   states, two suffice [1]. Studies with Gaussian states report four [2, 3].
   So "how many regimes are there" is not a property of the market alone.
3. **Gaussian HMMs on daily returns do not give persistent states.** Stated
   directly for crypto [2], partly visible in [3], and the starting point of
   the jump model literature [4, 5]. The same over-switching is known outside
   finance [6]. By contrast the Markov-switching **GARCH** regimes of [1] are
   persistent (0.93 to 0.94), because GARCH absorbs the short-term movement
   of volatility.
4. **Known cures for over-switching:** model the time spent in a state
   explicitly (HSMM [3]), favour self-transitions (sticky HMM [6]), penalize
   every switch (jump model [4]), or observe smoother features than the raw
   return [4, 5].
5. **Change point methods are a penalized likelihood problem**, and MDL is one
   way to set the penalty [7, 8]. The exact optimum is computable by dynamic
   programming [7].
6. **MDL is not BIC** when the number of parameters grows with the data [11],
   which is exactly the case in segmentation, where each new segment adds
   parameters.
7. **Model selection by information criteria is distrusted in the applied
   papers.** The recent ones choose hyperparameters by out-of-sample
   performance [2, 4, 9].
8. **Evaluation is out of sample and causal:** expanding or rolling windows,
   one-step-ahead forecasts, filtered state estimates, with trading delays
   and costs when a strategy is involved [1, 2, 3, 4].
9. **Offline and online are different problems.** Segmentation in hindsight
   [7, 8, 9] is a description; using it in real time needs an online method
   [10] or a rule for the current segment.
10. **The market is not stationary.** Volatility declines and memory changes
    as the market matures [12], and the authors of [3] warn that conclusions
    from a short history may not last.
11. **Regimes can live in correlations**, not only in the volatility of one
    asset [9].

## 4. What this means for our idea

**The first half of the idea is not new.** HMM regime detection on crypto
returns has been done several times, with Bayesian estimation, 2 to 5 states,
predictors and semi-Markov extensions [1, 2, 3]. Repeating it would add
nothing. The comparison with an MDL segmentation is where the project can
contribute: none of the papers found puts an MDL segmentation next to an HMM
on crypto data. This rests on a handful of searches, not a systematic one, so
it needs a wider check before it is claimed.

**The comparison must be designed around the persistence problem.** Insight 3
predicts that a Gaussian HMM on daily returns will switch state every few
days, while an MDL segmentation produces segments of weeks to months. The two
would then answer different questions and "do they agree" would have a
trivial answer: no. The first experiments in
[`../../src/notebooks/01_market_regimes_data_and_baseline.ipynb`](../../src/notebooks/01_market_regimes_data_and_baseline.ipynb)
confirm this on BTC (several hundred HMM switches against 14 MDL break points
in 5.4 years, agreement at chance level).

**Changes to the idea that follow from the literature:**

| Original plan | Change | Source |
| --- | --- | --- |
| Gaussian HMM on returns | Observe a smoother feature of volatility (we use log realized variance from hourly data). Add a persistence mechanism: sticky transitions, a jump penalty, or an HSMM. | [3, 4, 6] |
| Gaussian emissions | Add Student-t emissions, so that the number of states is not inflated by fat tails. | [1] |
| "The number of regimes is chosen by a criterion" | Report BIC and MDL, but decide by out-of-sample predictive score. Treat the number of regimes as dependent on the emission family. | [1, 2, 4, 9] |
| "Compare MDL and BIC" | Keep, and make it a real question: in segmentation the number of parameters grows with the data, where the two differ. Consider refined MDL (normalized maximum likelihood) next to the two-part code. | [7, 11] |
| Evaluate by held-out likelihood | Keep, in a causal, walk-forward form with periodic refitting. Add a variance forecast metric and persistence statistics. Compare against GARCH with Student-t errors, the standard that regime models have to beat. | [1, 2, 3] |
| Agreement between methods | Use the Rand index and break point distances from the change point literature. | [8] |
| MDL segmentation in hindsight | Add an online version for the predictive comparison: recompute daily, or use Bayesian online change point detection as a third method. | [10] |
| One coin at a time | Possible extension: segment several coins jointly, so that regimes can show in correlations. | [9] |
| Fit once | Refit as time moves forward. Volatility has declined as the market matured, so parameters from early years do not describe later ones. | [3, 12] |

**A sharper research question.** The literature suggests the interesting
question is not "which method finds the regimes" but: **an HMM assumes regimes
recur, a segmentation assumes each one is new. Which assumption describes
crypto volatility better, measured by code length and by out-of-sample
prediction?** If regimes recur, the HMM can reuse what it learned about a
state and should predict better after a switch. If the market keeps drifting
to new levels [12], the segmentation should win. This uses MDL as the common
currency for both models, fits the course topics, and has not been answered
in the papers reviewed.

## 5. Papers to obtain

Not open access, so not downloaded. All are cited by the papers above.

| Paper | Why it matters |
| --- | --- |
| Hamilton (1989), "A new approach to the economic analysis of nonstationary time series and the business cycle", Econometrica | The origin of Markov-switching models in economics. |
| Rydén, Teräsvirta, Åsbrink (1998), "Stylized facts of daily return series and the hidden Markov model", Journal of Applied Econometrics | Shows that an HMM with zero-mean Gaussian states reproduces most stylized facts of S&P 500 returns, but that its parameter estimates vary across subperiods. The classical reference for what our HMM on returns does. |
| Bulla, Bulla (2006), "Stylized facts of financial time series and hidden semi-Markov models", Computational Statistics & Data Analysis | The HSMM answer to the weaknesses found by Rydén et al. |
| Davis, Lee, Rodriguez-Yam (2006), "Structural break estimation for nonstationary time series models", JASA | Auto-PARM: the reference MDL segmentation method (piecewise autoregressive models, genetic algorithm, consistency proof). Our MDL cost should be checked against theirs. |
| Nystrup, Lindström, Madsen (2020), "Learning hidden Markov models with persistent states by penalizing jumps", Expert Systems with Applications | The original jump model, and the simulation evidence on HMM mis-estimation. |
| Patton (2011), "Volatility forecast comparison using imperfect volatility proxies", Journal of Econometrics | Justifies QLIKE and MSE as the loss functions that rank variance forecasts correctly when the proxy is noisy. |
| Thies, Molnár (2018), "Bayesian change point analysis of Bitcoin returns", Finance Research Letters | The closest prior work on the change point side: structural breaks in Bitcoin volatility. Must be read before claiming novelty. |
