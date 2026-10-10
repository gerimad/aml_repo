# Project ideas, round 3: projects with an impact on people, exchanges or markets

Third brainstorm. The first two rounds asked "what is a good machine learning
problem on crypto data". This one starts from the other end: **who would be
better off if the model worked**, and then asks whether the data and labels
exist.

Course topics are referred to by number:

| # | Topic |
| --- | --- |
| T1 | Classification, regression, generation |
| T2 | Problem domains: character recognition, identification, pattern classification, ranking, info extraction / retrieval, NLP |
| T3 | Statistics, linear algebra, optimization, information theory, PCA / LDA, feature engineering |
| T4 | Learner survey: linear, MaxEnt, HMM, kNN, max margin, evolutionary, boosting, trees, Bayesian, NN |
| T5 | NLP tasks: tagging, parsing, disambiguation, semantic similarity, dictionary building |
| T6 | Learning finite automata, transducers |
| T7 | Algorithmic information theory, MDL |
| T8 | Unsupervised learning, structure learning |

## What "impact" can honestly mean here

A course project will not change a market. It can do three useful things:

- **Show that a warning is possible:** that some loss people suffer today
  could have been flagged in advance, with a measured hit rate and false
  alarm rate.
- **Show that a common practice is wrong or unneeded:** as the earlier
  studies did for compression-based efficiency measures and for graph
  networks on Elliptic.
- **Leave a benchmark:** a labeled dataset and a baseline that others can
  build on.

Each idea below says which of these it aims at, and who benefits.

## Lessons carried over from the six studies

| Lesson | Consequence for these ideas |
| --- | --- |
| Labels must be exact and free | Prefer outcomes that the chain or the exchange records by itself: a liquidation, a delisting, a transfer to the wrong address. |
| "Exact" labels need checking | The prediction-market outcomes turned out to be inferred. Every label source below is rated. |
| Count independent events | Crashes, depegs and exchange failures are few. Ideas that depend on them are marked. |
| The simple baseline is strong | Each idea names the baseline it must beat. |
| One evaluation period misleads | Rolling evaluation is assumed wherever the world moves fast. |
| Keyless data only | "Access checked" means the source answered a request on 2026-10-10. It does not mean the data was downloaded or examined. |

## Overview

| # | Idea | Who benefits | Topics | Labels | Access checked | Risk |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Address poisoning: which attacks succeed? | Wallet users | T2, T6, T1 | Exact outcome, rule-based attack | Method checked | Medium |
| 2 | Delisting risk: survival analysis of exchange listings | Token holders, exchanges | T1, T3, T4 | Exact | Yes, data in repo | Low |
| 3 | Do positioning data improve tail-risk forecasts for margin? | Exchanges, leveraged traders | T1, T3, T4 | Exact | Data in repo | Low to medium |
| 4 | Liquidation early warning for DeFi borrowers | Borrowers, lending protocols | T1, T4 | Exact | Yes | Medium |
| 5 | Sandwich risk before you trade | DEX users | T1, T3 | Rule-based | Data in repo | Medium |
| 6 | Malicious approval warning | Wallet users | T1, T2, T8 | Blocklist | Yes | Medium to high |
| 7 | Sybil cluster detection for airdrops | Projects, honest users | T8, T2 | Published lists | Partly | Medium |
| 8 | Settlement-window manipulation detector | Market integrity | T1, T3, T8 | None (natural experiment) | Data in repo | Medium |
| 9 | What did this transaction do? Classification for accounting | Anyone filing taxes | T2, T5, T1 | Exact for known contracts | Yes | Low to medium |

Novelty ratings below are my estimate from memory, not from a literature
search.

## 1. Address poisoning: which attacks succeed?

A scammer sends a worthless or zero-value transfer to a victim from an
address that looks like one the victim uses (same first and last
characters). The fake address now sits in the victim's history. If the victim
later copies it, real funds go to the scammer. Detect the attacks, then
predict which of them will succeed.

- **Who benefits:** wallet users, and wallets that could show a warning.
  Individual losses to this attack have been reported in the tens of
  millions of dollars (from memory; not checked for this document).
- **Aim:** show that a warning is possible, and say which features of an
  attack make it dangerous.
- **Data:** stablecoin transfer events from public RPC endpoints (the same
  method that collected 2.7 million swaps for the earlier studies).
- **Labels:** the attack is identified by rule (a tiny transfer from an
  address matching a previous counterparty in its first and last
  characters). The outcome is exact: did the victim later send real funds to
  the look-alike?
- **Topics:** T2 (character-level similarity, pattern classification), T6
  (weighted edit distance as a transducer), T1 (classification with heavy
  imbalance).
- **Why high quality:** the outcome label is recorded by the chain itself,
  successes are rare but attacks number in the millions, and the question
  "what makes a look-alike convincing" has a concrete answer in characters
  matched and position in the history.
- **What you learn:** imbalanced classification, string similarity as a
  learned measure, building a dataset from raw logs.
- **Novelty:** medium. The attack is documented and has been measured; a
  predictive model of success is less common.
- **Baseline to beat:** number of matching characters alone.
- **Main risk:** successes are very rare, so the positive class may be a
  few hundred cases even in months of data. Scale of collection decides
  feasibility; check it first. The same model tells an attacker what works,
  so the write-up should stay on the defensive side.

## 2. Delisting risk: survival analysis of exchange listings

Of the 756 USDT pairs Binance has ever published, 249 are no longer traded.
Holders of a delisted token are often left with something hard to sell.
Model the time until a pair is delisted from its market quality, and produce
a risk score months ahead.

- **Who benefits:** token holders, who get a warning; exchanges, which run
  exactly this kind of monitoring.
- **Aim:** show that a warning is possible, and leave a benchmark.
- **Data:** Binance public candles for all pairs, already partly in
  `data/binance_spot_1h/`; the listing with first and last month per pair is
  cached in the repo.
- **Labels:** exact. A pair's files stop.
- **Topics:** T1 (classification and regression in the form of survival
  analysis, which no study so far has used), T3 (feature engineering,
  statistics of censored data), T4 (Cox regression, boosting, trees).
- **Why high quality:** 249 events is a workable number, the complexity
  study already built the features that should matter (traded value, share
  of zero returns, volatility), and censoring (pairs still alive) makes the
  statistics non-trivial in a way a plain classifier ignores.
- **What you learn:** survival analysis, time-varying covariates,
  evaluation by concordance and by calibration of predicted risk.
- **Novelty:** medium. Delisting effects on price are studied; predicting
  delisting from market quality less so.
- **Baseline to beat:** traded value alone.
- **Main risk:** some pairs vanish for reasons other than failure
  (renamed tickers, migrations, regulatory removals), as the complexity
  study found. They must be separated by hand or by rule. Exchanges also
  announce delistings in advance, so the last weeks before removal are not a
  fair prediction window.

## 3. Do positioning data improve tail-risk forecasts for margin?

Exchanges set margin requirements from forecasts of extreme price moves. The
regimes study found that a simple exponentially weighted volatility is hard
to beat using prices alone. Test whether futures positioning (open
interest, funding, the long / short ratios of all accounts and of top
traders) adds anything for the tails.

- **Who benefits:** exchanges' risk engines; leveraged traders, who are
  liquidated when margins are set too low for the next move.
- **Aim:** show that a common input is useful, or that it is not.
- **Data:** `data/binance_futures_main4/` already holds one-minute candles
  and five-minute positioning metrics for BTC, ETH, BNB and SOL from December
  2021 to January 2026. Binance publishes the same metrics, funding rates
  and order book depth for download. Liquidation records are not published.
- **Labels:** exact: the realized move over the next hour or day.
- **Topics:** T1 (quantile regression, classification of extreme moves), T3
  (statistics of tails, scoring of quantile forecasts), T4 (linear, boosting,
  Bayesian).
- **Why high quality:** four years, a strong baseline already built and
  tested, a sharp yes-or-no question, and a standard evaluation (Value-at-Risk
  backtests, the quantile loss).
- **What you learn:** quantile forecasting, backtesting of risk measures,
  and how to test whether a feature adds information beyond a baseline.
- **Novelty:** medium. Funding and open interest as predictors are studied
  for returns; less for margin-relevant tails with honest baselines.
- **Baseline to beat:** EWMA volatility with a Student-t quantile, and
  GARCH with Student-t errors, both from the regimes study.
- **Main risk:** extreme moves cluster in a handful of crash days, so the
  effective number of tail events is small, the Elliptic problem again.
  A likely result is "no reliable improvement".

## 4. Liquidation early warning for DeFi borrowers

On lending protocols a borrower whose collateral falls too far is liquidated
at a penalty. Predict, some hours ahead, which open positions will be
liquidated.

- **Who benefits:** borrowers, who could add collateral in time; protocols,
  which carry the risk of bad debt.
- **Aim:** show that a warning is possible beyond the obvious one.
- **Data:** the protocol's own events (deposits, borrows, repayments,
  liquidations) from public RPC endpoints. A request to the Aave V3 pool
  contract answered, with about one event per block.
- **Labels:** exact. The protocol emits an event for every liquidation.
- **Topics:** T1 (classification, time-to-event), T4 (boosting, logistic
  models).
- **Why high quality:** exact labels and a clear, hard baseline: the health
  factor itself, which already says how close a position is to liquidation.
  The question is whether borrower behaviour (how they reacted to earlier
  drops, what else they hold) adds to it.
- **What you learn:** reconstructing state from an event log, and beating a
  baseline that is nearly the definition of the target.
- **Novelty:** medium. Liquidations are well studied in aggregate; position-
  level early warning less.
- **Baseline to beat:** current health factor and recent volatility of the
  collateral.
- **Main risk:** reconstructing each position's health needs prices and
  interest indices at every moment, which is real engineering. And
  liquidations happen mostly on a few crash days.

## 5. Sandwich risk before you trade

A sandwich attack buys before a user's swap and sells after it, taking the
difference from the user. The bot study saw its footprint (two opposite
swaps in the same pool and block, 60 times more common among private
contracts). Predict, from what is known before a swap is sent, whether it
will be sandwiched, and how much it will lose.

- **Who benefits:** users of decentralized exchanges, and wallets that
  choose slippage settings and private routing for them.
- **Aim:** show that a warning is possible.
- **Data:** the two weeks of swaps already in `data/dex_swaps/`, with pool
  tokens.
- **Labels:** by rule (a swap enclosed by two opposite swaps of one
  contract in the same block and pool). The rule is standard and precise,
  but it is a rule.
- **Topics:** T1 (classification, regression of the loss), T3 (feature
  engineering).
- **Why high quality:** the label depends on what happens *after* the
  features are fixed, so it is not circular in the way bot-type labels were.
  The features are things a wallet knows: trade size relative to the pool,
  the pool's recent sandwich rate, the router used.
- **What you learn:** building labels from event patterns, cost-sensitive
  evaluation, and reasoning about what is knowable at decision time.
- **Novelty:** low to medium. Sandwich attacks are heavily measured;
  per-trade risk prediction for users is less common.
- **Baseline to beat:** trade size relative to pool liquidity.
- **Main risk:** the decisive variable is the user's slippage tolerance,
  which is in the transaction input and not in the swap event; it would have
  to be decoded. Most user trades now go through private channels, which
  changes who can be attacked. The model also describes profitable victims,
  so it is dual-use.

## 6. Malicious approval warning

Much theft from wallets happens through approvals: the user signs a
transaction that lets a contract spend their tokens. Classify the spender of
an approval as malicious or not, at the moment of signing.

- **Who benefits:** wallet users.
- **Aim:** show that a warning is possible for contracts not yet on any
  blocklist.
- **Data:** approval events from public RPC endpoints; features of the
  spender (age, code size, who funded it, how many approvals it collects and
  how fast).
- **Labels:** public blocklists. ScamSniffer's address blacklist and
  MetaMask's phishing list both answered a request.
- **Topics:** T1 (classification), T2 (pattern classification), T8
  (positive and unlabeled learning: the blocklist names some bad addresses,
  and says nothing about the rest).
- **Why high quality:** a real protection problem with a natural time split:
  can the model flag an address before the blocklist does?
- **What you learn:** learning from positive and unlabeled data, and
  evaluation under label delay, the issue that decided the Elliptic study.
- **Novelty:** medium. Commercial tools exist; open, evaluated models are
  rarer.
- **Baseline to beat:** age of the spender contract and number of approvals
  received.
- **Main risk:** label quality. A blocklist is incomplete and delayed, and
  how an address got on it is unknown. Attackers rotate addresses quickly,
  so the drift problem of Elliptic applies in full.

## 7. Sybil cluster detection for airdrops

Projects give tokens to early users, and some people create thousands of
accounts to collect many times. Detect groups of accounts run by one
operator.

- **Who benefits:** projects distributing tokens, and the honest users whose
  share is diluted.
- **Aim:** leave a benchmark; test whether behavioural similarity finds
  what the projects' own investigations found.
- **Data:** on-chain activity of the accounts; published lists of accounts
  that projects removed as sybils. Hop Protocol's list answered a request;
  the address I tried for a second project's list did not.
- **Labels:** the published lists. They are the output of someone else's
  detection, not ground truth.
- **Topics:** T8 (clustering, structure learning on the funding graph), T2
  (identification).
- **Why high quality:** it is the identification task of the bot study with
  external labels, and the bot study showed that behaviour alone identifies
  a trader far above chance.
- **What you learn:** clustering with partial labels, graph features, and
  evaluating against an imperfect reference.
- **Novelty:** low to medium. Sybil detection is an active area with
  commercial and community work.
- **Baseline to beat:** accounts funded from the same address.
- **Main risk:** the labels encode the method that produced them. A model
  that reproduces them has learned that method. And anything published
  becomes a guide to evasion.

## 8. Settlement-window manipulation detector

A 2026 paper reports that after Polymarket launched five-minute Bitcoin
markets, order flow on the spot market spikes in the last seconds of each
five-minute window, with price reversals afterwards. Build a detector for
such windows from spot data alone.

- **Who benefits:** market integrity: exchanges whose prices are being
  pushed, and the traders on the other side of those markets.
- **Aim:** reproduce a recent finding independently, and turn it into a
  monitor.
- **Data:** Binance one-second candles, already downloaded for the eight
  weeks of the prediction-market study and free for any other period,
  including the months before the markets launched on 12 February 2026.
- **Labels:** none. The comparison is before against after the launch, and
  the last seconds of a window against the rest of it.
- **Topics:** T1, T3 (statistics, tests of a difference between regimes),
  T8 (anomaly detection).
- **Why high quality:** a natural experiment with a known date, free data
  on both sides of it, and a claim specific enough to be checked.
- **What you learn:** event-study design, anomaly detection without labels,
  and replication.
- **Novelty:** low for the finding (it would be a replication), medium for
  an open detector.
- **Baseline to beat:** volume in the last ten seconds against the window
  average.
- **Main risk:** other things changed in early 2026 too, so "before and
  after" is not a clean comparison without a control such as the
  fifteen-minute boundaries. I read only the abstract and part of the
  introduction of the paper.

## 9. What did this transaction do? Classification for accounting

Anyone who must report crypto activity for tax has to sort thousands of
transactions into kinds: a trade, a transfer between own wallets, a
deposit into a lending protocol, an airdrop, a fee. Tools do this with
hand-written rules per protocol. Learn it from the transaction's events.

- **Who benefits:** ordinary users at tax time; accounting tools.
- **Aim:** show that a general classifier can replace per-protocol rules,
  and measure how well it handles contracts it has never seen.
- **Data:** transactions and their event logs from public RPC endpoints;
  function signatures from the 4byte directory and verified contracts from
  Sourcify, both of which answered a request in round 2.
- **Labels:** exact for transactions that call a known function of a known
  protocol (a swap on a known router is a trade). The test is on protocols
  held out of training.
- **Topics:** T2 (pattern classification), T5 (the event log as a sequence
  to tag or parse), T1.
- **Why high quality:** labels are free for the known cases, the held-out
  protocol split measures exactly the generalization that matters, and
  errors have a plain meaning.
- **What you learn:** classification over structured, variable-length input;
  designing a split that tests generalization and not memorization.
- **Novelty:** medium. Commercial tools do this; open benchmarks are scarce.
- **Baseline to beat:** rules on token flows alone (one token out, another
  in, is a trade).
- **Main risk:** the taxonomy. "What kind of transaction is this" has
  genuinely ambiguous cases, and what counts for tax differs by country. It
  needs a small, fixed set of classes decided up front.

## Ideas considered and set aside

| Idea | Why not |
| --- | --- |
| Stablecoin depeg early warning | A handful of events in history; nothing can be validated. |
| Exchange insolvency from on-chain flows | Essentially one large event. |
| Hack and exploit early warning | A few hundred events of very different kinds; the labels (a public list of hacks answered a request) say when, not what to look for beforehand. |
| Gas fee timing for users | Fees on Ethereum are currently very low, so there is little to save. |
| Rug pull detection, token impersonation, governance text against executed calls | Still good and still about protecting people. They are in the first two brainstorms. |

## Recommendation

- **Most feasible with real impact: idea 2 (delisting risk).** Exact labels,
  249 events, features already built in the complexity study, most of the
  data in the repo, and a method (survival analysis) no study has used. The
  result would be directly usable: a risk score per listed token.
- **Most direct human benefit: idea 1 (address poisoning).** The loss it
  addresses is large and concrete, and the outcome label is exact. It needs
  a data collection first, and the number of successful attacks must be
  counted before committing.
- **Lowest effort: idea 3 (positioning data and margin).** The data and the
  baseline exist. Expect a negative result, which would still be a finding
  about a widely used input.
- **Treat with care: ideas 5, 6 and 7.** Each protects users but also
  describes what works for an attacker or how to evade detection. They are
  defensible as research, and the write-up should be framed accordingly.
- Across all three brainstorms, the strongest options remain the follow-ups
  where results are already in hand (see part B of
  [`project_ideas_round2.md`](project_ideas_round2.md)); idea 2 here is the
  one I would add to that short list.
