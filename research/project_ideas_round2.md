# Project ideas, round 2: outside what we have looked at

Second brainstorm, written after the four feasibility studies (market
regimes, market complexity, illicit transaction drift, bot fingerprinting).
Part A has ideas in areas none of those studies touched. Part B lists the
follow-ups that the studies themselves suggest.

Course topics are referred to by number:

| # | Topic |
| --- | --- |
| T1 | Classification, regression, generation |
| T2 | Problem domains: speech and character recognition, identification, pattern classification, ranking / recommendation, info extraction / retrieval, NLP |
| T3 | Statistics, linear algebra, optimization, information theory, PCA / LDA, feature engineering |
| T4 | Learner survey: linear, MaxEnt, HMM, kNN, max margin, evolutionary, boosting, trees, Bayesian, NN |
| T5 | NLP tasks: tagging, chunking, parsing, disambiguation, language identification, role labeling, semantic similarity, dictionary building |
| T6 | Learning finite automata, transducers, Eilenberg machines |
| T7 | Algorithmic information theory, Kolmogorov complexity, MDL |
| T8 | Unsupervised learning, structure learning |

## What the four studies taught about choosing an idea

Each study ran into at least one of these, so the new ideas are rated on
them.

| Lesson | Where it came from |
| --- | --- |
| **Labels must be exact and free.** Proxy or rule-made labels cap what can be concluded. | Bot fingerprinting had no bot labels; identification, with the address as label, was the part that worked. |
| **Data must be reachable without an account or key.** | Half the sources named in the first brainstorm were not. |
| **Count independent examples, not rows.** | Elliptic: one drift event, 169 illicit transactions. Bots: 17 in the dev set. |
| **Find the simple baseline first.** It is usually strong. | EWMA, an order-1 Markov model, a random forest and symbol frequencies were all hard to beat. |
| **The named method must earn its place.** | Compressors and automata did not. |

The first four studies also left whole course topics unused: almost all of
T5 (tagging, parsing, language identification, information extraction),
transducers in T6, ranking and recommendation, character recognition,
generation, and PCA. Part A leans towards those.

Novelty ratings are my estimate from memory, not from a literature search.
"Access checked" means I requested the source on 2026-10-10 and it answered
without a key; it does not mean the full data was downloaded.

## Part A: new ideas

| # | Idea | Topics | Labels | Access checked | Risk |
| --- | --- | --- | --- | --- | --- |
| 1 | Calldata as a language: recovering argument types by tagging | T5, T6, T4 | Exact, free | Yes | Low |
| 2 | From proposal text to transaction: extracting governance actions | T5, T6, T2 | Exact, free | Partly | Medium |
| 3 | Token impersonation: confusable-aware name matching | T2, T6, T5 | Weak | Yes | Medium |
| 4 | ENS names as a small language: segmentation and language identification | T5, T6 | Synthetic exact, real small | No | Medium |
| 5 | Next-pool recommendation for traders | T2, T3, T4 | Exact, free | Data in repo | Low |
| 6 | A generative model of returns, judged by the stylized facts we measured | T1, T4, T3 | None needed | Data in repo | Medium |
| 7 | Compiler fingerprinting from contract bytecode | T2, T3, T4 | Exact, free | Yes | Low to medium |
| 8 | Factors of the option volatility surface | T3, T1 | None needed | Data in repo, one week | High |
| 9 | Are short-horizon prediction markets calibrated? | T1, T4, T3 | Exact, free | Data in repo | Medium |

### 1. Calldata as a language: recovering argument types by tagging

Every Ethereum transaction that calls a contract carries its arguments as a
string of 32-byte words, encoded by fixed rules. Given only that string,
label each word: address, integer, boolean, offset, length, raw bytes. This
is part-of-speech tagging on a language whose grammar is known exactly.

- **Data:** transaction inputs from public RPC endpoints (the bot study
  already fetched 940,000 of them and discarded this field), and function
  signatures from the 4byte directory and Sourcify, which give the true type
  of every word.
- **Topics:** T5 (sequence tagging, parsing), T6 (the encoding is a
  transducer; a learned tagger can be compared with the hand-written one),
  T4 (HMM, MaxEnt / CRF, NN on one task).
- **Why high quality:** labels are exact and unlimited, the metric is plain
  (per-word accuracy, whole-call exact match), and there is a clean ladder
  from a rule baseline through HMM and CRF to a neural tagger. Splitting by
  function signature tests generalization to functions never seen.
- **What you learn:** the full sequence-labeling stack on real data, how a
  known formal grammar constrains a statistical model, and where learned
  models beat rules (ambiguous words: a small integer and a boolean look
  alike).
- **Novelty:** medium. Tools that guess signatures exist, mostly rule-based
  and working from contract bytecode. A tagging study on calldata with a
  proper benchmark is less common.
- **Main risk:** it may be too easy. Most words are unambiguous, so the
  interest lies in the ambiguous minority and in nested dynamic types; the
  evaluation has to focus there. Also, the 4byte directory contains spam:
  the selector of the standard token transfer returned six candidate
  signatures, so labels should come from verified contracts where possible.

### 2. From proposal text to transaction: extracting governance actions

A DAO governance proposal is a text ("send 500,000 USDC to the grants
multisig") together with the exact contract calls that execute if it
passes. Extract the action (who receives, how much, which token) from the
text, and check it against the calls.

- **Data:** on-chain Governor contracts emit each proposal's description and
  its calls in one event, readable from public RPC. Snapshot's API is open
  for off-chain proposals, and governance forum dumps exist on Hugging Face.
- **Topics:** T5 (information extraction, semantic role labeling), T6 (a
  transducer that normalizes amounts: "500k", "0.5M", "500,000.00" to one
  number), T2.
- **Why high quality:** it is information extraction with ground truth that
  nobody had to annotate, and the result matters: a proposal whose text and
  calls disagree is a security problem.
- **What you learn:** extraction and role labeling, text normalization with
  finite-state methods, and evaluation when the truth is structured.
- **Novelty:** medium to high. Governance is studied for voting behaviour;
  text-to-action consistency much less.
- **Main risk:** sample size. A single DAO has a few hundred executable
  proposals, so many DAOs are needed, and proposals are heterogeneous. Count
  usable proposals before committing. The Snapshot API was checked; the
  on-chain event was not.

### 3. Token impersonation: confusable-aware name matching

Scam tokens copy the names and symbols of real ones, often with look-alike
characters (a Cyrillic "Т" in "USDТ"). Build a matcher that, for any token
name, finds the genuine token it imitates.

- **Data:** curated token lists as the set of genuine tokens (the Uniswap
  default list is open), names and symbols of all tokens read from the
  chain, and the Unicode consortium's table of confusable characters.
- **Topics:** T2 (character recognition, retrieval, ranking), T6 (weighted
  edit distance is a weighted transducer; its weights can be learned), T5
  (dictionary building).
- **Why high quality:** three families of similarity can be compared on one
  task: plain edit distance, confusable-weighted edit distance, and visual
  similarity of the rendered glyphs, which is character recognition in
  reverse.
- **What you learn:** weighted finite-state methods, learning edit costs,
  ranking metrics, and working with Unicode.
- **Novelty:** medium. Look-alike internet domains are well studied;
  look-alike tokens less.
- **Main risk:** labels. "On the curated list" is a sound definition of
  genuine, but "imitates token X" has to be established for the test set,
  partly by hand.

### 4. ENS names as a small language

Ethereum Name Service names are short strings without spaces
("vitalikbuterin", "notavalidname", "kryptowährung"). Segment them into
words and identify their language.

- **Data:** name registrations from the registrar contract's events through
  public RPC; word lists for training.
- **Topics:** T5 (language identification, chunking, dictionary building),
  T6 (segmentation with a language model is a finite-state search).
- **Why high quality:** unlimited synthetic training data (join known words,
  then split them again) with exact labels, plus a real test set that is
  small enough to label by hand. Classical methods (Viterbi over a unigram
  dictionary, character n-gram language identification) are strong and
  simple, so the comparison with neural models is fair.
- **What you learn:** language modelling at the character and word level,
  dynamic programming for segmentation, and the gap between synthetic and
  real data.
- **Novelty:** medium. Word segmentation is solved for hashtags and
  domains; ENS adds emoji, digits, mixed scripts and crypto slang.
- **Main risk:** it drifts away from crypto into general NLP. The event
  source was not checked.

### 5. Next-pool recommendation for traders

Which pool will a trader use next? Treat traders and pools as users and
items and build a recommender.

- **Data:** the week of 1.29 million swaps already in `data/dex_swaps/`:
  118,000 accounts, 18,500 pools. More weeks can be added with the existing
  script.
- **Topics:** T2 (ranking, recommendation), T3 (matrix factorization, SVD),
  T4 (nearest neighbour).
- **Why high quality:** labels are exact (the pools actually used later),
  the split is natural (first days, last days), and standard baselines
  exist: most popular pools, item-based nearest neighbours, matrix
  factorization. The bot study already showed that pool usage is individual
  enough to identify a trader in 35% of cases.
- **What you learn:** implicit-feedback recommendation, ranking metrics
  (recall at k, NDCG), cold start for new pools and new traders.
- **Novelty:** medium. Recommendation is rarely applied to on-chain trading.
- **Main risk:** popularity may dominate. A few pools take most of the
  volume, and "recommend the biggest pool" could be hard to beat.

### 6. A generative model of returns, judged by the stylized facts we measured

Train models that generate synthetic hourly return series and score them on
whether the output has the properties the real data has.

- **Data:** the hourly candles of 166 coins in `data/binance_spot_1h/`.
- **Topics:** T1 (generation), T4 (HMM, GARCH, NN), T3 (statistics).
- **Why high quality:** evaluating generative models is usually vague. Here
  there is a concrete test battery from our own studies: excess kurtosis of
  8 to 22, volatility memory lasting months, weekend volatility at 69 to 78%
  of weekdays, a sign-repeat probability of 0.475, no autocorrelation in
  returns. A model passes or fails each.
- **What you learn:** generative modelling, and how to evaluate it with
  statistics instead of by eye; why a model that matches the marginal
  distribution can still miss the dynamics.
- **Novelty:** low to medium. Generative models for financial series are a
  crowded area; a strict, pre-declared test battery is the contribution.
- **Main risk:** a bootstrap of GARCH residuals may pass most tests, leaving
  neural models little to add. That would repeat the pattern of the first
  four studies.

### 7. Compiler fingerprinting from contract bytecode

Given only a contract's deployed bytecode, say which compiler version and
settings produced it.

- **Data:** verified contracts from Sourcify, which publishes source,
  bytecode and compiler settings (the service answered without a key).
- **Topics:** T2 (identification, pattern classification), T3 (feature
  engineering: opcode n-grams), T4 (linear models, trees, kNN).
- **Why high quality:** exact labels at scale, a real use (unverified
  contracts with old compilers carry known bugs), and a leakage trap that
  makes the design interesting: the compiler writes its version into a
  metadata trailer, which must be stripped.
- **What you learn:** stylometry on machine code, leakage control, and
  high-dimensional sparse classification.
- **Novelty:** medium. Compiler provenance is studied for native binaries,
  less for the EVM.
- **Main risk:** near-duplicates. Forked contracts are everywhere, so the
  split must be by contract family, the same problem as the bursts of
  near-copies in Elliptic.

### 8. Factors of the option volatility surface

Reduce the implied volatility surface of BTC and ETH options to a few
factors (level, slope, curvature) and test whether they forecast realized
volatility.

- **Data:** `data/crypto_options_surface/` already holds 1.26 million option
  quotes with implied volatilities and Greeks, but covers one week (29
  August to 4 September 2026).
- **Topics:** T3 (PCA, the one syllabus method no study used), T1
  (regression).
- **Why high quality:** PCA has a clear interpretation here, and the link to
  the regimes study gives a test: do the surface factors forecast realized
  variance better than the EWMA baseline?
- **What you learn:** PCA on real, irregular data (strikes and expiries
  change), interpolation, and factor interpretation.
- **Novelty:** low. It is classic for equity options.
- **Main risk:** one week is far too short. This needs months of surface
  history, and I did not check whether that can be had freely.

### 9. Are short-horizon prediction markets calibrated?

Polymarket runs markets on whether BTC, ETH and others will be up in the
next 5 or 15 minutes. Test whether their prices are calibrated
probabilities, and whether a model built on spot data beats them.

- **Data:** `data/polymarket_5m_kachoio/` holds 15,682 five-minute BTC
  markets over 56 days (March to May 2026) with per-second quotes and
  outcomes. `data/polymarket_crypto_5m_15m/` adds four assets and two
  horizons, but its 9 million trades cover only about four and a half days
  in March 2026, not January to March as first written here. The repo has
  one-minute BTC spot candles for both periods. See
  [`idea_prediction_market_calibration.md`](idea_prediction_market_calibration.md).
- **Topics:** T1 (classification), T4 (logistic / MaxEnt, Bayesian), T3
  (proper scoring rules, calibration).
- **Why high quality:** exact labels (the resolutions), thousands of
  independent events, and a baseline that is as strong as a baseline can be:
  the market's own price. The complexity study found that the sign of an
  hourly return is predictable 52.4% of the time; whether the market prices
  that in is a specific, testable question.
- **What you learn:** calibration, proper scoring, and how to test a claim
  of market inefficiency honestly.
- **Novelty:** medium to high. These markets are recent.
- **Main risk:** the market may simply be right, giving a clean negative
  result. I listed this idea before; the difference now is that the data is
  confirmed to be in the repo.

### Wildcards

Further from the centre, not assessed:

- **Speech recognition of crypto talk.** Recognizers mangle tickers and
  jargon. Measure it on podcasts with human captions and repair it with a
  domain dictionary. Covers the speech topic, but needs audio.
- **Priority fee forecasting.** Fee history is free from any RPC endpoint.
  A clean regression problem, but a crowded one.
- **MDL summaries of transaction graphs.** Compress each Elliptic time step
  as a graph and ask whether illicit neighbourhoods compress differently.
  Speculative.

## Part B: follow-ups from the four studies

These reuse data and code already in the repo.

| # | Idea | Comes from | Topics |
| --- | --- | --- | --- |
| B1 | What can each complexity estimator detect? With context tree weighting as the constructive part | Market complexity | T7, T3 |
| B2 | A simulated drift benchmark: withhold clusters of illicit transactions, then let them appear; develop threshold adaptation on it | Illicit drift | T1, T3, T4 |
| B3 | Do volatility regimes recur, or is each one new? HMM against segmentation on realized variance | Market regimes | T4, T7, T8 |
| B4 | Trader identification with richer symbols and pool information | Bot fingerprinting | T2, T3 |
| B5 | Is the hourly reversal stronger in some volatility regimes? | Regimes and complexity | T4, T3 |
| B6 | One universal sequence model (context tree weighting) on return signs and on trader sequences | Complexity and bots | T7, T6 |

## Recommendation

- **Best new idea: 1 (calldata tagging).** It meets every lesson: exact free
  labels, keyless data partly already fetched, no shortage of examples, an
  obvious rule baseline, and a method (sequence tagging) that fits the
  problem by construction. It also covers T5 and T6, which nothing so far
  has.
- **Most original: 2 (proposal text to transaction).** The only idea where
  natural-language extraction has machine-checkable truth. Check the number
  of usable proposals first.
- **Lowest effort: 5 and 9.** The data is already in the repo and the labels
  are exact.
- **Best overall, new or not: still B1,** because its central result is
  already in hand. Idea 1 is the one I would pick if you want a fresh topic.
- **Avoid as a main project: 8,** until months of option data are available.
