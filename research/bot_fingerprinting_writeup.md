# Write-up: trading bot fingerprinting with probabilistic automata

Feasibility study for the idea in
[`idea_bot_fingerprinting_automata.md`](idea_bot_fingerprinting_automata.md).
It covers the literature, the data, the split, a baseline and a first test of
the proposed method.

- Literature review: [`literature/literature_review_bot_fingerprinting.md`](literature/literature_review_bot_fingerprinting.md)
- Notebook with all numbers and charts: [`../src/notebooks/04_bot_fingerprinting_data_and_baseline.ipynb`](../src/notebooks/04_bot_fingerprinting_data_and_baseline.ipynb)
- Code: [`../src/scripts/dex_swaps.py`](../src/scripts/dex_swaps.py), [`../src/scripts/trader_automata.py`](../src/scripts/trader_automata.py), tests in [`../tests/`](../tests/)

## Verdict

**Fingerprinting works; automata do not add to it.** From 100 swaps described
only by their timing and their relation to the previous swap, a trader can be
picked out of several hundred about a hundred times more often than by
chance. But a model that ignores the order of the swaps does this as well as
any sequence model tried, and the probabilistic automata that ALERGIA learns
turn out to be first-order Markov chains: one state per symbol, no deeper
memory.

**The idea as written was not feasible on the data side.** None of the label
sources it named could be used, so "which traders are bots, and of what
kind" could not be evaluated as planned. The study replaces it with two
tasks that have usable labels. The second of them, bot against non-bot, has
too few evaluation cases to say anything.

**Potential is limited, and lies in a different place than the idea put it:**
in the design of the symbols and in identification as a task, not in
automata learning.

Candidate models were scored on dev only; test was used for the fixed
baselines.

## 1. Data

**The planned sources were unavailable.** BigQuery and The Graph need
accounts or keys. The ZeroMEV label API no longer resolves. Flashbots and
EigenPhi labels have no open download.

**What was collected instead:**

| | |
| --- | --- |
| Source | Free public Ethereum RPC endpoints, no key |
| Content | Every swap event of Uniswap V2 and V3 style pools, with the signer and target of its transaction |
| Window | Seven days: blocks 26,100,000 to 26,150,399, 1 to 8 October 2026 |
| Size | 1,294,242 swaps in 940,245 transactions, on 18,514 pools, by 118,537 signing accounts; 180 MB on disk |
| Download time | About 1.5 hours, limited by the free endpoints |

**Validation.** 98.8% of blocks contain a swap, daily counts are even, there
are no duplicates or missing values, and the token amounts of a swap have
opposite signs in all but 335 events, as they must.

**Not in the data:** other exchange designs (Uniswap V4, Curve, Balancer),
token identities and prices, and any bot labels.

**Who is a trader.** A swap event names the address that called the pool.
Callers were sorted by who uses them:

| Kind of pool caller | Definition | Callers | Share of swaps |
| --- | --- | --- | --- |
| Public router | At least 50 signing accounts, at most 20 transactions each | 76 | 57% |
| Private contract | One or two signing accounts | 978 | 7% |
| Unclear | Everything else | 392 | 36% |

- A **bot** is a private contract. A **router user** is an account trading
  through public routers. Swaps through unclear callers are dropped.
- Dropping 36% of the swaps is a real loss. Much of it is contracts driven
  by fleets of signing accounts, which are probably professional systems.

**Features and labels.**

- **Features:** one symbol per swap, describing it relative to the same
  trader's previous swap: time since (same transaction, same block, within
  about a minute, within about an hour, later) and pool relation (same pool
  and direction, same pool reversed, a pool used before, a new pool). 20
  symbols. Addresses, pools and tokens are deliberately not part of them.
- **Labels, task A:** the trader's identity. Exact and free.
- **Labels, task B:** private contract against router user. A proxy,
  independent of the swap sequence but noisy.
- **Bot types** (arbitrage, sandwich) are not available. Rule-based type
  labels would also be circular, since the rules read the same patterns a
  model would.

**Are the sizes good?** Events, yes. Traders, no:

- The median router user swaps twice a week, the median private contract
  four times.
- Only 184 bots and 1,426 router users have at least 50 swaps.
- This is the "sparse sequences" risk the idea named, and it sets the scale:
  hundreds of traders.

## 2. Preprocessing and split

| Issue | Decision |
| --- | --- |
| Extremely skewed activity | Per-trader models only for traders with at least 50 symbols in train; queries truncated to 100 symbols and scored per symbol. |
| Identity would leak through pools and tokens | Kept out of the symbols; used only in a separate reference model. |
| Heavy router users are probably automated | Kept, noted as label noise, results reported per group. |
| An operator's account and its contract | Always in the same part of the trader split. |
| No prices | Swap size is its rank within its pool. |

**Split by time:** train is the first four days, dev the fifth, test the
last two.

- **Task A (identification):** a trader is enrolled with its train sequence
  and queried with its dev or test sequence. 306 traders qualify on dev (63
  bots), 494 on test (107 bots).
- **Task B (bot or not):** additionally split by trader, 60 / 20 / 20 by a
  hash of the operator's address, so that no trader or operator is on both
  sides.

**Leakage.** Symbols use only a trader's past (tested). Train and query
periods are disjoint. Two weaknesses remain: the public / private
classification of pool callers is computed from the whole week, test days
included; and for task B the label and the features are not fully
independent, since how a trader reaches a pool shapes its transactions.

## 3. Baseline

**Task A, identification (primary).** Given 100 later symbols, say which
enrolled trader produced them.

**Task B, bot or not (secondary).** Given 100 symbols from a trader never
seen, say whether it is a private contract.

**Metrics, and why.**

| Metric | Reason |
| --- | --- |
| **Top-1 accuracy**, top-5, mean reciprocal rank (task A) | Identification is ranking among all enrolled traders. Chance is one over their number. |
| **ROC AUC** and average precision (task B) | Threshold-free; average precision respects the class imbalance. |
| Bootstrap intervals over traders | The samples are small. |
| Number of states | Compactness is the argument for automata. |

**Protocol.** Fit on train-period symbols, score dev, then test, without
refitting.

**The fixed baseline** uses only how often each symbol occurs, not in what
order. A sequence model is justified only if it beats this.

| Task A | Dev (306 traders) | Test (494 traders) |
| --- | --- | --- |
| Chance, top-1 | 0.3% | 0.2% |
| **Symbol frequencies (fixed baseline), top-1** | **31%** (26 to 36) | **23%** (19 to 27) |
| Symbol frequencies, top-5 | 58% | 49% |
| Pool profile (reference, not behavioural), top-1 | 35% | 31% |

**How the baseline performs.**

- **About a hundred times better than chance**, with no address, pool or
  token in the input.
- **Where a trader trades identifies it somewhat better than how.** The two
  kinds of information would be combined in practice.
- **Lower on test than on dev.** Test has more candidates and is further
  from training; the two effects cannot be separated here.

## 4. Does the proposed method fit the task and data?

**Automata can be learned, but they do not do what the idea expected.**

| Task A on dev | Top-1 | Difference from baseline (95% interval) | Size |
| --- | --- | --- | --- |
| Symbol frequencies (baseline) | 31% | | 19 parameters |
| Bigram | 32% | +0.02 (-0.03 to +0.06) | 399 parameters |
| Trigram | 27% | -0.04 (-0.09 to +0.02) | 8,379 parameters |
| Automaton (ALERGIA) | 32% | +0.01 (-0.04 to +0.06) | median 13 states, 63 transitions |

- **Order of swaps adds almost nothing.** No sequence model is
  distinguishable from the baseline.
- **The automata are first-order Markov chains.** For 88% of traders the
  learned automaton has exactly one state per symbol the trader uses. The
  state-merging test found no deeper distinction worth keeping, even for the
  20 traders with more than 2,000 train symbols. An automaton here is a
  bigram model without the unused transitions.
- **They are compact:** 63 transitions against 399 parameters at equal
  accuracy. That is the one advantage found.
- **They are not interpretable in the sense hoped for.** A state means "the
  previous swap was of kind X". There are no hidden strategy states to read.
- **The alphabet matters more than the model.** Time gaps alone give 14%,
  pool relations alone 9%, both together 31%, and adding swap size 38% for
  the frequency model.
- **Longer queries help** (16% from 10 symbols, 31% from 100) and the curve
  has not flattened.
- **Bots are easier to recognize** than router users (43% against 28%). For
  the 63 bots alone the bigram model reaches 52%: a hint that order matters
  for some traders, on too small a sample to confirm.

**Task B gives no usable answer.**

| Bot or not | Dev (17 bots of 64) | Test (23 bots of 171) |
| --- | --- | --- |
| Symbol frequencies, logistic regression (fixed baseline), AUC | 0.75 (0.61 to 0.86) | 0.78 (0.71 to 0.85) |
| Symbol frequencies, likelihood ratio, AUC | 0.82 (0.69 to 0.92) | 0.64 (0.51 to 0.76) |
| Bigram likelihood ratio, AUC | 0.90 (0.80 to 0.97) | not run |
| Automaton likelihood ratio, AUC | 0.85 (0.72 to 0.95) | not run |

- Private contracts differ from router users above chance.
- With 17 and 23 bots the intervals overlap for every pair of models, and
  the same model moves from 0.82 on dev to 0.64 on test.

**Why the method does not fit.** Three reasons, in order of importance:

1. **Too little data per trader.** State merging keeps a distinction only
   when the data supports it. With a median of about 260 symbols over 20
   symbol types, nothing beyond one step is supported.
2. **The symbols already carry the short-term structure.** "Same
   transaction" and "same block" put the most telling sequence information
   into single symbols, leaving little for a model of order to add.
3. **The behaviour may simply be close to first-order** at this level of
   description. This cannot be separated from reason 1 with one week of
   data.

## 5. What to expect from more complex models

| Extension | Expected effect | Basis |
| --- | --- | --- |
| Richer symbols (swap size, position in block, number of pools in the transaction, gas behaviour) | The most promising direction. Size alone raised the baseline from 31% to 38%. | Alphabet comparison; Niedermayer et al. (2024) on timing and gas features. |
| Combining behaviour with pool profile | A clear gain for identification: the two are different information. | Both reach about 31 to 35% separately. |
| More data per trader (a month instead of a week) | Needed before any statement about memory beyond one step. May let automata grow real states. | 88% of automata have no extra state. |
| Automata with looser merging, or other learners (FlexFringe, timed automata) | More states, but no gain in accuracy in a first check. Timed automata would replace the hand-made time-gap symbols, which is the more interesting change. | Exploration; Pellegrino et al. (2017). |
| Hidden Markov models | Would test directly for hidden strategy states. Given that order adds nothing measurable, a gain is unlikely at this data size. | Bigram against baseline. |
| Neural sequence models | Not at this scale: a few hundred traders with a few hundred symbols each. | Trigram already overfits. |
| Per-transaction instead of per-swap symbols | May separate arbitrage (cycles inside one transaction) from routed trades, which look alike now. | Both groups have many "same transaction" symbols. |
| Real bot labels | Required for the idea's original goal. Would need a key-based source or the rule-based detectors, with the circularity that brings. | Section 1. |

**A caution on expectations.** The identification result is already driven
by symbol frequencies. Improvements will come from describing swaps better,
which is feature engineering, not from the automata that give the idea its
name and its link to the course topic.

## 6. Conclusions

**Dataset.** Usable for behaviour, not for bot detection. Swap events are
free to collect, clean, and enough for identification among hundreds of
traders. They contain no labels, cover part of the market, and yield few
traders with enough activity. Building the bot classification the idea
describes needs a label source that this study could not reach.

**Baseline.** Simple and strong: symbol frequencies identify a trader among
300 to 500 in 23 to 31% of cases. It shows that the symbolization captures
individual habits, which was the idea's main risk ("alphabet design can
fail"). It did not fail.

**Method.** Probabilistic automata learned by state merging are the wrong
tool at this data size. They reduce to bigram models, score like bigram
models, and their states carry no meaning beyond the last symbol. Their only
measured merit is compactness.

**Is there potential?**

- **For the idea as written: little.** Its two pillars were bot labels,
  which are not available, and interpretable automata, which did not
  emerge.
- **For trader identification from behaviour: some.** The task has exact
  labels, the baseline works, and there is a clear path to better results
  through richer symbols and pool information. It connects to the
  de-anonymization literature. It is, however, a feature-engineering
  project more than an automata-learning one, and it raises a privacy
  question that a write-up would have to address.
- **For the course topic (learning automata): the honest result is
  negative**, and a negative result on one week of data is weak. A month of
  data for the most active traders would be the minimum to make it firm.

**Recommended next steps, if the idea is pursued.**

1. Collect four weeks for the 200 most active traders and repeat the test
   for memory beyond one step.
2. Extend the symbols with size, position in block and transaction shape.
3. Try timed automata in place of time-gap symbols.
4. Obtain a real label source before returning to bot classification.

Compared with the other three ideas studied, this one has the weakest case:
the data lacks labels, the method shows no advantage, and the positive
finding does not depend on the method.

## Limitations of this study

- **One week of data, and one particular week.** No statement about
  stability over months.
- **No bot labels.** Task B uses a structural proxy, and its evaluation sets
  have 17 and 23 bots.
- **36% of swaps were dropped** as unattributable, probably including many
  professional traders.
- **The router / private classification looks at the whole week,** test
  period included.
- **Only part of the market:** Uniswap V2 and V3 style pools.
- **One alphabet design** was explored in depth, chosen before seeing
  results but refined once (the definition of a public router was tightened
  after a first look at the data showed fleets of signers being counted as
  routers).
- **The statement that looser merging does not help** comes from an
  exploratory run on about a third of the data, not from the notebook.
- **ALERGIA through one library with one main setting.** Other state-merging
  algorithms were not tried.
- **The literature review is the thinnest of the four:** most papers at
  abstract level, and the core automata-learning references not obtained.
