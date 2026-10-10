# Idea: next-pool recommendation for traders

Which liquidity pool will a trader use next? Treat trading accounts as users
and pools as items, and build a recommender that ranks, for each account,
the pools it has not traded yet by how likely it is to trade them in the
following days.

## Summary

- **Data:** the week of Uniswap-style swap events already collected in
  `data/dex_swaps/`; more weeks with the existing download script.
- **Topics covered:** ranking and recommendation; linear algebra (matrix
  factorization, SVD) and feature engineering; the learner survey (nearest
  neighbour, linear models, boosting, neural nets).
- **Why high quality:** exact labels (the pools actually traded later), a
  natural split by time, standard baselines and metrics, and a first check
  that the obvious baseline does not already solve it.
- **What you learn:** implicit-feedback recommendation, ranking metrics,
  cold start, and how to evaluate a recommender without fooling yourself.
- **Novelty:** medium. Recommendation is rarely applied to on-chain trading.
  This is my estimate, not the result of a literature search.

## Data

One week of swaps, 1 to 8 October 2026, collected for the bot fingerprinting
study with `src/scripts/dex_swaps.py` from free public endpoints:

- 1,294,242 swaps by 118,537 signing accounts on 18,514 pools.
- Validated in
  [`../src/notebooks/04_bot_fingerprinting_data_and_baseline.ipynb`](../src/notebooks/04_bot_fingerprinting_data_and_baseline.ipynb):
  no duplicates, no missing values.

A first look at the properties that matter for recommendation, splitting the
week into its first five days and its last two:

| Property | Value |
| --- | --- |
| Share of swaps in the 10 / 100 / 1,000 most used pools | 31% / 59% / 90% |
| Pools with fewer than 10 swaps in the week | 14,366 of 18,514 |
| Accounts that used at least 2 / 5 / 20 different pools | 50,261 / 11,639 / 1,831 |
| Median number of pools per account | 1 |
| Distinct account-pool pairs in the first five days | 294,848 (0.02% of all possible pairs) |
| Accounts active in both periods | 18,063 |
| Their account-pool pairs in the last two days | 93,311, of which 48% were already seen |
| **New pairs to predict** | **48,115, from 11,403 accounts** |
| New pairs whose pool existed in the first five days | 87% |
| New pairs that go to one of the 10 / 100 most used pools | 10% / 27% |
| Pools in the last two days that did not exist before | 1,975 of 7,765, carrying 6% of the swaps |

> **Update after the feasibility study.** The table above counts pairs, so
> very active accounts weigh heavily. Measured per user, as a recommender is
> scored, popularity is a much stronger reference than the 10% suggests
> (recall@10 of about 0.16), and the item should be the token, not the pool.
> See [`pool_recommendation_writeup.md`](pool_recommendation_writeup.md).

Three properties shape the project:

- **The data is very sparse and very skewed.** Half the accounts trade one
  pool, and most pools are barely used.
- **Popularity does not solve the task.** The main risk I named when
  proposing the idea was that "recommend the biggest pools" would be hard
  to beat. Only 10% of the new pairs go to a top-10 pool, so a popularity
  ranking of length 10 can find at most a tenth of them.
- **Half of what a trader does next is a repeat.** Recommending pools the
  account already used is trivially good and must be scored separately from
  recommending new ones.

## Research questions

1. How well can an account's next *new* pools be predicted from the pools it
   and similar accounts have used?
2. Do methods that learn from co-usage (neighbours, matrix factorization)
   beat popularity, and by how much for accounts with few past pools?
3. Does *when* and *how* an account trades (order of pools, recency, the
   behavioural symbols of the bot study) add to *which* pools it traded?
4. How can brand-new pools be recommended, when nobody has traded them yet?
5. Are the learned pool and account embeddings meaningful, for example do
   pools sharing a token, or accounts of the same kind (bots, router users),
   lie together?

## Approach

1. **Define users and items.** Users: signing accounts. Items: pools. Unlike
   the bot study, no swaps need to be dropped, since every swap has a signer.
   Decide whether private bot contracts are users of their own or are merged
   with their operator.
2. **Build the interaction data.** For each account and pool: number of
   swaps, first and last time. Implicit feedback: there are no ratings and
   no negative examples.
3. **Baselines.**
   - Most popular pools overall, and most popular recently.
   - "Repeat": the account's own most used pools (for the repeat task only).
   - Item-based nearest neighbours on pool co-usage.
4. **Models.**
   - Matrix factorization for implicit feedback (weighted ALS, or BPR).
   - A truncated SVD of the interaction matrix as the linear-algebra
     reference.
   - A sequence-aware model (next item given the last few pools), to answer
     question 3.
   - Optionally a two-stage ranker: candidates from the above, re-ranked by
     gradient boosting with account and pool features.
5. **Cold start for new pools.** Add pool content: its two tokens (one
   contract call per pool) and its age. A new pool that pairs a known token
   can then be recommended to accounts that trade that token.
6. **Inspect the embeddings** (question 5), using the bot / router user
   labels from the bot study as a check.

## Evaluation

- **Split by time:** train on the first days, tune on a middle period, test
  on the last. With one week this means roughly four days, one day, two
  days, as in the bot study. More weeks would allow a rolling evaluation.
- **Two tasks, scored separately:** new pools only (the main one), and all
  pools including repeats.
- **Metrics:** recall at 10 and NDCG at 10, averaged over accounts. Recall
  says how many of the pools actually used appear in the list; NDCG also
  rewards putting them near the top. Also coverage: how many different
  pools the system ever recommends, to detect a model that has collapsed to
  popularity.
- **By account activity:** results for accounts with 1, 2 to 4, and 5 or
  more past pools, since the difficulty differs completely.
- **Uncertainty:** bootstrap over accounts.
- **Protocol detail that matters:** rank against all pools, not against a
  small sample of negatives. Sampled evaluation is known to distort
  comparisons between recommenders.

## Risks

- **One week is short.** The numbers above come from a single week and a
  five-day history. Preferences of low-activity accounts cannot be learned
  from that; more weeks should be downloaded early (about 1.5 hours per week
  with the existing script).
- **Most accounts cannot be served.** Half have a single pool. The
  evaluation will rest on the 11,000 or so accounts that return and try
  something new.
- **Bots dominate the interactions.** A few accounts and contracts make
  thousands of swaps across hundreds of pools. They could dominate both
  training and metrics; cap or separate them.
- **Token launches drive new pairs.** Many "new pool" events are probably
  many accounts rushing into one freshly launched token. That is closer to
  trend detection than to personal taste, and a recency-weighted popularity
  baseline may capture much of it. This has not been checked.
- **Leakage through time.** Any feature of a pool (its popularity, its age)
  must be computed from the training period only.
- **Is it useful, and is it harmless?** A recommender for speculative tokens
  is not obviously something to build for real users. As a study of trading
  behaviour it is fine; the write-up should say which it is.
- **Coverage of the market.** Only Uniswap V2 and V3 style pools are in the
  data.

## Relation to other ideas

- Reuses the data and download code of
  [trading bot fingerprinting](idea_bot_fingerprinting_automata.md). That
  study found that the set of pools a trader uses identifies it among 306
  traders in 35% of cases, which is the evidence that pool usage is
  personal enough to recommend from.
- Listed as idea 5 in [`project_ideas_round2.md`](project_ideas_round2.md).

## References

To be verified before use; cited from memory.

- Hu, Koren and Volinsky (2008), "Collaborative filtering for implicit
  feedback datasets".
- Rendle et al. (2009), "BPR: Bayesian personalized ranking from implicit
  feedback".
- Krichene and Rendle (2020), "On sampled metrics for item recommendation".
- Dacrema, Cremonesi and Jannach (2019), "Are we really making much
  progress? A worrying analysis of recent neural recommendation approaches".
