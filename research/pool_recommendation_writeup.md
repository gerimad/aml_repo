# Write-up: next-pool recommendation for traders

Feasibility study for the idea in
[`idea_next_pool_recommendation.md`](idea_next_pool_recommendation.md). It
covers the literature, the data, the split, a baseline and a first test of
the proposed method.

- Literature review: [`literature/literature_review_pool_recommendation.md`](literature/literature_review_pool_recommendation.md)
- Notebook with all numbers and charts: [`../src/notebooks/05_pool_recommendation_data_and_baseline.ipynb`](../src/notebooks/05_pool_recommendation_data_and_baseline.ipynb)
- Code: [`../src/scripts/pool_recsys.py`](../src/scripts/pool_recsys.py), [`../src/scripts/dex_swaps.py`](../src/scripts/dex_swaps.py), tests in [`../tests/`](../tests/)

## Verdict

**The idea is feasible and the method shows a real, consistent gain, in a
changed form.** The item to recommend has to be the token, not the pool,
because routers choose pools. Collaborative filtering alone is no better
than recommending the most popular tokens, but blended with popularity it
beats the baseline at every one of seven evaluation points, by a third on
average.

**The main weakness is statistical.** Results swing by a factor of almost
two from one day to the next, far more than the sampling uncertainty over
users. Two weeks of data give seven overlapping development points and one
test point, which is enough to see the pattern and not enough to rely on its
size.

Candidate models were scored at the development points only. During
exploration I also scored them once at the test point; see "Limitations".

## 1. Data

| | |
| --- | --- |
| Source | Free public Ethereum RPC endpoints, no key |
| Content | Every swap on Uniswap V2 and V3 style pools, with the signing account; and the two tokens of every pool |
| Window | 14 days: blocks 26,049,600 to 26,150,399, 24 September to 8 October 2026 |
| Size | 2,710,140 swaps in 1,965,611 transactions, by 211,547 accounts, on 24,630 pools holding 20,014 tokens |

The second week and the pool tokens were downloaded for this study; the
first week comes from the bot fingerprinting study.

**Validation.** 99.1% of blocks contain a swap, daily counts are even, there
are no duplicates or missing values. The tokens of all but 10 pools could be
read, covering 99.98% of swaps.

**Is it enough and relevant?**

- Events and users: yes.
- Users with something to personalize on: thousands, not hundreds of
  thousands. 56% of accounts used one pool in two weeks.
- Time: not enough. See section 4.

**What the descriptive statistics say.** Everything is skewed. Ten pools
carry 29% of the swaps; 70% of pools have fewer than five users; the hundred
most active accounts make 27% of all swaps.

**Features and labels.**

- **Features:** the user-token matrix of the history (who traded what).
  Nothing else is used yet.
- **Labels:** the tokens a user trades for the first time in the following
  two days. Exact and free.

## 2. Preprocessing and split

**The central preprocessing decision: tokens, not pools.**

- A fifth of all transactions pass through more than one pool, and they
  contain 41% of the swaps. A user swapping ETH for a token may be routed
  through ETH/USDC and USDC/TOKEN. The user chose the token; the router chose
  the pools.
- Of the user-pool pairs that count as "new" at pool level, 16% are pools of
  two base currencies and 12% are another pool for a token the user had
  already traded. More than a quarter of the pool-level labels are not new
  choices.
- Each transaction is therefore reduced to the tokens its sender ended up
  selling and buying, with pass-through tokens dropped. This also removes
  arbitrage cycles, which end where they start.

| Other issue | Decision |
| --- | --- |
| Wrapped ETH and two dollar stablecoins are in almost every trade | Tokens traded by more than 10% of users are removed (three tokens). |
| A few accounts make a large share of the swaps | Interactions are binary, and metrics are averaged over users. |
| Half of what users do next is a repeat | New tokens and all tokens are separate tasks. |
| Tokens that did not exist in the history | Counted as misses, not removed. |

After this: 157,060 users and 19,631 tokens. The median user has one token;
11,000 users have five or more.

**Split, on one global timeline.** Train is the first ten days, dev days 11
and 12, test days 13 and 14. A recommendation is judged on the two days that
follow it.

- **Development:** seven rolling evaluation points, after day 4 to day 10.
  At each, all earlier days are the history and the next two days the
  target. None reaches into the test days.
- **Test:** one point, after day 12.
- Each point evaluates 6,700 to 8,600 users who have a history and trade at
  least one token new to them in the target.

**What the split shows about the task.**

- 15 to 25% of the tokens users pick up next did not exist in the history.
  They were launched inside the two target days.
- Almost all the rest are established tokens. Tokens that first appeared in
  the last three days of the history account for about 1%.
- The adoptions are spread out: the ten most picked tokens cover 14% of
  them.

**Leakage.** One point in time for all users; users and items indexed from
the history only; popularity computed from the history only; every model
ranks all tokens, with no sampled negatives; the user's own history removed
from the "new tokens" ranking. Not handled: one person with several accounts.

## 3. Baseline

**Task (primary).** For each user, rank the tokens it has not traded. A hit
is a token it trades for the first time in the next two days.

**Metrics, and why.**

| Metric | Reason |
| --- | --- |
| **Recall@10**, averaged over users | A user can act on a short list. Averaging over users keeps a few very active accounts from dominating. |
| NDCG@10, hit rate | Position in the list; and the plain reading "was the list any use". |
| Coverage | Detects a model that shows everyone the same few tokens. |
| **Spread over evaluation points** | The period turned out to be the main source of uncertainty. |

**Protocol.** At each point: fit on the history, rank all known tokens for
every evaluated user, score against the next two days.

**The fixed baseline** recommends the tokens with the most users in the
history. It has no setting and is the same list for everyone.

| New tokens, recall@10 | Development (mean of 7) | Lowest | Highest | Test |
| --- | --- | --- | --- | --- |
| **Popularity (fixed baseline)** | **0.155** | 0.114 | 0.201 | **0.202** |
| Recent popularity (last day only) | 0.156 | 0.100 | 0.197 | 0.173 |

**How the baseline performs.**

- **It finds a real share:** 16% of first-time tokens on average, and at
  least one right token for about one user in four.
- **It depends strongly on the period:** from 0.11 to 0.20. The sampling
  uncertainty over users at one point is about ±0.007, so the period effect
  is several times larger.
- **Recency does not reliably help.** Popularity over the last day wins at 2
  of the 7 development points and loses at the test point.
- **It is not a recommender:** under 1% of tokens ever appear in a list.
- **With repeats included,** a user's own history (0.27) beats popularity
  (0.17), as expected. That task was not pursued.

**A methodological point.** My first exploration used a single evaluation
point, after day 10. There, recent popularity beats plain popularity by
0.04 with tight intervals, and collaborative filtering looks clearly worse
than both. Neither conclusion holds on average over the seven points. Only
the rolling evaluation showed it.

## 4. Does the proposed method fit the task and data?

**Partly: not alone, but as a complement to popularity, yes.**

| New tokens, 7 development points | Mean recall@10 | Lowest | Highest | Beats baseline at | Coverage |
| --- | --- | --- | --- | --- | --- |
| Popularity (fixed baseline) | 0.155 | 0.114 | 0.201 | | 0.7% |
| Item nearest neighbours | 0.158 | 0.119 | 0.216 | 3 of 7 | 22% |
| Truncated SVD, rank 32 | 0.128 | 0.092 | 0.187 | 2 of 7 | 4% |
| Blend: 85% popularity + neighbours | 0.181 | 0.155 | 0.227 | 7 of 7 | 0.8% |
| **Blend: 50% popularity + neighbours** | **0.207** | 0.161 | 0.278 | **7 of 7** | 18% |

- **Alone, collaborative filtering matches popularity and no more.**
- **Blended with popularity it is consistently better:** +0.05 recall on
  average, a third above the baseline, at all seven points. The two carry
  different information: what many people trade, and which of it fits this
  user.
- **The blend also recommends far more of the catalogue** (18% of tokens
  against 0.7%).
- **History helps the collaborative models, as theory says.** For users with
  one token they are level with popularity or below; for users with five or
  more they are ahead (0.14 against 0.12). The blend leads in every group.
- **The result holds under the variants tried:** first-time buys only as the
  target, and without the 5% of users with the largest histories.
- **With pools as items, as the idea was first written,** all scores are
  higher (popularity 0.25) because the task includes non-choices, and item
  neighbours alone fall behind popularity. The blend still leads.

**What does not fit.**

- **Cold start.** 15 to 25% of the target cannot be reached by any model
  that only knows past interactions.
- **Stability.** The ranking of the single models changes from day to day.
  Seven overlapping points are few, and the blend weight was chosen on the
  same points on which it is reported.

## 5. What to expect from more complex models

| Extension | Expected effect | Basis |
| --- | --- | --- |
| More weeks of data and a longer rolling evaluation | No gain in score, but the precondition for trusting any. The first priority. | Day-to-day spread of 0.11 to 0.20. |
| A properly weighted combination (a learned ranker over popularity, neighbour and recency scores) | A further modest gain over the 50 / 50 blend, which was barely tuned. | Two weights tried, both ahead at every point. |
| Token features for cold start: age, pool size at launch, deployer, the launch platform | The largest untapped part: up to 15 to 25% of the target is new tokens. Hard, because a model has minutes to hours of information. | Share of future-only tokens; Choi et al. (2024) on item features. |
| Implicit matrix factorization (iALS, BPR), properly tuned | Probably level with item neighbours, perhaps a little better. Truncated SVD, its untuned relative, was worse. | Rendle et al. (2021); SVD result. |
| Sequence-aware models (what was traded last) | Uncertain. Most users have one or two tokens, so there is little sequence. | 56% of users with one pool. |
| Neural recommenders | Unlikely to help at this sparsity and scale. | Dacrema et al. (2019), Rendle et al. (2020). |
| Separating bots from people | May sharpen the result: users with large histories are the hardest for every model. | Recall by history size. |

**A caution on expectations.** The gain measured is from 0.155 to 0.207.
In practical terms the best model still misses four of every five tokens a
user will pick up. A large part of the remainder is brand-new tokens and
day-to-day fashion that no history predicts.

## 6. Conclusions

**Dataset.** Free, clean and large in events, and now including token
identities. Its limits for this task are the short time span, the extreme
sparsity per user, and items that are born inside the prediction window.

**Baseline.** Popularity is strong and unstable: strong enough to equal
collaborative filtering alone, unstable enough that a single evaluation is
misleading.

**Method.** Collaborative filtering fits as a component, not as the
solution. The simplest neighbour model, mixed half and half with popularity,
gives a consistent gain. That is a positive result for the method, and it
was obtained with almost no tuning.

**Is there potential?**

- **Yes, as a recommendation study with an unusual twist.** The task has
  exact labels, a working baseline, a method that improves on it, clear next
  steps, and no prior work found on fungible tokens. It covers course topics
  the other studies did not: ranking and recommendation, nearest neighbours,
  matrix factorization.
- **The evaluation question is itself a contribution.** That model rankings
  flip from day to day in a fast-moving market, and that a single split
  points to the wrong baseline, is a concrete case for the evaluation
  literature.
- **As a product: questionable.** Recommending speculative tokens to traders
  is not obviously desirable, and the best model is right about one time in
  five. The write-up should present this as a study of trading behaviour.

**Recommended next steps.**

1. Download four to six more weeks (about 20 minutes to 1.5 hours per week
   with the existing script) and repeat the rolling evaluation with
   non-overlapping targets.
2. Fix the blend on the first half of that period and confirm it on the
   second.
3. Add tuned implicit matrix factorization as the reference model.
4. Collect launch features for new tokens and attempt the cold-start part.
5. Separate automated accounts and report results for the rest.

## Limitations of this study

- **Two weeks, one market phase.** Seven development points whose two-day
  targets overlap, and one test point.
- **The blend was chosen and reported on the same development points.** Its
  size of gain is optimistic; its sign (7 of 7 points for both weights
  tried) is more trustworthy.
- **Test-period scores of the candidates were seen once.** While checking
  the stability of the baselines I ran the neighbour model and two blends at
  the points after day 11 and day 12. They were not used for any choice and
  are not reported, but the test point is no longer untouched for those
  models.
- **Earlier evaluation points have shorter histories** (4 days against 10),
  so time and history length are mixed in the day-to-day variation.
- **A user is an address.** People with several addresses, and bots, are not
  separated.
- **Only part of the market:** Uniswap V2 and V3 style pools. A token bought
  elsewhere looks "new" here.
- **Two collaborative models, default settings.** No matrix factorization
  with proper weighting and tuning was run.
- **The 10% threshold for base currencies** was set by inspection. It
  separates three tokens (traded by 81%, 40% and 29% of users) from the
  fourth (5%), so the result should not be sensitive to it, but that was not
  tested.
- **The literature review is at abstract level,** and three of the
  downloaded papers were not read.
