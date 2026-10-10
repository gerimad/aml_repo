# Literature review: next-pool recommendation for traders

Review for the idea in
[`../idea_next_pool_recommendation.md`](../idea_next_pool_recommendation.md):
treat trading accounts as users and liquidity pools as items, and recommend
to each account the pools it is likely to trade next.

## 1. Scope and method

The review covers four questions:

1. How is recommendation done when there are only observed actions, no
   ratings?
2. How strong are simple methods compared with complex ones?
3. How should a recommender be evaluated, and what goes wrong?
4. Has recommendation been applied to on-chain assets?

Fifteen open-access papers were downloaded to `papers/` (PDFs are ignored by
git, the links below are the source of record). The "Read" column says how much
of each was read for this review.

| # | Paper | File in `papers/` | Read |
| --- | --- | --- | --- |
| 1 | Hu, Koren, Volinsky (2008), [Collaborative filtering for implicit feedback datasets](http://yifanhu.net/PUB/cf.pdf) | `hu2008_implicit_feedback_cf.pdf` | Abstract |
| 2 | Rendle et al. (2009), [BPR: Bayesian personalized ranking from implicit feedback](https://arxiv.org/abs/1205.2618) | `rendle2009_bpr.pdf` | Abstract |
| 3 | Hidasi et al. (2015), [Session-based recommendations with recurrent neural networks](https://arxiv.org/abs/1511.06939) | `hidasi2015_gru4rec.pdf` | Abstract |
| 4 | Steck (2019), [Embarrassingly shallow autoencoders for sparse data](https://arxiv.org/abs/1905.03375) | `steck2019_ease.pdf` | Abstract |
| 5 | Dacrema, Cremonesi, Jannach (2019), [Are we really making much progress? A worrying analysis of recent neural recommendation approaches](https://arxiv.org/abs/1907.06902) | `dacrema2019_worrying_analysis.pdf` | Abstract |
| 6 | Rendle et al. (2020), [Neural collaborative filtering vs. matrix factorization revisited](https://arxiv.org/abs/2005.09683) | `rendle2020_ncf_vs_mf.pdf` | Abstract |
| 7 | Rendle et al. (2021), [Revisiting the performance of iALS on item recommendation benchmarks](https://arxiv.org/abs/2110.14037) | `rendle2021_ials_revisited.pdf` | Abstract |
| 8 | Rendle (2019), [Evaluation metrics for item recommendation under sampling](https://arxiv.org/abs/1912.02263) | `rendle2019_metrics_under_sampling.pdf` | Abstract |
| 9 | Ji et al. (2020), [A critical study on data leakage in recommender system offline evaluation](https://arxiv.org/abs/2010.11060) | `ji2020_data_leakage_recsys.pdf` | Abstract |
| 10 | Meng et al. (2020), [Exploring data splitting strategies for the evaluation of recommendation models](https://arxiv.org/abs/2007.13237) | `meng2020_data_splitting_recsys.pdf` | Abstract |
| 11 | Li et al. (2021), [A next basket recommendation reality check](https://arxiv.org/abs/2109.14233) | `li2021_next_basket_reality_check.pdf` | Abstract, conclusions |
| 12 | Choi et al. (2024), [A recommender system for NFT collectibles with item feature](https://arxiv.org/abs/2403.18305) | `choi2024_nft_recsys_item_features.pdf` | Abstract, conclusion |
| 13 | Li (2023), [NFT.mine: an xDeepFM-based recommender system for non-fungible token (NFT) buyers](https://arxiv.org/abs/2306.03942) | `nftmine2023_xdeepfm.pdf` | Not read (search summary only) |
| 14 | Ludewig, Jannach (2018), [Evaluation of session-based recommendation algorithms](https://arxiv.org/abs/1803.09587) | `ludewig2018_session_evaluation.pdf` | Not read |
| 15 | Anelli et al. (2022), [Top-N recommendation algorithms: a quest for the state-of-the-art](https://arxiv.org/abs/2203.01155) | `anelli2022_topn_quest.pdf` | Not read |

**This review is at abstract level.** Twelve papers were read only in their
abstract (one with its conclusions), and three not at all. That is enough to
place the idea and to take over the main warnings about evaluation, which
the abstracts state plainly. It is not enough to cite any of them for
details. The item-based neighbour method and the truncated SVD used in the
experiments are described from general knowledge; their original papers are
listed in section 5.

## 2. Summaries

### 2.1 Recommendation from implicit feedback

**[1] Hu, Koren, Volinsky (2008).** With implicit feedback (purchases,
views) "we lack substantial evidence on which products consumer dislike".
The paper proposes to treat the data "as indication of positive and negative
preference associated with vastly varying confidence levels", which leads to
a weighted matrix factorization fitted by alternating least squares, with
cost linear in the data size.

**[2] Rendle et al. (2009), BPR.** Earlier methods for implicit feedback are
"designed for the item prediction task of personalized ranking, [but] none
of them is directly optimized for ranking". BPR is a ranking criterion
derived from a Bayesian analysis, with a stochastic gradient learner that
can train matrix factorization or neighbour models.

**[3] Hidasi et al. (2015), GRU4Rec.** When only short sessions are known
instead of long user histories, "the frequently praised matrix factorization
approaches are not accurate" and practice falls back on item-to-item
recommendations. The paper models the whole session with a recurrent network.

**[4] Steck (2019), EASE.** A linear item-item model with a closed-form
solution that "achieves better ranking accuracy than various
state-of-the-art collaborative-filtering approaches, including deep
non-linear models, on most of the publicly available data-sets".

### 2.2 How strong are simple methods?

**[5] Dacrema, Cremonesi, Jannach (2019).** Of 18 neural recommendation
algorithms from top conferences, only 7 could be reproduced. Six of those
"can often be outperformed with comparably simple heuristic methods, e.g.,
based on nearest-neighbor or graph-based techniques", and the seventh "did
not consistently outperform a well-tuned non-neural linear ranking method".

**[6] Rendle et al. (2020).** Revisits neural collaborative filtering: "with
a proper hyperparameter selection, a simple dot product substantially
outperforms the proposed learned similarities".

**[7] Rendle et al. (2021).** Matrix factorization by implicit alternating
least squares had been reported as uncompetitive. "With proper tuning, iALS
is highly competitive and outperforms any method on at least half of the
comparisons."

**[11] Li et al. (2021): next-basket recommendation.** The closest setting
to ours: predict the set of items a user will consume next.

- The next basket consists of **repeat items** and **explore items**, and the
  two must be evaluated separately.
- "No state-of-the-art NBR method, deep learning-based or otherwise,
  consistently shows the best performance across datasets. Compared to a
  simple frequency-based baseline, the improvements are modest or even
  absent."
- "There is a clear difficulty gap and trade-off between the repeat task and
  the explore task." Being biased towards the easier repeat task boosts
  overall scores.
- Deep methods "achieve relatively good explore performance, but are not
  able to outperform the simple frequency-based baseline" on repeats.

### 2.3 How to evaluate, and what goes wrong

**[8] Rendle (2019): sampled metrics.** To save time, many papers rank the
true item against a small random sample of other items. Such sampled metrics
"are inconsistent with their exact version" and "do not persist relative
statements, e.g., algorithm A is better than B, not even in expectation".
With very small samples all metrics collapse to one.

**[9] Ji et al. (2020): data leakage.** "Data leakage is caused by not
observing global timeline in evaluating recommenders." With the common
leave-last-item-out split per user, a model trains on interactions that
happened after the ones it is tested on, and "all models indeed recommend
future items that are not available at the time point of a test instance".
The amount of leaked data changes the ranking of models. The authors call
for evaluation along a single timeline.

**[10] Meng et al. (2020): data splitting.** "There is no standard splitting
strategy", and the choice "can markedly alter the ranking of
state-of-the-art recommender systems, making much of the currently published
literature non-comparable".

### 2.4 Recommendation for on-chain assets

**[12] Choi et al. (2024).** A graph-based recommender for NFTs using
transaction records plus image, text and price features of the items. The
authors report that item features improve the result and that graph models
beat their baselines. They note that recommendation for NFTs has received
"comparatively less attention".

**[13] NFT.mine (2023)** applies a deep factorization model to marketplace
data. Not read.

The search found work on NFTs only. It found **no paper on recommending
fungible tokens or liquidity pools from on-chain trading histories.** This
was one search, so the absence is not established.

## 3. Insights gathered

1. **Implicit feedback has positives only.** Not trading a token is not a
   sign of dislike [1, 2].
2. **Standard, well-understood models exist:** item neighbours, matrix
   factorization (iALS, BPR), linear item-item models [1, 2, 4].
3. **Simple models, properly tuned, are hard to beat,** and many reported
   gains of complex models do not survive a careful comparison [5, 6, 7].
4. **A frequency or popularity baseline can be as good as anything** in
   next-basket settings [11].
5. **Separate repeats from new items.** They are different tasks of very
   different difficulty, and mixing them hides what a model does [11].
6. **Split on one global timeline.** Per-user leave-one-out leaks the
   future [9].
7. **Rank against all items.** Sampled negatives can reverse the ranking of
   models [8].
8. **The split changes the conclusions** [10], so it must be stated and
   justified.
9. **With short histories, item-to-item methods are the practical choice**
   [3].
10. **On-chain recommendation has been tried for NFTs,** where item features
    help with sparse interactions [12]. Fungible tokens appear untouched.

## 4. What this means for our idea

**The idea is plausible and, as far as one search shows, new in its domain.**
The methods are standard; what is new is the data: pseudonymous accounts,
items that appear and disappear within days, and "users" that are partly
machines.

**The literature mostly tells us how not to fool ourselves.** The first
experiments in
[`../../src/notebooks/05_pool_recommendation_data_and_baseline.ipynb`](../../src/notebooks/05_pool_recommendation_data_and_baseline.ipynb)
follow insights 5 to 7 and confirm insights 3, 4 and 8 on this data:

- Popularity is level with item neighbours and ahead of SVD when each is
  used alone, in line with [5, 11].
- Half of what a user trades next is a repeat, and a user's own history
  predicts that best, in line with [11].
- The data adds a warning of its own that goes beyond [10]: the ranking of
  models changes with the *day* of the evaluation, and a first comparison
  made at a single point in time pointed to the wrong baseline.

**Changes to the idea that follow from the literature and the first look at
the data:**

| Original plan | Change | Source |
| --- | --- | --- |
| Items are pools | Items are tokens. A router, not the user, chooses pools. | Data (notebook, section 3.1) |
| Score "new pools" and "all pools" | Keep, as the explore and repeat tasks. Focus on explore. | [11] |
| Baselines: popularity, repeat, item neighbours | Keep. Expect popularity to be strong. | [5, 11] |
| Models: weighted ALS or BPR, truncated SVD, a sequence model, a two-stage ranker | Start with item neighbours and SVD. Add iALS with proper tuning before anything neural. A sequence model only if order is shown to matter. | [3, 6, 7] |
| Split by time: train, tune, test | Keep the global timeline, and evaluate at many points in time instead of one. | [9, 10]; data |
| Rank against all pools | Keep. | [8] |
| Cold start through pool content (its two tokens, its age) | More important than planned: 15 to 25% of what users pick up next did not exist before. Item features are the only way to reach it. | [12]; data |

**A sharper research question.** *When items are short-lived and interest
moves fast, what can personalization add to popularity, and how many
evaluation periods does it take to tell?* The second half is as much a part
of the question as the first.

## 5. Papers to obtain

| Paper | Why it matters |
| --- | --- |
| Sarwar et al. (2001), "Item-based collaborative filtering recommendation algorithms", WWW | The neighbour method used in the experiments. |
| Cremonesi, Koren, Turrin (2010), "Performance of recommender algorithms on top-N recommendation tasks", RecSys | The truncated SVD recommender used in the experiments, and an early statement that popularity distorts top-N evaluation. |
| Krichene, Rendle (2020), "On sampled metrics for item recommendation", KDD | The published version of [8]. The idea file cites it; what was read here is the preprint. |
| Schein et al. (2002), "Methods and metrics for cold-start recommendations", SIGIR | The cold-start problem, which is a fifth to a quarter of our task. |
| Papers [13], [14] and [15] of the table | Downloaded but not read. [14] is the reference for comparing sequence-aware methods with neighbour methods. |
