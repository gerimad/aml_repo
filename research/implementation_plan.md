# Implementation plan

How the project plan in [`crypt_project_plan.pdf`](crypt_project_plan.pdf) is
turned into code, notebooks and results. The PDF says *what* is done and how
it is judged. This file says *how* it is built: modules, functions, file
formats, settings, checks and the order of work.

- **Rule of precedence.** Where this file and the PDF differ, the PDF is right.
  A difference that is needed is recorded under "Changes to the plan" in
  [`project_direction.md`](project_direction.md) and in section 11 below.
- **Guidelines.** The implementation guidelines in `project_direction.md`
  apply throughout: simple, pedantic, explained code; assumptions written down
  where they are made; reusable logic in `src/scripts/` with tests; notebooks
  for exploration and results only.

## Contents

0. [Starting point](#0-starting-point)
1. [Fixed definitions](#1-fixed-definitions)
2. [Repository layout and modules](#2-repository-layout-and-modules)
3. [WP1 Data](#3-wp1-data)
4. [WP2 Protocol](#4-wp2-protocol)
5. [WP3 Nearest neighbours](#5-wp3-nearest-neighbours-core-problem-1)
6. [WP4 Time weighting](#6-wp4-time-weighting-core-problem-2)
7. [WP5 Cold start](#7-wp5-cold-start-core-problem-3)
8. [WP6 Learned ranker](#8-wp6-learned-ranker)
9. [WP7 Test and write-up](#9-wp7-test-and-write-up)
10. [Order of work, effort and checkpoints](#10-order-of-work-effort-and-checkpoints)
11. [Open points found while writing this plan](#11-open-points-found-while-writing-this-plan)
12. [WP8 Trading strategy backtest (extension)](#12-wp8-trading-strategy-backtest-extension)

## 0. Starting point

| Item | State on 21 November 2026 |
| --- | --- |
| `src/scripts/dex_swaps.py` | Download, decode, pool-token lookup, `load_swaps`. Resumable, one Parquet file per 1,000 blocks. |
| `src/scripts/pool_recsys.py` | `trade_legs`, `interactions`, `to_matrix`, `build_task`, `Popularity`, `OwnHistory`, `ItemKNN`, `PureSVD`, `Blend`, `top_k`, `ranking_metrics`, `evaluate`, `coverage`, `bootstrap_mean`. |
| Tests | `tests/test_dex_swaps.py`, `tests/test_pool_recsys.py`. 34 pass (`uv run pytest`). |
| Local data | `data/dex_swaps/chunks/`: 203 chunks, blocks 25,948,800 to 26,150,399 (exploration data only). `pool_tokens.parquet` cache. |
| Project data (blocks 25,344,000 to 25,948,799) | **Not downloaded.** |

What already matches the PDF and is reused as is: the token-level trade legs,
the shrunk cosine `ItemKNN` (its formula is exactly the one in section 6.1 of
the PDF), `Blend` (the per-user max scaling of the fixed combination), `top_k`
with exclusion, and the metric functions.

## 1. Fixed definitions

These are fixed before any project data is looked at. Code reads them from one
place (`protocol.py`, section 4) and never re-types them.

### 1.1 Periods and evaluation points

All times are UTC. An evaluation point *t* is a midnight. The history of *t* is
every trade with timestamp `< t`; its target is every trade with timestamp in
`[t, t + 2 days)`.

| Part | Period (2026) | Evaluation points *t* | Number |
| --- | --- | --- | --- |
| Burn-in | 18 Jun 00:00 to 2 Jul 00:00 | none | 0 |
| Dev A | 2 Jul to 22 Jul | 2, 4, 6, …, 20 Jul | 10 |
| Dev B | 22 Jul to 13 Aug | 22, 24, 26, 28, 30 Jul, 1, 3, 5, 7, 9, 11 Aug | 11 |
| Test | 13 Aug to 10 Sep | 13, 15, …, 31 Aug, 2, 4, 6, 8 Sep | 14 |

- Targets do not overlap and do not cross a period boundary: the last Dev A
  target ends exactly where Dev B begins, and so on.
- The data is read by block number (25,344,000 to 25,948,799) and cut by
  timestamp. **Check in WP1:** the first block must be at or before 18 Jun
  00:00 and the last at or after 10 Sep 00:00. If not, the missing blocks are
  downloaded (the block range is approximate, the dates are not).

### 1.2 Users, tokens, interactions

| Term | Definition | Assumption stated in code |
| --- | --- | --- |
| User | The signing account `tx_from` of a transaction. | One account is one decision maker. False for people with several accounts and for intent-based systems where a solver signs; this makes histories look shorter, it cannot leak labels. |
| Trade leg | One row per (transaction, token, side) from `pool_recsys.trade_legs`; tokens only passed through are dropped. | A pool's two tokens never change (they are immutable in the contract). |
| Base currency | A token traded by more than 10% of the users of the **burn-in** period. Removed from all legs, all periods. | Expected: WETH, USDC, USDT. If the list differs, it is recorded, not adjusted. |
| Interaction | User *u* traded token *i* (buy or sell) at least once. Binary. | Trade count and size carry no preference information we can use. |

### 1.3 Universe, label and evaluated users at a point *t*

| Object | Definition |
| --- | --- |
| Past of *u* | All tokens *u* traded from the start of the data up to *t*. |
| Token universe at *t* | All non-base tokens traded by anyone from the start of the data up to *t*. Every model ranks this whole set. |
| Reference window | The 14 days before *t*. |
| Positive pair (*u*, *i*) | *u* trades *i* in the target and *i* is not in the past of *u*. |
| Cold pair | A positive pair whose token is not in the universe at *t*. It stays in the denominator of recall and NDCG. |
| Evaluated users | Users with at least one token in the reference window and at least one positive pair. **The same set for every model at a point**, whatever history length a model uses. This is what makes the comparison paired. |
| Excluded from a list | The user's whole past. |

The PDF defines the label as "never traded before *t*", which uses the whole
past. The baseline uses the 14-day reference window. See 11.2 for the
consequence.

### 1.4 Metrics and statistics

| Name | Definition | Function |
| --- | --- | --- |
| Recall@10 (primary) | Hits in the list / positive pairs of the user (cold pairs included), averaged over evaluated users | `pool_recsys.ranking_metrics` |
| NDCG@10, hit rate@10 | As in `ranking_metrics` | same |
| Coverage | Distinct tokens in all lists / size of the universe | `pool_recsys.coverage` |
| Recall@10 without the head | Recall computed after removing from the target the most popular tokens of the history that together hold a third of its interactions (Cremonesi 2010) | new: `protocol.head_tokens` |
| Recall by history size | Separately for users with 1, 2 to 4, and 5 or more tokens in the reference window | new: `protocol.by_history_size` |
| Paired difference | *d_p* = recall@10(model) − recall@10(baseline) at point *p* | new: `stats.paired_differences` |
| Interval over points | Percentile bootstrap of the mean of *d_p*, 10,000 resamples of points, seed 0 | new: `stats.bootstrap_over_points` |
| Sign test | Exact two-sided binomial test of the number of points with *d_p* > 0 (ties dropped and reported) | new: `stats.sign_test` (wraps `scipy.stats.binomtest`) |
| Interval within a point | Percentile bootstrap over users, 1,000 resamples | `pool_recsys.bootstrap_mean` |

**Success criterion (from the PDF, repeated so the code can check it):** on the
14 test points, the mean of *d_p* for the primary model is positive and its
95% bootstrap interval over points excludes zero.

## 2. Repository layout and modules

New code goes into a few plain modules. Each serves one work package; each has
a test file. Existing modules keep their names ("pool" in `pool_recsys` is a
historical name, see the PDF).

| File | Work package | Contents |
| --- | --- | --- |
| `src/scripts/dex_swaps.py` | 1 | Existing. Add: block-range filter in `load_swaps`; `validate_swaps`. |
| `src/scripts/token_data.py` | 1 | **New.** Project trade legs, base-currency list, token first-seen table, launch pool. Writes `data/project/`. |
| `src/scripts/protocol.py` | 2 | **New.** The fixed definitions of section 1; evaluation points; tasks per point; running a model over points; result tables. |
| `src/scripts/stats.py` | 2 | **New.** Paired differences, bootstrap over points, sign test, summary table. |
| `src/scripts/pool_recsys.py` | 3, 4 | Existing. Add: time-weighted matrix, `TimeWeightedPopularity`, `TimeWeightedItemKNN`. |
| `src/scripts/cold_start.py` | 5 | **New.** Refreshed-list protocol, token features, trivial heuristics. |
| `src/scripts/ranker.py` | 6 | **New.** Candidates, features, training rows, logistic regression, boosted trees, cross-validation over points, ablation. |
| `src/scripts/run_test.py` | 7 | **New.** Scores the frozen configuration on the test points, once. |
| `configs/*.json` | 3 to 7 | **New.** Chosen settings, written by the tuning notebooks and read by later ones. Committed. |
| `results/<wp>/*.csv` | 2 to 7 | **New.** One row per (model, point) and per (model, point, history group). Small, committed. |
| `src/notebooks/09_` … `15_` | one each | See the work packages. |

Data written under `data/project/` (local, not committed):

| File | Rows | Columns |
| --- | --- | --- |
| `legs.parquet` | one per (transaction, token, side), base currencies removed | `block`, `timestamp`, `tx_hash`, `user`, `token`, `side`, `pool` |
| `base_tokens.json` | — | the list, the burn-in shares of the top 10 tokens, and the date it was computed |
| `tokens.parquet` | one per token | `token`, `first_block`, `first_timestamp`, `launch_pool`, `launch_base` (the other token of the launch pool), `censored` (first seen in the first 24 hours of the data, so the true age is unknown) |
| `validation.csv` | one per week | the checks of 3.2 |

**Dependency to add:** `lightgbm` (for the boosted trees with a logistic and a
LambdaMART objective). Everything else is already in `pyproject.toml`.

## 3. WP1 Data

**Goal:** 12 weeks of validated swaps and trade legs.

### 3.1 Download

```bash
uv run python src/scripts/dex_swaps.py --start 25344000 --blocks 604800 --workers 2
```

- 605 chunks (the last one has 800 blocks). The existing chunks start at
  25,948,800, so the ranges do not collide.
- Expected 4 to 18 hours of download (20 min to 1.5 h per week), about 16
  million swaps, 2.5 to 3 GB.
- Then fetch the tokens of the new pools: `load_pool_tokens(swaps["pool"].unique())`.

### 3.2 Changes to `dex_swaps.py`

| Function | Spec |
| --- | --- |
| `load_swaps(data_dir, first_block=None, last_block=None)` | Reads only the chunk files whose range lies inside `[first_block, last_block]`, decided from the file names before reading. Raises if any chunk of the requested range is missing (lists the missing ranges). Default with no bounds: all chunks, as now, so notebooks 05 and 08 still run. **Why:** the project must never load the exploration chunks by accident. |
| `validate_swaps(swaps, pool_tokens) -> pd.DataFrame` | One row per ISO week. Columns and thresholds: `blocks_with_swap` (share, ≥ 0.98), `duplicate_events` (on `tx_hash`, `log_index`; must be 0), `missing_values` (must be 0), `swaps_with_known_tokens` (share, ≥ 0.999), `swaps_per_day_min`, `swaps_per_day_max`, `users_per_day_min`, `users_per_day_max`, `passed` (bool). Daily counts are reported, not thresholded, because the market changes; a day under half the week's median is flagged. |

### 3.3 `token_data.py`

| Function | Spec |
| --- | --- |
| `PROJECT_FIRST_BLOCK = 25_344_000`, `PROJECT_LAST_BLOCK = 25_948_799` | Constants. |
| `base_tokens(legs, start, end, threshold=0.10) -> list[str]` | Tokens traded by more than `threshold` of the distinct users with timestamp in `[start, end)`. Called once with the burn-in period. |
| `project_legs(swaps, pool_tokens, base) -> pd.DataFrame` | `pool_recsys.trade_legs`, plus `timestamp` and `pool` carried over, base tokens removed. Asserts: no null token, no null user, sorted by block. |
| `token_table(legs, data_start) -> pd.DataFrame` | One row per token: first block and time, the pool of its first leg (`launch_pool`) and that pool's other token (`launch_base`), `censored = first_timestamp < data_start + 24 h`. |
| `build_project_data(out_dir)` | Runs the three steps above in order, writes the files of section 2, prints the counts at each step. |

### 3.4 Tests

`trade_legs` is already tested. New tests on small hand-built frames:
`load_swaps` reads only the requested range and raises on a gap;
`validate_swaps` counts a planted duplicate and a planted missing value;
`base_tokens` at threshold exactly 10% (strictly more is removed);
`token_table` picks the first leg and sets `censored` correctly at the boundary.

### 3.5 Notebook `09_project_data.ipynb`

Validation table per week; the base-currency list with the burn-in shares;
the descriptive statistics of PDF Table 4 recomputed on 12 weeks; the
timestamps of the first and last block against section 1.1.

**Done when:** 12 weeks pass validation, the period check of 1.1 holds, and
`data/project/` is written. Any failed check is either fixed by re-downloading
or recorded as a change to the plan.

## 4. WP2 Protocol

**Goal:** the baseline table on the 21 development points, and the machinery
that every later package uses.

### 4.1 `protocol.py`

| Function | Spec |
| --- | --- |
| `POINTS: dict[str, list[pd.Timestamp]]` | `{"dev_a": [...10], "dev_b": [...11], "test": [...14]}` as in 1.1. Built from the dates, then asserted: counts 10/11/14, spacing exactly 2 days, no target crosses a period boundary. |
| `REFERENCE_WINDOW = pd.Timedelta(days=14)`, `HORIZON = pd.Timedelta(days=2)`, `K = 10` | Constants. |
| `make_task(legs, t, history_length=None) -> dict` | Builds one evaluation at *t*, following 1.3. Returns everything `pool_recsys.build_task` returns plus: `t`, `universe` (index of tokens), `history_matrix` (binary, users × universe, trades in `[t − history_length, t)`; `None` means since the data start), `past` (binary, users × universe, whole past, used for exclusion and the label), `window_size` (tokens per user in the reference window), `last_timestamp` (per user-token pair, for time weighting). Users are indexed by the reference window plus the history; tokens by the universe. **Asserts:** no row of `legs` with timestamp ≥ *t* enters any history object; positives are disjoint from `past`; the evaluated set does not depend on `history_length`. |
| `evaluate_point(model, task) -> pd.DataFrame` | Fits nothing: the caller fits. Scores evaluated users in batches, excludes `past`, takes top 10, returns one row per user with `recall`, `ndcg`, `hit`, `recall_no_head`, `window_size`, and the lists in `.attrs`. |
| `run(model_factory, legs, part, **task_kwargs) -> pd.DataFrame` | For each point of `part`: build the task, `model = model_factory(task)` (fits on the task's history), evaluate. Returns one row per (point, user). Caches tasks on disk under `data/project/tasks/` keyed by (*t*, history length), since tuning rebuilds the same tasks many times. |
| `summarize(per_user) -> pd.DataFrame` | One row per point: mean recall, NDCG, hit rate, coverage, recall without the head, number of evaluated users, positive pairs, share of cold pairs, and the within-point bootstrap interval of recall. Also one row per (point, history group). |
| `head_tokens(history_matrix, share=1/3)` | The most popular tokens that together hold `share` of the history's interactions. |

### 4.2 `stats.py`

| Function | Spec |
| --- | --- |
| `paired_differences(a, b, metric="recall")` | Joins two `summarize` tables on point; returns *d_p*. Raises if the point sets differ. |
| `bootstrap_over_points(d, n_boot=10_000, seed=0, level=0.95)` | Percentile interval of the mean of *d*. |
| `sign_test(d)` | Wins, losses, ties, two-sided p-value. |
| `compare(table, baseline="popularity")` | For every model: mean, lowest, highest, mean *d*, interval, wins out of points, p-value. This is the standard results table of every notebook. |

### 4.3 Tests

`POINTS` matches 1.1 by count and spacing; `make_task` on a toy log with known
answers (a trade one second before *t* is history, one at *t* is target, a
token first seen in the target is a cold pair, a repeat is not a positive);
the evaluated set is unchanged when `history_length` changes;
`bootstrap_over_points` on constant input gives a zero-width interval;
`sign_test(12 wins of 14)` gives p < 0.05 and 11 of 14 does not.

### 4.4 Notebook `10_protocol_and_baseline.ipynb`

Popularity on the 14-day reference window (the fixed baseline, PDF 5.4) and
recent popularity (last day) on all 21 development points; per-point table and
chart; by history size; the sizes of PDF Table 7 per point; share of cold
pairs. Writes `results/wp2/baseline.csv`.

**Done when:** the baseline table exists on 21 points and every later
notebook can produce a `compare` table with one call.

## 5. WP3 Nearest neighbours (core problem 1)

**Goal:** a tuned fixed combination *w·p̃ + (1 − w)·r̃*, compared with the
baseline on Dev B.

- **Model.** `Blend(Popularity on the 14-day window, ItemKNN(k, λ), w)`, all
  existing code. The neighbour model is fitted on the same 14-day history in
  this package; history length belongs to WP4.
- **Grid on Dev A** (fit once per (*k*, λ) per point, the blend weight is cheap):

  | Setting | Values |
  | --- | --- |
  | *k* | 10, 25, 50, 100, 200 |
  | λ | 0, 1, 5, 10, 25, 50 |
  | *w* | 0, 0.1, 0.2, …, 1.0 (0 is neighbours alone, 1 is the baseline) |

- **Choice rule.** Highest mean recall@10 over the 10 Dev A points. If two
  settings are within 0.002, take the simpler one: smaller *k*, then larger
  λ, then larger *w*. Written to `configs/wp3_knn.json`.
- **Reported on Dev B:** `compare` table against the baseline; by history
  size; coverage; recall without the head.
- **Notebook** `11_nearest_neighbours.ipynb`; results in `results/wp3/`.
- **Tests:** existing `ItemKNN` and `Blend` tests; add one that `Blend` with
  `w = 1` gives exactly the popularity ranking.

## 6. WP4 Time weighting (core problem 2)

**Goal:** a half-life *h* and a history length *L*, and their effect.

### 6.1 Weighting

An interaction (*u*, *i*) gets the weight *ω_ui* = 2^(−a/h), where *a* is the
time from *u*'s **last** trade of *i* to *t*, in days. With *h* = ∞ every weight
is 1 (the WP3 model).

| Place | Formula |
| --- | --- |
| Popularity | *p_i* = Σ_u *ω_ui* |
| Similarity | the shrunk cosine of `ItemKNN` computed on the weighted matrix *W* = [*ω_ui*] |
| User profile | *r_ui* = Σ_{j ∈ H_u} *ω_uj s_ji* |

### 6.2 Code in `pool_recsys.py`

| Function / class | Spec |
| --- | --- |
| `time_weights(last_timestamp, t, half_life_days)` | Sparse matrix of *ω* with the same pattern as the binary history. `half_life_days=None` means ∞. Asserts all weights in (0, 1]. |
| `TimeWeightedPopularity(half_life)` | `fit(task)` uses the weighted column sums. |
| `TimeWeightedItemKNN(k, shrink, half_life)` | Similarity on *W*; `scores` multiplies the weighted user rows by it. |

The existing `fit(matrix)` interface is kept: the weighted matrix is passed as
`matrix`, so `Blend`, `top_k` and `evaluate` need no change.

### 6.3 Grid on Dev A

| Setting | Values |
| --- | --- |
| *h* (days) | 0.5, 1, 2, 4, 7, 14, ∞ |
| *L* (days) | 7, 14, 28, all |
| *k*, λ, *w* | the WP3 values first; then *w* re-tuned on the same grid |

Choice rule as in WP3, ties to the larger *h*. All-time popularity (*h* = ∞,
*L* = all) and popularity of the last day are reported as the two ends.
**Constraint:** at the first Dev A point only 14 days exist, so *L* = 28 and
*L* = all are equal to *L* = 14 there (see 11.1). Notebook
`12_time_weighting.ipynb`; config `configs/wp4_time.json`; results `results/wp4/`.

**Tests:** weights at age 0 are 1, at age *h* are 0.5; *h* = ∞ reproduces
`ItemKNN` and `Popularity` exactly.

## 7. WP5 Cold start (core problem 3)

**Goal:** the share of cold pairs that becomes reachable when lists are
refreshed, with token features and trivial heuristics as references. This
package is independent of WP3 and WP4 and can run in parallel with them.

### 7.1 Refreshed protocol

- At a point *t*, lists are rebuilt at *t*, *t + R*, *t + 2R*, …, up to *t* + 2
  days, with refresh interval *R* ∈ {1, 3, 6, 12} hours.
- Each positive pair (*u*, *i*) with first trade at time τ is scored against
  the list of *u* built at the last refresh time ≤ τ. The pair is a hit if *i*
  is in that list. The universe and the exclusion grow with each refresh (all
  data before the refresh time).
- **Pair-level recall@10:** the share of positive pairs that are hits,
  averaged over evaluated users as in 1.3. Reported for all pairs and for cold
  pairs alone. Under *R* = 2 days it equals the main protocol (test).
- The baseline is refreshed the same way, so the comparison is fair.

### 7.2 Token features (from the data only)

| Feature | Source |
| --- | --- |
| Age at the refresh time | `tokens.parquet` |
| Users in the first hour, first day | legs |
| Base currency of the launch pool (WETH, USDC, USDT, other) | `tokens.parquet` |
| Deploying account | **Not in the data.** See 11.3. |

### 7.3 Models compared

| Model | Spec |
| --- | --- |
| Baseline, refreshed | Popularity on the 14-day window before each refresh |
| Newest-popular (trivial heuristic, Schein 2002) | Among tokens younger than *A* hours, the most users since launch; *A* ∈ {6, 24} |
| Mixed list | The best WP3/WP4 model's top 10 − *m*, plus the top *m* tokens of the newest-popular list, *m* ∈ {1, 2, 3} |
| Feature score | Logistic regression on the token features of 7.2, trained on Dev A pairs, used in place of newest-popular in the mixed list |

`cold_start.py` holds `refresh_times`, `refreshed_evaluate(model_factory, legs, t, R)`,
`token_features(legs, tokens, at)` and `newest_popular(...)`. Tests on a toy log
where a token launches inside the target window. Notebook
`13_cold_start.ipynb`; results `results/wp5/`. Settings (*R*, *A*, *m*) chosen
on Dev A, reported on Dev B.

## 8. WP6 Learned ranker

**Goal:** a ranker over the signals of WP3 to WP5, compared with the best fixed
combination on Dev B; the ablation by feature group.

### 8.1 Candidates

For each evaluated user: the union of the top *N* tokens by popularity and the
top *N* by neighbour score (best WP4 settings), past excluded, *N* = 200.
**Candidate recall** (share of positive pairs, cold pairs included, that are
among the candidates) is measured on Dev A first, for *N* = 200 and *N* = 500.
If 500 adds more than 0.02 to candidate recall, *N* = 500 is used. Candidate
recall is the ceiling of the ranker and is reported.

### 8.2 Features (all computed from data before *t*)

| Group | Features | Core problem |
| --- | --- | --- |
| Popularity | users of the token in the 14-day window, in the last day, with *h* = 1, 7 days; rank of the token by popularity | 2 |
| Neighbours | *r_ui* plain and time-weighted; the largest single *s_ji*; the number of the user's tokens with *s_ji* > 0 | 1, 2 |
| User | tokens in the reference window, tokens in the past, days since the last trade, share of buys | 1 |
| Token | age (days, with the `censored` flag), users in the first hour and first day, launch base (one-hot: WETH, USDC, USDT, other) | 3 |

### 8.3 Training rows

- One row per (point, user, candidate). Label 1 for a positive pair.
- Dev A: 10 points × about 8,000 users × up to 400 candidates ≈ 30 million
  rows. For **training only**, all positives are kept and 50 negatives per
  (point, user) are sampled with seed 0. Scoring on Dev B and test always uses
  the full candidate set.
- Rows are written to `data/project/ranker/rows_<t>.parquet`, one file per point.

### 8.4 Models

| Order | Model | Library | Settings searched |
| --- | --- | --- | --- |
| 1 | Logistic regression on standardized features | scikit-learn | *C* ∈ {0.01, 0.1, 1, 10} |
| 2 | Boosted trees, logistic loss | LightGBM `binary` | leaves ∈ {7, 15, 31}, learning rate ∈ {0.03, 0.1}, min rows per leaf ∈ {100, 1000}; trees by early stopping |
| 3 | Boosted trees, LambdaMART | LightGBM `lambdarank`, group = (point, user), metric NDCG@10 | as 2 |

- **Cross-validation:** 5 folds of 2 consecutive Dev A points each. A fold's
  points are held out whole. The score of a setting is the mean recall@10 over
  the held-out points. After the choice, the model is refitted on all 10 Dev A
  points with the chosen number of trees.
- **Selection on Dev B** (paired, `stats.compare`): baseline, best fixed
  combination (WP3 or WP4), and the three rankers.
- **Ablation on Dev B:** the chosen ranker retrained without one feature group
  at a time (4 runs) and with only the popularity group.

`ranker.py` holds `candidates`, `features`, `training_rows`, `fit_logistic`,
`fit_gbdt`, `cv_over_points`, `rank`. Tests: candidates never contain the
user's past; every feature at point *t* is unchanged when trades at or after
*t* are deleted from the input (the leakage test); `cv_over_points` never puts
rows of one point in both training and validation. Notebook
`14_learned_ranker.ipynb`; config `configs/wp6_ranker.json`; model files in
`data/project/models/`.

## 9. WP7 Test and write-up

### 9.1 Freezing

Before the test period is touched:

1. `configs/frozen.json` names **one primary model** (the one the success
   criterion is about) and the secondary models to report, with all settings
   and the paths of trained model files. Chosen on Dev B by the largest mean
   paired difference against the baseline.
2. All tests pass; the working tree is clean; the commit is tagged
   `test-freeze`.

### 9.2 `run_test.py`

- Refuses to run if the tree is dirty, `HEAD` is not the `test-freeze` tag, or
  `results/test/` already exists.
- Builds the 14 test tasks, scores every frozen model, writes
  `results/test/per_point.csv`, `results/test/compare.csv`, the commit hash
  and the start and end time to `results/test/run.json`.
- Prints whether the success criterion of 1.4 holds for the primary model.
- Models are not refitted on Dev B: the ranker is the one trained on Dev A, as
  the PDF says. The first-stage models are fitted at each test point on its own
  history, which is part of the method, not tuning.

### 9.3 Notebook `15_test_results.ipynb` and the report

Reads `results/test/` only: per-point table and chart, the `compare` table, by
history size, without the head, the cold-start result, the ablation, and an
error analysis (which tokens were missed most; recall by token age; users with
many tokens). Then the final report.

## 10. Order of work, effort and checkpoints

| Step | Depends on | Rough effort | Checkpoint |
| --- | --- | --- | --- |
| WP1 code (load filter, validation, token data) | none | 1 day | tests pass |
| WP1 download | WP1 code not needed; can start now | 4–18 h unattended | 605 chunks |
| WP1 build and validate | both above | 0.5 day | notebook 09, validation passed |
| WP2 protocol and stats | WP1 | 2 days | baseline on 21 points |
| WP3 neighbours | WP2 | 1–2 days | `configs/wp3_knn.json` |
| WP5 cold start | WP2 | 3 days | in parallel with WP3/WP4 |
| WP4 time weighting | WP3 | 1–2 days | `configs/wp4_time.json` |
| WP6 ranker | WP3, WP4, WP5 | 4 days | Dev B comparison and ablation |
| WP7 freeze, test, write-up | WP3–WP6 | 3 days | tag `test-freeze`, `results/test/` |
| WP8 strategy backtest (extension, section 12) | WP7 (frozen recommender) | 1.5 weeks + 1–2 days of download | tag `strategy-freeze`, `results/wp8/heldout/` |

- **Minimum complete result:** WP1 to WP3 and WP7. That is a tuned neighbour
  model compared with the baseline on 14 clean test points. WP4 to WP6 add to
  it but are not needed for a valid result.
- **Start the download first.** It needs no new code and is the longest step.
- Branch per work package; each merges into `main` only when its tests pass
  and its notebook runs from top to bottom with outputs saved.

## 11. Open points found while writing this plan

These are decisions the PDF leaves open or that turn out not to work as
written. Each needs a decision before the package that uses it, recorded in
`project_direction.md` under "Changes to the plan".

1. **History length on Dev A is capped by the burn-in.** The first Dev A point
   has 14 days behind it, the last 32. Settings *L* = 28 and *L* = all
   therefore differ only at the later points, so the effect of *L* is mixed
   with the position of the point. Proposal: tune *L* on all Dev A points as
   planned, and also report the effect of *L* on Dev B alone, where every
   point has at least 34 days.
2. **The label uses a past that grows.** "Never traded before *t*" with the
   whole past means early points count more pairs as new than late points,
   because older trades are invisible to them. It does not leak, but it adds a
   trend across points. Proposal: keep the PDF definition and report, per
   point, the share of positives whose token the user traded before the
   reference window; if the trend is large, add the reference-window label as
   a variant.
3. **The deploying account is not in the data.** The swaps do not say who
   created a token contract, and finding the creator through the free RPC
   endpoints needs a trace or an indexer. Proposal: drop the feature and
   record the change; the launch pool and early users remain.
4. **"The proposed method" must be one model.** The PDF lists a fixed
   combination and three rankers. Testing several against the baseline and
   claiming success for the best would be a multiple comparison. Proposal: as
   in 9.1, name one primary model before the test; report the others as
   secondary.
5. **The sign test is weak at 14 points.** A model that truly wins 80% of
   points passes 12 of 14 only 45% of the time. The success criterion uses the
   bootstrap interval, not the sign test; the sign test is reported alongside
   it. A bootstrap over 14 points is itself rough, so the 14 differences are
   always shown.
6. **`load_swaps()` with no arguments loads the exploration chunks.** Every
   project notebook must call it with the project block range (3.2), or read
   `data/project/legs.parquet`, which is the recommended path.

## 12. WP8 Trading strategy backtest (extension)

**Goal:** test whether the recommender's gain over the popularity baseline is
worth money: a strategy that buys tail tokens with high predicted new demand,
backtested after costs against strategies that use only visible demand. The
strategy, its rationale and its traps are described in
[`trading_strategy_backtest.md`](trading_strategy_backtest.md); this section
fixes how it is built and judged. Added on 23 November 2026, after the
project's test was scored (see `project_direction.md`, "Changes to the plan").

**What does not change.** The recommender is the one frozen in WP7
(`configs/frozen.json`): the untimed fixed blend for the tail strategy, the
LambdaMART ranker for the whole-market variant. WP8 tunes only the strategy's
own settings.

### 12.1 Periods

| Part | Period (2026) | Use |
| --- | --- | --- |
| Strategy development | Dev A and Dev B (2 July – 13 August) | Tune the settings of 12.4 |
| Strategy choice | the project's test period (13 August – 10 September) | Choose between the tail strategy, the whole-market variant and their settings. This period was scored for recall in WP7 but never for returns. |
| History only | 24 September – 8 October (the feasibility window, downloaded earlier) | 14-day history for the second held-out part; never evaluated |
| **Held-out** | **10 – 24 September and 8 October – 5 November** | Scored once, from a commit tagged `strategy-freeze`. Never downloaded or looked at before WP8. |

Decision times are every 6 hours. The bootstrap of 12.6 resamples **days**,
not decisions, because positions overlap and trades on one day are not
independent. The held-out part has 42 days.

### 12.2 Data

| Step | Detail | Code |
| --- | --- | --- |
| Download | swaps for 10 Sep – 24 Sep and 8 Oct – 5 Nov with the WP1 pipeline (`dex_swaps.py`), validated per week as in WP1 | existing |
| Token decimals | `decimals()` of every token through Multicall3, as for the pool tokens; tokens whose call fails are excluded and counted | `prices.py` |
| V3 fee tiers | `fee()` of every V3 pool through Multicall3; V2 pools are 0.3% | `prices.py` |
| ETH price | per block from the WETH/USDC and WETH/USDT pools: volume-weighted over the block's swaps, carried forward | `prices.py` |
| Price per swap | for each swap with a base token: base amount / token amount, using the decimals and the sign convention of V2 and V3 events | `prices.py` |
| Buy volume per account | each account's buys in ETH over the 14 days before $s$ | `strategy.py` |

Price impact is estimated from the data, because pool reserves are not
downloaded: for a trade of size $q$ in a pool, the median relative price move
from the swap before to the swap after, over observed swaps of size near $q$ in
that pool in the 7 days before. Downloading V2 `Sync` events and V3 `slot0` for
an exact impact is optional (12.7).

### 12.3 Code

| Module | Functions | Spec |
| --- | --- | --- |
| `prices.py` | `token_decimals`, `pool_fees`, `eth_price_per_block`, `swap_prices`, `impact_estimate` | 12.2 |
| `strategy.py` | `tail_tokens(task)`, `predicted_inflow(task, lists, hit_rates, activity)`, `surprise(inflow, recent_volume)`, `universe(...)`, `select(scores, k)` | the signal and the rules of `trading_strategy_backtest.md` |
| `backtest.py` | `run(decisions, prices, costs, settings) -> trades, periods` | opens positions at the first swap at least one block after $s$; exits at the first swap after $s + H$ or after the stop is crossed; a position that cannot be exited is worth zero; returns one row per trade and one row per 6-hour period with all open positions marked to the last swap price |

**Tests** (toy pools and swaps written by hand):

- a swap's price is the same whichever token is `token0`, for V2 and V3 sign conventions;
- entry never uses a swap in or before the decision block (the look-ahead test: deleting every swap after $s$ leaves the signal unchanged, as in the ranker's leakage test);
- the stop exits at the first swap after the fall, not at the stop level;
- a token with no swap after entry is valued at zero;
- costs: a round trip in a 0.3% pool with no price change loses 0.6% plus gas;
- the head is removed exactly as `protocol.head_tokens` defines it.

### 12.4 Settings tuned on development data

| Setting | Values |
| --- | --- |
| $K$ tokens per decision | 3, 5, 10 |
| $H$ holding time | 6, 24, 48 hours |
| Stop | none, −50% |
| Activity weight $a_u$ | 1 (count accounts), buy volume |
| Position cap | 1% of the pool's 24-hour buy volume (fixed, not tuned) |

Choice rule on development data: the highest mean net return per period minus
that of tail popularity (12.5); ties within one standard error go to the
simpler setting (smaller $K$, then $a_u = 1$, then no stop).

### 12.5 Benchmarks

Hold ETH; popularity; momentum; flow; random (all over the whole market); and
**tail popularity** and **tail flow** (the same tail universe chosen by visible
demand). The **primary comparison** is the tail strategy against **tail
popularity**: it isolates the recommender's information from the universe it
trades in. All others are secondary.

### 12.6 Success criterion (stated before the held-out period is touched)

On the 42 held-out days: the mean net return per 6-hour period of the tail
strategy minus that of tail popularity is positive, and its 95% bootstrap
interval over days (10,000 resamples, seed 0) excludes zero. Reported
alongside, never instead: the same before costs, the capacity (capital usable
under the position cap), the return with and without tokens whose buyers are
mostly in coordinated groups, and every benchmark.

### 12.7 Notebooks, configs and order

| Step | Output | Effort |
| --- | --- | --- |
| Download and validate the held-out weeks (start first, unattended) | `data/dex_swaps/chunks/`, validation table | 1–2 days unattended |
| `prices.py` with tests; notebook `16_prices.ipynb` (prices, ETH price, impact estimate, checked against known pools) | price tables in `data/project/prices/` | 1 day |
| `strategy.py` with tests; notebook `17_demand_signal.ipynb`: does predicted inflow match realized new buyers, and do realized buyers move prices? (both links of the strategy, on development data) | `results/wp8/signal_dev.csv` | 1 day |
| `backtest.py` with tests; notebook `18_backtest_development.ipynb`: settings on Dev A/B, choice on the project's test period | `configs/wp8_strategy.json` | 2–3 days |
| Freeze (`configs/wp8_frozen.json`, tag `strategy-freeze`), then `run_backtest.py` on the held-out days once; notebook `19_backtest_heldout.ipynb` reads `results/wp8/heldout/` only | the verdict of 12.6 | 1 day |
| Optional: `Sync`/`slot0` download for exact price impact | re-run of 18 and 19 with exact impact, reported next to the estimate | 4–18 h unattended |

