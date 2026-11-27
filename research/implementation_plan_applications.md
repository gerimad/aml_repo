# Implementation plan: applications of the recommender

How eight applications of the project's recommender are built, checked and
judged. They use what the project already produced: the frozen models of
`configs/frozen.json`, the neighbour similarities, the per-point tasks, and
the farm groups found in notebook 11. Each is a work package with its own
code, tests, notebook, success criterion and stated limits.

| WP | Application | What it delivers | Ground truth |
| --- | --- | --- | --- |
| 9 | Expectedness scores (shared base) | for every trade at every point: how far down the account's ranking the traded token was | exact (the model's own ranking) |
| 10 | Farm detection from behaviour | groups of accounts that act as one, beyond identical 14-day sets | none; checked by held-out co-movement |
| 11 | Account segmentation | a small set of trader types with profiles | none; checked by stability and usefulness |
| 12 | Token similarity map | clusters of tokens and a "related tokens" panel | partial: held-out co-adoption |
| 13 | Scam-push detection | a token risk score from adoption the model cannot explain | exact outcome from prices (collapse) |
| 14 | Unusual-activity alerts | a per-account alert for trades far outside its own pattern | weak: needs transfer logs and a stated rule |
| 15 | Likely buyers of a token | for a token, the accounts most likely to pick it up (airdrop or promotion audience) | exact (who buys in the next two days) |
| 16 | Safer recommendations | how often lists contain tokens that collapse, and a re-ranking that lowers it | exact outcome from prices (collapse) |
| 17 | Forecasting new buyers | the number of new buyers per token in the next 48 hours | exact (new buyers) |
| 18 | Real-time lists on the test | the frozen time-weighted blend rebuilt every hour, scored on the test | exact (the project's label) |
| 19 | Buyers of brand-new tokens | one hour after a launch, the accounts most likely to buy next, from the first buyers | exact (who buys in the next 24 hours) |

**Rules that carry over from the project** ([`project_direction.md`](project_direction.md)):
simple, explained code; reusable logic in `src/scripts/` with tests;
notebooks for results; every choice made on development data and every claim
checked once on data not used for it; changes recorded under "Changes to the
plan".

**Privacy.** The data is public and pseudonymous, but 13 and 14 label
accounts and tokens as suspicious. Results are reported in aggregate. No
list of flagged addresses is published, in the repository or in a page;
examples in notebooks show shortened addresses only, and case reviews stay
local.

## 0. Periods and data

| Part | Period (2026) | Use |
| --- | --- | --- |
| History | 18 June – 2 July (burn-in) | history only |
| Fit | Dev A, 2 – 22 July | fit clusterings, choose thresholds and settings |
| Check | Dev B, 22 July – 13 August | choose between versions |
| Report | the project's test period, 13 August – 10 September | report once per work package |
| Extra held-out | 10 – 24 September, 8 October – 5 November (downloading for WP8) | second report where a package needs more points |

None of these packages changes a project result. Where one re-scores the
recommender (WP10's sensitivity analysis), it is a reported sensitivity, not
a new test.

| Data | Have? | Needed by |
| --- | --- | --- |
| Legs, tasks, frozen models, similarities | yes | all |
| Token symbols and decimals | yes (`dex_swaps.load_token_metadata`) | 12, 13 |
| Swap `tx_from`, `recipient`, position in block | yes, in the raw chunks; not in the legs | 10, 14 |
| Prices per swap | from WP8's `prices.py` | 13 |
| ERC-20 `Transfer` logs of the tokens involved | **no**; download, as for swaps | 14 |

Swap recipients are mostly router contracts: in a sample chunk the recipient
equals the signing account in only 12% of swaps. Where tokens finally go
therefore cannot be read from swaps; WP14 needs transfer logs.

**New dependency:** `networkx` (pure Python) for community detection.
Everything else uses scikit-learn and SciPy, which are installed.

---

## 9. WP9 Expectedness scores (shared base)

**Goal:** for every positive pair (account, token) at every evaluation point,
how expected the trade was under the recommender. WP13 and WP14 are built on
it; WP11 uses it as a feature.

- **Definition.** At point $t$, the frozen fixed blend scores every token of
  the universe for the account. The **rank percentile** of a picked-up token
  is its rank among the tokens the account has not traded, divided by their
  number: 0 is the model's first choice, 1 its last. Cold pairs (token did not
  exist at $t$) get no rank and are kept as their own group.
- **Why the fixed blend, not the ranker.** It scores every token (the ranker
  only scores ~680 candidates and puts the rest at one level), it is refitted
  at each point, and on the test it is as good as the ranker.
- **Also stored:** whether the token is in the popular head; the largest
  similarity between the token and any of the account's tokens (0 means no
  co-trading path); the token's popularity rank.

| Code | Spec |
| --- | --- |
| `expectedness.py`: `rank_percentiles(task, scorer, pairs)` | ranks computed batch by batch from the scorer's dense scores, past tokens excluded exactly as in `protocol.evaluate_point` |
| `expectedness.py`: `pair_table(legs, task, scorer)` | one row per positive pair: point, user, token, cold, rank_percentile, in_head, max_similarity, popularity_rank, first_trade_time |

**Tests:** a token ranked first has percentile 0 and one ranked last 1; past
tokens never count in the denominator; on a toy market the percentile agrees
with `protocol.evaluate_point`'s hits (a pair is a hit exactly when its rank
is below 10); deleting trades after $t$ changes nothing (leakage test).

**Output:** `results/wp9/pairs_<part>.parquet` for Dev A, Dev B and the test
(local, ~300k rows per part, not committed); notebook `20_expectedness.ipynb`
with the distribution of percentiles by part, by token age, and for hits
against misses.

**Effort:** 1 day.

---

## 10. WP10 Farm detection from behaviour

**Goal:** find groups of accounts that act as one, beyond the identical
14-day token sets of notebook 11, and measure how much they affect the
project's evaluation.

### 10.1 Method

1. **Co-action graph.** Two accounts are linked when, in a window of 14 days,
   they first-traded at least $m$ of the same tokens, each time within
   $\Delta$ minutes of each other. Weight = the number of such shared first
   trades. Built with a sorted sweep per token (accounts' first trades of a
   token in time order), so it never forms all pairs.
2. **Groups.** Connected components of the graph with weight ≥ $m$, then
   Louvain communities (`networkx`) inside components larger than 500
   accounts, so that one weak link does not merge two farms.
3. **Group features** (for the report, not the definition): size, share of
   members with one token, spread of first-trade times, number of distinct
   signing contracts (`tx_to`), share of members funded by the same address
   (not available without transfer logs; left out and stated).

Settings tuned on Dev A: $\Delta \in \{5, 30, 120\}$ minutes, $m \in \{2, 3, 5\}$.

### 10.2 Check without labels

There is no list of known farms for 2026. A grouping is real if it **keeps
acting together on data it was not built from**:

- Groups are built on Dev A. For each group, the **co-action rate on Dev B**:
  of the pairs of members that both trade in Dev B, the share that first-trade
  a common token within $\Delta$ minutes.
- Compared with **matched random pairs**: pairs of accounts with the same
  number of trades and the same first-trade days, not in a common group.
- **Success criterion (stated now):** the median group's co-action rate on
  Dev B is at least 10 times that of matched random pairs, with a bootstrap
  interval over groups whose lower end is above 10. Reported again on the test
  period with groups built on Dev A + Dev B.
- Notebook 11's identical-set rule is reported side by side as the baseline
  method: how many of its groups the co-action graph recovers, and how many
  more accounts it finds.

### 10.3 Effect on the project's evaluation (sensitivity)

The project's Dev B and test comparisons are recomputed with farm members
removed from the evaluated accounts, every model alike. Reported as a
sensitivity analysis next to the scored result, never in place of it:
recall of each model, the gain over the baseline, and the ranker against the
fixed blend. This answers the report's main limitation.

| Code | Spec |
| --- | --- |
| `farms.py`: `first_trades(legs, start, end)` | each account's first trade of each token in a window, with time and `tx_to` |
| `farms.py`: `coaction_edges(first, delta, min_shared)` | sparse weighted account graph |
| `farms.py`: `groups(edges, max_component)` | group id per account |
| `farms.py`: `coaction_rate(groups, first_later, delta)` and `matched_pairs(...)` | the check of 10.2 |

**Tests:** two accounts with three shared first trades 2 minutes apart are
linked for $\Delta$ = 5 and $m$ = 3, not for $\Delta$ = 1; a single shared
trade never links; matched pairs have equal trade counts; the sweep gives the
same edges as a brute-force pair loop on a small market.

**Notebook:** `21_farms.ipynb`. **Effort:** 2–3 days.

---

## 11. WP11 Account segmentation

**Goal:** a small number of trader types that differ in behaviour and in how
well the recommender serves them, with a profile of each.

### 11.1 Features (per account, at a point, from the past only)

| Group | Features |
| --- | --- |
| Activity | trades and tokens in the last 14 days; days active; days since the first trade in the data |
| What it trades | share of tokens in the popular head; median age of tokens at the account's first trade of them; share of trades in tokens younger than a day |
| How it trades | median time from first buy to first sell of a token; share of buys; trades per active day |
| What the model expects | median rank percentile of its pick-ups (WP9, from the previous point, so not leaking) |
| Farm | member of a WP10 group (a label used for the profile, not a clustering feature) |

### 11.2 Method

Standardized features (log for counts), then k-means and a Gaussian mixture
for $k$ from 3 to 10 on Dev A accounts. The choice of $k$: silhouette, and
**stability**: the adjusted Rand index between clusterings of two random
halves of the accounts, assigned to the nearest centre. The smallest $k$
within one standard error of the best stability is used.

### 11.3 What counts as useful

- **Stable over time:** the same account's segment at Dev A and at Dev B
  agrees more than chance (adjusted Rand index reported); segment sizes do
  not swing by more than a factor of 2.
- **Different for the recommender:** recall of the fixed blend and of the
  baseline per segment on Dev B.
- **Success criterion (stated now):** a fixed blend whose popularity weight
  $w$ is chosen per segment on Dev A beats the single-$w$ blend on Dev B,
  paired by point, with a 95% interval that excludes zero. If it does, this is
  the one place where the applications could improve the recommender itself;
  it is then reported on the test once.

| Code | Spec |
| --- | --- |
| `segments.py`: `account_features(legs, task, pairs_previous)` | the table of 11.1 |
| `segments.py`: `fit(features, k, kind)`, `assign(model, features)` | k-means or mixture on standardized features |
| `segments.py`: `split_stability(features, k, seed)` | adjusted Rand index of two halves |
| `segments.py`: `per_segment_blend(task, segments, weights)` | the blend with one $w$ per segment |

**Tests:** features use only legs before $t$ (leakage test); a toy set of two
separated clouds gives two segments and stability 1; `per_segment_blend` with
the same $w$ everywhere equals the plain blend.

**Notebook:** `22_segments.ipynb` (profiles as a table of feature medians per
segment, named after the fact by their profile, e.g. "one-token sellers").
**Effort:** 2 days.

---

## 12. WP12 Token similarity map

**Goal:** a map of tokens by who trades them together: clusters
(ecosystems, copycats, farm targets) and a "related tokens" list per token.

### 12.1 Method

- **Similarity:** the frozen neighbour model's similarity on the 14-day
  window ($k$ = 400, $\lambda$ = 200), symmetrized, at each evaluation point.
- **Clusters:** Louvain communities (`networkx`) on the similarity graph of
  tokens with at least 20 accounts in the window. Resolution tuned on Dev A
  for stability (below).
- **Map:** a 2-D layout by t-SNE (scikit-learn) on the similarity rows of the
  top 3,000 tokens by accounts, for the explorer page.
- **Related tokens of token $i$:** its 10 most similar tokens.
- **Copycats:** pairs of tokens in the same cluster whose symbols are equal or
  one edit apart (symbols from `load_token_metadata`), with their launch times.
- **Farm targets:** clusters whose accounts are mostly WP10 group members.

### 12.2 Checks

- **Stability:** adjusted Rand index of the clusters of tokens present at two
  points two weeks apart; the resolution with the best stability on Dev A is
  used.
- **Related tokens predict co-adoption** (held-out): for an account that
  picks up token $i$ for the first time on Dev B, is its *next* new token among
  $i$'s 10 related tokens? Compared with the 10 most popular tokens as the
  baseline list. **Success criterion (stated now):** the hit rate of the
  related list beats the popular list, paired by point, interval excluding
  zero, on Dev B; reported on the test.
- **Copycats:** counts reported, with the share of copycat tokens that
  collapse (from WP13's label) against other tokens of the same age.

| Code | Spec |
| --- | --- |
| `token_map.py`: `similarity_graph(task, settings, min_accounts)` | symmetrized sparse graph |
| `token_map.py`: `clusters(graph, resolution, seed)` | cluster id per token |
| `token_map.py`: `layout(graph, n, seed)` | x, y per token |
| `token_map.py`: `related(graph, token, n)` | the related list |
| `token_map.py`: `copycats(clusters, symbols, max_edit)` | candidate pairs |

**Tests:** two disjoint groups of co-traded tokens form two clusters; the
graph is symmetric; `related` never returns the token itself; edit distance
on symbols ignores case.

**Notebook:** `23_token_map.ipynb`; a fourth view "Token map" in
`explorer/`. **Effort:** 2 days.

---

## 13. WP13 Scam-push detection

**Goal:** a token-level score for adoption that the recommender cannot
explain, and a test of whether it warns of collapse.

### 13.1 The score

For token $i$ in the 24 hours after a decision time $s$ (6-hourly), among
accounts that first-trade $i$:

- **unexplained share** = share of these accounts for which $i$ had no
  co-trading path (max similarity 0, WP9) and was not in the popular head;
- **low-rank share** = share whose rank percentile of $i$ is above 0.9;
- **burst** = new accounts in the last hour divided by new accounts in the
  last 24 hours;
- **farm share** = share of new accounts in WP10 groups.

The score is a logistic regression on these four, trained on Dev A tokens.

### 13.2 The outcome (exact, from prices)

**Collapse:** the token's price in its most active base pool (WP8's
`prices.py`) falls more than 90% below its peak since $s$ within the next 72
hours, and its trading stops (fewer than 5 swaps in the last 24 of those
hours). Defined from prices only, so it does not reuse any feature of the
score (the lesson of the earlier memecoin study, where the label contained the
predictor).

### 13.3 Baseline and criterion

- **Baseline score:** logistic regression on what anyone sees without the
  recommender: token age, accounts so far, accounts in the last hour, buy
  share of trades.
- **Success criterion (stated now):** adding the four recommender-based
  features raises the AUC for collapse on Dev B, and on the test, with a 95%
  bootstrap interval over days of the AUC difference that excludes zero.
  Precision among the top 1% of scored tokens is reported alongside.
- Only tokens with at least 20 new accounts in the 24 hours are scored (fewer
  give no share worth computing).

| Code | Spec |
| --- | --- |
| `scam_push.py`: `token_window_features(pairs, legs, s, farms)` | the four features and the baseline features per token |
| `scam_push.py`: `collapse_label(prices, token, s, hours)` | the outcome of 13.2 |
| `scam_push.py`: `auc_difference_by_day(scores_a, scores_b, labels, days)` | the bootstrap of 13.3 |

**Tests:** a token whose price falls 95% and stops trading is labelled
collapsed, one that falls 95% and recovers is not; features use only data
before $s$ + 24 h and labels only data after it; the bootstrap resamples days,
not tokens.

**Depends on:** WP9, WP10, WP8's `prices.py`. **Notebook:** `24_scam_push.ipynb`.
**Effort:** 2–3 days after `prices.py`.

---

## 14. WP14 Unusual-activity alerts for accounts

**Goal:** flag account activity that fits a compromised key: the account
suddenly trades tokens far outside its own pattern and its holdings leave to
an address it never used before.

### 14.1 Data first

The pattern "sold, then sent to a new address" needs where the proceeds
went. Swap recipients are mostly routers, so this needs **ERC-20 `Transfer`
logs** for the accounts and tokens involved and plain ETH transfers (the
latter need traces, which the free endpoints do not serve; WETH transfers
are covered by the logs). Plan: download `Transfer` logs filtered by the
WETH, USDC and USDT contracts (where proceeds usually go) for the project's
weeks, with the WP1 pipeline. This is the largest download of the
applications; size it first on one week.

### 14.2 The alert

At a 6-hourly decision time, for an account with at least 5 tokens in its
history (the model needs a pattern to deviate from):

- **deviation:** the mean rank percentile (WP9) of the account's first trades
  in the last 6 hours, against its own previous median;
- **liquidation:** the share of the account's tokens sold in those 6 hours;
- **exit:** the proceeds (WETH/USDC/USDT) transferred to an address the
  account never sent to before, within an hour of the sales;
- **silence:** no activity in the following 7 days (used only to describe
  alerts after the fact, never as an input).

An alert fires when deviation, liquidation and exit are all above
thresholds set on Dev A to fire for at most 1 in 10,000 account-days.

### 14.3 Checking it, honestly

There is no ground truth. What can be reported:

- the alert rate per period and its stability;
- the share of alerted accounts that go silent for 7 days, against matched
  accounts with the same liquidation but no deviation (does deviation add
  anything to "sold everything"?);
- a manual review of the 50 highest alerts on Dev B by the authors, each
  classified with a stated checklist (exit address receiving from many other
  accounts; exit address labelled on a public explorer; tokens involved);
  kept local.

No success criterion beyond "deviation adds to liquidation alone in
predicting silence, interval over days excluding zero", because the real
target (was this account compromised?) is not observable in the data.

| Code | Spec |
| --- | --- |
| `transfers.py`: download and load of `Transfer` logs, as in `dex_swaps.py` | |
| `alerts.py`: `account_window(pairs, legs, transfers, s)` | deviation, liquidation, exit, silence |
| `alerts.py`: `fire(window, thresholds)` | alert flags |

**Tests:** an account that sells all its tokens and sends the proceeds to a
fresh address within an hour fires with all three signals; one that sells and
keeps the proceeds does not; "new address" means never a recipient of the
account before $s$.

**Depends on:** WP9, the transfer download. **Notebook:** `25_alerts.ipynb`.
**Effort:** 3–4 days plus the download.

---

## 15. WP15 Likely buyers of a token (the recommender read by token)

**Goal:** for a token, rank the accounts most likely to pick it up in the
next two days. This is the project's score matrix read by column instead of by
row. Uses: choosing who receives an airdrop or a promotion, or whom a wallet
shows a token to. Added 24 November 2026 (`project_direction.md`).

### 15.1 Method

At each evaluation point $t$:

- **Audience:** the accounts with at least one token in their 14-day window
  that have never traded the token before $t$ (the same accounts a list could
  go to).
- **Tokens scored:** the tokens of the universe with at least 10 new buyers
  among the audience in the target window $[t, t + 2\text{ days})$.
  Conditioning on the outcome only chooses which tokens are scored, as the
  project chooses its evaluated accounts; every method is scored on the same
  tokens.
- **Model score:** the frozen fixed blend's score $s_{ui}$, read down the
  column of token $i$. The blend's popularity part is the same for every
  account (each part is scaled to a maximum of 1 per account, and popularity
  does not depend on the account), so the order comes from the neighbour part.
  Accounts with no co-trading path to $i$ tie at the bottom and are ordered by
  the activity baseline.
- **Baselines:** (a) **activity:** accounts ordered by the number of tokens in
  their 14-day window, the same order for every token (the obvious "send it to
  the most active accounts"); (b) **random** order.
- **Metrics per token:** recall@1000 (share of the token's new buyers in the
  first 1,000 accounts) and precision@100; mean over the scored tokens of a
  point. Reported separately for head and tail tokens (`protocol.head_tokens`).

The LambdaMART ranker is left out: it scores only 500 candidate tokens per
account, so most of a token's column is missing.

### 15.2 Criterion (stated now)

Nothing is tuned (the blend is frozen), so Dev B and test are both reported.
**Success:** on the 14 test points, the mean paired difference in
recall@1000 between the model and the activity baseline is positive, with a
95% bootstrap interval over points that excludes zero. Reported again with
the WP10 farm members removed from the audience and from the buyers.

| Code | Spec |
| --- | --- |
| `audience.py`: `token_columns(task, scorer, tokens, rows)` | the blend's scores for the given tokens and accounts, scored in account batches as in `evaluate_point` |
| `audience.py`: `rank_accounts(scores, tiebreak)` | the audience order per token |
| `audience.py`: `audience_metrics(order, buyers, k)` | recall@k and precision@k per token |

**Tests:** a token's column equals the matching entries of the row scores; an
account that traded the token before $t$ is never in its audience; with a
perfect score every buyer is in the first places (recall@k = 1 when buyers
≤ k); ties are broken by activity, not by account id.

**Depends on:** frozen models; WP10 for the farm-free version.
**Notebook:** `26_audience.ipynb`. **Effort:** 1 day.

---

## 16. WP16 Safer recommendations

**Goal:** measure, and then reduce, how often the recommender lists tokens
that collapse soon after. A wallet could not show recommendations that steer
people into rug pulls. The question is what removing them costs in recall.
Added 24 November 2026.

### 16.1 Measuring the harm

At each evaluation point, for the fixed blend's top-10 lists:

- **collapsed share:** the share of listed tokens that collapse within 72
  hours after $t$, by WP13's label (13.2, from prices only);
- **harmful hits:** the share of hits (listed and then bought) whose token
  collapses.

Both are reported for the frozen models on Dev B and test before anything
is changed.

### 16.2 The re-ranking

The blend's first 50 tokens per account are re-ordered by
$s_{ui} - \lambda \, r_i(t)$, with $s_{ui}$ scaled to a maximum of 1 per
account and $r_i(t)$ the token's collapse risk at $t$:

- $r_i(t)$ is WP13's chosen score, computed from the 24 hours before $t$ only.
  Tokens WP13 cannot score (fewer than 20 new accounts in those 24 hours) get
  the Dev A collapse rate of such tokens in their age band (under 1 day,
  1 – 7 days, over 7 days).
- $\lambda \in \{0, 0.1, 0.2, 0.5, 1, 2\}$. The WP13 score is trained on Dev
  A, so $\lambda$ is chosen on Dev B: the largest fall in collapsed share
  whose loss in recall@10 is at most 5% of the unchanged recall.

### 16.3 Criterion (stated now)

**Success:** on the test points, with the chosen $\lambda$, the collapsed
share falls (mean paired difference over points negative, 95% bootstrap
interval excluding zero) and recall@10 falls by at most 5% of its unchanged
value. The whole trade-off curve over $\lambda$ is reported in either case.
The last test points need prices up to 72 hours after 8 September. These
come from the downloaded held-out weeks and are used only as labels.

| Code | Spec |
| --- | --- |
| `safer.py`: `list_outcomes(lists, task, collapsed)` | collapsed share and harmful hits per point |
| `safer.py`: `rerank(scores, risk, lam, depth)` | the re-ordered top-10 |
| `safer.py`: `tradeoff(points, lams)` | recall and collapsed share per $\lambda$ and point |

**Tests:** $\lambda = 0$ gives the frozen lists exactly; a risk of 1 on one
token with a large $\lambda$ removes it from every list; re-ranking only
reorders the first 50 tokens; the risk at $t$ uses no trade at or after $t$.

**Depends on:** WP13 (score and label), WP8's `prices.py`.
**Notebook:** `27_safer_recommendations.ipynb`. **Effort:** 1–2 days after
WP13.

---

## 17. WP17 Forecasting a token's new buyers

**Goal:** forecast how many new buyers each active token gets in the next 48
hours, by adding up the accounts' predictions. Buyer counts drive volume and
fees, which matter to liquidity providers and token teams. WP8 showed that
predicted demand does not give trading profit. This asks only whether the
count itself is forecastable, which needs no prices. Added 24 November 2026.

### 17.1 Method

At each evaluation point $t$, for every token with at least one buyer in the
14 days before $t$:

- **Target:** $N_i$ = accounts whose first trade of $i$ ever falls in
  $[t, t + 2\text{ days})$, counted among accounts with a history before $t$
  (the accounts the recommender can see). The count among all accounts,
  brand-new ones included, is reported alongside.
- **Recommender feature:** $L_i$ = the number of accounts with $i$ in their
  top-50 list under the frozen fixed blend (the demand signal of notebook 17,
  for every active token).
- **Baseline features:** new buyers in the 48 hours before $t$ (persistence),
  accounts in the last 24 hours, and token age.
- **Models:** linear regressions of $\log(1 + N_i)$, (a) on the baseline
  features and (b) on the baseline features plus $\log(1 + L_i)$. Both are
  fitted on Dev A points.

### 17.2 Criterion (stated now)

**Success:** on the test points, model (b) has a lower mean absolute error in
$\log(1 + N_i)$ than model (a). The test is the mean paired difference over
points, with a 95% bootstrap interval that excludes zero. Dev B is reported
first. Spearman correlation per point and the error for the tail tokens
alone are reported alongside.

| Code | Spec |
| --- | --- |
| `demand_forecast.py`: `new_buyers(legs, t, hours, known_only)` | $N_i$ and the persistence feature |
| `demand_forecast.py`: `list_counts(task, scorer, depth)` | $L_i$ per token |
| `demand_forecast.py`: `fit_and_score(train, test, features)` | per-point errors of a model |

**Tests:** a buyer who traded the token before $t$ is not counted; the
persistence feature uses only the 48 hours before $t$; $L_i$ counts each
account at most once per token; tokens with no buyer in the 14 days are not
scored.

**Depends on:** frozen models; reuses notebook 17's new-buyer counting.
**Notebook:** `28_buyer_forecast.ipynb`. **Effort:** 1 day.

---

## 18. WP18 Real-time lists on the test

**Goal:** confirm on the test the largest gain seen in development. Notebook 13
measured on Dev B that rebuilding the lists every hour inside the two target
days lifts the time-weighted blend from 0.143 to 0.187, while the refreshed
baseline stays at 0.096. This was never scored on the test. Added 27 November
2026.

- **Models:** the frozen time-weighted blend (`configs/frozen.json`) and the
  popularity baseline, both rebuilt every hour with `cold_start.refreshed_evaluate`
  (the protocol of notebook 13, unchanged), and both at 48 hours (the main
  protocol) for the paired comparison. Nothing is tuned.
- **Criterion (stated now):** on the 14 test points, (a) the hourly blend
  beats the hourly baseline and (b) the hourly blend beats the same blend
  built once per two days; for both, the mean paired difference is positive
  with a 95% bootstrap interval over points that excludes zero.
- **Reported alongside:** the share of cold pairs reached, and the hourly
  blend against the scored test baseline (0.108).

## 19. WP19 Buyers of brand-new tokens

**Goal:** WP15 for tokens the model has never seen. A token launched after the
evaluation point has no column in the similarity matrix; its first buyers give
it one. Added 27 November 2026.

- **Tokens:** first traded ever in $[t, t + 24\text{ h})$, with at least 3
  *seed* accounts (audience accounts that trade it in its first hour) and at
  least 10 buyers.
- **Audience:** the accounts with at least one token in the 14 days before $t$
  (as WP15), minus the seeds. **Buyers:** audience accounts whose first trade
  of the token falls in (launch + 1 h, launch + 25 h]. Everything used to
  score is known at launch + 1 h: account histories before $t$, and the seeds.
- **Scores** (variants chosen on Dev A by mean recall@1000):
  - *seed profile*: $p_j$ = number of seeds with token $j$ in their 14-day
    window; score $= W p$ ($W$: accounts × tokens window matrix);
  - *seed profile, idf*: the same with $p_j$ divided by $\log(1 + \text{accounts with } j)$;
  - *recommender neighbours*: $p$ passed through the frozen blend's similarity
    matrix, score $= W (S p)$: the recommender's own co-trading similarities
    read for the new token.
- **Baselines** (the better chosen on Dev A): *activity* (window size, as
  WP15) and *launch hunters* (the number of window tokens the account first
  traded within a day of their launch). Random is reported.
- **Criterion (stated now):** on the 14 test points, the chosen score beats
  the chosen baseline in mean recall@1000 per point, 95% bootstrap interval
  over points excluding zero.
- **Checks reported alongside:** precision@100; the same without WP10's farm
  groups (as WP15); the same counting only buyers who arrive more than 6 hours
  after launch, so that bots buying with the seeds cannot carry the result.

## Order of work and effort

| Step | Depends on | Effort | Checkpoint |
| --- | --- | --- | --- |
| WP9 expectedness | frozen models | 1 day | `results/wp9/`, notebook 20 |
| WP10 farms | WP9 not needed | 2–3 days | co-action check on Dev B; sensitivity of the project result |
| WP11 segments | WP9, WP10 | 2 days | stability; per-segment blend on Dev B |
| WP12 token map | WP10 (farm targets) | 2 days | related-tokens check; explorer view |
| WP13 scam push | WP9, WP10, WP8 `prices.py` | 2–3 days | AUC difference on Dev B and test |
| Transfer download (start early, unattended) | none | size on one week first | logs for the project weeks |
| WP14 alerts | WP9, transfers | 3–4 days | alert rate, silence comparison, case review |
| WP15 likely buyers | frozen models (WP10 for the farm-free version) | 1 day | recall@1000 against activity on test |
| WP17 buyer forecast | frozen models | 1 day | error with and without the list counts on test |
| WP16 safer recommendations | WP13 | 1–2 days | collapsed share against recall, $\lambda$ chosen on Dev B |

About three weeks in total, most of it parallel to WP8.

- **Cheapest with the clearest result:** WP10 (it answers the report's main
  limitation) and WP12 (mostly built from existing similarities).
- **Most novel:** WP13; its label is exact and its criterion is stated in
  advance.
- **Added 24 November (15 – 17):** each has an exact outcome from the data
  and uses the recommender itself. WP15 and WP17 are cheap and can follow
  WP10; WP16 follows WP13.
- **Weakest evidence:** WP14; it can describe and rank suspicious activity
  but cannot measure how often it is right.
