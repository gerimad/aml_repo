# Project direction

**From 11 October 2026 the project work has one goal: to carry out the plan
in [`crypt_project_plan.pdf`](crypt_project_plan.pdf) as it is written.**

The exploration of topics is over. The topic is next-token recommendation
for decentralized-exchange traders, and the PDF is the reference for what is
done, in what order, and how it is judged.

- **Authors:** Gergely Madarász, Levente Tóth, Béla Jakab
- **Course:** BME Applied Machine Learning, project work

## What the plan commits us to

This is a summary. Where it and the PDF differ, the PDF is right.

| | |
| --- | --- |
| **Task** | For each trading account, rank the tokens it has not traded. A hit is a token it trades for the first time in the next two days. |
| **Data** | Uniswap V2 and V3 style swaps on Ethereum. 12 weeks: blocks 25,344,000 to 25,948,799, 18 June to 10 September 2026. |
| **Split** | One global timeline with non-overlapping two-day targets: burn-in, Dev A (10 points), Dev B (11 points), test (14 points). |
| **Fixed baseline** | The 10 most popular tokens of the history that the user has not traded. No settings. |
| **Main metric** | Recall@10 averaged over users, per evaluation point and as a mean with its spread. |
| **Method** | Two stages: candidates from popularity and item-based nearest neighbours, re-ranked by a learned model. |
| **Core problems** | Personalization from very short histories, time weighting, and cold start. |
| **Success criterion** | The method improves mean recall@10 over the baseline on the test points, with a bootstrap interval of the paired difference that excludes zero. |

### The split

| Part | Period (2026) | Evaluation points | Use |
| --- | --- | --- | --- |
| Burn-in | 18 June to 2 July | none | History only |
| Dev A | 2 to 22 July | 10 | Tune settings, train the ranker |
| Dev B | 22 July to 13 August | 11 | Choose between models |
| Test | 13 August to 10 September | 14 | Scored once, from a frozen commit |

### The work packages

| # | Work package | Output | Depends on |
| --- | --- | --- | --- |
| 1 | **Data.** Download the 12 weeks and validate each week. | Validated swaps and trade legs | none |
| 2 | **Protocol.** Rolling evaluation, burn-in, paired comparison; re-measure the baseline. | Baseline table on the 21 development points | 1 |
| 3 | **Nearest neighbours.** Tune the neighbour settings and the blend weight. | First model comparison with the baseline | 2 |
| 4 | **Time weighting.** Half-life and history length. | Time-weighted model | 3 |
| 5 | **Cold start.** Refreshed lists and token features. | Share of the cold-start part that becomes reachable | 2 |
| 6 | **Learned ranker.** Logistic regression, then boosted trees; ablation by feature group. | Ranker compared with the best fixed blend | 3, 4, 5 |
| 7 | **Test and write-up.** Score the frozen models once on the test points. | Final report | 3 to 6 |
| 8 | **Strategy backtest (extension).** Trade tail tokens on the recommender's predicted demand; backtest after costs against visible-demand benchmarks on held-out weeks. | Whether the gain over the baseline is worth money | 7 |
| 9–14 | **Applications (extension, after 8).** Expectedness scores, farm detection, segmentation, token map, scam-push detection, unusual-activity alerts. | See `implementation_plan_applications.md` | 7, 8 |

## How we work from here

- **New work belongs to a work package.** A notebook, script or document
  should say which package it serves.
- **The test period is scored once.** Nothing from 13 August to 10 September
  is looked at with a model before the models and their settings are frozen
  and committed.
- **Settings come from Dev A, model choices from Dev B.**
- **Every comparison includes the fixed baseline** and is reported per
  evaluation point, not as a single number.
- **A change to the plan is made in the plan.** If something in the PDF
  turns out to be wrong or impossible, the change and its reason are
  recorded in this file under "Changes to the plan", so that the plan and
  the work do not drift apart.

## Guidelines for the implementation

### 1. Simple, pedantic, explained

- **Simple.** Choose the plainest implementation that does the job. A
  longer, obvious solution is preferred to a short, clever one.
- **Pedantic.** Do each step explicitly and in order. Check inputs and
  intermediate results (shapes, counts, value ranges) instead of trusting
  them.
- **Code blocks are explained.** Every function has a docstring that says
  what it does, what it takes and what it returns. Every notebook code cell
  is preceded by text that says what the cell does and why.
- **Assumptions are always stated.** Wherever the code or an analysis relies
  on an assumption, it is written down at that place: what is assumed, why
  it is reasonable, and what would break if it were false. Examples: "a
  pool's two tokens never change", "a user is one signing account".

### 2. What notebooks are for

- **Notebooks do exploratory work and display results:** tables, charts and
  the reading of them.
- **Reusable logic lives in `src/scripts/`,** with tests in `tests/`.
  Notebooks import it and do not redefine it.
- **A notebook runs from top to bottom** on the data in `data/` and shows
  its outputs when committed, so that the results can be read without
  running it.

## What happens to the earlier material

- **The other topics are closed.** The idea files, write-ups and notebooks
  01 to 04, 06 and 07 stay in the repository as a record of the exploration.
  They are not continued.
- **The feasibility study stays as background.**
  [`pool_recommendation_writeup.md`](pool_recommendation_writeup.md) and
  notebook 05 are where the baseline numbers of the plan come from.
- **The markdown plan is superseded.**
  [`project_plan_token_recommendation.md`](project_plan_token_recommendation.md)
  is an earlier draft. It reports model results and a test period that the
  PDF no longer uses.
- **The 28-day check is exploration only.**
  [`token_recommendation_28_days.md`](token_recommendation_28_days.md) and
  notebook 08 looked at 10 September to 8 October. That period is not part
  of the project data and is not used for tuning, selection or testing.
- **Reusable code:** `src/scripts/dex_swaps.py` (download) and
  `src/scripts/pool_recsys.py` (preprocessing, models, metrics), with their
  tests.

## Next step

Work package 1: download the 12 weeks.

```bash
uv run python src/scripts/dex_swaps.py --start 25344000 --blocks 604800 --workers 2
```

The download is resumable. If the free endpoint reports a rate limit, run
the command again and it continues where it stopped.

## Changes to the plan

| Date | Change | Reason |
| --- | --- | --- |
| 21 Nov 2026 | The download starts at block 25,340,000 instead of 25,344,000. Legs outside 18 June 00:00 to 10 September 00:00 UTC are dropped, so the periods are exactly those of the plan. | Block 25,344,000 is at 18 June 10:58 UTC, so the plan's block range missed 11 hours of the first burn-in day. The plan's block numbers were approximate; its dates are not. Dropping legs after 10 September 00:00 also keeps the 11 hours of exploration-period data at the end of the block range out of the project. |
| 22 Nov 2026 | WP3's grid is extended by *k* = 400, 800 and λ = 100, 200 (616 settings instead of 330). The choice rule and Dev A points are unchanged. | On the planned grid the best setting was *k* = 200, λ = 50, the largest value of both, with recall still rising towards that edge. The best setting could lie outside the grid. Dev B was not looked at for this decision beyond the one comparison already in notebook 11. |
| 22 Nov 2026 | No change: accounts in groups with identical histories (likely automated farms) stay in the data and in the evaluation. They are reported as a limitation (notebook 11, section 2b). | Found while checking WP3: one group of 573 accounts moves recall at 2 July by 0.06, and such groups hold 15% of evaluated accounts on Dev A and 9% on Dev B. Removing them was considered and declined, to keep the plan as written. |
| 23 Nov 2026 | WP6's tree grid is smaller than planned: 15 or 31 leaves (not 7), learning rate 0.05 only (not 0.03 and 0.1), and 100 or 300 trees as a grid setting instead of early stopping. Minimum rows per leaf 100 or 1000 as planned. The best fixed combination on Dev B is taken as both the untimed (WP3) and the time-weighted (WP4) blend, which are tied there. | Early stopping needs a validation split inside each training fold; a fixed number of trees keeps the procedure the same for every model kind. The smaller grid keeps cross-validation to about an hour. The best tree setting (31 leaves, 300 trees) is at the edge of this grid; this is reported, and the grid is not extended. |
| 23 Nov 2026 | Open points 11.2–11.5 of the implementation plan are decided as proposed there. 11.2: the label stays as in the PDF; the per-point share of positives whose token the account traded before the reference window is reported. 11.3: the deploying-account feature is dropped. 11.4: one primary model, the others secondary. 11.5: success is the 95% bootstrap interval over the 14 test points excluding zero; the sign test and all 14 differences are reported alongside. | Decided by the authors before the freeze. |
| 23 Nov 2026 | Primary model for the test: the LambdaMART ranker, by the rule of implementation plan 9.1 (largest mean paired difference against the baseline on Dev B: +0.052). Secondary: the untimed and the time-weighted fixed blends, and the boosted-trees and logistic-regression rankers. | The rule as written. Its advantage over the fixed blends faded across Dev B (notebook 14, section 6); the secondary comparison with the blends reports how this carries over to the test. |
| 23 Nov 2026 | The trained ranker files are kept in `models/` in the repository, not only in `data/project/models/`. | `data/` is not tracked, so without this the `test-freeze` commit would not contain the models it freezes. |
| 23 Nov 2026 | A work package 8 is added after the test: a backtest of a trading strategy built on the frozen recommender ([`trading_strategy_backtest.md`](trading_strategy_backtest.md), implementation plan section 12). It uses new held-out weeks (10 – 24 September, 8 October – 5 November) that are downloaded for it. | Decided by the authors after the test was scored. It does not change any project result: the recommender stays frozen, and the strategy has its own periods, settings and success criterion, stated before its held-out weeks are touched. |
| 23 Nov 2026 | Work packages 9–14 (applications of the recommender: expectedness scores, farm detection, segmentation, token map, scam-push detection, unusual-activity alerts) are added, to start after WP8 is finished ([`implementation_plan_applications.md`](implementation_plan_applications.md)). | Decided by the authors. They change no project result; WP10's re-scoring without farm accounts is reported as a sensitivity analysis next to the scored test. |
| 23 Nov 2026 | WP8 gets a second signal version, "filtered visible demand" (the recommender ranks the tail tokens with the most visible demand), next to the planned surprise signal (implementation plan 12.4). | The explorer page showed that visible demand picks tail tokens with more new buyers than the recommender's signal. Added before any WP8 setting was tuned and before the held-out weeks were looked at. |
| 23 Nov 2026 | WP8 accounting: the return of a decision is the mean net return of its positions from entry to exit, and a day's return is the mean over its decisions; a decision without an entered position counts as 0. Not marked to market every 6 hours. Entry at the first buy swap after the decision block and exit at the first sell swap after the holding time, so observed prices already contain the pool fee; our own price impact and gas are added on top. | Simpler and stated before any tuning; overlapping positions are handled by bootstrapping over days. Using the prices real buyers and sellers got avoids assuming we trade at a mid price. |
| 23 Nov 2026 | WP8 performance indicators and portfolio accounting: daily returns come from a portfolio of sleeves (`performance.equity_curve`), and the indicators of implementation plan 12.6 (Sharpe with interval, deflated Sharpe, Sortino, drawdown, Calmar, Kelly, trade statistics, capacity) are reported for every strategy and benchmark. The success criterion uses these portfolio daily returns. | Requested by the authors before any tuning. Per-decision averages are not a portfolio; the deflated Sharpe ratio accounts for the number of settings tried. |
| 23 Nov 2026 | WP8 gets a third version, "launch momentum" (implementation plan 12.8): hourly, tiny positions in tokens 15 minutes to 6 hours old picked by order flow, a take-profit ladder, judged by log growth against random new tokens. | Proposed by the authors before any WP8 tuning: the fat tail of new tokens may pay for many losers. It does not use the recommender and has its own benchmark and criterion. |
| 23 Nov 2026 | WP8 gets a fourth tail version, "unexplained flow" (visible ÷ (1 + predicted demand)). | Proposed by the authors before any tuning: order flow the recommender did not predict may mark an outside catalyst. It is the trading side of WP13's scam-push hypothesis, which predicts the opposite. |
| 23 Nov 2026 | Five more days of swaps (blocks 26,354,400 to 26,390,399, about 6 – 10 November) are downloaded, for prices only. | Positions opened late in the held-out weeks need prices after 5 November to exit (up to 72 hours plus 24 hours of grace). Without them they would count as unsellable. These days are used for exits only, never for a decision or a signal. |
| 23 Nov 2026 | WP8: predicted demand is not weighted by a hit rate per history group; every list counts 1 (or the account's buy volume). | The signal tables count lists per history group but outcomes are not tracked per account, so a tail-specific hit rate per group cannot be estimated from them. Decided before any tuning. |
| 24 Nov 2026 | WP8 entry: a position is opened at the decision time $s$ at the price of the last buy swap before the decision block (what the last real buyer paid, fee and token tax included), plus our own price impact and gas. Exits use only swaps after $s$; a token never traded again is worth zero. Replaces "entry at the first buy swap after the decision block" (23 Nov). | Found in notebook 17 before any backtest: 55-64% of the young tokens the launch rules pick get no buy after $s$ (many are rug pulls). The old rule marked them "not entered" and dropped them, so the backtest hid its worst trades. Our own buy is the entry, so we cannot wait for another buyer. |
| 24 Nov 2026 | WP8 development backtests (notebook 18, `dev_backtest.py`): only prices up to 10 September are read; tail decisions end on 7 September and launch decisions on 6 September, so that every position (longest holding time plus 24 hours of grace) exits before the held-out weeks. Launch positions are 0.05 ETH, capped at 1% of the buy volume since launch. Launch choice rule (12.8 gives none): the highest mean daily difference in log growth against `launch_random` with the same N, K, H and ladder on Dev A/B; settings within one standard error go to the simpler one (smaller K, then larger N, then b = 0.5, then no ladder). The whole-market variant (LambdaMART ranker) and the farm-group split of 12.6 are not part of this run. | Fixed before any backtest result was seen. Using prices after 10 September for exits would let the held-out weeks into the development results. The launch rule mirrors the tail rule of 12.4. The whole-market variant needs its own signal tables, and farm groups come with WP10. |
| 24 Nov 2026 | WP8 backtest prices and costs: only swaps of at least 0.01 ETH count as prices (entry, exits, impact); gas is charged per swap (one buy, one per sale) and the return is on what the position cost, so it never falls below −100% (a sale that would not cover its gas is not made); positions below 0.01 ETH are not opened. | The first run of notebook 18 gave impossible results. After rug pulls, pools had only dust swaps (down to 1e-10 ETH) at prices up to a billion times off, and these were taken as exits. Gas charged on top of total losses put returns below −100%, which turned portfolio values negative. Found and fixed before any result was used for a choice. |
| 24 Nov 2026 | WP8 is closed as a negative result on development data: no strategy is frozen and the held-out weeks (10 – 24 September, 8 October – 5 November) are not scored for returns. The report states the development result (notebook 18): every tail version and benchmark loses after costs (about −7% to −14% a day at the chosen settings); the recommender versions lose less than tail popularity on the tuning days (+0.08 to +0.13 a day, standard error about 0.03) and on the choice period once the single best day is dropped; every launch-momentum portfolio ends near zero and momentum does not pick better than random. | Decided by the authors at the checkpoint of implementation plan 12.7 step 10. The outcome is clear on development data, so a held-out run would not change the conclusion. The held-out weeks stay unused for returns and remain available to the applications. |
| 24 Nov 2026 | WP10 settings choice (section 10 of the applications plan gives none): Dev A is split into a build half (2 – 12 July) and a check half (12 – 22 July); the check of 10.2 is run for each of the 9 settings ($\Delta$ ∈ {5, 30, 120} minutes, $m$ ∈ {2, 3, 5}); among the settings that pass it there, the one with the most grouped accounts is chosen (ties: larger $m$, then smaller $\Delta$). The co-action rate of matched random pairs is pooled over all groups' matched pairs (up to 1,000 per group); the ratio is the median group rate over that pooled rate. "First trade" is an account's first trade of the token in the whole data. The group feature "distinct signing contracts (`tx_to`)" is left out: the legs do not carry it. | Settings must be chosen on Dev A, but the check needs data after the build window, so Dev A is split. Only $\Delta$ = 5, $m$ = 5 passes on the split (ratio 97, interval 34 – 195; 771 accounts in 67 groups); every $m$ = 2 setting fails (median group rate 0). Fixed before Dev B was looked at. |
| 24 Nov 2026 | WP10 sensitivity (10.3): three sets of accounts are removed in turn: members of the chosen groups (built on Dev A for Dev B, on Dev A + Dev B for test); every account in the loosest graph ($\Delta$ = 120, $m$ = 2) as an upper bound; and notebook 11's identical-set groups (at least 20 accounts) at each point. Models: the baseline, the fixed blend and the LambdaMART ranker. | The chosen groups are small (hundreds of accounts), so on their own they can barely move recall; the loose graph and the identical-set rule bound how much coordinated accounts could matter. |
| 24 Nov 2026 | Three applications added to the applications plan: WP15 likely buyers of a token (the blend read by token, against "most active accounts"), WP16 safer recommendations (collapsed share of the lists and a risk re-ranking, after WP13), WP17 forecasting a token's new buyers (list counts added to a persistence model). Each has its success criterion stated in the plan before any of its data is looked at. Order: WP15 and WP17 after WP10, WP16 after WP13. | Requested by the authors. WP8 showed that the recommender's demand prediction does not give trading profit, so the added applications use what the project did show (who adopts what), with outcomes read exactly from the data and no price prediction. |
| 24 Nov 2026 | WP15: three variant scores are added next to the plan's score (the blend's column, which stays the primary and is reported against its criterion as stated): the raw neighbour score (without the blend's per-account scaling), the plan's score × window size, and × log(1 + window size). One variant is chosen on the 10 Dev A points (highest mean recall@1000 over points) and held to the same criterion against activity on test, as a secondary result. Dev A points are now also scored. | At a first look (one Dev B point, plan's score and activity only) the plan's score caught 7% of buyers in the first 1,000 accounts and activity 44%: the blend scales each account's scores to a maximum of 1, which ranks a token within an account's options but says nothing about whether the account buys at all, and buyers are mostly very active accounts. The variants were compared on one Dev A point only before this was written; the variants were not looked at on Dev B or test. |
| 24 Nov 2026 | WP13 "low-rank share": the share of a token's new buyers for whom it ranked in the worst 10% **among the tokens active in the 14 days before the decision** (traded by at least one account in that window), instead of among the whole universe. | Decided by the authors. WP9 showed that, over the whole universe, almost no pair has a percentile above 0.9 (most of the universe is inactive tokens, which every active token outranks), so the feature as planned would be zero almost everywhere. |
| 24 Nov 2026 | WP14 is sized before it is committed to: the `Transfer` logs of one week are downloaded first, and the authors decide on the rest from its size and download time. Order of the remaining applications confirmed: WP12, WP11, WP13, WP16, WP14. | Decided by the authors. WP14 has the largest download and the weakest evidence of the applications. |
