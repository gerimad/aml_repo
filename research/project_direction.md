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
