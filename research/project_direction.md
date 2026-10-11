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

None so far.
