# Write-up: the first seconds of a memecoin launch

Feasibility study for the idea in
[`idea_memecoin_launch_outcomes.md`](idea_memecoin_launch_outcomes.md).
It covers the literature, the data, the split, a baseline and a first test of
whether bundling and copied names say anything about what happens next.

- Literature review: [`literature/literature_review_memecoin_launches.md`](literature/literature_review_memecoin_launches.md)
- Notebook with all numbers and charts: [`../src/notebooks/07_memecoin_launch_data_and_baseline.ipynb`](../src/notebooks/07_memecoin_launch_data_and_baseline.ipynb)
- Code: [`../src/scripts/pumpfun.py`](../src/scripts/pumpfun.py), tests in [`../tests/test_pumpfun.py`](../tests/test_pumpfun.py)

## Verdict

**The data is the best of the seven studies, and the idea's two headline
effects mostly do not survive a closer look.**

- **Bundling.** Heavily bundled launches are "rugged" 93% of the time
  against 37% for unbundled ones. Most of that is built in: the dataset's
  rug label is close to "the curve reached 5 SOL and did not graduate", and
  bundle wallets are what puts SOL into the curve. With the curve's size
  held fixed, bundling adds 0.003 of AUC.
- **Copied names.** The first look found that the first use of a popular
  name graduates five times as often as the twenty-first. That came from
  selecting names by their later popularity. Counted properly, graduation is
  about the same at every position, and name features add 0.003 of AUC or
  less.

**Two things do work.**

- **Whether the creator sells within a minute can be predicted well:** AUC
  0.89 on the test days against 0.80 for a fixed baseline, with the
  difference well outside its interval. Nearly all of the gain is one
  fact: creators who sold fast before do so again.
- **The rare graduation can be ranked usefully:** of the 1% of launches
  scored highest, 31% graduate, against 1.4% overall. The simple baseline
  already does this, and nothing tested improves on it.

**As a project it is feasible, with a different headline than planned.** The
honest title is closer to "what the first three seconds of a launch do and
do not tell you" than to "bundles and copycats predict rugs". The open
modelling problem is the third of launches by first-time creators, where
the model has little to go on (AUC 0.72).

## 1. Data

| | |
| --- | --- |
| Source | Public dataset `loopholetape/pumpfun-launches` on Hugging Face (CC BY 4.0), release of 10 October 2026, downloaded at a pinned revision into `data/pumpfun_launches/` (157 MB) |
| Content | 1,180,869 launches, 31 August to 3 October 2026, 267,328 creators. One row per launch: creator, name, an early snapshot, outcomes with their timing |
| Fully recorded part | 321,041 launches from 27 September on. Before that the recorder missed about half of the buys |
| Study population | 199,986 launches by 81,986 creators: fully recorded, classic (not "mayhem" mode), quoted in SOL, outside three hours whose outcomes were never resolved |

**Validation.** The files match the author's documentation on every count I
checked: rows per regime, the share of mayhem launches (25 to 30% per day),
launches not quoted in SOL (6 to 12%), snapshot coverage, and the
unresolved hours of 30 September. No mint appears twice. The author's
`KNOWN_ISSUES.md` lists 17 issues with counts and is the basis for most
decisions here.

**Outcomes** in the study population, one per launch:

| Outcome | Share |
| --- | --- |
| Creator buyout: the creator buys the whole curve in the launch transaction | 1.4% |
| Graduated otherwise | 1.7% |
| Rug rule, creator-dump branch | 30.6% |
| Rug rule, drain branch | 32.0% |
| Faded | 33.5% |
| Unresolved | 0.8% |

**Three corrections to the idea file's first look.**

- **Nearly half of all graduations (45%) are creator buyouts.** They
  complete at 0.00 seconds and are known at launch from the creator's
  purchase. The first look counted them as graduations within ten seconds.
  A quarter of the remaining graduations also complete within three
  seconds. Graduation after the first seconds is about 1.1% of launches.
- **The rug label is mostly the size of the curve.** Of the curves that
  ever held 5 SOL and did not graduate, 99.99% are labelled rugged. An
  unfinished curve ends at zero, so the rule "fell to a quarter of its
  peak" is true of all of them. Below 5 SOL the label is the creator-dump
  branch alone. The author documents the label as an observation rule; the
  consequence for a bundling study is that the label partly contains the
  treatment.
- **Repeated names are mostly many creators.** One creator makes over half
  the launches in 10% of the heavily repeated names, not 27%.

**Timing.** Half of the creators who sell make their first sale within 21
seconds of launch and a quarter within 3 seconds. The rug rule fires at a
median of about 30 seconds.

**Is it enough and relevant?** More than enough: 165,000 launches enter the
prediction task, with 16,000 events of the common kind and 750 of the rare
kind on the test days alone. Directly relevant: it is the launches
themselves. Narrow in time: one week is fully recorded.

**Features and labels.**

- **Labels:** the creator's first sale and its time; graduation and its
  time. The rug label is reported, not modelled.
- **Features, all known 3 seconds after launch:** the creator's purchase,
  the SOL in the curve, bundle wallets and the insiders' share, the number
  of buys; the creator's record and pace, built from 34 days of history; the
  history of the launch's name, built the same way.

## 2. Preprocessing and split

**The moment of prediction is 3 seconds after launch.** Every early
snapshot in the fully recorded data is at most 3 seconds old, so everything
in it is known by then.

**Launches already decided at that moment are left out** (17% of the study
population): creator buyouts, curves already full, and launches rugged or
graduated within 3 seconds. They are described, not predicted. 165,381
launches remain; in 120,362 of them the creator holds tokens.

**Targets.**

| Target | Role |
| --- | --- |
| The creator, holding tokens at 3 seconds, sells within the next 60 seconds | Primary. A behaviour with a time, not defined through the curve's size. Base rate 44 to 46% |
| The launch graduates | Secondary. Label checked by the dataset's author against the chain (recall 98.6%). Base rate 1.4% |
| The dataset's rug rule fires | Not a target. One number, the SOL in the curve, "predicts" it with an AUC of 0.87 |

**Histories from the past only.** For each launch, the creator's and the
name's earlier launches are counted, and an earlier rug, graduation or sale
counts only if it had already happened when the present launch started
(`pf.count_before`, tested). The dataset's own reputation columns are not
used, because their timing is not documented.

**Names.** Normalized by Unicode form, case, and removal of everything but
letters and digits: 95,937 written names become 79,980. Looking further
with character n-grams, about 8% of single-use names are close to a name
used five times or more, usually that name with text added. Fuzzy matching
would move about 2% of launches into an existing meme.

**Split by time:** 27 to 30 September train (85,433 launches), 1 October dev
(24,726), 2 and 3 October test (55,222).

**Creators.** The idea file asked that no creator be on both sides. I did
not do that: a returning creator is the case in which a record helps, and
the record is built from the past. Instead, results are reported
separately for first-time creators, intervals resample whole creators, and
no creator identity is a feature. 1% of creators make 33% of the launches.

**Leakage checks.**

- The trap of the first look is confirmed and removed: on all launches the
  SOL in the curve gives an AUC of 0.946 for graduation because buyouts are
  already complete; in the task it gives 0.868.
- No creator's sale before the moment remains among the holders.
- Models trained on shuffled labels score between 0.45 and 0.58, around 0.5
  but more widely than independent launches would allow. That is the
  clustering by creator.
- The days differ: a model can tell a dev launch from a train launch with
  an AUC of 0.87. Part is built in, since histories lengthen over the data.
  This is drift, not leakage.

**The sampling problem the author warns of is small in this week.** 94% of
resolved launches have a full snapshot, and those with and without one have
the same outcome rates.

## 3. Baseline

**Fixed baseline:** logistic regression on three things a careful buyer
already looks at: the creator's purchase, the SOL in the curve, and the
creator's record of rugged and graduated launches.

**Metrics.**

| Metric | Why |
| --- | --- |
| AUC | ranking quality, comparable across days and base rates |
| Average precision | the same for the rare event |
| Log loss | whether the probabilities can be believed |
| Event rate in the top 10% or top 1% of scores | what acting on the highest scores gives |

**Protocol.** Fit on train; compare ten candidates on dev; choose one per
task by average precision; score it and the baseline on test, with 95%
intervals that resample creators. The boosted models' settings were fixed
in advance and not tuned.

**Primary task on the test days** (36,556 launches, base rate 44%):

| | AUC | Average precision | Sells, in the top 10% of scores |
| --- | --- | --- | --- |
| Fixed baseline | 0.803 (0.785 to 0.817) | 0.742 | 88% |
| Chosen: boosting on basics, bundling and creator pace | 0.886 (0.878 to 0.894) | 0.872 | 99% |
| Difference | +0.083 (+0.072 to +0.098) | +0.130 (+0.113 to +0.151) | |

The same on each test day (0.891 and 0.881). The chosen model's
probabilities match the observed shares closely.

| Test launches by | Baseline AUC | Chosen AUC |
| --- | --- | --- |
| Returning creators (65%) | 0.854 | 0.939 |
| First-time creators (35%) | 0.682 | 0.717 |
| All but the 100 most prolific creators | 0.795 | 0.871 |

**Secondary task on the test days** (55,222 launches, 753 graduations):

| | AUC | Average precision | Graduates, in the top 1% of scores |
| --- | --- | --- | --- |
| Fixed baseline | 0.876 | 0.207 | 32% |
| Chosen: logistic regression on all features | 0.885 | 0.209 | 31% |
| Difference | +0.008 (+0.006 to +0.011) | +0.002 (-0.004 to +0.008) | |

The top tenth of scores holds 65% of all graduations.

## 4. Does the proposed method fit the task and data?

**Bundling as a dose: the raw relation is strong, and mostly not about
bundling.**

| Bundle wallets | Launches | Median SOL in the curve at 3 s | Rug rule | Graduates | Creator sells within 60 s |
| --- | --- | --- | --- | --- | --- |
| 0 | 73,856 | 0.01 | 37% | 0.2% | 25% |
| 1 to 2 | 45,769 | 1.2 | 69% | 0.9% | 46% |
| 3 to 5 | 31,169 | 6.2 | 91% | 2.3% | 71% |
| 6 or more | 14,587 | 10.6 | 93% | 6.8% | 71% |

Among launches with 5 to 20 SOL in the curve, the rug rule fires on 98%
whatever the number of bundle wallets, and graduation is 2 to 4%. What
survives is the creator's behaviour in small launches: with under 1 SOL in
the curve, the creator sells within a minute in 21%, 33% and 60% of cases
as bundle wallets go from none to three or more. In the models, bundling
features add 0.003 of AUC to the basics on dev, and about 0.01 for
first-time creators.

**Copied names: the features can be built cleanly and carry little.** With
position counted over 34 days of history, graduation is 1.5% for a new
name and 1.4 to 1.9% for copies. A copy of a name that had already
graduated does somewhat better (2.3% against 1.7%). In the models, names
add 0.003 or less.

**Survival models: the framing fits, the Cox model does not.**

- The classifier trained for the 60-second question ranks sellers by time
  better than a Cox model (concordance 0.77 against 0.74) and is better at
  every horizon from 10 seconds to 30 minutes.
- Effects are not proportional over time: the creator's record of fast
  sales multiplies the hazard by 3.7 in the first 30 seconds and by 1.6
  after.
- There is no censoring: every curve was followed to its end. Competing-risk
  methods are not needed for the same reason.
- What carries over is the setup: a fixed moment, the launches at risk at
  that moment, a horizon.

**What does carry the prediction.** One feature, the share of the creator's
earlier launches that were sold within a minute, is worth 0.19 of AUC in
the primary model; nothing else is worth more than 0.03. For graduation,
the SOL in the curve is worth 0.34 and nothing else more than 0.01. This
agrees with Marino et al. (2026) on the SOL in the curve.

## 5. What to expect from more complex models

**On this table, little.** Boosting on all features and boosting without
the name features differ by 0.0003. For graduation, boosting is worse than
logistic regression: 1,137 positive examples are not enough for it.

**Gains would need new information, not a larger model.**

- **Who the first-time creators are.** A third of launches come from
  wallets with no history. If many are known operators under fresh
  wallets, the funding of the wallet would link them. That is the
  [sybil idea](idea_sybil_cluster_detection.md) applied here, and it needs
  transfer data that this dataset does not have.
- **The first trades as a sequence.** Marino et al. find that reaching a
  curve level in few trades is the strongest predictor of graduation. The
  snapshot gives only counts. Hu et al. report that designed features beat
  sequence models on launch data, so the expected gain is modest.
- **A later moment.** At 60 seconds far more is known, but half of the
  creators who sell have sold by 21 seconds. A warning at 60 seconds comes
  after most of what it warns about.
- **Name embeddings** are unlikely to matter, since exact name history is
  worth almost nothing.

## 6. Conclusions

**On the dataset.**

- Large, clean, and documented with unusual honesty. The week of complete
  recording is enough for every question asked here.
- Its `rugged` column should not be used as a target, and its `graduated`
  column only after creator buyouts are separated.
- One week on one platform. The last day already looks different from the
  others.

**On the baseline.**

- Logistic regression on three basics is a strong baseline: 0.80 and 0.88.
  For graduation it is as good as anything tested.
- It should be kept as the reference. Much of the apparent gain of boosting
  on the primary task is non-linearity in the same three features (0.825 to
  0.874 on dev).

**On the method.**

- A fixed prediction moment with strict "known by then" features is what
  makes the task meaningful. Without it the first look reached an AUC of
  0.96 by reading the answer.
- The two signals in the idea's title contribute almost nothing once the
  basics are in. The creator's record contributes almost everything.
- Horizon-specific classifiers fit; proportional-hazards models do not.

**On the opportunity.**

- **As a course project it is feasible and has two publishable-quality
  lessons in it:** a label that contains the treatment, and an effect
  created by selecting on the outcome. Both were live mistakes in the idea
  file, found and measured.
- **As a warning tool the result is narrower than hoped.** It says "this
  creator sells fast" for creators it has seen. For new creators it adds
  little to the basics.
- **The original question, how long a bundled launch survives before a
  rug, has no clean answer in this data,** because survival as labelled is
  a function of how much SOL was put in. A proper answer needs a
  buyer-centred outcome, such as whether a buyer at 3 seconds could ever
  have sold at a profit, and that needs the price path, not in this
  dataset.
- **Recommendation.** If the aim is a strong positive result, this is a
  weaker choice than it looked. If the aim is a careful study with a large
  dataset and clear, defensible findings, it is a good one, and the next
  step is the price-path outcome.

## Limitations of this study

- **One week of fully recorded data.** 27 September to 3 October 2026.
  Nothing here says how stable the results are over months.
- **The targets are behaviours, not fraud.** "The creator sells within a
  minute" includes any first sale of any size. The dataset does not give
  the size.
- **A bundle wallet is defined by timing:** a first purchase in the launch
  block or the next. Fast independent buyers meet the same definition.
- **Decisions were made with the whole week in view.** The prediction
  moment, the exclusions and the targets were chosen after exploring all
  seven days, test days included. The candidate models and the rule for
  choosing among them were fixed before any model was scored, and the
  notebook was run in full three times without changing them.
- **Histories are weaker early on.** Creator records before 27 September
  come from the legacy recording, in which rugs are undercounted.
- **38% of the fully recorded launches are set aside:** mayhem mode,
  launches not quoted in SOL, and three unresolved hours. Nothing here
  applies to them.
- **3 October differs** from the other days in several base rates. Whether
  that is the market or late settlement of outcomes is unknown.
- **The near-duplicate analysis was not validated by hand.** I looked at
  about thirty pairs; there is no precision figure. Many names are
  offensive, and none are printed in the notebook.
- **The Cox model was fitted on a sample** of 30,000 training launches with
  nine covariates. A richer hazard model was not tried.
- **The literature was read in parts.** One paper at length, the rest by
  abstract and selected passages. One of the closest papers has had its
  numbers withdrawn by its author.
