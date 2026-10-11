# Write-up: illicit transaction classification under temporal drift

Feasibility study for the idea in
[`idea_illicit_transaction_drift.md`](idea_illicit_transaction_drift.md).
It covers the literature, the data, the split, a baseline and a first look at
the drift.

- Literature review: [`literature/literature_review_illicit_drift.md`](literature/literature_review_illicit_drift.md)
- Notebook with all numbers and charts: [`../src/notebooks/03_illicit_drift_data_and_baseline.ipynb`](../src/notebooks/03_illicit_drift_data_and_baseline.ipynb)
- Code: [`../src/scripts/elliptic.py`](../src/scripts/elliptic.py), tests in [`../tests/test_elliptic.py`](../tests/test_elliptic.py)

## Verdict

**The idea is feasible and its drift angle is open, but the evidence base is
thin.** The data is clean, the published baseline reproduces, and the
collapse after the dark market shutdown can be described much more precisely
than the literature does: a new kind of illicit transaction replaces the old
one, while licit transactions do not change. Retraining recovers the ranking
of the new transactions but not the decisions.

The limit is statistical: the whole phenomenon is **one event with 169
illicit transactions after it**. A project built on it can produce a careful
case study, not a general result about drift, unless it finds more events.

**The model-comparison half of the idea should be dropped.** The literature,
including two re-evaluations from 2026, has settled that a random forest on
the published features beats graph networks under a clean protocol.

Unlike in the other two studies, **test-period labels were used here**, for
the fixed baseline and for the retraining experiments, which are the object
of the study. Nothing was tuned on them. See "Limitations".

## 1. Data

| | |
| --- | --- |
| Source | The Elliptic dataset (Weber et al., 2019). Original on Kaggle; downloaded from the PyTorch Geometric mirror, which needs no account |
| Content | 203,769 Bitcoin transactions in 49 time steps about two weeks apart, 234,355 payment flows between them |
| Features | 165 anonymized features per transaction (93 local, 72 aggregated over direct neighbours), plus the time step |
| Labels | 4,545 illicit, 42,019 licit, 157,205 unknown |
| Size on disk | 150 MB download |

**Validation.** Every count matches the published description. There are no
missing values, no duplicate transactions or edges, and no edge between two
time steps.

**Is it enough and relevant?**

- For classification: yes. 46,564 labeled transactions are enough for
  tabular models.
- For the drift question: barely. After the shutdown at step 43 there are
  169 illicit transactions in seven time steps, and two of those steps have
  5 and 2.

**Features and labels.** The 165 published features and the graph are the
input; the time step is excluded. Labels are licit or illicit for the 23% of
transactions whose entity is known. Unlabeled transactions are excluded from
supervised training and scoring, and never treated as licit. The raw
transactions are not published, so no new transaction features can be built.

**What the descriptive statistics say.**

- The illicit share of labeled transactions swings between 0.3% and 36% from
  step to step, long before the shutdown.
- Features are standardized over the whole dataset, heavy-tailed (values up
  to 289 standard deviations) and redundant (71 features in pairs correlated
  above 0.99).

## 2. Preprocessing and split

**Findings from the exploratory analysis.**

- **Six features are clocks.** The time step explains 63 to 99% of their
  variance and their mean rises linearly. Time is in the features even
  though the time step column is removed.
- **Illicit transactions come in bursts of near-copies.** A third of them
  have another transaction at a distance below 0.1 (licit: 2%), usually in
  the same time step and also illicit. A random split would therefore leak,
  and the effective number of illicit examples is far below 4,545.
- **Standardization over all periods** is a mild leak built into the
  published data. It cannot be undone.

**Decisions.** Split by time only. Drop the time step. Keep the clock
features in the baseline for comparability (they do not change its score),
remove them for drift detection. No scaling or outlier treatment for tree
models.

**Split, by time step:**

| Part | Steps | Labeled | Illicit | Illicit share |
| --- | --- | --- | --- | --- |
| Train | 1 to 26 | 24,717 | 2,433 | 9.8% |
| Dev | 27 to 34 | 5,177 | 1,029 | 19.9% |
| Test, before the shutdown | 35 to 42 | 9,983 | 914 | 9.2% |
| Test, after the shutdown | 43 to 49 | 6,687 | 169 | 2.5% |

The test period is the one used in the literature. Train and dev together
are the literature's training period. The test period is reported in two
parts, because one average over both hides what happens.

**A limitation no split can fix:** the only shutdown lies in the test period.
Methods that adapt to drift can be tuned only on dev, which has no such
event.

**Leakage.** Checked and handled: time-ordered split, no time step feature,
no labels inside the features, graph confined to time steps, walk-forward
code that provably uses past labels only. Not removable: the global
standardization.

## 3. Baseline

**Task.** Classify labeled transactions of a later period as illicit or
licit, after training on earlier periods only.

**Metrics, and why.**

| Metric | Reason |
| --- | --- |
| **Illicit F1**, with precision and recall, at threshold 0.5 | The standard on this dataset. Accuracy is useless at 90% licit. |
| **Average precision** | Judges the ranking without a threshold. It separates "ranks badly" from "threshold in the wrong place", which is the key distinction after the shutdown. |
| **Per time step, and before / after the shutdown** | The question is about change over time. |
| **Bootstrap intervals** | To show how little the post-shutdown period can support. They are optimistic, since transactions in a burst are not independent. |

**Protocol.** Fit on train, score dev. Refit on train and dev, score test
once (the published protocol). No refitting within test, threshold 0.5.

**The fixed baseline** is a random forest with the published settings (50
trees, 50 features per split) on all 165 features.

| Illicit class | Dev F1 | Test F1 | Test precision | Test recall | Test average precision |
| --- | --- | --- | --- | --- | --- |
| Flag everything | 0.33 | 0.12 | 0.07 | 1.00 | 0.07 |
| Logistic regression | 0.74 | 0.44 | 0.32 | 0.70 | 0.34 |
| **Random forest (fixed baseline)** | **0.96** | **0.80** | **0.90** | **0.72** | **0.79** |
| Random forest, local features only | 0.94 | 0.77 | 0.84 | 0.71 | 0.77 |
| Random forest, without the clock features | 0.96 | 0.81 | 0.93 | 0.72 | 0.78 |

**How the baseline performs.**

- **It reproduces the published result:** F1 0.80 against 0.79 in Weber et
  al., 0.80 to 0.81 over five other seeds, bootstrap interval 0.78 to 0.82.
  Logistic regression gives 0.44 against a published 0.48. The local-feature
  forest differs more (0.77 against 0.69), possibly because the time step is
  excluded here.
- **It is strong where the world has not changed:** F1 0.96 on dev and 0.90
  on test before the shutdown.
- **After the shutdown it finds 3 of 169 illicit transactions:** F1 0.03.
- **The headline 0.80 is an average of 0.90 and 0.03.** It describes neither
  regime.

## 4. Does the proposed method fit the task and data?

The idea has three parts. They fit to different degrees.

### What kind of drift is it? This part fits well.

| Fixed baseline on the test period | Before the shutdown | After the shutdown |
| --- | --- | --- |
| Illicit share of labeled | 9.2% | 2.5% |
| Recall | 85% | 2% |
| False positive rate | 0.5% | 0.7% |
| Mean score given to illicit transactions | 0.83 | 0.09 |
| ROC AUC | 0.97 | 0.62 |
| Median distance of an illicit transaction to the nearest illicit one in training | 1.3 | 8.7 |
| Illicit transactions closer to a training illicit than to a training licit | 78% | 9% |

- The licit class is unchanged. The illicit class is replaced.
- The class balance shifts too, but that alone would leave recall intact.
- The post-shutdown illicit transactions lie in a region of feature space
  where training had only licit ones. It is a change of the illicit class's
  own distribution: not pure covariate shift, not a relabeling of old inputs.
- Read together with the bursts of near-copies, the likely explanation is
  that the classifier had learned the patterns of particular illicit
  entities, and those entities stopped operating. This is an interpretation;
  the data has no entity identifiers to confirm it.

The literature describes this only as "the emerging event has not been
learned". The measurement above is, as far as the papers reviewed go, new.

### Can it be detected without labels? The standard tool does not fit.

A classifier trained to tell one period from another succeeds perfectly for
**any** two periods, including two neighbouring time steps inside the
training data (AUC 1.00; without the clock features 0.93 to 0.96). Step 43 is
not more different from step 42 than step 21 is from step 20. The harmful
change affects 2% of the transactions and does not show in the overall
feature distribution.

### Does adaptation help? Partly, and under a strong assumption.

Random forest retrained before every test step on all earlier labels:

| After the shutdown (169 illicit) | F1 at 0.5 | 95% interval | Average precision | 95% interval |
| --- | --- | --- | --- | --- |
| Static (trained on steps 1 to 34) | 0.03 | 0.00 to 0.06 | 0.04 | 0.03 to 0.07 |
| Retrained every step, all history | 0.23 | 0.15 to 0.32 | 0.51 | 0.43 to 0.60 |
| Retrained every step, last 5 steps | 0.31 | 0.23 to 0.39 | 0.40 | 0.32 to 0.48 |
| Retrained every step, labels 2 steps late | 0.03 | 0.00 to 0.07 | 0.10 | 0.08 to 0.14 |

- **Retraining restores the ranking** (average precision 0.04 to 0.51) **but
  not the decisions** (F1 0.23, recall 14%). The new illicit transactions
  get scores above the licit ones but below 0.5. In hindsight, the best
  threshold on the same scores would give F1 0.55.
- This qualifies the original paper's statement that a retrained random
  forest cannot capture the new illicit transactions: true for its
  decisions, not for its ranking.
- **Everything depends on fast labels.** With labels two steps (about a
  month) late, almost nothing is gained. Real labels for illicit activity
  come from investigations and take far longer.
- Before the shutdown retraining adds almost nothing (0.91 against 0.90).
- A diagnostic upper bound, a model that has seen every other step including
  later ones, reaches an average precision of 0.56 after the shutdown. The
  new pattern is hard to learn from 169 examples whatever the method.

### The graph. Not tested, and not the priority.

No graph network was trained. Illicit transactions are linked to other
illicit ones far more often than chance (54% of their labeled neighbours,
against a base rate of 10%), but the graph has about two edges per
transaction, 60% of an illicit transaction's neighbours are unlabeled, and
the aggregated features already summarize direct neighbours. The literature
reports graph networks below the random forest under clean evaluation.

## 5. What to expect from more complex models

| Extension | Expected effect | Basis |
| --- | --- | --- |
| Threshold or calibration adaptation after a detected change | The largest available gain: up to F1 0.55 from the same retrained scores, if the threshold can be set without hindsight. | Best-threshold diagnostic. |
| Recency weighting of new illicit examples | Likely to raise scores of the new kind faster. Trade-off against noise from few examples. | The 5-step window: higher F1, lower average precision. |
| Gradient boosting instead of random forest | Small gain before the shutdown, none after it by itself. | Literature ranks tree ensembles closely; the failure is in the data, not the learner. |
| Graph features: distance to known illicit transactions | Reported to help most after the shutdown (10 to 16 F1 points). The most promising use of the graph. | Oliveira et al. (2021). |
| Graph neural networks | No gain over the random forest expected, and a high risk of leakage. | Weber et al. (2019), Maganti (2026), Akter et al. (2026). |
| Semi-supervised use of unlabeled transactions | Possibly useful when labels are late. Unsupervised anomaly detection is known to fail. | Lorenz et al. (2020). |
| Label-shift correction without labels | Not applicable: it requires each class to look the same, and the illicit class does not. | Section 4; Lipton et al. (2018). |
| Drift detection on model outputs | Worth trying: the number of flagged transactions drops sharply after step 42. Ambiguous by itself (fewer illicit, or missed illicit?). | Baseline chart. |

**A caution on expectations.** After the shutdown, a method is judged on 169
illicit transactions from, by the look of the data, very few sources. The
interval for an F1 of 0.23 is 0.15 to 0.32 even when transactions are
treated as independent, which they are not. Differences between adaptation
methods of a few points will not be distinguishable.

## 6. Conclusions

**Dataset.** Clean, well documented and a standard benchmark, which makes
results comparable. Three properties limit it: anonymized features (changes
can be measured but not explained), near-duplicate bursts (few independent
illicit examples), and a single drift event with few illicit transactions
after it.

**Baseline.** The random forest reproduces the published 0.80 and is hard to
beat where nothing has changed. Its single test score is a misleading
summary. The baseline to report is the pair: 0.90 before the shutdown, 0.03
after.

**Method.** The three parts of the idea come out differently. Characterizing
the drift works and already yields a clear, apparently new description.
Detection with the standard two-sample test does not work on this data.
Adaptation by retraining works for ranking, fails for decisions, and depends
entirely on label delay.

**Is there potential?**

- **As a model benchmark: no.** That question is answered in the literature.
- **As a drift case study: yes, with modest scope.** A precise account of
  what changed at step 43, why standard detection is blind to it, and how
  recovery depends on the number and delay of new labels would be a real
  contribution, since every paper mentions the shutdown and none analyses
  it. The result that retraining fixes the ranking but not the threshold
  points to a concrete method to develop.
- **As general evidence about drift: only if more events are found.** Two
  ways: simulate them, by withholding a cluster of illicit transactions from
  training and treating its appearance as a new event, which gives many
  events with known ground truth; and replicate on Elliptic++ or Elliptic2.

**Recommended next steps.**

1. Build the simulated-drift benchmark from held-out illicit clusters, so
   that adaptation methods can be tuned and compared on many events.
2. Develop threshold adaptation that uses only past labels, and evaluate
   recovery as a function of the number and delay of labels.
3. Add distance-to-known-illicit graph features.
4. Use the Elliptic++ entity graph to test whether post-shutdown illicit
   transactions come from new entities.
5. Read Vassallo et al. (2021) and Šafář et al. (2026) before claiming
   novelty for the adaptation and the leakage findings.

## Limitations of this study

- **Test-period labels were used.** The fixed baseline was scored on test, as
  in the literature. The retraining runs use test labels causally (only from
  earlier steps). Two diagnostics look at the labels being scored and are
  marked as such: the best-threshold F1 and the model that has seen all
  other steps. No setting was chosen on test, but having looked at the
  post-shutdown period this closely, it can no longer serve as an untouched
  test set for methods developed from here on. This is a further reason for
  the simulated benchmark.
- **One event, 169 illicit transactions.** All post-shutdown numbers are
  noisy, and the bootstrap intervals understate that.
- **No graph model was trained.** The statement that graph networks would
  not help rests on the literature, two of whose key papers are from 2026,
  one a single-author preprint.
- **The entity explanation is an inference** from distances and bursts, not
  an observation.
- **One model family.** Only logistic regression and random forests were
  run.
- **Features are anonymized,** so the six "clock" features are identified by
  their behaviour, not their meaning.
- **The literature review is partial.** Most papers were read in part (see
  the table in the review).
