# Literature review: illicit transaction classification under temporal drift

Review for the idea in
[`../idea_illicit_transaction_drift.md`](../idea_illicit_transaction_drift.md):
classify Bitcoin transactions as licit or illicit on the Elliptic dataset,
with the focus on why models collapse after the dark market shutdown and
whether that drift can be detected and adapted to.

## 1. Scope and method

The review covers four questions:

1. What does the dataset contain and what did its authors find?
2. How have later papers modelled it, and how sound are their evaluations?
3. What has been done about the drift at the shutdown?
4. What does the general literature on distribution shift offer?

Sixteen open-access papers were downloaded to `papers/` (PDFs are ignored by
git, the links below are the source of record). The "Read" column says how much
of each was read for this review.

| # | Paper | File in `papers/` | Read |
| --- | --- | --- | --- |
| 1 | Weber et al. (2019), [Anti-money laundering in Bitcoin: experimenting with graph convolutional networks for financial forensics](https://arxiv.org/abs/1908.02591) | `weber2019_elliptic_aml_gcn.pdf` | Dataset, methods, experiments, start of discussion |
| 2 | Pareja et al. (2020), [EvolveGCN: evolving graph convolutional networks for dynamic graphs](https://arxiv.org/abs/1902.10191) | `pareja2020_evolvegcn.pdf` | Passages on Elliptic |
| 3 | Lorenz et al. (2020), [Machine learning methods to detect money laundering in the Bitcoin blockchain in the presence of label scarcity](https://arxiv.org/abs/2005.14635) | `lorenz2020_aml_label_scarcity.pdf` | Abstract, conclusion |
| 4 | Oliveira et al. (2021), [GuiltyWalker: distance to illicit nodes in the Bitcoin network](https://arxiv.org/abs/2102.05373) | `oliveira2021_guiltywalker.pdf` | Abstract, results passage |
| 5 | Alarab, Prakoonwit (2022), [Graph-based LSTM for anti-money laundering: experimenting temporal graph convolutional network with Bitcoin data](https://doi.org/10.1007/s11063-022-10904-8) | `alarab2022_graph_lstm_aml.pdf` | Abstract |
| 6 | Lo et al. (2022), [Inspection-L: self-supervised GNN node embeddings for money laundering detection in Bitcoin](https://arxiv.org/abs/2203.10465) | `lo2022_inspection_l.pdf` | Abstract |
| 7 | Elmougy, Liu (2023), [Demystifying fraudulent transactions and illicit nodes in the Bitcoin network for financial forensics](https://arxiv.org/abs/2306.06108) (Elliptic++) | `elmougy2023_elliptic_plus_plus.pdf` | Abstract |
| 8 | Bellei et al. (2024), [The shape of money laundering: subgraph representation learning on the blockchain with the Elliptic2 dataset](https://arxiv.org/abs/2404.19109) | `bellei2024_elliptic2.pdf` | Abstract |
| 9 | Deprez et al. (2025), [Advances in continual graph learning for anti-money laundering systems: a comprehensive review](https://arxiv.org/abs/2503.24259) | `arxiv2503_24259_continual_graph_aml.pdf` | Abstract, passage on Elliptic |
| 10 | Maganti (2026), [When graph structure becomes a liability: a critical re-evaluation of graph neural networks for Bitcoin fraud detection under temporal distribution shift](https://arxiv.org/abs/2604.19514) | `arxiv2604_19514_graph_liability.pdf` | Abstract, conclusion |
| 11 | Akter et al. (2026), [Temporal-structural stress testing and cross-granularity robustness evaluation of graph AI for cryptocurrency AML detection](https://eudl.eu/doi/10.4108/airo.13658) | `eudl_airo13658_temporal_tree_ensembles.pdf` | Abstract |
| 12 | Lu et al. (2018), [Learning under concept drift: a review](https://arxiv.org/abs/2004.05785) | `lu2018_concept_drift_review.pdf` | Abstract, taxonomy of drift |
| 13 | Rabanser, Günnemann, Lipton (2019), [Failing loudly: an empirical study of methods for detecting dataset shift](https://arxiv.org/abs/1810.11953) | `rabanser2019_failing_loudly.pdf` | Abstract |
| 14 | Lipton, Wang, Smola (2018), [Detecting and correcting for label shift with black box predictors](https://arxiv.org/abs/1802.03916) | `lipton2018_label_shift.pdf` | Abstract |
| 15 | Grinsztajn, Oyallon, Varoquaux (2022), [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815) | `grinsztajn2022_trees_vs_deep_tabular.pdf` | Abstract |
| 16 | Saito, Rehmsmeier (2015), [The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets](https://doi.org/10.1371/journal.pone.0118432) | `saito2015_precision_recall_imbalanced.pdf` | Abstract |

Most papers were read in part only. Papers 10 and 11 are from 2026 and very
recent; paper 10 is a single-author preprint. Their claims are reported as
claims. Papers that could not be downloaded are listed in section 5.

## 2. Summaries

### 2.1 The dataset and its first results

**[1] Weber et al. (2019).** Introduces the Elliptic dataset and the
benchmark that almost all later work follows.

- **Data:** 203,769 transactions (nodes), 234,355 payment flows (edges), 49
  time steps about two weeks apart. Each time step is one connected
  component of transactions that appeared within about three hours; there
  are no edges between time steps. 4,545 transactions (2%) are labeled
  illicit, 42,019 (21%) licit, the rest are unlabeled.
- **Labels** follow the entity that made the transaction: exchanges, wallet
  providers, miners and licit services against scams, malware, terrorist
  organizations, ransomware and Ponzi schemes.
- **Features:** 166 per transaction. 94 "local" ones (including the time
  step, numbers of inputs and outputs, fee, volumes) and 72 "aggregated"
  ones: maximum, minimum, standard deviation and correlation coefficients of
  the same information over the direct neighbours. They are anonymized.
- **Protocol:** a temporal 70:30 split, training on time steps 1 to 34 and
  testing on 35 to 49, "because it reflects the nature of the task".
- **Models and illicit F1:** logistic regression 0.48, multilayer perceptron
  0.65, **random forest 0.79** (50 trees, 50 features per split), graph
  convolutional network 0.63, with a skip connection 0.71. Adding the
  network's node embeddings to the random forest gives 0.80. The random
  forest on the 94 local features alone scores 0.69, so the neighbour
  aggregates matter.
- **The dark market shutdown:** "the sudden closure of a dark market
  occurring during the time span of the data (at time step 43)... causes all
  methods to perform poorly after the shutdown. Even a Random Forest model
  re-trained after every test time step, assuming the availability of ground
  truth after each time, is not able to reliably capture new illicit
  transactions". The paper names robustness to such events "a major
  challenge".

**[2] Pareja et al. (2020), EvolveGCN.** A graph network whose weights are
evolved over time by a recurrent network. On Elliptic it uses a
train / validation / test split along time (31 / 5 / 13 steps). Dynamic
models do somewhat better than the static one, but "all methods perform
poorly starting at step 43... Even dynamic models are not able to perform
reliably, because the emerging event has not been learned".

### 2.2 Later models on Elliptic

**[3] Lorenz et al. (2020): few labels.** Unsupervised anomaly detection
performs poorly on Elliptic, and the authors "present evidence that anomalies
in the feature-space are not indicative of illicit behaviour". With active
learning, a few hundred labels (5% of the total) match the fully supervised
baseline.

**[4] Oliveira et al. (2021), GuiltyWalker.** New features from random walks
on the graph: the distance from a transaction to transactions already known
to be illicit. Added to the random forest of [1] they improve illicit
classification by about 5 points in the region of interest. The authors
report that after step 43 the plain random forest "cannot achieve an Illicit
F1-score value above 0.25", that their features help most there (10 to 16
points on average over steps 43 to 49), and that "for time steps 48 and 49,
both of these models still perform poorly".

**[5] Alarab and Prakoonwit (2022)** combine a graph convolutional network
with an LSTM and study active learning with uncertainty estimates.
**[6] Lo et al. (2022), Inspection-L** train graph embeddings with a
self-supervised objective and feed them to a random forest, reporting results
above the earlier state of the art. Both read at abstract level only.

**[7] Elmougy and Liu (2023), Elliptic++.** Extends the dataset with 822,000
wallet addresses (56 features each) and 1.27 million interactions, giving
four graphs: transaction to transaction, address to address, address to
transaction, and user entities. Enables classification of addresses as well
as transactions.

**[8] Bellei et al. (2024), Elliptic2.** A different, much larger dataset
from the same company: 122,000 labeled *subgraphs* within a background graph
of 49 million nodes. The premise is that money laundering is a property of a
group of connected transactions, not of a single one.

**[9] Deprez et al. (2025): continual learning.** A review and benchmark of
continual graph learning methods (replay, regularization, architecture) for
anti-money laundering, with Elliptic as the real dataset. The authors call it
"a well-suited case-study for continual learning" because of the shutdown,
which they describe as a "sudden shift in feature distribution of the illicit
cases".

### 2.3 Re-evaluations: are graph networks really better?

**[10] Maganti (2026).** Argues that the common claim that graph networks
beat feature-only models on Elliptic does not survive a leakage-free
protocol.

- Under a strictly inductive protocol (the network is trained only on the
  graph of steps up to 34), with 10 seeds: **random forest on the raw
  features F1 0.821, best graph network (GraphSAGE) 0.689**, multilayer
  perceptron 0.549.
- The same GraphSAGE, same seeds, scores 0.294 when trained transductively
  on the full graph and 0.689 inductively: a 39.5 point difference
  attributed to "training-time exposure to test-period adjacency".
- A hybrid of graph embeddings and raw features, reported at 0.807 in an
  earlier draft, falls to 0.699 under the clean protocol. The graph's
  contribution over an equally sized perceptron is +0.018 F1.
- Randomly rewired edges do better than the real ones by 8.9 F1 points.
- The paper notes that the aggregated features already give non-graph models
  first-order neighbourhood information.

**[11] Akter et al. (2026).** Compares classical models, tree ensembles,
graph-derived features and graph networks under a temporal split, with
metrics aligned to an investigator's budget. On the Elliptic test period:
**Extra Trees average precision 0.570, GraphSAGE 0.311**, a strictly temporal
GraphSAGE 0.381. The difference has a bootstrap interval of 0.22 to 0.30.
The same ranking holds on Elliptic2.

A third 2026 paper (Šafář et al., DFRWS USA) could not be downloaded. Its
abstract reports that commonly used train / test splits of Elliptic are not
properly isolated and that the leakage inflates reported performance.

### 2.4 Distribution shift in general

**[12] Lu et al. (2018): concept drift review.** With the joint distribution
written as P(X) × P(y | X), drift has three sources: a change in P(X) alone
("virtual drift", which does not move the decision boundary), a change in
P(y | X) alone ("actual drift"), or both. The field divides into drift
detection, drift understanding and drift adaptation.

**[13] Rabanser et al. (2019): detecting shift.** Machine learning systems
"tend to fail silently". Among methods for detecting dataset shift, two-sample
tests on a reduced representation from a pre-trained classifier work best,
and domain-discriminating classifiers help to characterize a shift and to
judge whether it is harmful.

**[14] Lipton et al. (2018): label shift.** When the class proportions change
but each class looks the same (P(y) changes, P(x | y) does not), the new
proportions can be estimated from an existing classifier's predictions on
unlabeled data, and the classifier corrected, without new labels.

**[15] Grinsztajn et al. (2022).** Across 45 tabular datasets, tree-based
models remain better than deep learning on medium-sized data. Among the
reasons: neural networks are not robust to uninformative features.

**[16] Saito and Rehmsmeier (2015).** On imbalanced data, ROC plots can be
"deceptive" about a classifier's reliability, and precision-recall plots are
more informative.

## 3. Insights gathered

1. **There is a standard benchmark protocol:** train on steps 1 to 34, test
   on 35 to 49, report illicit precision, recall and F1 [1]. Departing from
   it makes results incomparable.
2. **The random forest on the published features is the model to beat,** and
   after seven years it still is, once evaluation is clean [1, 10, 11].
3. **Many reported gains of graph networks come from evaluation choices,**
   mainly letting the network see test-period graph structure during
   training [10], and from random instead of temporal splits [11].
4. **The graph adds little beyond the aggregated features** [1, 10]. Where
   graph information helps, it is as features relating a transaction to
   known illicit ones [4].
5. **Every method fails after step 43,** including dynamic graph models and a
   random forest retrained every step [1, 2, 4]. No paper reviewed solves it.
6. **The cause of the failure is described only loosely:** "the emerging
   event has not been learned" [2], a "shift in feature distribution of the
   illicit cases" [9]. None of the papers reviewed separates the kinds of
   shift or measures them.
7. **Illicit behaviour is not anomalous in feature space** [3]. Unsupervised
   detection does not work; labels are needed.
8. **Labels are the scarce resource** [3, 5]. In practice they arrive late,
   through investigations. Retraining experiments assume they arrive at once
   [1].
9. **The drift vocabulary to use** is P(X), P(y | X) and P(y) [12, 14], and
   the standard detection tool is a classifier two-sample test [13].
10. **Label shift can be corrected without labels** if the classes themselves
    do not change [14], a condition that must be checked on Elliptic.
11. **Evaluate with precision-recall measures** on imbalanced data [16], and
    with operating points an investigator would use [11].
12. **The field is moving to other units of analysis:** addresses and
    entities [7], subgraphs [8].

## 4. What this means for our idea

**The benchmark side of the idea is closed.** The idea rated its novelty as
"low to medium" and placed the opening in the drift angle. The literature
confirms both halves. Another model comparison, with or without a graph
network, would repeat [1, 10, 11]. The drift, by contrast, is noted in
every paper and analysed in none.

**What the first experiments add** (see
[`../../src/notebooks/03_illicit_drift_data_and_baseline.ipynb`](../../src/notebooks/03_illicit_drift_data_and_baseline.ipynb)):

- The drift is in the illicit class only: recall falls from 85% to 2% while
  the false positive rate stays at about 0.5%. Post-shutdown illicit
  transactions are far, in feature space, from every earlier illicit one.
  This sharpens insight 6 and means the label-shift correction of [14] does
  not apply, since its condition fails.
- The classifier two-sample test of [13] is saturated: any two time steps
  are perfectly separable, partly because six features encode time.
- Retraining every step restores the *ranking* of post-shutdown transactions
  (average precision 0.04 to 0.51) but not the decisions at a fixed
  threshold, which qualifies the statement in [1]. With a label delay of two
  steps the gain disappears, which makes insight 8 concrete.

**Changes to the idea that follow from the literature:**

| Original plan | Change | Source |
| --- | --- | --- |
| Compare linear, boosting, trees, neural and graph models | Reduce to a reproduction of the standard baselines. Take the model ranking from the literature. | [1, 10, 11] |
| "Add the graph: a GCN or GraphSAGE" | Optional. If done, strictly inductive training. The more promising graph input is distance-to-known-illicit features. | [4, 10] |
| "How much of the drop is covariate shift and how much is concept shift?" | Keep as the first research question, stated with P(X), P(y), P(x \| y), and answered per class. | [12, 14] |
| "Can the drift be detected from unlabeled data alone?" | Keep, but the standard test fails on this data. Detection must be calibrated against normal step-to-step variation, or work on model outputs. | [13] |
| "Which adaptation strategy recovers the most?" | Keep, and add label delay as an explicit experimental factor. Evaluate ranking and decisions separately. | [1, 3] |
| "Use the unlabeled share (semi-supervised)" | Keep as an option for adaptation when labels are late. Unsupervised anomaly detection is known not to work. | [3] |
| Metrics: illicit precision, recall, F1, PR-AUC per time step | Keep. Add scores at an investigation budget. | [11, 16] |
| Elliptic++ as optional extra data | Useful for a different reason than planned: the entity graph could test whether post-shutdown illicit transactions come from new entities. | [7] |

**A sharper research question.** *After an abrupt change in illicit
behaviour, how quickly can a detector recover, as a function of how many new
labels it gets and how late they arrive?* This is answerable on Elliptic, has
practical meaning, and is not addressed in the papers reviewed. Its main
weakness is that Elliptic contains exactly one such event.

## 5. Papers to obtain

Not downloaded (paywalled or blocked).

| Paper | Why it matters |
| --- | --- |
| Šafář, Pluskal, Veselý, Ryšavý (2026), DFRWS USA, on information leakage between train and test splits of Elliptic | Directly about evaluation soundness on this dataset. Known here only from its abstract. |
| Vassallo, Vella, Ellul (2021), "Application of gradient boosting algorithms for anti-money laundering in cryptocurrencies", SN Computer Science | To my recollection it proposes an adaptive boosting scheme aimed at concept drift on Elliptic. If so, it is the closest prior work on adaptation and must be read before claiming novelty. Not verified. |
| Alarab, Prakoonwit, Nacer (2020), "Competence of graph convolutional networks for anti-money laundering in Bitcoin blockchain" | Cited as the source of an improved graph network result (F1 0.74 according to [10]). |
| Gama et al. (2014), "A survey on concept drift adaptation", ACM Computing Surveys | The standard reference on adaptation methods and on evaluation under drift. |
| Moreno-Torres et al. (2012), "A unifying view on dataset shift in classification", Pattern Recognition | Clean definitions of covariate, prior and concept shift. |
