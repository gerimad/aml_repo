# Idea: illicit transaction classification under temporal drift

Classify Bitcoin transactions as licit or illicit on the Elliptic dataset. The
focus is not on a higher score but on why models collapse in the later time
steps, after a dark market shutdown changes what illicit activity looks like,
and on whether that drift can be detected and adapted to.

## Summary

- **Data:** Elliptic dataset, public on Kaggle. Optionally Elliptic++ for
  wallet level data.
- **Topics covered:** classification; descriptive statistics, PCA, LDA and
  feature engineering; the learner survey (linear, max margin, boosting,
  decision trees, Bayesian, neural nets, plus graph neural nets).
- **Why high quality:** a standard benchmark with a published baseline, a
  strict temporal split and heavy class imbalance, so the evaluation
  methodology is the real content.
- **What you learn:** imbalanced learning, concept drift, honest temporal
  validation, and when graph structure adds anything over tabular features.
- **Novelty:** low to medium. The dataset is heavily used; the drift angle is
  where there is room.

## Data

The Elliptic dataset (Weber et al., 2019, "Anti-Money Laundering in Bitcoin:
Experimenting with Graph Convolutional Networks for Financial Forensics"):

- About 204k transactions (nodes) and 234k payment flows (edges).
- 49 time steps, each roughly two weeks apart and each a separate connected
  component.
- 166 features per transaction: 94 local ones and 72 aggregated from
  neighbors.
- Labels: about 2% illicit, 21% licit, the rest unknown.

Two properties shape the project:

- **The features are anonymized.** There is no meaning attached to the
  columns, so feature engineering is limited to transformations, selection and
  graph-derived features.
- **A dark market closes around time step 43.** Models trained on earlier
  steps lose most of their recall on illicit transactions afterwards.

Check the exact figures against the dataset itself before quoting them.

## Research questions

1. How much of the performance drop after the shutdown is covariate shift
   (the inputs change) and how much is concept shift (the label rule changes)?
2. Can the drift be detected from unlabeled data alone, before labels arrive?
3. Which adaptation strategy recovers the most: retraining on a sliding
   window, sample reweighting, or online updates?
4. Does graph structure help once the tabular baseline is tuned properly?
5. Can the large unlabeled share be used (semi-supervised learning) without
   leaking future information?

## Approach

1. **Reproduce the baseline.** Train on the early time steps and test on the
   later ones, as in the original paper (steps 1 to 34 and 35 to 49). Logistic
   regression, random forest, gradient boosting, MLP.
2. **Describe the data.** Per-step class balance, feature distributions, PCA
   and LDA projections colored by time step to make the drift visible.
3. **Measure drift.** Per-step illicit F1, a domain classifier (can a model
   tell early steps from late ones?), and distribution distances per feature.
4. **Adapt.** Compare sliding window retraining, importance weighting and
   incremental learning.
5. **Add the graph.** A GCN or GraphSAGE model, and graph-derived features fed
   to the tabular models.

## Evaluation

- Illicit-class precision, recall, F1 and PR-AUC, reported per time step and
  not only as one aggregate number.
- Splits by time only. No random splits, since they leak the future.
- Several seeds and confidence intervals for every comparison.

## Risks

- **Crowded benchmark.** A new headline score is unlikely; the contribution
  has to be the drift analysis.
- **Anonymized features** limit interpretation of what changed.
- **Few illicit examples** after the shutdown make late-step metrics noisy.

## References

- Weber et al. (2019), "Anti-Money Laundering in Bitcoin: Experimenting with
  Graph Convolutional Networks for Financial Forensics".
- Elmougy and Liu (2023), "Demystifying Fraudulent Transactions and Illicit
  Nodes in the Bitcoin Network for Financial Forensics" (Elliptic++).
- Dataset: <https://www.kaggle.com/datasets/ellipticco/elliptic-data-set>
