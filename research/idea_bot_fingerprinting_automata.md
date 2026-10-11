# Idea: trading bot fingerprinting with probabilistic automata

Learn a probabilistic automaton per trader from its sequence of on-chain
actions on a decentralized exchange (DEX). Use the automata as behavioral
fingerprints: to cluster traders, to tell bots from humans, and to identify
the bot type (arbitrage, sandwich, market making).

## Summary

- **Data:** DEX swap events for Uniswap pools, with MEV bot labels from public
  MEV datasets for evaluation.
- **Topics covered:** learning finite automata (probabilistic automata,
  state merging); identification and pattern classification; unsupervised
  learning; hidden Markov models as a baseline.
- **Why high quality:** interpretable models whose states can be read and
  explained, compared head to head with HMM and neural sequence baselines.
- **What you learn:** stochastic grammatical inference, sequence
  discretization, and the trade-off between interpretable and black box
  sequence models.
- **Novelty:** high. Bot detection usually uses hand-built features or
  heuristics; automata-based behavioral fingerprints are uncommon.

## Data

- **Events:** swap, mint and burn events for a set of Uniswap V2 / V3 pools
  over a fixed period, from the Ethereum public dataset on BigQuery, The
  Graph, or an archive node. Each event gives the sender, pool, direction,
  size, block and position in the block.
- **Labels:** known MEV bots and their type from public MEV data (for example
  Flashbots `mev-inspect`, EigenPhi, ZeroMev), plus Etherscan address tags.
  Check which of these are still available and how they define each bot type.
- **Scope:** start with a few high-volume pools and a few weeks of data; the
  full chain history is far more than the project needs.

Two properties shape the project:

- **Labels are partial and heuristic.** They come from rule-based detectors,
  so they cover the bots those rules can see and nothing else.
- **Activity is heavily skewed.** A few addresses produce most events, and
  most addresses have too few to learn an automaton from.

## Research questions

1. Can a probabilistic automaton learned from one trader's action sequence
   separate bots from humans, and one bot type from another?
2. How does it compare with an HMM and a neural sequence model on the same
   sequences, in accuracy and in the number of events needed?
3. Are the learned states interpretable, for example does a sandwich bot's
   automaton show the buy, victim, sell pattern?
4. How sensitive are the results to the alphabet, that is, to how raw events
   are turned into symbols?
5. Does clustering the automata without labels recover the known bot types,
   or find groups that the heuristic labels miss?

## Approach

1. **Design the alphabet.** Map each event to a symbol from a small set, for
   example direction (buy / sell), size bucket, time gap bucket (same block,
   next block, later), and whether the pool is the same as the previous one.
   This is the main design decision; try several and report all of them.
2. **Learn automata.** Fit a probabilistic deterministic finite automaton per
   trader, or per group of traders, with state merging (ALERGIA, or MDI).
   Existing libraries such as AALpy and FlexFringe implement these.
3. **Compare automata.** Define a distance between traders, for example the
   likelihood of one trader's sequences under another's automaton, or a
   divergence between the distributions the automata generate.
4. **Classify and cluster.** Supervised: assign a sequence to the class whose
   automaton gives it the highest likelihood. Unsupervised: cluster traders by
   the distance from step 3.
5. **Baselines.** Hand-built features with gradient boosting, n-gram models,
   one HMM per class, and a small recurrent or transformer model.
6. **Interpret.** Draw the automata for each bot type and check them against
   how the strategies are known to work.

## Evaluation

- Bot vs. human and bot type classification: precision, recall and F1 per
  class, with traders (not events) split between train and test so that no
  address appears in both.
- A later time period as a second test set, since bot strategies change.
- Clustering against labels: adjusted Rand index and purity.
- Sample efficiency: accuracy as a function of the number of events per
  trader.
- Model size (number of states) next to accuracy, since interpretability is
  part of the claim.

## Risks

- **Alphabet design can fail.** Too coarse and all traders look alike; too
  fine and there is not enough data per symbol. This is the main risk.
- **Sparse sequences.** Most addresses have very few events, so per-trader
  automata only work for active addresses; grouping is needed for the rest.
- **Label quality.** Results can only be as good as the heuristic labels, and
  a model may be penalized for finding bots the labels miss.
- **Bots split activity** across many addresses and contracts, and a sandwich
  spans several transactions around a victim's, so one address's sequence may
  not show the whole strategy.
- **Fallback:** if per-trader automata do not work, the same data and
  alphabet still support the HMM and feature-based classifiers, which gives a
  weaker but complete project.

## References

- Carrasco and Oncina (1994), "Learning stochastic regular grammars by means
  of a state merging method" (ALERGIA).
- Thollard, Dupont and de la Higuera (2000), "Probabilistic DFA inference
  using Kullback-Leibler divergence and minimality" (MDI).
- de la Higuera (2010), "Grammatical Inference: Learning Automata and
  Grammars".
- Daian et al. (2020), "Flash Boys 2.0: Frontrunning in Decentralized
  Exchanges, Miner Extractable Value, and Consensus Instability".

Citations are from memory; verify them before use.
