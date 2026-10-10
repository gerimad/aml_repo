# Idea: sybil cluster detection for airdrops

Projects reward early users with tokens, and some people run hundreds or
thousands of accounts to collect many times. A few projects investigated
this and published which accounts they removed. Use such a list to test
whether groups of accounts run by one operator can be found from on-chain
behaviour.

## Summary

- **Data:** the published investigation of the Hop Protocol airdrop, which
  names 14,195 removed accounts and the groups they were assigned to; plus
  the on-chain activity of all candidate accounts.
- **Topics covered:** unsupervised learning and structure learning
  (clustering, community detection on a funding graph); identification and
  pattern classification; feature engineering.
- **Why high quality:** it is a clustering problem with an external
  reference to score against, which is rare, and the reference has group
  structure, not only a yes / no label.
- **What you learn:** clustering with partial labels, graph features, and
  how to evaluate against a reference that is itself the output of a method.
- **Who benefits:** projects distributing tokens, and the honest users whose
  share is diluted.
- **Novelty:** low to medium. Sybil detection is an active area with
  commercial and community work. This is my estimate, not the result of a
  literature search.

## Data

**The reference.** The repository `hop-protocol/hop-airdrop` on GitHub
answered requests on 2026-10-10. I looked at file names and at the first
lines of four files, not at their full contents.

| File | What it holds |
| --- | --- |
| `eliminatedSybilAttackers.csv` | 14,195 addresses removed as sybils |
| `eligibleAddresses.txt` | 28,857 lines: the addresses that stayed eligible |
| `groups.json` | groups with a name (an address), the number of connected addresses, the number removed, and the removed addresses of each group |
| `eliminatedGroups.ts` | which groups were removed |
| `addressesMetadataAllUsers.csv` | per address: number of transactions and total volume on the protocol |
| `finalDistribution.csv` | tokens finally given to each address |

The repository states no licence. It was last updated in September 2023.

**The behaviour.** Not collected. It has to come from chain data: for each
candidate account, its bridge transactions, who funded it, and where its
funds went. The airdrop covered activity in 2021 and 2022 on Ethereum and
several second-layer networks.

Three properties shape the project:

- **The labels are someone else's detection result.** The project and its
  community decided which accounts were sybils, by methods of their own. A
  model that reproduces the list has learned those methods, not the truth.
- **There is group structure.** The groups file says which removed accounts
  belong together, so clusterings can be scored as clusterings.
- **The history is old and spread over several networks.** The free
  endpoints used in the earlier studies serve old blocks only in small
  windows, or not at all. Collecting 2021-22 activity on five networks is
  the expensive part.

## Research questions

1. Do accounts that the investigation grouped together look alike in
   behaviour (timing, amounts, routes), beyond being funded from the same
   source?
2. Which signal finds the groups: the funding graph, behavioural similarity,
   or both?
3. How many of the removed accounts can be recovered without using the
   list, and at what cost in honest accounts wrongly flagged?
4. Are there coherent clusters among the accounts that were *not* removed?
   These are either missed sybils or false alarms, and looking at them is
   the interesting part.
5. Does the behavioural fingerprint of the bot study (symbols describing
   each action relative to the previous one) carry over to this setting?

## Approach

1. **Decide the scope by what can be collected.** Start with one network and
   the bridge contract's own events, which name sender, amount, destination
   and time for every transfer. Extend only if that works.
2. **Features per account:** number of transfers, amounts and how round they
   are, hours of activity, time between first funding and first use, routes
   taken, and whether the account went quiet after qualifying.
3. **Graph:** an edge between two accounts when one funded the other or both
   were funded by the same address, with care for exchanges, which fund
   everyone.
4. **Baseline:** connected components of the funding graph.
5. **Methods:** clustering on behavioural features (density-based, since the
   number of clusters is unknown and most accounts belong to none);
   community detection on the graph; the two combined.
6. **Semi-supervised variant:** use part of the list as labels and predict
   the rest, split by group so that no group is on both sides.
7. **Inspect** the clusters that the investigation did not flag.

## Evaluation

- **Against the groups:** adjusted Rand index and pair-level precision and
  recall (of all pairs the model puts together, how many did the
  investigation; and the reverse).
- **Against the list:** precision and recall of "flagged as sybil", with the
  honest eligible addresses as negatives.
- **At a fixed false-alarm rate.** Wrongly removing an honest user is the
  costly error, so recall at 1% false positives says more than an average.
- **Split by group** for anything supervised.
- **Agreement is not correctness.** High agreement means the method
  reproduces the investigation; clusters outside the list are not thereby
  wrong. Report both, and examine a sample by hand.

## Risks

- **Data collection may not be feasible with free access.** This is the
  first thing to test. If old history on second-layer networks cannot be
  read, the project reduces to the Ethereum side, or to the per-address
  totals in the repository, which is too little.
- **Circularity.** If the investigation removed accounts because they shared
  a funding source, then "funding graph finds them" is true by
  construction. The value is in what behaviour adds, and in the clusters the
  investigation missed.
- **Exchange-funded accounts** look connected and are not. One group in the
  file has 194,922 connected addresses of which 7,423 were removed, which
  looks like such a hub.
- **One airdrop, 2022.** Sybil operators have changed methods since. A
  second published list would be needed to say anything general; the one
  other source I tried was not found at the address I guessed.
- **Dual use.** A description of what gets caught is a guide to not being
  caught. Report at the level of signals, not recipes.
- **No licence** is stated for the repository. Using a public list of
  addresses for research is ordinary, but check before redistributing it.

## Relation to other ideas

- Extends [trading bot fingerprinting](idea_bot_fingerprinting_automata.md),
  where behaviour alone identified a trader among several hundred in 31% of
  cases, and where the missing piece was external labels. Here labels
  exist.
- Shares the difficulty found in the
  [illicit transaction study](idea_illicit_transaction_drift.md): labels
  that come from an investigation are incomplete and reflect its methods.
- Listed as idea 7 in [`project_ideas_round3.md`](project_ideas_round3.md).

## References

To be verified before use; cited from memory.

- Douceur (2002), "The sybil attack".
- Ester et al. (1996), "A density-based algorithm for discovering clusters
  in large spatial databases with noise" (DBSCAN).
- Blondel et al. (2008), "Fast unfolding of communities in large networks"
  (Louvain).
- Victor (2020), "Address clustering heuristics for Ethereum".
