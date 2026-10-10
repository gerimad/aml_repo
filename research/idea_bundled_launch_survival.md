# Idea: how bundled is a token launch, and how long does it survive?

When a new memecoin launches, its creator often buys a large share of the
supply in the same instant through several wallets, a "bundle", so that the
token looks in demand while insiders hold most of it. They can then sell
everything at once: a rug pull. Measure how bundled each launch is, and
model how the time until a rug depends on it.

> **Update.** This idea was extended into
> [`idea_memecoin_launch_outcomes.md`](idea_memecoin_launch_outcomes.md),
> which adds copycat names and is based on a first look at four days of the
> data. That look changed two things said below: rugs are much faster than
> assumed (median 26 seconds), and heavily bundled launches both rug more
> often and graduate more often.

## Summary

- **Data:** a public dataset of 1.18 million pump.fun launches on Solana
  with, for each, the number of bundle wallets, the insiders' share of the
  supply, whether and when a rug happened, and whether the token
  "graduated".
- **Topics covered:** classification and regression in the form of survival
  analysis with competing outcomes; statistics and feature engineering; the
  learner survey (Cox regression, boosting, trees).
- **Why high quality:** hundreds of thousands of events instead of the
  dozens or hundreds of the earlier ideas, a clear dose-response question,
  and a dataset whose author documents its flaws in unusual detail.
- **What you learn:** survival analysis, competing risks, working with
  selection bias in sampled features, and the gap between prediction and
  cause.
- **Who benefits:** retail buyers of new tokens, who lose money to rugs, and
  the tools that warn them.
- **Novelty:** medium. Bundle checkers are common trading tools; a measured
  relation between degree of bundling and survival time is less so. This is
  my estimate, not the result of a literature search.

## Data

**Main source:** `loopholetape/pumpfun-launches` on Hugging Face (licence
CC BY 4.0). I read its documentation on 2026-10-10. I did not download or
examine the data, so every number below is the author's.

| | |
| --- | --- |
| Launches | 1,180,869, from 31 August to 3 October 2026 (34 days) |
| Complete subset | 321,661 launches from 27 September on, recorded after a decoder fix |
| Legacy subset | 859,208 earlier launches, in which about 48% of buys were not recorded |

Columns that this idea needs, as documented:

| Column | Meaning |
| --- | --- |
| `f10_bundle_wallets` | number of wallets, other than the creator, whose first purchase landed in the launch slot or the next one; measured in the first 10 seconds |
| `f10_insider_share`, `f60_insider_share` | creator's share plus the bundle wallets' share of the supply, at 10 and at 60 seconds |
| `dev_share`, `dev_buy_sol` | the creator's own first purchase |
| `rugged`, `rug_dt_s` | whether the dataset's rug rule fired, and seconds from launch to the rug |
| `rug_branch_inferred` | which kind: a drain of the pool, or the creator selling at least half within five minutes |
| `graduated`, `migrate_dt_s` | whether the token completed its launch curve and moved to a regular pool, and when |
| `dev_first_sell_dt_s` | seconds to the creator's first sale |
| `f10_rep_n`, `f10_rep_rug` | the creator's number of earlier launches, and the share of them that were rugged |
| `peak_rs`, `terminal_rs` | highest and last amount of SOL in the launch curve |

Three properties shape the project, all stated in the dataset's own notes:

- **"Rugged" is an observation rule, not verified fraud.** It fires on a
  drain (the curve held at least 5 SOL and ended at 25% of its peak or less)
  or on a creator dump. Most rugs in the data are creator dumps on small
  curves.
- **The early features exist for only about 63% of launches, and that part
  is not a random sample:** it is rugged in 73% of resolved cases against
  41% for the rest. The author's warning is "never model the sampled rows as
  the population".
- **Only the last week is fully recorded.** In the legacy subset the bundle
  and buy counts are undercounted. The clean analysis has about 320,000
  launches over seven days.

**A first look on Ethereum.** To see whether the phenomenon is visible in
data already in the repo, I took the pools that first traded on days 8 to 10
of the two-week swap window and pair a new token with ETH or a dollar
stablecoin (5,186 pools):

- 84% stopped trading within an hour, and 89% within a day. The median such
  pool has a single trader.
- Only 83 pools had five or more different buyers in their first trading
  block. None of them was still trading at the end of the window.
- Pools where more of the first hour's buying came in the first block
  stopped trading sooner (rank correlation -0.33).

This is a rough proxy (a pool "first seen" is not the same as a pool
created, and "stopped trading" is not a rug), and the launch-bundle practice
is rare on Ethereum mainnet. It suggests the question is worth asking and
that Solana, where the practice is common and the data exists, is the place
to ask it.

## Research questions

1. **Dose and response.** Does the time to a rug shorten as the insiders'
   share at launch rises? Is the relation steady, or is there a threshold?
2. **Wallets or share?** Which matters more: how many bundle wallets there
   are, or how much of the supply they hold together?
3. **Competing outcomes.** A launch can be rugged, can graduate, or can
   simply fade. How does bundling shift the chances among the three?
4. **Does it add to what is already known at launch:** the creator's own
   purchase, and the creator's record of earlier rugs?
5. **How early is the warning?** With the 10-second snapshot alone, how well
   can a rug within the first five minutes, hour and day be predicted?
6. **Do bundled launches that survive differ** from those that do not?

## Approach

1. **Use the complete subset** for anything involving bundle features, and
   the legacy subset only for outcomes and creator histories.
2. **Define the clock.** Time from launch to rug. A launch that graduates,
   or is still unresolved when observation ends, is censored for the rug
   outcome. Graduation is treated as a competing event, not as "survived
   for ever".
3. **Describe.** Survival curves by level of insider share (for example
   none, under 10%, 10 to 30%, over 30%) and by number of bundle wallets.
4. **Baselines.** The creator's own share alone; the creator's past rug
   rate alone.
5. **Models.** Cox regression with insider share, bundle wallets, creator
   purchase, creator record, time of day; a discrete-time hazard model
   fitted with boosting, to allow thresholds and interactions; a
   competing-risks version for rug against graduation.
6. **Handle the sampling.** For features that exist only on sampled rows,
   either restrict claims to the sampled population and say so, or reweight
   towards the full population using what is known for every launch.
7. **Check the rule.** Repeat the main analysis separately for the two kinds
   of rug, since "creator sold half within five minutes" and "the pool was
   drained" are different events.

## Evaluation

- **Split by time:** fit on the earlier days of the complete subset, test on
  the later ones. With one week, for example five days and two.
- **Concordance index** for the ranking of launches by risk.
- **Time-dependent AUC** for "rugged within 5 minutes / 1 hour / 1 day".
- **Calibration** of predicted rug probabilities by group.
- **The headline as a plain table:** median time to rug, and share rugged
  within an hour, for each level of bundling, with intervals.
- **Creators, not launches, are the independent units.** One creator can
  launch hundreds of tokens. Standard errors should be clustered by creator,
  and a creator should not be on both sides of a split.

## Risks

- **The label is partly mechanical.** One branch of the rug rule needs the
  curve to have held at least 5 SOL. Bundles put SOL into the curve at
  launch, so a bundled launch is more likely to be *eligible* for that
  branch at all. Part of any relation between bundling and rugs could come
  from the rule. Analysing the two branches separately, and conditioning on
  the SOL in the curve at 10 seconds, addresses this.
- **Selection in the sampled features,** as described above.
- **One week of fully recorded data,** on one platform, in one market mood.
- **Prediction is not cause.** Bundling may mark creators who intend to rug,
  without bundling itself changing anything. The project can show an
  association and a useful warning signal; it cannot show that preventing
  bundles would prevent rugs.
- **A bundle is defined by timing** (first purchase in the launch slot or
  the next). Fast independent buyers ("snipers") meet the same definition.
  The dataset cannot tell them apart, and neither can the model.
- **Most launches die at once.** If nearly everything is rugged within
  minutes, there is little variation in survival time to explain, and the
  question becomes "rug or not" more than "how long".
- **Naming and framing.** The write-up should describe a risk signal for
  buyers. It should not read as advice on how to bundle without being
  noticed.

## Relation to other ideas

- Uses the same method as
  [delisting risk](idea_delisting_risk_survival.md), survival analysis, at
  the opposite time scale: seconds to hours instead of months to years, and
  hundreds of thousands of events instead of two hundred. Doing both would
  make a coherent pair.
- The [recommendation study](idea_next_pool_recommendation.md) found that 15
  to 25% of the tokens traders pick up next were launched within the
  previous two days, which is the population this idea studies.
- Related to "rug pull and scam token early detection" in the first
  brainstorm, with a sharper question and far more data.

## References

To be verified before use; cited from memory.

- Cox (1972), "Regression models and life-tables".
- Fine and Gray (1999), "A proportional hazards model for the
  subdistribution of a competing risk".
- Mazorra, Adan and Daza (2022), "Do not rug on me: zero-dimensional scam
  detection".
- Xia et al. (2021), "Trade or trick? Detecting and characterizing scam
  tokens on Uniswap decentralized exchange".
