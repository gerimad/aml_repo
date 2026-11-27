# Possible applications of the swap data

**What this is.** A list of questions that the 12 weeks of Ethereum swaps
collected for the project could answer beyond next-token recommendation. For
each: what exactly would be predicted or detected, where the labels come from,
what data we already have and what is missing, a baseline, how it would be
evaluated, the effort, and the main risk.

**What this is not.** Part of the project plan. The project carries out
[`crypt_project_plan.pdf`](crypt_project_plan.pdf) as written
([`project_direction.md`](project_direction.md)). These are follow-ups and
material for the report's outlook. Where an earlier idea file or feasibility
study already covers a topic, it is linked, and its verdict is taken into
account.

## What we have

| Data | Content | Where |
| --- | --- | --- |
| Raw swaps | 609 chunks, blocks 25,340,000–25,948,799 (18 June – 10 September 2026), Uniswap V2 and V3 style pools. Per swap: block, position in the block, pool, `amount0`, `amount1`, `sender`, `recipient`, `tx_from` (the signing account), `tx_to`, transaction hash | `data/dex_swaps/chunks/` |
| Trade legs | 10.0 million legs: block, time, account, token, side (buy or sell), pool. Base currencies (WETH, USDC, USDT) removed | `data/project/legs.parquet` |
| Tokens | 57,411 tokens: first trade, launch pool, launch currency, whether the first trade is before the data starts (`censored`) | `data/project/tokens.parquet` |
| Pools | token pair of every pool | `data/project/pools.parquet` |
| Accounts | 618,812 accounts | `data/project/users.parquet` |

**What we do not have,** and what it would take to get it:

| Missing | Needed for | How to get it |
| --- | --- | --- |
| Liquidity events (`Mint`, `Burn`, V3 `Collect`) | Rug pulls in the strict sense: the liquidity is pulled | The same `eth_getLogs` download as the swaps, with two more event signatures. Roughly the same 4–18 hours. |
| The account that deployed each token | Creator behaviour, linking tokens of one operator | Needs traces or an indexer; not possible with the free endpoints (implementation plan, open point 11.3). |
| Prices in USD | Flows in money terms | Derivable from the data: WETH/USDC and WETH/USDT pools give the ETH price per block. Tokens are priced through their pool against a base currency. |
| Transfers outside swaps | Funding links between accounts (who sent ETH to whom) | ERC-20 `Transfer` logs and plain ETH transfers; large, ETH transfers need traces. |

## Findings from the project that point to these questions

- **Coordinated groups exist and are large.** Groups of 20 or more accounts
  with exactly the same 14-day token set hold 15% of evaluated accounts on
  Dev A (up to 48% at 6 July) and 9% on Dev B. One group of 573 accounts each
  traded one token on 28–29 June and all bought the same next token through
  one pool within 5.5 hours on 3 July (notebook 11, section 2b).
- **New tokens are adopted within minutes.** 77% of the first trades of new
  tokens by evaluated accounts happen within an hour of the token's first
  trade (notebook 13).
- **Early accounts are the one useful signal about a new token.** In the
  feature score of notebook 13, accounts in the first day carry nearly all the
  weight; age, first hour and launch currency hardly matter.
- **The market changes between periods.** The baseline's recall@10 was 0.155
  in the feasibility weeks and 0.075–0.094 in June–August (notebook 10).

---

## 1. Coordinated account groups (farms, sybils)

**Question.** Which accounts are run together by one operator, and what share
of the activity do they make up?

- **Label.** No ground truth in the data. Two options:
  - *Published lists:* projects that filtered airdrop farmers have published
    the removed addresses (see [`idea_sybil_cluster_detection.md`](idea_sybil_cluster_detection.md)).
    Only useful if one of those campaigns falls into our 12 weeks.
  - *Self-supervised check:* a group found on the first half of an account's
    history should keep acting together on the second half (same tokens, same
    hours). This tests whether a grouping is real without labels.
- **Features.** Same token sets, first trades within minutes of each other,
  same pools, same transaction shape (`tx_to` contract, gas pattern not
  downloaded), amounts that are exact copies or fixed fractions.
- **Baseline.** The rule we already used: identical 14-day token sets, groups
  of 20 or more (`protocol.same_history_groups`).
- **Method.** Graph of accounts linked by co-trading the same token within
  a short window; community detection; or the timing fingerprint of the
  earlier feasibility study ([`bot_fingerprinting_writeup.md`](bot_fingerprinting_writeup.md):
  a trader is identified from 100 swaps far better than chance).
- **Why it matters here.** It is the cleanest follow-up of the project: the
  groups bias any recommender evaluation, and the plan lists "a separation of
  automated accounts by behavioural features" as optional work.
- **Effort.** Medium. All data is in hand.
- **Main risk.** Without labels, "found a group" and "found a real operator"
  cannot be told apart for groups that only trade one token.

## 2. Cohort behaviour prediction

**Question.** Given the accounts that bought a token in its first hour (its
early cohort), what will they do next: sell within a day, hold, or move on
together to the same next token?

- **Label.** From the legs directly: the cohort members' sells of the token
  in the next 6 / 24 hours (share of the cohort that sold), and their next
  first-time token.
- **Features.** Cohort size, share of the cohort in coordinated groups (from
  1), the cohort's past behaviour (how fast its members sold earlier tokens),
  the token's launch currency and age.
- **Baseline.** The cohort's average past sell speed; for the next token, the
  recommender of this project applied to the cohort as one account.
- **Evaluation.** The same global timeline as the project: train on Dev A
  cohorts, choose on Dev B, test once. Metrics: AUC for "most of the cohort
  sells within 24 hours"; recall@10 for the cohort's next token.
- **Prior evidence.** The memecoin study found that *whether a creator sells
  within a minute* is predictable mostly from the creator's own past
  ([`memecoin_launch_writeup.md`](memecoin_launch_writeup.md): AUC 0.89 against
  0.80). The same "past behaviour repeats" effect is likely for cohorts.
- **Effort.** Low to medium. All data is in hand; most code exists
  (`cold_start.EarlyTrades`, the protocol).
- **Main risk.** Cohorts of farms will dominate the easy cases; results should
  be reported with and without them.

## 3. Rug pulls and token collapse

**Question.** From the first hour of a new token, can we tell whether its
price collapses (or its liquidity is pulled) within a day?

- **Label, two versions.**
  - *Collapse (data in hand):* the token's price in its launch pool, from
    `amount0`/`amount1`, falls more than 90% below its peak within 24 hours
    and trading stops. Measurable now.
  - *Rug in the strict sense (needs a download):* a `Burn` event that removes
    most of the pool's liquidity, usually soon after launch.
- **Features.** Accounts in the first minutes and hour, their share in
  coordinated groups, size distribution of the first buys (one big buyer vs
  many small ones), how many of the first buyers sell within minutes, launch
  currency, imbalance of the first hour's flow (see 4).
- **Baseline.** Number of accounts in the first hour (the signal that won in
  notebook 13).
- **Evaluation.** Tokens launched in Dev A for training, Dev B for choice,
  test tokens once; AUC and precision among the top 1% scored.
- **Prior evidence.** The memecoin study on Solana found the headline effect
  (bundled launches rug more) mostly built into the label: with the curve's
  size fixed, bundling added 0.003 of AUC. The lesson for us: define the
  collapse label without using the features that will predict it, and compare
  with a size-only baseline.
- **Effort.** Medium for the collapse version (price series per pool must be
  built from the raw swaps); plus the download for the strict version.
- **Main risk.** Most new tokens die quickly anyway, so "collapses" may be the
  default outcome and the interesting question becomes the rare survivor.

## 4. Flow imbalance and unusual inflows

**Question.** When the buy volume of a token suddenly exceeds its sell volume
by far, what follows: more new buyers, a price rise, or a collapse?

- **Measure.** Per token and hour: net flow = buys minus sells, in the base
  currency (ETH or USD), from the swap amounts. A z-score against the token's
  own recent hours or against tokens of the same age marks unusual inflows.
- **Labels for prediction.** Over the next 6 / 24 hours: new accounts trading
  the token (the recommender's target), the price change, a collapse (from 3).
- **Uses.**
  - *As an anomaly detector:* list the largest imbalances and check by hand
    what they are (a listing, a farm, one whale, a wash-trading loop).
  - *As a recommender feature:* add "net inflow in the last hour" to the
    ranker of WP6 and measure it with the same ablation. This is the cheapest
    test of the whole list, because the protocol and the ranker already exist.
- **Baseline.** Raw buy count (accounts), which the recommender already uses.
- **Effort.** Low to medium: a price per block for ETH, a volume table per
  token and hour, then features.
- **Main risk.** Amounts in token units need decimals per token
  (an `eth_call` per token, like the pool tokens); without them only ratios
  within one token are meaningful.

## 5. Adoption forecasting for tokens

**Question.** How many new accounts will trade a token in the next two days?

- **Why.** It is the token-level half of the recommendation task: a good
  forecast of adoption is a better popularity score, and for new tokens it is
  the cold-start problem. Notebook 13 shows that a token's first-day accounts
  are the strongest signal we have.
- **Label.** Counted from the legs, exactly as the recommender's target.
- **Baseline.** Accounts in the last day (the "recent popularity" of
  notebook 10).
- **Method.** Count regression (Poisson, boosted trees) on the token's recent
  accounts, inflow (4), cohort features (2), age.
- **Evaluation.** Error on the log count; and, as the real test, the
  recommender's recall@10 with the forecast in place of popularity.
- **Effort.** Low. Everything is in hand.
- **Main risk.** Adoption is concentrated in few tokens per day; a model can
  look good by getting those few right.

## 6. Wash trading and round trips

**Question.** Which trading is not a real change of hands: one operator buying
and selling the same token to itself to create volume?

- **Signals.** The same account (or group from 1) buys and sells the same token
  within minutes for near-equal amounts; trades whose `sender` and `recipient`
  or `tx_from` link back to the same accounts; volume without a change in the
  number of holders.
- **Label.** No ground truth. Report the share of volume flagged by a strict,
  stated rule and check samples by hand.
- **Effort.** Low for a rule-based measure; high for anything validated.
- **Main risk.** Market makers and arbitrage bots also buy and sell the same
  token quickly; they must be separated (for example by trading many tokens and
  pools, which wash traders do not).

## 7. Bots and sandwich attacks (MEV)

**Question.** Which accounts are automated, and how much of the swap flow is
sandwiching (a bot trading just before and just after a victim's swap in the
same block)?

- **Signals.** The position in the block (`tx_index`, `log_index`) is in the
  raw swaps: a sandwich is a buy and a sell by one account around another
  account's swap in the same pool and block. This is close to a definition, so
  the label is nearly exact.
- **Use for the project.** Removing sandwich bots and arbitrage bots from the
  accounts the recommender is evaluated on is a cleaner version of the farm
  limitation.
- **Effort.** Low for sandwiches (a pattern match on block order); the earlier
  fingerprinting study covers the rest
  ([`bot_fingerprinting_writeup.md`](bot_fingerprinting_writeup.md)).
- **Main risk.** Little; this is well studied, so the contribution would be the
  measurement on our data, not a new method.

## 8. Account churn

**Question.** Will an account trade again in the next 7 days?

- **Label.** From the legs.
- **Baseline.** Days since the last trade.
- **Why.** Most accounts trade one token and leave; knowing which ones return
  says whom a recommender can help at all. It also explains part of the gap
  between Dev A and Dev B.
- **Effort.** Low.
- **Main risk.** A standard problem with a strong baseline; likely a small gain.

## 9. Market phases

**Question.** Why is the baseline half as strong in June–August as in the
feasibility weeks, and can the phase of the market be detected as it changes?

- **Measures.** Daily: new tokens launched, share of trades in new tokens,
  share of accounts in groups (1), concentration of adoption (top-10 tokens'
  share of new pairs).
- **Prior evidence.** The regime study on Binance prices
  ([`market_regimes_writeup.md`](market_regimes_writeup.md)) is the closest
  earlier work.
- **Effort.** Low as a descriptive analysis; it would be a useful section of
  the project report in its own right.

---

## Comparison

| # | Application | Labels | Data in hand? | Effort | Reuses the project | Strength of the likely result |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Coordinated groups | none (self-check) or published lists | yes | medium | yes (found the groups) | medium; hard to validate |
| 2 | Cohort behaviour | exact, from legs | yes | low–medium | yes | good; past behaviour repeats |
| 3 | Collapse / rug | exact (collapse); strict rug needs a download | partly | medium | partly | uncertain; label must be defined carefully |
| 4 | Flow imbalance | exact, from amounts | yes (decimals needed) | low–medium | yes (ranker ablation) | good as a ranker feature test |
| 5 | Adoption forecast | exact, from legs | yes | low | yes | good; directly improves the recommender |
| 6 | Wash trading | none | yes | low (rule) | no | descriptive only |
| 7 | Sandwich bots | near-exact from block order | yes | low | partly (cleaner evaluation) | solid measurement, little novelty |
| 8 | Churn | exact | yes | low | partly | small gain over baseline |
| 9 | Market phases | descriptive | yes | low | yes | useful context for the report |

## Suggested order, if any of this is done

1. **Within the project, as small additions to the report:** 9 (market phases)
   to explain the change in the baseline, and 7 (sandwich bots) to describe who
   the evaluated accounts are. Both are descriptive and cheap.
2. **The first real follow-up:** 4 or 5 tested as a feature of the existing
   ranker. The protocol, the ablation and the success criterion already exist,
   so the result would be as clean as the project's own.
3. **A separate study:** 2 (cohorts) and 1 (groups) together, since cohorts
   mixed with farms are the main risk of 2 and the main object of 1.
4. **Only after downloading liquidity events:** 3 in its strict form.

Anything that touches the test period (13 August – 10 September) must wait
until the project's test has been scored once from the `test-freeze` commit.
