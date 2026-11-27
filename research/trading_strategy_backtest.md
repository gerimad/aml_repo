# A trading strategy built on the recommender, and how to backtest it

**What this is.** A proposal for a follow-up study: turn the token recommender
of the project into a trading signal and test, on historical swaps, whether it
would have made money after costs. It states the idea, the exact strategy, the
data it needs, the backtest protocol, the benchmarks it must beat, and the ways
such a backtest usually fools itself.

**What this is not.** Part of the project plan, or advice to trade. The tokens
involved are mostly new, illiquid and speculative; many collapse within hours
([`possible_applications.md`](possible_applications.md), section 3). The study
is about whether the signal has economic content, not about running it.

## The idea

The recommender predicts which accounts will trade which tokens for the first
time in the next two days. Summed over accounts, that is a forecast of **new
demand** for each token. If new demand is not yet in the price, buying the
tokens with the most predicted new buyers, before those buyers arrive, should
earn a return.

The strategy is built so that **its result depends on the recommender's gain
over the popularity baseline**, not on what popularity already knows. A
strategy that buys the most-demanded tokens overall would mostly trade the
popular head, where the baseline is already strong and where everyone else
sees the same demand. That would test popularity, not the recommender.

## Where the gain over the baseline matters most

From the project's test period (14 points, 13 August – 10 September;
notebook 15):

| | Popularity baseline | Untimed fixed blend | LambdaMART ranker |
| --- | --- | --- | --- |
| Recall@10, all tokens | 0.108 | 0.153 | 0.154 |
| **Recall@10 without the head** (the most popular tokens holding a third of the history's trades) | **0.000** | **0.033** | 0.028 |
| Coverage: share of all tokens on at least one list | 0.2% | 3.8% | 2.1% |
| Recall@10, accounts with 5+ tokens | 0.057 | 0.094 | 0.096 |

- **Over all tokens** the recommender is 42% better than popularity: a real
  gain, but on tokens whose demand is visible to anyone.
- **Outside the head the baseline finds nothing, by construction**, and the
  recommender finds 3% of the pick-ups. Every correct prediction there is
  information popularity does not have. The recommender also names about 18
  times as many distinct tokens.
- **Among active accounts** (5+ tokens) the gain is +66%. These are the
  accounts whose trades are larger and more frequent.

So the strategy trades **only tail tokens** (outside the head), where the
recommender's edge over popularity is the whole signal, and weights demand by
**active accounts**, where its edge is largest. In the tail the untimed fixed
blend is better than the ranker (0.033 against 0.028) and it is refitted at
every point, so it cannot drift from a training period; it is the model used.

Two facts make the test doubtful, and the backtest must show them:

- **Tail tokens are illiquid.** A correct demand forecast for a token with a
  few thousand dollars of daily volume is worth little after price impact. The
  position caps and impact estimate below decide whether the edge is tradable.
- **A first-time trade is not always a buy,** and part of the demand comes from
  coordinated groups (notebook 11, section 2b), which pump and then dump
  together.

## The signal

At a decision time $s$:

1. Fit the untimed fixed blend at $s$ on the 14 days before $s$ (settings of
   `configs/frozen.json`, the same code as the project's test run).
2. Make the list of 10 tokens for every account that traded in the 14 days
   before $s$ and at least once in the last 7 days.
3. **Remove the head:** the most popular tokens of the 14-day history that
   together hold a third of its trades (`protocol.head_tokens`, the definition
   of the project's "recall without the head"). Only tokens outside it can be
   selected.
4. **Predicted new buyers** of tail token $i$, weighted by how active each
   account is:

   $$D_i(s) = \sum_u \mathbb{1}[\,i \in \text{list}_u(s)\,]\cdot \hat p_u \cdot a_u$$

   - $\hat p_u$ is the chance that a tail token on $u$'s list is picked up,
     estimated on Dev A/B by the account's history size (1, 2–4, 5+ tokens).
   - $a_u$ is the account's activity: its buy volume in ETH over the last 14
     days, capped at the 99th percentile so that one whale does not decide
     alone. With $a_u = 1$ the signal counts accounts; with volume it
     estimates the ETH that will flow in, which is what moves a price.
5. **Surprise, not level:** $S_i = D_i(s) / (1 + V_i(s))$, where $V_i(s)$ is the
   token's buy volume in ETH over the last 24 hours. The aim is tokens whose
   predicted inflow is large against the inflow already seen.

Only **buys** count as demand. The recommender's label counts any first trade;
its tail hit rates $\hat p_u$ are re-measured on first *buys* before use.

## The strategy

| Part | Rule (first version; every number is a setting tuned on development data only) |
| --- | --- |
| Decision times | every 6 hours (notebook 13: lists rebuilt more often find more) |
| Universe | tail tokens only (step 3) with a WETH, USDC or USDT pool, at least 24 hours old, sold by at least 20 different accounts in the last 24 hours (a token nobody can sell is a trap: honeypots and transfer-tax tokens) |
| Selection | the $K = 5$ tokens with the highest surprise $S_i$ |
| Entry | buy at the price of the first swap in the token's most active base pool at least 1 block after $s$ (no trading in the block the signal is computed from) |
| Size | equal weight; each position at most 1% of the pool's buy volume of the last 24 hours. For tail tokens this cap is binding: it decides how much capital the strategy can use at all, which is reported |
| Exit | sell after $H = 24$ hours at the price of the first swap after $s + H$; earlier if the price falls 50% below entry (priced at the first swap after the fall, not at the stop level) |
| Costs | the pool's fee (0.3% for V2 pools, the fee tier for V3 pools), an estimate of price impact (below), and gas per swap |

**Variant for comparison: the whole market.** The same rules without step 3
(head included) and with the LambdaMART ranker. Comparing the two variants
shows directly whether the money, if any, comes from the part of the
recommender that popularity lacks.

## Data

| Need | Have? | How |
| --- | --- | --- |
| Swaps with amounts and order in the block | yes | `data/dex_swaps/chunks/`: `amount0`, `amount1`, block, position in block |
| Token decimals | no | one `eth_call` per token (`decimals()`), like the pool-token lookup; minutes with Multicall3 |
| Price per swap | derivable | base amount / token amount from the two amounts and the decimals; ETH in USD from the WETH/USDC pools |
| Pool fee tier (V3) | no | one `eth_call` per V3 pool (`fee()`) |
| Pool reserves / liquidity | no | V2 `Sync` events or V3 `slot0` and `liquidity`; needed for an honest price-impact estimate. Without them, impact is estimated from how much prices moved after observed swaps of similar size. |
| Gas prices | no | block base fee from block headers (one call per block, or a public dataset) |
| The recommender | yes | `configs/frozen.json` (the untimed fixed blend, and the ranker for the whole-market variant), `src/scripts/tuning.py`, `src/scripts/ranker.py` |
| The head at each time | yes | `protocol.head_tokens` on the 14-day history |
| Account buy volume in ETH | derivable | from the swap amounts and the ETH price |

## Backtest protocol

The same discipline as the project, because the same failure (tuning on the
period you report) is even easier with returns than with recall.

1. **Split by time.** Settings ($K$, $H$, the decision interval, the stop, the
   surprise normalization) are chosen on Dev A, the version is chosen on Dev B,
   and the result is reported once on a held-out period.
2. **Which held-out period.** The project's test period (13 August –
   10 September) has already been looked at for recommendation, though never for
   returns. The clean choice is **new data after 10 September**, downloaded with
   the same pipeline; the project's test period is then a second, slightly less
   clean check.
3. **No look-ahead.** Every input at $s$ uses only swaps before $s$: the
   recommender (as in the project), the universe filters, the volume caps. Entry
   is at a later swap than $s$; exit prices are the first swap after the exit
   time, never a price chosen afterwards.
4. **No survivorship.** Tokens that collapse or stop trading stay in. If a
   position cannot be exited because no one trades the token any more, it is
   valued at zero.
5. **One pre-stated success criterion,** as in the project: the mean return per
   6-hour period after costs, minus the best benchmark's, is positive with a 95%
   bootstrap interval over periods that excludes zero (bootstrap over days, not
   over trades, because trades on one day are not independent).

## Benchmarks it must beat

A recommender-based strategy is only interesting if it beats strategies that
need no recommender:

| Benchmark | Rule | Why |
| --- | --- | --- |
| Hold ETH | hold WETH over the same periods | the asset the strategy trades out of |
| Popularity | buy the 5 tokens with the most buyers in the last 24 hours | the recommender's baseline as a trading rule |
| Momentum | buy the 5 tokens with the largest price rise in the last 6 hours | the obvious alternative signal |
| Flow | buy the 5 tokens with the largest net buy inflow in the last hour ([`possible_applications.md`](possible_applications.md), section 4) | the visible demand the surprise term tries to look past |
| Tail popularity | buy the 5 tail tokens with the most buyers in the last 24 hours | the same universe, chosen by visible demand: the baseline restricted to where the strategy trades |
| Tail flow | buy the 5 tail tokens with the largest net buy inflow in the last hour | the same universe, chosen by visible inflow |
| Random | buy 5 random tokens from the same universe | shows what the universe alone does after costs |

The decisive comparisons are with **tail popularity** and **tail flow**: the
same universe of tail tokens, picked by visible demand instead of predicted
demand. They isolate the recommender's information from the universe: if tail
tokens simply rise or fall as a group, those benchmarks do the same. The
whole-market benchmarks (popularity, flow, momentum over all tokens) show
whether the tail is worth trading at all.

## What is reported

- Return per period after costs, its spread, and the cumulative curve, next to
  the benchmarks.
- Hit rate (share of trades with a positive return), average gain and loss,
  maximum drawdown.
- The same numbers **before costs**: if the signal earns before costs and loses
  after, the honest result is "the information is real but not tradable at this
  size".
- **Does the signal do what it claims?** The correlation between $D_i(s)$ and the
  realized number of new buyers in the next $H$ hours, and between realized new
  buyers and returns. The strategy needs both links: the first is what the
  project measured, the second is the new claim.
- The return by token age and with the coordinated groups' share of the buyers,
  to see whether gains come from farms' pumps (likely to vanish) or from broad
  adoption.
- **Capacity:** the capital the strategy could have used under the 1%-of-volume
  cap. A tail strategy can be right and still too small to matter; this number
  says how small.

## How such a backtest fools itself

| Trap | Here | Guard |
| --- | --- | --- |
| Prices you could not get | the swap price is the price someone else got, before your trade's own impact | entry at a later swap; impact estimate; position cap at 1% of volume |
| Tokens you could not sell | honeypots and transfer-tax tokens show buys and few or no sells | the universe needs sells by 20+ accounts; a failed exit is valued at zero |
| Tuning on the reported period | dozens of settings, one good-looking curve | settings on Dev A, version on Dev B, held-out period once |
| Overlapping positions | a 24-hour hold with 6-hourly decisions overlaps four positions | returns accounted per period with all open positions; bootstrap over days |
| Being first | the backtest assumes nobody else trades on the same visible information | report how much of the return is earned in the first block after entry; if most of it, assume competitors take it |
| Farm pumps | coordinated groups buy together and sell together | report returns with and without tokens whose buyers are mostly in groups |

## Effort

| Step | Effort |
| --- | --- |
| Decimals, fee tiers, price per swap, ETH price | 1 day |
| Signal at 6-hourly decision times on all periods (the recommender is ready) | 1 day plus compute (one ranker run per decision time) |
| Backtest engine (positions, costs, exits, accounting per period) with tests | 2 days |
| Benchmarks, tuning on Dev A/B, report | 2 days |
| Optional: reserves / liquidity download for price impact | 4–18 hours unattended, as for the swaps |

About a week of work. The most likely result is honest and useful either way:
either the personalized part of the recommender carries tradable information
beyond aggregate popularity, or it does not survive costs and competition.
