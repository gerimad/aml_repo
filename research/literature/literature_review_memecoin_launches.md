# Literature review: the first seconds of a memecoin launch

Review for the idea in
[`../idea_memecoin_launch_outcomes.md`](../idea_memecoin_launch_outcomes.md):
from what is visible in the first seconds of a pump.fun launch, how bundled
it is and whether its name is a copy, model what happens next.

## 1. Scope and method

The review covers five questions:

1. What is already known about pump.fun launches and what predicts their
   outcome?
2. How do other studies define and label a rug pull, and how early do they
   claim to detect one?
3. What is known about bundles, snipers and serial creators?
4. What is known about copied token names?
5. Which methods for time-to-event data and for tracking copies apply?

Fifteen open-access papers were downloaded to `papers/` (PDFs are ignored by
git, the links below are the source of record). The "Read" column says how
much of each was read for this review.

| # | Paper | File in `papers/` | Read |
| --- | --- | --- | --- |
| 1 | Marino, Naviglio, Tarantelli, Lillo (2026), [Predicting the success of new crypto-tokens: the Pump.fun case](https://arxiv.org/abs/2602.14860) | `marino_predicting_success_pumpfun_2602.14860.pdf` | Introduction, platform, dataset, start of the dump section, conclusions (about a third of 29 pages) |
| 2 | Li, Kuznetsov, Yanovich, Nott-Whaley, Vodolazov (2026), [Catching the rug: early prediction of fraudulent memecoins on Solana via machine learning](https://arxiv.org/abs/2608.20271) | `li_catching_the_rug_2608.20271.pdf` | Abstract, related work, label definition, conclusion. Result tables not read |
| 3 | Hu, Tekin, Xu, Liu (2026), [MELT: a behavioral trace dataset for high-risk memecoin launch detection](https://arxiv.org/abs/2602.13480) | `memetrans_2602.13480.pdf` | Abstract, parts of the data and experiment sections, limitations |
| 4 | Chen et al. (2026), [From hype to collapse: investigating rug pull scams on Solana](https://arxiv.org/abs/2603.24625) | `solrugdetector_2603.24625.pdf` | Abstract, contributions, one table, limitations |
| 5 | Kamat (2026), [Coordinated sniper cohorts on pump.fun](https://arxiv.org/abs/2607.02795) | `pumpfun_sniper_cohorts_2607.02795.pdf` | Correction notice, abstract, introduction |
| 6 | Mancino (2025), [The memecoin phenomenon: an in-depth study of Solana's blockchain trends](https://arxiv.org/abs/2512.11850) | `memecoin_phenomenon_solana_2512.11850.pdf` | Abstract |
| 7 | Alhaidari, Kalal, Palanisamy, Sural (2025), [SolRPDS: a dataset for analyzing rug pulls in Solana decentralized finance](https://arxiv.org/abs/2504.07132), CODASPY | `alhaidari_solrpds_2504.07132.pdf` | Abstract |
| 8 | Cernera, La Morgia, Mei, Sassi (2023), [Token spammers, rug pulls, and sniper bots](https://arxiv.org/abs/2206.08202), USENIX Security | `cernera_token_spammers_rug_pulls_sniper_bots_2206.08202.pdf` | Abstract, contributions, the passage on sniper bots |
| 9 | Xia et al. (2021), [Trade or trick? Detecting and characterizing scam tokens on Uniswap decentralized exchange](https://arxiv.org/abs/2109.00229) | `xia_trade_or_trick_2109.00229.pdf` | Abstract, the passages on names and on the classifier |
| 10 | Mazorra, Adan, Daza (2022), [Do not rug on me: zero-dimensional scam detection](https://arxiv.org/abs/2201.07220) | `mazorra_do_not_rug_on_me_2201.07220.pdf` | Abstract, the passages on concentration and on evaluation points |
| 11 | Gao et al. (2020), [Tracking counterfeit cryptocurrency end-to-end](https://arxiv.org/abs/2011.02673) | `gao_counterfeit_cryptocurrency_2011.02673.pdf` | Abstract, definitions |
| 12 | Leskovec, Backstrom, Kleinberg (2009), [Meme-tracking and the dynamics of the news cycle](https://cs.stanford.edu/people/jure/pubs/quotes-kdd09.pdf), KDD | `leskovec_meme_tracking_kdd09.pdf` | Abstract, the passage on clustering variants |
| 13 | Cheng, Adamic, Dow, Kleinberg, Leskovec (2014), [Can cascades be predicted?](https://arxiv.org/abs/1403.4608), WWW | `cheng_can_cascades_be_predicted_1403.4608.pdf` | Abstract, problem definition, the passage on content features |
| 14 | Reimers, Gurevych (2019), [Sentence-BERT](https://arxiv.org/abs/1908.10084), EMNLP | `reimers_sentence_bert_1908.10084.pdf` | Abstract |
| 15 | Ishwaran, Kogalur, Blackstone, Lauer (2008), [Random survival forests](https://arxiv.org/abs/0811.1645), Annals of Applied Statistics | `ishwaran_random_survival_forests_0811.1645.pdf` | Abstract, the passage on the concordance index |

**This is a reading of parts, not of whole papers.** Only [1] was read at
length. Papers [1] to [6] are preprints from 2025 and 2026 and, as far as I
could see, not peer reviewed; their claims are reported as claims. The
numerical results of [5] have been withdrawn by its author (see 2.3).

The dataset's own documentation (`README.md` and `KNOWN_ISSUES.md` of
[`loopholetape/pumpfun-launches`](https://huggingface.co/datasets/loopholetape/pumpfun-launches))
was read in full and is used throughout the study. It is documentation, not
literature, but it shaped more decisions than any paper here.

## 2. Summaries

### 2.1 Pump.fun launches and what predicts their outcome

**[1] Marino et al.** is the closest study to this idea.

- **Data.** One month of pump.fun (September 2025): 655,770 tokens from
  243,123 creators, decoded from the chain by the authors. 0.63% graduate.
- **Mechanism.** A token starts on a "virtual" bonding curve with 30 SOL of
  synthetic reserve. It graduates when about 85 real SOL have been paid in,
  and trading moves to an ordinary pool. Liquidity cannot be added or
  removed during the curve phase, so there is no pool for a creator to
  drain.
- **Method.** The probability of graduation is estimated as a function of
  the SOL already in the curve, then conditioned on further variables: the
  share of bot-like trades, the number of trades it took to get there, the
  presence of previously successful traders, and whether the creator is
  among the most prolific. Every variable uses only what happened up to the
  current point on the curve.
- **Findings.** The SOL in the curve is the natural state variable.
  Reaching a level in few trades is the strongest additional predictor.
  Bot-heavy tokens graduate less. Successful traders add little. Prolific
  creators add nothing.
- **Dumps.** Among tokens with at least 30 swaps, 92% show at least one
  "dump" by the authors' price-based rule.
- **Stated future work:** a hazard-rate formulation, and features from the
  token's name and image.

**[6] Mancino** describes the platform in late 2024: it issued up to 71% of
all tokens minted on Solana, and fewer than 2% moved on to an exchange.

### 2.2 Rug pull labels and early detection

- **[2] Li et al.** assemble 6.4 million Solana tokens over seven months and
  predict a one-hour outcome from the first five minutes of trading. A token
  is labelled a rug pull if its locked value collapses or if it is idle for
  most of its life. By this definition "a vast majority" of tokens are rug
  pulls within an hour. Split by time. Boosted trees do best. Their raw
  data includes bundle fields and the mayhem flag.
- **[3] Hu et al. (MELT)** publish 41,000 launches *that graduated*, with
  200 million transactions parsed into typed records, and trace bundles by
  linking accounts of one entity. On average 36.5% of the supply sits with
  coordinated accounts. 98.7% of creations include a developer purchase in
  the same transaction. The risk label comes from the price path after
  graduation. Models on designed features beat sequence models, and
  transformers do worst. Their stated limitation: tokens that never
  graduate are not covered.
- **[4] Chen et al.** verify 117 rug pulls by hand, derive three patterns
  (freezing accounts, withdrawing liquidity, pump and dump), and label
  76,469 of 100,063 tokens on three Solana exchanges. The median rug pull
  token lives for 0.012 days, about 17 minutes. They state that the labels
  are retrospective and a lower bound. Of 78 large groups of scam tokens,
  16 are a single address deploying repeatedly.
- **[7] Alhaidari et al.** publish 62,895 suspicious Solana pools from 2021
  to 2024, annotated by inactivity.
- **[9] Xia et al.** flag about half of the tokens listed on Uniswap V2 as
  scams. Their classifier scores 96% precision and recall under ten-fold
  cross-validation on randomly shuffled tokens.
- **[10] Mazorra et al.** extend that dataset and report 99% accuracy in
  detecting scams "before the malicious maneuver". For a scam token, the
  moments at which it is evaluated are drawn from before its largest drop,
  which is itself known only afterwards. They use the concentration of
  holdings as a feature.

### 2.3 Bundles, snipers and serial creators

- **[8] Cernera et al.** study Ethereum and BNB Chain up to March 2022.
  About 60% of tokens are active for less than a day. 1% of addresses create
  20 to 25% of all tokens, as disposable tokens for what they call the
  one-day rug pull. A single attempt succeeds 40 to 60% of the time, and
  the low cost makes repeating worthwhile. Sniper bots buy a token the
  moment it is listed.
- **[5] Kamat** looks for small groups of wallets that repeatedly appear
  among the first buyers of pump.fun launches, and estimates their effect on
  later buying. **The numerical results are withdrawn.** A notice of October
  2026 on the paper states that 43% of the rows labelled as buys were sells
  and that only 23% of real buys were recorded. Two methodological points
  stand without the numbers: an outcome must not count the treated wallets'
  own purchases, and most of a naive difference disappears once launches
  are matched on quality.
- **[3]** (above) is the only study here that measures bundles directly.

### 2.4 Copied names

- **[11] Gao et al.** find 2,117 counterfeit tokens on Ethereum imitating 94
  of the 100 largest tokens. Two kinds: the identical name from a different
  creator, and a known name with characters or words added
  ("combo-squatting").
- **[9]** notes that scam tokens share names heavily: 429 tokens carry the
  name of one well-known project.
- **[4]** includes an analysis of the names of rug pull tokens, which I did
  not read.

None of these studies looks at names as a cascade in time: who is first,
how fast copies follow, and whether position matters for the outcome.

### 2.5 Methods

- **[12] Leskovec et al.** track short phrases through 90 million news and
  blog articles. Variants of a phrase are linked when their word-level edit
  distance is small or they share a long run of words, and the linked graph
  is cut into clusters. A meme rises and falls over days.
- **[13] Cheng et al.** predict the growth of reshare cascades. They pose the
  question as "will a cascade of size k reach 2k", which keeps the classes
  balanced at every k. Timing and structure predict; content hardly does.
  The same content posted independently grows to very different sizes.
- **[14] Reimers and Gurevych** produce sentence embeddings that can be
  compared by cosine similarity, which makes similarity search cheap.
- **[15] Ishwaran et al.** extend random forests to censored survival times
  and evaluate with the concordance index: the share of pairs in which the
  case that fails first was given the worse prediction.

## 3. Insights gathered

1. **The SOL in the curve is the state variable, and it is strong.** [1]
   builds everything on it. Any feature has to be judged by what it adds to
   it.
2. **"Rug pull" has no agreed definition, and loose ones saturate.** [2]
   finds a vast majority of tokens are rug pulls, [9] half, [4] three
   quarters, each under its own rule. On pump.fun there is no liquidity to
   withdraw [1], so rules built for exchange pools do not carry over.
3. **Very high scores in this literature often come with a weak protocol.**
   [9] shuffles tokens at random across time. [10] places its evaluation
   points using the outcome. The time-split studies [2] [3] report far more
   modest numbers.
4. **Bundles have been measured on graduated tokens only** [3]. The long
   tail of launches that never graduate, which is nearly all of them, is
   named as a gap by the authors.
5. **Serial creators are the norm on every chain** [8] [4]. A few addresses
   produce a large share of tokens and treat each as disposable.
6. **Decoding pump.fun trades is error-prone.** [5] was withdrawn over
   misread buys, and the dataset used here missed half the buys until a fix
   on 27 September 2026. Counts of early buyers deserve suspicion wherever
   they come from.
7. **An outcome must not contain the treatment.** [5]'s one robust point:
   counting a cohort's own purchases in "later buying" doubles the apparent
   effect.
8. **Names are reused heavily by scam tokens** [9] [11], but nobody has
   studied the order of copies.
9. **In cascades, timing predicts and content does not** [13], and how a
   prediction task is posed decides whether it is trivially unbalanced.
10. **Simple tabular models beat sequence models** on launch data [3], and
    [1] names hazard models and name features as open work.

## 4. What this means for our idea

**The idea is mostly new in what it asks, and well covered in how to ask
it.** [1] did the graduation question a year earlier on the same platform
and names both of this idea's additions, a hazard formulation and token
names, as future work. Bundling on ungraduated launches is the gap [3]
names. That supports the choice of topic. It also sets the bar: the
additions have to beat the SOL in the curve.

The feasibility study (notebook 07) followed the literature at each of the
points below.

| Point | From | What was done, and what came of it |
| --- | --- | --- |
| Judge features against the SOL in the curve | [1] | It is in the fixed baseline. Bundling then adds +0.003 AUC |
| Do not trust a rug label before looking at it | [2] [4] [9] | The dataset's label turned out to be close to "the curve reached 5 SOL and did not graduate". It was dropped as a target |
| An outcome must not contain the treatment | [5] | The same fault in a new place: bundle wallets put SOL into the curve, and the rug label depends on the SOL in the curve |
| Use only what is known at the prediction moment | [1] [10] | The moment is fixed at 3 seconds. Launches decided by then are excluded. Histories count an earlier outcome only once it had happened |
| Split by time | [2] [3] | Four days train, one dev, two test |
| Serial creators | [8] [4] | 1% of creators make 33% of launches. Intervals resample creators, and first-time creators are reported separately |
| Tabular models first | [3] | Logistic regression and boosting. Boosting helped on one task and hurt on the other |
| Hazard formulation | [1] [15] | Tried. A proportional-hazards model fits worse than a classifier per horizon, and there is no censoring to handle |
| Names as features | [1] [9] [11] | Tried. Worth 0.003 AUC or less |
| Variants of a name | [11] [12] | Normalization, then character n-grams. Most near-duplicates are a known name with text added, as in [11] |
| How the prediction task is posed | [13] | The "first is best" pattern of the first look came from selecting names by their later popularity, the trap [13]'s formulation avoids |

**Three findings of the first look are contradicted by the literature or by
the closer look it prompted.**

- **Graduation rates.** The first look found 2.2% graduating, against 0.63%
  in [1] a year earlier. Nearly half of the difference is launches in which
  the creator buys the whole curve at once, a practice [1] does not
  describe. Without those and without the instant fills, about 1.1%
  graduate.
- **Prolific creators.** [1] finds that knowing a creator is prolific does
  not help predict graduation. This study agrees for graduation. For the
  creator's own selling it is the opposite: the creator's record is the
  strongest feature by far.
- **Bundling as a risk signal.** [3] presents bundle traces as the key to
  risk. On ungraduated launches, with the curve's size held fixed, this
  study finds almost no added signal. The two are not in conflict: [3]
  studies price paths after graduation, with bundles traced through
  transfers, while this dataset counts wallets that bought in the first
  block.

**What the literature cannot settle.** Whether a token is a fraud. Every
label in section 2.2 is a rule about observable behaviour, and so are the
targets of this study.

## 5. Papers to obtain

Not downloaded; cited in the idea file from memory.

- Cox (1972), "Regression models and life-tables", Journal of the Royal
  Statistical Society B. Behind a paywall.
- Fine and Gray (1999), "A proportional hazards model for the
  subdistribution of a competing risk", JASA. Behind a paywall.
- Harrell et al. (1982), "Evaluating the yield of medical tests", JAMA: the
  concordance index, known here through [15].
- Solidus Labs (2025), "Rug pulls and pump-and-dumps on Solana": an industry
  report claiming about 99% of pump.fun tokens show such patterns. Seen only
  through news coverage.
- Srifa et al. on Uniswap V3 rug timing, Cao et al. on wash trading in BNB
  Chain meme tokens, and Yaremus et al. on the TON chain, all cited by [2].

Worth reading in full before a project: [1] (the conditioning method and
the dump section), [3] (how bundles are traced), and the name analysis in
[4].
