# Literature review: trading bot fingerprinting with probabilistic automata

Review for the idea in
[`../idea_bot_fingerprinting_automata.md`](../idea_bot_fingerprinting_automata.md):
learn a probabilistic automaton from each trader's sequence of on-chain
actions on a decentralized exchange, and use the automata to cluster traders
and to identify bots.

## 1. Scope and method

The review covers four questions:

1. What kinds of bots trade on decentralized exchanges, and how are they
   detected today?
2. Has machine learning been used to detect bots or to recognize users from
   their on-chain behaviour?
3. How are probabilistic automata learned from sequences, and where has that
   been used for behavioural fingerprinting?
4. What does this imply for labels, features and evaluation?

Thirteen open-access papers were downloaded to `papers/` (PDFs are ignored by
git, the links below are the source of record). The "Read" column says how much
of each was read for this review.

| # | Paper | File in `papers/` | Read |
| --- | --- | --- | --- |
| 1 | Daian et al. (2019), [Flash Boys 2.0: frontrunning, transaction reordering, and consensus instability in decentralized exchanges](https://arxiv.org/abs/1904.05234) | `daian2019_flash_boys_2.pdf` | Abstract |
| 2 | Zhou et al. (2020), [High-frequency trading on decentralized on-chain exchanges](https://arxiv.org/abs/2009.14021) | `zhou2020_hft_dex_sandwich.pdf` | Abstract |
| 3 | Qin, Zhou, Gervais (2021), [Quantifying blockchain extractable value: how dark is the forest?](https://arxiv.org/abs/2101.05511) | `qin2021_quantifying_bev.pdf` | Abstract |
| 4 | Torres, Camino, State (2021), [Frontrunner Jones and the raiders of the dark forest](https://arxiv.org/abs/2102.03347) | `torres2021_frontrunner_jones.pdf` | Abstract |
| 5 | Weintraub et al. (2022), [A flash(bot) in the pan: measuring maximal extractable value in private pools](https://arxiv.org/abs/2206.04185) | `weintraub2022_flashbot_in_the_pan.pdf` | Introduction (start) |
| 6 | Heimbach et al. (2023), [Ethereum's proposer-builder separation: promises and realities](https://arxiv.org/abs/2305.19037) | `heimbach2023_pbs.pdf` | Title only (downloaded, not read) |
| 7 | Victor, Weintraud (2021), [Detecting and quantifying wash trading on decentralized cryptocurrency exchanges](https://arxiv.org/abs/2102.07001) | `victor2021_wash_trading_dex.pdf` | Abstract |
| 8 | Park et al. (2023), [Unraveling the MEV enigma: ABI-free detection model using graph neural networks](https://arxiv.org/abs/2305.05952) | `park2023_mev_detection_gnn.pdf` | Abstract |
| 9 | Niedermayer, Saggese, Haslhofer (2024), [Detecting financial bots on the Ethereum blockchain](https://arxiv.org/abs/2403.19530) | `niedermayer2024_financial_bots_ethereum.pdf` | Abstract, definition of a bot, features, conclusions |
| 10 | Béres et al. (2020), [Blockchain is watching you: profiling and deanonymizing Ethereum users](https://arxiv.org/abs/2005.14051) | `beres2020_profiling_ethereum_users.pdf` | Abstract, contributions, conclusion |
| 11 | Verwer, Hammerschmidt (2022), [FlexFringe: modeling software behavior by learning probabilistic automata](https://arxiv.org/abs/2203.16331) | `verwer2022_flexfringe.pdf` | Abstract, introduction (start) |
| 12 | Hammerschmidt et al. (2016), [Interpreting finite automata for sequential data](https://arxiv.org/abs/1611.07100) | `hammerschmidt2016_interpreting_automata.pdf` | Abstract |
| 13 | Mao et al. (2012), [Learning Markov decision processes for model checking](https://arxiv.org/abs/1212.3873) | `mao2012_learning_mdp_model_checking.pdf` | Title only (downloaded, not read) |

**This review is thinner than the other three.** Most papers were read at
abstract level, two were not read at all, and the core references on automata
learning could not be downloaded (section 5). The summaries of the MEV papers
rest on their abstracts; the statements about automata learning in sections
2.3 and 3 rest partly on general knowledge of the field and are marked where
that is the case.

## 2. Summaries

### 2.1 Bots on decentralized exchanges

**[1] Daian et al. (2019).** The paper that named the phenomenon. It
documents "the widespread and rising deployment of arbitrage bots" on
decentralized exchanges, which pay high fees and optimize latency "to
frontrun, i.e., anticipate and exploit, ordinary users' DEX trades". Bots
compete in "priority gas auctions", bidding up fees for an early position in
the block. The value available from ordering transactions is called miner
extractable value (MEV).

**[2] Zhou et al. (2020).** Formalizes the **sandwich attack**: a bot places
one trade before and one after a victim's trade in the same block. Its
feasibility depends on "the relative positioning of a transaction within a
blockchain block".

**[3] Qin et al. (2021).** Measures the value extracted by sandwich attacks,
liquidations and arbitrage over 32 months: 540 million USD, "divided among
11,289 addresses". The detection is by rules applied to on-chain data.

**[4] Torres et al. (2021).** Rule-based detection of three kinds of
frontrunning (displacement, insertion, suppression) over 11 million blocks,
finding almost 200,000 attacks.

**[5] Weintraub et al. (2022)** and **[6] Heimbach et al. (2023)** concern
the move of this activity from the public transaction pool to private
channels and block builders. Read only minimally; relevant because bots'
transactions increasingly never appear in public before inclusion, which
affects what can be observed.

**[7] Victor and Weintraud (2021).** Detects wash trading on two order-book
exchanges from trading structures between accounts. More than 30% of traded
tokens were affected. An example of defining illicit behaviour through a
structural pattern.

**[8] Park et al. (2023), ArbiNet.** A graph neural network that detects
arbitrage and other MEV transactions from token-transfer graphs, without
knowledge of contract interfaces. The authors state that existing rule-based
detectors have "low recall" and need "continuous monitoring of new DeFi
services".

### 2.2 Machine learning on account behaviour

**[9] Niedermayer et al. (2024): the closest prior work.** Machine-learning
detection of financial bots on Ethereum.

- **Definition:** "A bot is an EOA [externally owned account] that has sent
  at least one transaction that was compiled and sent by a software program
  without human oversight."
- **Taxonomy:** 7 categories and 24 subcategories of financial bots.
- **Ground truth:** 133 human and 137 bot addresses, collected by hand. The
  authors acknowledge that the sample is small.
- **Features:** 83 per account, in four groups: from the address itself, from
  transactions, from function calls, and from events. Many are aggregates
  over the account's history (for example a test against Benford's law, and
  a "sleepiness" measure of gaps in activity).
- **Results:** a random forest reaches 83% accuracy on bot against human;
  a Gaussian mixture model reaches 82.6% average cluster purity. Tree models
  also work on a four-class task (three kinds of MEV and non-MEV).
- **Most informative features:** those related to "time, frequency, gas
  price, and gas limit of outgoing transactions". Bots differ from humans in
  their "activity / inactivity patterns".
- The paper notes that rule-based MEV detectors can be read as bot detectors,
  by taking the senders of detected transactions as bots.

**[10] Béres et al. (2020).** Profiling and de-anonymizing Ethereum users.

- Behavioural "quasi-identifiers" (time-of-day activity, the gas price a user
  sets, position in the transaction graph) can link addresses belonging to
  the same user.
- **Ground truth without manual labeling:** addresses registered under the
  same Ethereum Name Service name belong to the same user.
- Graph embeddings outperform the activity-based profiles.
- The task is posed as ranking: for an address, rank candidate addresses by
  similarity and see where the true partner lands.

### 2.3 Learning probabilistic automata

**[11] Verwer and Hammerschmidt (2022), FlexFringe.** An open-source package
for learning probabilistic deterministic finite automata by **state
merging**: start from a tree of all observed sequences and merge states whose
futures are statistically compatible. Applied to software logs, the learned
automata serve as interpretable models and as anomaly detectors, "competitive
with an existing solution based on neural nets". The authors also report a
tension: "learning smaller, more convoluted models improves the performance"
on anomaly detection, at the cost of interpretability.

**[12] Hammerschmidt et al. (2016).** Discusses what "interpretable" means
for automata and shows uses for "prediction, classification, and clustering
on sequential data".

**[13] Mao et al. (2012)** was downloaded as the open reference for the
ALERGIA family of algorithms applied to system models, but not read.

Not downloaded, known from the search results: **Pellegrino et al. (2017),
"Learning behavioral fingerprints from netflows using timed automata"**. It
learns automata from the network traffic of hosts infected with malware; the
automata "encode patterns of short-term interactions" and serve as "compact
behavioral fingerprints" that also detect previously unseen malware. This is
the nearest analogue of our idea in another domain: botnets in place of
trading bots, network flows in place of swaps.

From general knowledge of the field, not from a paper read here: in the
PAutomaC competition on learning probabilistic automata (Verwer et al.,
2014), simple n-gram models were strong baselines, and state-merging methods
did not uniformly beat them. This should be verified in the source.

## 3. Insights gathered

1. **Bot activity on decentralized exchanges is large, structured and well
   described** [1, 2, 3, 4]. The main types (arbitrage, sandwich, other
   frontrunning) have precise definitions.
2. **Detection today is rule-based** [3, 4]. Rules find what they were
   written for and miss new variants [8, 9].
3. **Labels are the bottleneck.** The only hand-labeled set found has 270
   addresses [9]. Everything else is produced by rules [3, 4, 8].
4. **Labels produced by rules are functions of the same data a model would
   use.** A model trained on them learns the rule back. None of the papers
   reviewed discusses this in the passages read, but it follows directly from
   how the labels are made.
5. **Existing machine-learning detectors summarize an account by aggregate
   features** [9]. The order of actions is not modelled. That is the gap the
   idea aims at.
6. **Timing is the most informative signal for bot against human** [9], and
   position within the block defines sandwiches [2].
7. **Behaviour can identify a user, not only a type** [10], and identity gives
   ground truth for free when a linking source exists.
8. **Identification is evaluated as ranking** [10].
9. **Automata have been used as behavioural fingerprints** for malware in
   network traffic (Pellegrino et al.) and for software [11], with
   interpretability as the main argument [12].
10. **Interpretability and accuracy pull in different directions** for
    learned automata [11].
11. **Bots increasingly act through private channels and their own
    contracts** [5, 6]. The contract a bot uses is a more stable unit than
    the account that signs for it.

## 4. What this means for our idea

**The gap is real.** No paper found models the *sequence* of a trader's
actions on a decentralized exchange, with automata or with anything else.
Aggregate features [9] and transaction-level graph models [8] are the state
of the art. This rests on a limited search and should be checked further
before it is claimed.

**The labeling problem decides the design.** Insights 3 and 4 mean that a
"bot type" classifier trained on rule-based MEV labels would be circular, and
that no large independent label set exists. Three consequences, which the
first experiments in
[`../../src/notebooks/04_bot_fingerprinting_data_and_baseline.ipynb`](../../src/notebooks/04_bot_fingerprinting_data_and_baseline.ipynb)
adopt:

- **Make identification the primary task.** "Is this the same trader as that
  one?" has exact, free ground truth (the address), as in [10], and it is
  what "fingerprinting" means. It needs no bot labels at all.
- **For bot against non-bot, use a label that does not come from the action
  sequence:** whether the trader calls pools through its own contract or
  through a public router. This follows insight 11 and is independent of the
  features, though it is a proxy and a noisy one.
- **Leave bot types (arbitrage, sandwich) for later,** and if used, as
  descriptions of what a learned automaton shows, not as training labels.

**Changes to the idea that follow from the literature:**

| Original plan | Change | Source |
| --- | --- | --- |
| Labels from public MEV datasets | Not available as planned (the public sources named in the idea were unreachable or need keys), and circular if used for training. Replaced as above. | [3, 4, 9] |
| Classify bots and bot types | Identification first, bot against router user second, types last. | [9, 10] |
| Alphabet from direction, size, time gap, pool | Keep, with timing given the most resolution, including "same transaction" and "same block". | [2, 9] |
| Unit of analysis: the trader's address | For bots, the contract that calls the pools; for others, the signing account. | [5, 6] |
| Baselines: hand-built features, n-grams, HMM, neural models | Keep n-grams as the main baseline, expected to be strong. Add a reference that uses only *where* a trader trades. | General knowledge on PAutomaC; [10] |
| Evaluate by precision, recall, F1 | For identification: top-1 and top-5 accuracy and mean reciprocal rank. | [10] |
| "Interpretable models whose states can be read" | Keep as the main argument for automata, but treat it as a claim to be demonstrated, and report model size next to accuracy. | [11, 12] |

**A sharper research question.** *Does the order of a trader's actions carry
identifying information beyond the frequencies of those actions, and does a
learned automaton capture it more compactly or more readably than an n-gram
model?* This is specific, has exact ground truth, and separates the two
things the idea bundles: whether sequence matters, and whether automata are
the right way to use it.

## 5. Papers to obtain

Not downloaded (paywalled or blocked).

| Paper | Why it matters |
| --- | --- |
| Carrasco, Oncina (1994), "Learning stochastic regular grammars by means of a state merging method" | The ALERGIA algorithm, used here through the AALpy library. Its compatibility test and parameter should be understood from the source. |
| Verwer, Eyraud, de la Higuera (2014), "PAutomaC: a probabilistic automata and hidden Markov models learning competition", Machine Learning | The benchmark for exactly this kind of model, and the source for the claim that n-grams are strong baselines. |
| Pellegrino, Lin, Hammerschmidt, Verwer (2017), "Learning behavioral fingerprints from netflows using timed automata", IFIP/IEEE IM | The nearest analogue of the idea. Its handling of time (timed automata instead of time-gap symbols) may be directly reusable. |
| Muškardin et al. (2022), "AALpy: an active automata learning library", Innovations in Systems and Software Engineering | The library used for ALERGIA in the experiments. |
| de la Higuera (2010), "Grammatical Inference: Learning Automata and Grammars" | Textbook for the method. |
| Mao et al. (2016), "Learning deterministic probabilistic automata from a model checking perspective", Machine Learning | The journal version of [13], with the analysis of ALERGIA variants. |
