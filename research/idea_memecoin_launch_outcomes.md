# Idea: the first seconds of a memecoin launch, bundles, copycats and outcomes

More than forty thousand tokens are launched on pump.fun every day. Almost
all are dead within minutes. Two things are visible in the first seconds of a
launch: **how bundled it is** (how much of the supply insiders bought at once
through several wallets) and **whether it is a copy** (the same name as a
token that just took off). Model what those two say about what happens next:
a rug, and how fast; or one of the rare successes.

This combines two ideas: how the degree of bundling relates to the time
until a rug (see
[`idea_bundled_launch_survival.md`](idea_bundled_launch_survival.md)), and
copycat cascades in token names.

> **Update after the feasibility study**
> ([`memecoin_launch_writeup.md`](memecoin_launch_writeup.md)). Three
> statements below are wrong or misleading and are kept as written:
>
> - **"Graduated within ten seconds" is mostly the creator buying the whole
>   curve at launch** (45% of all graduations), not a fast success.
> - **The "rugged" label is close to "the curve reached 5 SOL and did not
>   graduate".** The table of bundling against rugs below largely reflects
>   that bundles put SOL into the curve.
> - **"The first use of a popular name graduates five times as often"** came
>   from selecting names that later became popular. Over all names and 34
>   days of history, position makes almost no difference.
>
> The share of repeated names dominated by one creator is 10%, not 27%.

## Summary

- **Data:** a public dataset of 1.18 million pump.fun launches on Solana, with
  each launch's name, creator, bundle wallets, insider share, and outcomes
  with their timing.
- **Topics covered:** classification and survival analysis with competing
  outcomes; NLP on short texts (near-duplicate detection, semantic
  similarity, clustering into memes); unsupervised learning; feature
  engineering; the learner survey (logistic and Cox regression, boosting,
  nearest neighbours).
- **Why high quality:** over a hundred thousand labeled events per week
  instead of the dozens or hundreds of the earlier ideas, effects that are
  visible in a first look, and a dataset whose author documents its flaws.
- **What you learn:** survival analysis and competing risks, text clustering
  at scale, separating selection from effect, and avoiding label leakage
  when outcomes arrive within seconds.
- **Who benefits:** retail buyers of new tokens, who lose money to rugs and
  to late copies, and the tools that warn them.
- **Novelty:** medium to high. Bundle checkers and copy-trading tools exist;
  I am not aware of a study of naming cascades. This is my estimate, not the
  result of a literature search.

## Data

**Source:** `loopholetape/pumpfun-launches` on Hugging Face (licence
CC BY 4.0): 1,180,869 launches from 31 August to 3 October 2026. According
to its documentation, the 321,661 launches from 27 September on are fully
recorded, and the earlier ones undercount buys.

**A first look.** I downloaded four days of the fully recorded part (28
September to 1 October 2026, 184,985 launches) to a scratch folder, not into
`data/`, and computed the numbers below. They are a first pass on four days,
not a study.

| | |
| --- | --- |
| Launches per day | 42,000 to 50,000 |
| Creators | 54,203; 77% launch once; the top 1% make 43% of all launches; one made 2,625 |
| Rugged (of launches with a resolved outcome) | 73% |
| Time from launch to rug | median 26 seconds; 90% within 3 minutes |
| Kind of rug | 66% creator dump, 34% drain of the curve |
| Graduated (completed the launch curve) | 2.2% |
| Graduations within 10 seconds of launch | 59% of all graduations |
| Graduations later than 60 seconds | 0.63% of launches |
| Launches whose exact name was already used in these four days | 66% |
| Names used 20 times or more | 1,100 names, covering 62,019 launches |

**Bundling and outcome** (launches with early features and a resolved
outcome):

| Bundle wallets at 10 seconds | Launches | Rugged | Graduated |
| --- | --- | --- | --- |
| 0 | 87,904 | 66% | 1.9% |
| 1 to 2 | 31,123 | 74% | 0.7% |
| 3 to 5 | 21,171 | 92% | 3.3% |
| 6 or more | 9,774 | 92% | 8.2% |

**Copying and outcome** (names used 20 times or more, by order of
appearance within the four days):

| Position | Launches | Graduated | Rugged | Median peak in the curve |
| --- | --- | --- | --- | --- |
| 1st | 1,100 | 9.7% | 72% | 2.5 SOL |
| 2nd to 5th | 4,400 | 4.4% | 76% | 1.6 SOL |
| 6th to 20th | 16,500 | 3.3% | 75% | 1.1 SOL |
| 21st and later | 40,019 | 1.7% | 84% | 0.6 SOL |
| For comparison: names used once | 42,689 | 0.7% | 61% | |

Five properties shape the project:

- **Bundling cuts both ways.** Heavily bundled launches are rugged far more
  often, and they also graduate far more often. They are the extremes in
  both directions, so "more bundled, shorter life" is too simple.
- **"Graduated" mostly does not mean "succeeded".** More than half of the
  graduations happen within ten seconds: the curve is bought out at once,
  presumably by the launcher. Organic success, a graduation after the first
  minute, is under 1% of launches. The two must be separate outcomes.
- **Everything happens in seconds.** With a median rug at 26 seconds, a
  feature measured at 10 seconds is often measured during the event. This is
  the main leakage trap.
- **Copies are the rule.** Two thirds of launches reuse a recent name. Half
  the copies of a popular name come more than a day after its first use, and
  a tenth within four minutes.
- **A few creators make much of the data.** In 27% of the repeated names, a
  single creator made more than half of the launches. Those are one account
  launching the same token over and over, not a crowd imitating a hit.

## Research questions

1. **Bundling as a dose.** How do the chances of a rug, an instant buyout
   and an organic graduation change with the number of bundle wallets and
   the insiders' share? Is there a threshold?
2. **How fast.** Given that a launch will be rugged, does bundling change
   how soon?
3. **What is a meme.** Can launches be grouped into memes by their names,
   beyond exact matches: spelling variants, added words, symbols?
4. **The life of a meme.** How does a cascade grow and die: how many copies,
   how quickly, from how many distinct creators?
5. **Is it worth being early?** The first use of a popular name graduates
   five times as often as the twenty-first. How much of that is real, and how
   much is that names get copied *because* the first one succeeded?
6. **Do the two interact?** Are late copies more bundled? Is an unbundled
   original the pattern of the rare organic successes?
7. **A usable warning.** From what is known a few seconds after launch,
   how well can "rugged within a minute" be predicted, and what does a
   buyer gain by avoiding the riskiest tenth?

## Approach

1. **Fix the outcomes.** Four mutually exclusive ones: rugged, bought out at
   once (graduation within the first seconds), organic graduation, and faded
   or unresolved. Analyse the two kinds of rug separately.
2. **Fix the moment of prediction, and what is known then.** A feature
   measured at 10 seconds may only predict what happens after 10 seconds.
   Launches already rugged or bought out by then are a separate group,
   described but not predicted.
3. **Bundling.** Survival curves and competing-risk models by level of
   bundling, with the creator's own purchase, the creator's record and the
   SOL in the curve as covariates. A boosted discrete-time hazard model to
   allow thresholds.
4. **Names into memes.** Normalize names (case, spacing, look-alike
   characters), link near-duplicates by edit distance and character n-grams,
   and compare with sentence embeddings for names that mean the same in
   different words. Evaluate a sample by hand.
5. **Cascades.** For each meme: first appearance, number of copies over
   time, number of distinct creators, share from its largest creator.
   Separate crowd cascades from one-creator spam.
6. **Early against late.** Compare outcomes by position in the cascade.
   To deal with the selection problem: condition on what was knowable when
   each copy launched (how the original was doing at that moment), and
   compare copies launched while the original was still rising with copies
   launched after it had died.
7. **The warning model.** Boosting on bundle features, creator record and
   cascade features (is this a copy, how old is the meme, how did earlier
   copies end), against the baselines below.

## Evaluation

- **Split by time:** fit on the earlier days, tune on the next, test on the
  last. The fully recorded part gives about a week; the legacy part adds four
  more weeks for outcomes, names and creators, which the cascade analysis
  mostly needs.
- **Creators are the independent units.** A creator must not be on both
  sides of a split, and standard errors are clustered by creator. Results
  are also reported without the most prolific creators.
- **Survival metrics:** concordance, and time-dependent AUC for "rugged
  within 30 seconds / 5 minutes" among launches alive at the prediction
  moment.
- **Classification metrics** for the rare organic graduation: average
  precision and the hit rate in the top 1% of scores, since the base rate is
  under 1%.
- **Baselines:** the creator's own purchase alone; the creator's past rug
  rate alone; the SOL in the curve at the snapshot alone. In a first look,
  that last feature by itself separated graduations almost perfectly, which
  is the instant-buyout leak described above, so it is the baseline to be
  suspicious of.
- **Clustering of names:** precision and recall of the meme assignment on a
  hand-labeled sample; stability under the similarity threshold.
- **The plain reading:** a table of outcomes by bundling level and by
  position in the cascade, with intervals, on the test days.

## Risks

- **Leakage through time.** The largest risk. Outcomes arrive within
  seconds, some before the features are measured. Every result must state
  what was known when.
- **Selection in "being first".** Names are copied because the first launch
  did well. Without care, the cascade analysis only rediscovers that.
- **The window cuts cascades at both ends.** A name's "first use" in the
  data may be its thousandth in reality, and late copies fall outside. The
  34 days of the full dataset help; old memes need to be flagged by their
  rate of appearance on the first day.
- **The rug label is a rule,** and partly mechanical. One branch requires the
  curve to have held at least 5 SOL, which bundles themselves supply.
- **Sampling of early features.** Over the whole dataset the author reports
  them for about 63% of launches, a non-random part, though they are present
  for 93% of the four days I looked at. The pattern has to be checked per
  day.
- **Special launch types.** 28% of the sampled launches are in a "mayhem"
  mode with its own rules, and 11% are not quoted in SOL. They probably need
  separate treatment; I have not looked into what they are.
- **One platform, one month.**
- **Association, not cause.** Bundling may mark creators who intend to rug
  without changing anything by itself.
- **Framing.** The output is a risk signal for buyers. It should not read as
  a guide to launching copies or to bundling unnoticed.

## Relation to other ideas

- Supersedes [`idea_bundled_launch_survival.md`](idea_bundled_launch_survival.md)
  as the broader version; that file has more detail on the dataset's columns
  and the survival setup.
- Shares its method with
  [delisting risk](idea_delisting_risk_survival.md): survival analysis at a
  time scale of seconds instead of years.
- The name-matching part overlaps with "token impersonation" in
  [`project_ideas_round2.md`](project_ideas_round2.md): look-alike
  characters and learned edit distance.
- The [recommendation study](idea_next_pool_recommendation.md) found that 15
  to 25% of the tokens traders pick up next were launched within the previous
  two days. This idea studies that population.

## References

To be verified before use; cited from memory.

- Cox (1972), "Regression models and life-tables".
- Fine and Gray (1999), "A proportional hazards model for the
  subdistribution of a competing risk".
- Leskovec, Backstrom and Kleinberg (2009), "Meme-tracking and the dynamics
  of the news cycle".
- Mazorra, Adan and Daza (2022), "Do not rug on me: zero-dimensional scam
  detection".
- Reimers and Gurevych (2019), "Sentence-BERT: sentence embeddings using
  siamese BERT-networks".
