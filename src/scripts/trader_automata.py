"""Turn on-chain swap histories into symbol sequences and model them.

Building blocks for the "trading bot fingerprinting with probabilistic
automata" experiments. The pipeline is:

    swaps  ->  traders  ->  one symbol per swap  ->  a sequence model per trader

From a notebook in src/notebooks:

    import sys; sys.path.append("../scripts")
    import dex_swaps as ds
    import trader_automata as ta

    swaps = ds.load_swaps()
    callers = ta.classify_callers(swaps)                 # routers vs. private contracts
    events = ta.add_symbols(ta.assign_traders(swaps, callers))
    sequences = ta.sequences_by_trader(events)           # trader -> array of symbols

    model = ta.NGramModel(order=1).fit(sequences[some_trader])
    model.mean_log_likelihood(other_sequence)            # nats per symbol

    automaton = ta.PdfaModel().fit(sequences[some_trader])   # learned with ALERGIA
    automaton.n_states, automaton.mean_log_likelihood(other_sequence)

All models share `fit(sequence)` and `mean_log_likelihood(sequence)`, so they
can be swapped in `identification_scores`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Who is trading: routers and private contracts
# --------------------------------------------------------------------------

PUBLIC_MIN_SIGNERS = 50  # a public router is used by at least this many accounts ...
PUBLIC_MAX_TXS_PER_SIGNER = 20  # ... each of which sends few transactions through it
PRIVATE_MAX_SIGNERS = 2  # a pool caller used by at most this many accounts is a private contract


def classify_callers(swaps: pd.DataFrame) -> pd.DataFrame:
    """Classify every address that calls pools (`sender`) by who uses it.

    A router or aggregator is used by many different signing accounts, each a
    few times. A trading bot's own contract is used by one or two accounts.
    Returns one row per sender with n_swaps, n_txs, n_signers,
    txs_per_signer and kind:
      "public"   many signers, few transactions each: a router
      "private"  one or two signers: a trader's own contract
      "unclear"  everything in between, including contracts driven by a
                 fleet of signing accounts
    """
    callers = swaps.groupby("sender").agg(
        n_swaps=("tx_hash", "size"), n_txs=("tx_hash", "nunique"), n_signers=("tx_from", "nunique")
    )
    callers["txs_per_signer"] = callers["n_txs"] / callers["n_signers"]
    callers["kind"] = "unclear"
    public = (callers["n_signers"] >= PUBLIC_MIN_SIGNERS) & (
        callers["txs_per_signer"] <= PUBLIC_MAX_TXS_PER_SIGNER
    )
    callers.loc[public, "kind"] = "public"
    callers.loc[callers["n_signers"] <= PRIVATE_MAX_SIGNERS, "kind"] = "private"
    return callers.sort_values("n_swaps", ascending=False)


def assign_traders(swaps: pd.DataFrame, callers: pd.DataFrame) -> pd.DataFrame:
    """Attribute each swap to a trader and drop the swaps that cannot be attributed.

    Adds two columns:
      trader       the private contract that called the pool ("bot"), or the
                   account that signed a transaction sent through a public
                   router ("router_user")
      trader_kind  "bot" or "router_user"
    Swaps whose pool caller is "unclear" are dropped. The result is in chain order.
    """
    kind = swaps["sender"].map(callers["kind"])
    out = swaps[kind.isin(["public", "private"])].copy()
    is_bot = (kind[out.index] == "private").to_numpy()
    out["trader"] = np.where(is_bot, out["sender"], out["tx_from"])
    out["trader_kind"] = np.where(is_bot, "bot", "router_user")
    return out.sort_values(["block", "tx_index", "log_index"], ignore_index=True)


# --------------------------------------------------------------------------
# From swaps to symbols
# --------------------------------------------------------------------------

GAP_NAMES = ["same tx", "same block", "within 5 blocks", "within 300 blocks", "later"]
RELATION_NAMES = ["same pool, same direction", "same pool, reversed", "known pool", "new pool"]
ALPHABETS = {"gap": 5, "relation": 4, "gap_relation": 20, "gap_relation_size": 60}


def add_symbols(events: pd.DataFrame) -> pd.DataFrame:
    """Describe every swap relative to the same trader's previous swap.

    `events` needs the columns trader, block, tx_hash, pool, amount0 and must
    be in chain order. Adds:
      gap       0 same transaction, 1 same block, 2 within 5 blocks (about a
                minute), 3 within 300 blocks (about an hour), 4 later
      relation  0 same pool and direction as the previous swap, 1 same pool,
                opposite direction, 2 another pool the trader used before,
                3 a pool new to the trader
      size      0, 1, 2: tercile of the swap's size among all swaps of its pool
      symbol    gap * 4 + relation, a number from 0 to 19
    The first swap of each trader has no predecessor and gets -1 in gap,
    relation and symbol. Only the past of a trader is used, never its future.
    """
    out = events.copy()
    by_trader = out.groupby("trader", sort=False)
    block_gap = out["block"] - by_trader["block"].shift()
    same_tx = out["tx_hash"] == by_trader["tx_hash"].shift()
    gap = np.select([same_tx, block_gap == 0, block_gap <= 5, block_gap <= 300], [0, 1, 2, 3], 4)

    token0_in = out["amount0"] > 0
    same_pool = out["pool"] == by_trader["pool"].shift()
    same_direction = token0_in == token0_in.groupby(out["trader"], sort=False).shift()
    seen_before = out.duplicated(["trader", "pool"])
    relation = np.select([same_pool & same_direction, same_pool, seen_before], [0, 1, 2], 3)

    first = by_trader.cumcount() == 0
    out["gap"] = np.where(first, -1, gap)
    out["relation"] = np.where(first, -1, relation)
    out["symbol"] = np.where(first, -1, gap * 4 + relation)
    rank = out["amount0"].abs().groupby(out["pool"]).rank(pct=True)
    out["size"] = np.minimum((rank * 3).astype(int), 2)
    return out


def symbol_column(events: pd.DataFrame, alphabet: str = "gap_relation") -> pd.Series:
    """The symbol of each event under one of the ALPHABETS (-1 for a trader's first event)."""
    if alphabet == "gap":
        return events["gap"]
    if alphabet == "relation":
        return events["relation"]
    if alphabet == "gap_relation":
        return events["symbol"]
    if alphabet == "gap_relation_size":
        return pd.Series(
            np.where(events["symbol"] < 0, -1, events["symbol"] * 3 + events["size"]),
            index=events.index,
        )
    raise ValueError(f"unknown alphabet {alphabet!r}")


def sequences_by_trader(events: pd.DataFrame, alphabet: str = "gap_relation") -> dict:
    """Map each trader to its symbol sequence (an int array), in chain order.

    A trader's first event has no symbol and is left out.
    """
    symbols = symbol_column(events, alphabet)
    keep = symbols >= 0
    grouped = symbols[keep].groupby(events.loc[keep, "trader"], sort=False)
    return {trader: values.to_numpy(dtype=np.int64) for trader, values in grouped}


def symbol_name(symbol: int) -> str:
    """Readable name of a gap_relation symbol, for example "same tx / new pool"."""
    return f"{GAP_NAMES[symbol // 4]} / {RELATION_NAMES[symbol % 4]}"


# --------------------------------------------------------------------------
# Sequence models
# --------------------------------------------------------------------------


class NGramModel:
    """Markov model of fixed order with additive smoothing.

    Order 0 ignores the order of symbols (a bag of symbols), order 1 conditions
    on the previous symbol, order 2 on the previous two. The first symbols of a
    sequence, which lack a full context, are conditioned on a "start" marker.
    """

    def __init__(self, order: int = 1, n_symbols: int = 20, alpha: float = 0.5):
        self.order, self.n_symbols, self.alpha = order, n_symbols, alpha
        self.log_probs = None

    def _contexts(self, sequence: np.ndarray) -> np.ndarray:
        # Encode the `order` previous symbols as one integer; start marker = n_symbols.
        base = self.n_symbols + 1
        padded = np.concatenate([np.full(self.order, self.n_symbols), sequence])
        context = np.zeros(len(sequence), dtype=np.int64)
        for offset in range(self.order):
            context = context * base + padded[offset : offset + len(sequence)]
        return context

    def fit(self, sequence) -> NGramModel:
        """Fit on one sequence."""
        return self.fit_many([sequence])

    def fit_many(self, sequences) -> NGramModel:
        """Fit on several sequences, for example all traders of one class.

        Each sequence starts from the "start" marker, so no transition is
        counted across the boundary between two sequences.
        """
        n_contexts = (self.n_symbols + 1) ** self.order
        counts = np.zeros((n_contexts, self.n_symbols))
        for sequence in sequences:
            sequence = np.asarray(sequence, dtype=np.int64)
            np.add.at(counts, (self._contexts(sequence), sequence), 1)
        counts += self.alpha
        self.log_probs = np.log(counts / counts.sum(axis=1, keepdims=True))
        return self

    def mean_log_likelihood(self, sequence) -> float:
        """Average log probability per symbol (nats) of a sequence under the model."""
        sequence = np.asarray(sequence, dtype=np.int64)
        return float(self.log_probs[self._contexts(sequence), sequence].mean())


class PdfaModel:
    """Probabilistic deterministic finite automaton learned by state merging (ALERGIA).

    A single behaviour trace is cut into overlapping windows of `window`
    symbols. Their prefix tree is the starting automaton, and ALERGIA merges
    states whose observed futures are statistically compatible (`eps` is the
    confidence parameter of its test: smaller values merge more). The result
    is a compact machine with loops that can be run over sequences of any
    length. Uses the implementation in the AALpy library.

    When scoring, transition probabilities are mixed with a uniform
    distribution (weight `smoothing`) so that unseen symbols are not
    impossible. On a symbol with no transition from the current state, the
    machine restarts from that symbol's entry state.
    """

    START = "start"

    def __init__(self, window: int = 4, eps: float = 0.05, n_symbols: int = 20, smoothing: float = 0.05):
        self.window, self.eps, self.n_symbols, self.smoothing = window, eps, n_symbols, smoothing
        self.transitions, self.initial, self.n_states = None, None, 0

    def fit(self, sequence) -> PdfaModel:
        """Learn the automaton from one sequence."""
        return self.fit_many([sequence])

    def fit_many(self, sequences) -> PdfaModel:
        """Learn one automaton from several sequences; windows never span two sequences."""
        from aalpy.learning_algs import run_Alergia

        windows = []
        for sequence in sequences:
            symbols = [int(s) for s in sequence]
            length = min(self.window, len(symbols))
            windows += [
                [self.START] + symbols[i : i + length] for i in range(len(symbols) - length + 1)
            ]
        machine = run_Alergia(windows, automaton_type="mc", eps=self.eps, print_info=False)
        ids = {state.state_id: i for i, state in enumerate(machine.states)}
        # transitions[state][symbol] = (next state, probability)
        self.transitions = [
            {target.output: (ids[target.state_id], prob) for target, prob in state.transitions}
            for state in machine.states
        ]
        self.initial = ids[machine.initial_state.state_id]
        self.n_states = len(machine.states)
        self.state_symbols = [state.output for state in machine.states]  # symbol that leads into it
        return self

    def describe(self, top: int = 3) -> pd.DataFrame:
        """One row per state: the symbol that leads into it and its most likely next symbols.

        `next` lists up to `top` (symbol, probability, next state) triples,
        most likely first. This is the readable form of the automaton.
        """
        rows = []
        for state, moves in enumerate(self.transitions):
            ranked = sorted(moves.items(), key=lambda item: -item[1][1])[:top]
            rows.append({
                "state": state,
                "entered_by": self.state_symbols[state],
                "n_next": len(moves),
                "next": [(symbol, round(prob, 3), target) for symbol, (target, prob) in ranked],
            })  # fmt: skip
        return pd.DataFrame(rows)

    def mean_log_likelihood(self, sequence) -> float:
        """Average log probability per symbol (nats) of a sequence under the automaton."""
        floor = self.smoothing / self.n_symbols
        keep = 1 - self.smoothing
        entry = self.transitions[self.initial]
        state, total = self.initial, 0.0
        for symbol in sequence:
            symbol = int(symbol)
            step = self.transitions[state].get(symbol)
            if step is None:
                total += np.log(floor)
                restart = entry.get(symbol)
                state = restart[0] if restart else self.initial
            else:
                total += np.log(keep * step[1] + floor)
                state = step[0]
        return total / max(len(sequence), 1)


class PoolProfile:
    """Reference "model" that ignores behaviour: which pools a trader uses, and how often.

    `fit` takes a sequence of pool addresses. The score of another sequence is
    the cosine similarity of the two pool-count vectors. It answers a different
    question (where a trader trades, not how) and marks what is achievable
    without any sequence modelling.
    """

    def fit(self, pools) -> PoolProfile:
        self.counts = pd.Series(list(pools)).value_counts()
        self.norm = float(np.sqrt((self.counts**2).sum()))
        return self

    def mean_log_likelihood(self, pools) -> float:
        other = pd.Series(list(pools)).value_counts()
        dot = float((self.counts.reindex(other.index).fillna(0) * other).sum())
        return dot / (self.norm * float(np.sqrt((other**2).sum())))


# --------------------------------------------------------------------------
# Identification: whose sequence is this?
# --------------------------------------------------------------------------


def identification_scores(enrolled: dict, queries: dict, make_model) -> pd.DataFrame:
    """Score every query sequence under a model fitted to every enrolled trader.

    `enrolled` and `queries` map trader -> sequence. Returns a frame with one
    row per query trader and one column per enrolled trader; the entry is the
    query's mean log-likelihood under that trader's model (higher = closer).
    """
    models = {trader: make_model().fit(sequence) for trader, sequence in enrolled.items()}
    scores = {
        query: {trader: model.mean_log_likelihood(sequence) for trader, model in models.items()}
        for query, sequence in queries.items()
    }
    return pd.DataFrame(scores).T


def identification_ranks(scores: pd.DataFrame) -> pd.Series:
    """Rank of the true trader for each query (1 = identified correctly).

    Ties count against the true trader, so a model that scores everyone the
    same gets the worst rank, not the best.
    """
    values = scores.to_numpy()
    own = np.array([scores.columns.get_loc(query) for query in scores.index])
    own_score = values[np.arange(len(values)), own]
    return pd.Series((values >= own_score[:, None]).sum(axis=1), index=scores.index, name="rank")


def identification_metrics(scores: pd.DataFrame) -> dict[str, float]:
    """How well the true trader is ranked among all enrolled ones.

    Every query row must have its own trader among the columns. Returns
    top-1 and top-5 accuracy, the mean reciprocal rank, the number of queries
    and candidates, and the top-1 accuracy of random guessing.
    """
    values = scores.to_numpy()
    rank = identification_ranks(scores).to_numpy()
    return {
        "top1": float((rank == 1).mean()),
        "top5": float((rank <= 5).mean()),
        "mrr": float((1 / rank).mean()),
        "n_queries": len(values),
        "n_candidates": values.shape[1],
        "chance_top1": 1 / values.shape[1],
    }


def assign_groups(keys, fractions=(0.6, 0.2, 0.2), names=("train", "dev", "test")) -> pd.Series:
    """Assign each key to a group by a hash of the key, reproducibly and independently of order.

    Used to split traders into disjoint train / dev / test sets: the same
    address always lands in the same set, in every run and on every machine.
    """
    import zlib

    keys = pd.Index(keys).unique()
    position = np.array([zlib.crc32(str(k).encode()) for k in keys]) / 2**32
    edges = np.cumsum(fractions)[:-1]
    return pd.Series(np.array(names)[np.searchsorted(edges, position, side="right")], index=keys)
