"""The evaluation protocol of the project (work package 2).

Everything that decides *how* models are judged lives here, written once:
the evaluation points, what a task at one point contains, which users are
scored, and how a model's lists are turned into a result table. The
definitions are those of the implementation plan, section 1, which follows
the project plan (crypt_project_plan.pdf, sections 4 and 5).

Typical use in a notebook:

    import protocol as pc
    legs = token_data.load_legs()
    result = pc.run("popularity", pc.popularity_scorer, legs, pc.POINTS["dev_a"])
    result.per_point          # one row per evaluation point
    result.per_user           # one row per (point, evaluated user)

A *scorer factory* is a function that takes a Task and returns a *scorer*:
a function from a list of user rows of that task to a dense array of scores
(rows x tokens of the task's universe). Higher scores are recommended first.
The protocol removes each user's past tokens and keeps the top 10.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse

import pool_recsys as pr
from token_data import BURN_IN_END, DATA_END, PROJECT_DIR

# --------------------------------------------------------------------------
# Fixed definitions (implementation plan, section 1)
# --------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"  # small result tables, committed

HORIZON = pd.Timedelta(days=2)  # the target: the two days after a point
REFERENCE_DAYS = 14  # the baseline's history, and who is evaluated
K = 10  # length of a recommendation list
DEV_B_START = pd.Timestamp("2026-07-22", tz="UTC")
TEST_START = pd.Timestamp("2026-08-13", tz="UTC")


def evaluation_points() -> dict[str, list[pd.Timestamp]]:
    """The 35 evaluation points, by part: every second midnight (UTC) of each period.

    Checked here, so that a typing error in a date stops everything:
    10 / 11 / 14 points, exactly two days apart, and no target (two days
    after a point) reaches into the next part.
    """
    periods = {"dev_a": (BURN_IN_END, DEV_B_START), "dev_b": (DEV_B_START, TEST_START), "test": (TEST_START, DATA_END)}
    points = {}
    for part, (start, end) in periods.items():
        points[part] = list(pd.date_range(start, end - HORIZON, freq=HORIZON))
        assert points[part][-1] + HORIZON == end, f"the last target of {part} must end where the part ends"
    assert [len(p) for p in points.values()] == [10, 11, 14], "the plan has 10, 11 and 14 points"
    return points


POINTS = evaluation_points()


def part_of(t: pd.Timestamp) -> str:
    """The part ("dev_a", "dev_b" or "test") that an evaluation point belongs to."""
    for part, points in POINTS.items():
        if t in points:
            return part
    raise ValueError(f"{t} is not an evaluation point")


# --------------------------------------------------------------------------
# One evaluation point
# --------------------------------------------------------------------------


@dataclass
class Task:
    """Everything known and to be predicted at one evaluation point t.

    Rows are users, columns are tokens. Both are indexed from the data
    *before t only*, so nothing about the future can enter a model through
    the index.

      t               the evaluation point
      users           user id of each row (every user with a trade before t)
      tokens          token id of each column: the universe, every token traded before t
      past_age_days   users x tokens: for each pair the user traded before t, the
                      age of its last such trade at t, in days (always > 0).
                      All history matrices are derived from this one.
      new             users x tokens, binary: positive pairs whose token is in
                      the universe (the user trades it in the target and never before t)
      cold_new        per user row: positive pairs whose token is *not* in the
                      universe (it did not exist before t). Counted as misses.
      window_size     per user row: tokens traded in the reference window (14 days)
      eval_rows       rows of the evaluated users: window_size >= 1 and at least
                      one positive pair (new or cold). This set depends only on the
                      data, never on a model's settings, so every model at a point
                      is scored on the same users.
    """

    t: pd.Timestamp
    users: np.ndarray
    tokens: np.ndarray
    past_age_days: sparse.csr_matrix
    new: sparse.csr_matrix
    cold_new: np.ndarray
    window_size: np.ndarray
    eval_rows: np.ndarray

    def history(self, days: float | None = None) -> sparse.csr_matrix:
        """Binary users x tokens matrix of the pairs traded in the last `days` days before t.

        days=None means since the start of the data (the whole past).
        """
        return _keep_recent(self.past_age_days, days, binary=True)

    def history_ages(self, days: float | None = None) -> sparse.csr_matrix:
        """Like history(), but each entry is the age in days of the pair's last trade."""
        return _keep_recent(self.past_age_days, days, binary=False)

    @property
    def past(self) -> sparse.csr_matrix:
        """Binary matrix of every pair traded before t; removed from every list."""
        return self.history(None)

    def describe(self) -> dict:
        """The sizes of the task, as in Table 7 of the project plan."""
        rows = self.eval_rows
        n_new = int(self.new[rows].sum())
        n_cold = int(self.cold_new[rows].sum())
        return {
            "point": self.t,
            "part": part_of(self.t) if any(self.t in p for p in POINTS.values()) else "",
            "users_in_history": len(self.users),
            "tokens_in_history": len(self.tokens),
            "users_evaluated": len(rows),
            "positive_pairs": n_new + n_cold,
            "cold_share": n_cold / max(n_new + n_cold, 1),
        }


def _keep_recent(ages: sparse.csr_matrix, days: float | None, binary: bool) -> sparse.csr_matrix:
    """Keep the entries of an age matrix that are at most `days` old (all if None)."""
    kept = ages.copy()
    if days is not None:
        kept.data[kept.data > days] = 0
        kept.eliminate_zeros()
    if binary:
        kept.data[:] = 1.0
    return kept


def make_task(legs: pd.DataFrame, t: pd.Timestamp) -> Task:
    """Build the Task at evaluation point t from the trade legs (implementation plan 1.3).

    `legs` needs columns timestamp, user, token (integer ids). Steps:
      1. Split the legs into the past (timestamp < t) and the target
         (t <= timestamp < t + 2 days). Later legs are not looked at.
      2. Index users and tokens from the past.
      3. For every past (user, token) pair, keep the time of its last trade.
      4. Positive pairs: target pairs of known users that are not past pairs.
         Those whose token is outside the universe are cold.
      5. Evaluated users: a token in the last 14 days and at least one positive pair.
    """
    t = pd.Timestamp(t)
    past_legs = legs[legs["timestamp"] < t]
    target_legs = legs[(legs["timestamp"] >= t) & (legs["timestamp"] < t + HORIZON)]
    assert past_legs["timestamp"].max() < t, "no leg at or after t may be in the past"

    # Steps 2 and 3
    past_pairs = past_legs.groupby(["user", "token"], observed=True)["timestamp"].max().reset_index()
    users = np.unique(past_pairs["user"].to_numpy())
    tokens = np.unique(past_pairs["token"].to_numpy())
    rows = np.searchsorted(users, past_pairs["user"].to_numpy())
    cols = np.searchsorted(tokens, past_pairs["token"].to_numpy())
    age_days = (t - past_pairs["timestamp"]).dt.total_seconds().to_numpy() / 86400
    assert (age_days > 0).all(), "every past trade is strictly before t"
    shape = (len(users), len(tokens))
    past_age_days = sparse.csr_matrix((age_days, (rows, cols)), shape=shape)

    # Step 4
    target_pairs = target_legs[["user", "token"]].drop_duplicates()
    target_pairs = target_pairs[np.isin(target_pairs["user"].to_numpy(), users)]
    seen = target_pairs.merge(past_pairs[["user", "token"]], on=["user", "token"], how="left", indicator=True)
    positive = seen[seen["_merge"] == "left_only"]
    in_universe = np.isin(positive["token"].to_numpy(), tokens)

    known = positive[in_universe]
    new = sparse.csr_matrix(
        (np.ones(len(known)), (np.searchsorted(users, known["user"]), np.searchsorted(tokens, known["token"]))),
        shape=shape,
    )
    cold = positive[~in_universe].groupby("user").size()
    cold_new = np.zeros(len(users))
    cold_new[np.searchsorted(users, cold.index.to_numpy())] = cold.to_numpy()

    # Step 5
    window = _keep_recent(past_age_days, REFERENCE_DAYS, binary=True)
    window_size = np.asarray(window.sum(axis=1)).ravel()
    n_positive = np.asarray(new.sum(axis=1)).ravel() + cold_new
    eval_rows = np.flatnonzero((window_size >= 1) & (n_positive > 0))

    task = Task(t, users, tokens, past_age_days, new, cold_new, window_size, eval_rows)
    assert task.new.multiply(task.past).nnz == 0, "a positive pair can never be a past pair"
    return task


def load_task(legs: pd.DataFrame, t: pd.Timestamp, cache_dir: Path = PROJECT_DIR / "tasks") -> Task:
    """make_task, cached on disk: building a task takes seconds and tuning needs each one many times.

    The cache file is named by the point. Delete the folder whenever legs.parquet is rebuilt.
    """
    path = cache_dir / f"task_{pd.Timestamp(t):%Y%m%d}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    task = make_task(legs, t)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(task, f)
    return task


# --------------------------------------------------------------------------
# Scoring one point
# --------------------------------------------------------------------------

Scorer = Callable[[np.ndarray], np.ndarray]


def popularity_scorer(task: Task, days: float = REFERENCE_DAYS) -> Scorer:
    """The fixed baseline: tokens with the most distinct users in the last `days` days.

    With the default of 14 days this is exactly the baseline of the project
    plan (section 5.4). days=1 gives "recent popularity".
    """
    window = task.history(days)
    model = pr.Popularity().fit(window)
    return lambda rows: model.scores(window[rows])


def head_tokens(window: sparse.csr_matrix, share: float = 1 / 3) -> np.ndarray:
    """Columns of the most popular tokens that together hold `share` of the window's interactions.

    Used for "recall without the head" (Cremonesi et al. 2010): these tokens
    are removed from the target so that a model cannot score well on popular
    tokens alone. Always at least one token.
    """
    users_per_token = np.asarray(window.sum(axis=0)).ravel()
    order = np.argsort(-users_per_token, kind="stable")
    cumulative = np.cumsum(users_per_token[order]) / users_per_token.sum()
    n_head = int(np.searchsorted(cumulative, share) + 1)
    return order[:n_head]


def history_group(window_size: np.ndarray) -> np.ndarray:
    """The history-size group of each user: "1", "2-4" or "5+" tokens in the reference window."""
    return np.select([window_size <= 1, window_size <= 4], ["1", "2-4"], "5+")


def evaluate_point(scorer: Scorer, task: Task, k: int = K) -> tuple[pd.DataFrame, np.ndarray]:
    """Score every evaluated user of a task and compare the lists with the target.

    Returns (per_user, lists):
      per_user   one row per evaluated user: user, recall, ndcg, hit,
                 recall_no_head, window_size, group
      lists      the recommended token columns, evaluated users x k

    Users are scored in batches, sized so that one batch of dense scores
    stays near 100 MB whatever the size of the universe.
    """
    rows_all = task.eval_rows
    past = task.past
    head = head_tokens(task.history(REFERENCE_DAYS))
    keep_column = np.ones(len(task.tokens))
    keep_column[head] = 0
    new_no_head = (task.new @ sparse.diags(keep_column)).tocsr()
    new_no_head.eliminate_zeros()

    batch = max(64, 25_000_000 // max(len(task.tokens), 1))
    parts, lists = [], []
    for start in range(0, len(rows_all), batch):
        rows = rows_all[start : start + batch]
        scores = scorer(rows)
        assert scores.shape == (len(rows), len(task.tokens)), "a scorer must score every token of the universe"
        recommended = pr.top_k(scores, k, exclude=past[rows])
        metrics = pr.ranking_metrics(recommended, task.new[rows], task.cold_new[rows])
        metrics["recall_no_head"] = pr.ranking_metrics(recommended, new_no_head[rows], task.cold_new[rows])["recall"]
        parts.append(metrics)
        lists.append(recommended)

    per_user = pd.concat(parts, ignore_index=True)
    per_user.insert(0, "user", task.users[rows_all])
    per_user["window_size"] = task.window_size[rows_all].astype(int)
    per_user["group"] = history_group(task.window_size[rows_all])
    return per_user, np.vstack(lists)


# --------------------------------------------------------------------------
# Many points
# --------------------------------------------------------------------------


@dataclass
class Result:
    """The result of one model on a list of points.

      per_point   one row per point: model, point, part, users, recall (mean
                  over users) with its bootstrap interval over users, ndcg,
                  hit_rate, coverage, recall_no_head, and recall per history group
      per_user    one row per (point, evaluated user)
    """

    per_point: pd.DataFrame
    per_user: pd.DataFrame


def summarize_point(name: str, task: Task, per_user: pd.DataFrame, lists: np.ndarray) -> dict:
    """One row of Result.per_point."""
    low, high = pr.bootstrap_mean(per_user["recall"], n_boot=1000, seed=0)
    row = {
        "model": name,
        "point": task.t,
        "part": part_of(task.t),
        "users": len(per_user),
        "recall": per_user["recall"].mean(),
        "recall_low": low,
        "recall_high": high,
        "ndcg": per_user["ndcg"].mean(),
        "hit_rate": per_user["hit"].mean(),
        "coverage": pr.coverage(lists, len(task.tokens)),
        "recall_no_head": per_user["recall_no_head"].mean(),
    }
    for group in ("1", "2-4", "5+"):
        row[f"recall_{group}"] = per_user.loc[per_user["group"] == group, "recall"].mean()
    return row


def run(
    name: str,
    scorer_factory: Callable[[Task], Scorer],
    legs: pd.DataFrame,
    points: list[pd.Timestamp],
    verbose: bool = True,
) -> Result:
    """Evaluate one model at each of the given points.

    At each point the scorer factory is called with that point's task, so a
    model is refitted on each point's own history, as the protocol requires.
    """
    summaries, users = [], []
    for t in points:
        task = load_task(legs, t)
        per_user, lists = evaluate_point(scorer_factory(task), task)
        summaries.append(summarize_point(name, task, per_user, lists))
        users.append(per_user.assign(model=name, point=t))
        if verbose:
            print(f"  {name:<28} {t:%d %b}: recall@10 = {summaries[-1]['recall']:.3f}  ({len(per_user):,} users)")
    return Result(pd.DataFrame(summaries), pd.concat(users, ignore_index=True))


def task_sizes(legs: pd.DataFrame, points: list[pd.Timestamp]) -> pd.DataFrame:
    """Task.describe() for each point, as one table."""
    return pd.DataFrame([load_task(legs, t).describe() for t in points])
