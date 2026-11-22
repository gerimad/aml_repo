"""Cold start: refreshed lists and new-token heuristics (work package 5, core problem 3).

Under the main protocol a list is made once, at the evaluation point t, and
judged on the next two days. A token launched inside those two days cannot
be on it: 15 to 25% of the target is out of reach. This module adds a second
protocol in which lists are rebuilt every R hours inside the target window
(implementation plan, section 7.1):

  - refresh times s = t, t + R, t + 2R, ... before t + 2 days
  - each positive pair (u, i) of the main task, first traded at time tau,
    is judged against u's list built at the last refresh s <= tau
  - recall of a user = pairs found / positive pairs, averaged over the
    evaluated users of the main task (the same users as under the main protocol)

With R = 48 hours this is exactly the main protocol (a test checks it).

A scorer factory here takes the state at the refresh time (a protocol.Task
built at s) and the token table, so that heuristics can use token ages.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

import pool_recsys as pr
import protocol as pc

RefreshFactory = Callable[[pc.Task, pd.DataFrame], pc.Scorer]


# --------------------------------------------------------------------------
# The refreshed protocol
# --------------------------------------------------------------------------


def first_trades_in_target(legs: pd.DataFrame, task: pc.Task) -> pd.DataFrame:
    """Every positive pair of a task with the time its user first traded the token.

    Columns: user, token, first_time, cold (token not in the universe at t).
    Only pairs of the task's evaluated users are returned.
    """
    t = task.t
    window = legs[(legs["timestamp"] >= t) & (legs["timestamp"] < t + pc.HORIZON)]
    first = window.groupby(["user", "token"], observed=True)["timestamp"].min().rename("first_time").reset_index()
    evaluated = set(task.users[task.eval_rows])
    first = first[first["user"].isin(evaluated)]

    # Keep the positive pairs only: drop pairs the user had traded before t.
    past = task.past.tocoo()
    past_pairs = pd.DataFrame({"user": task.users[past.row], "token": task.tokens[past.col]})
    first = first.merge(past_pairs, on=["user", "token"], how="left", indicator=True)
    first = first[first["_merge"] == "left_only"].drop(columns="_merge")
    first["cold"] = ~np.isin(first["token"].to_numpy(), task.tokens)

    expected = int(task.new[task.eval_rows].sum() + task.cold_new[task.eval_rows].sum())
    assert len(first) == expected, "the pairs found here must be exactly the task's positive pairs"
    return first.reset_index(drop=True)


def refresh_times(t: pd.Timestamp, hours: float) -> list[pd.Timestamp]:
    """t, t + R, t + 2R, ... strictly before the end of the target window."""
    step = pd.Timedelta(hours=hours)
    times, s = [], pd.Timestamp(t)
    while s < t + pc.HORIZON:
        times.append(s)
        s += step
    return times


def refreshed_evaluate(
    factories: dict[str, RefreshFactory],
    legs: pd.DataFrame,
    task: pc.Task,
    tokens: pd.DataFrame,
    hours: float,
    k: int = pc.K,
) -> dict[str, pd.DataFrame]:
    """Recall@k of models whose lists are rebuilt every `hours` hours (section 7.1).

    `factories` maps a model name to its scorer factory. All models are
    scored in one pass over the refresh times, so the state at each refresh
    (a Task built at s, which takes seconds) is built once, not once per model.

    Returns, per model, one row per evaluated user of `task` with user,
    positives, hits, recall, cold_positives, cold_hits.
    """
    pairs = first_trades_in_target(legs, task)
    times = refresh_times(task.t, hours)
    # The last refresh at or before tau is the number of whole intervals between t and tau.
    pairs["refresh"] = ((pairs["first_time"] - task.t) // pd.Timedelta(hours=hours)).astype(int)
    assert pairs["refresh"].between(0, len(times) - 1).all()
    hit = {name: np.zeros(len(pairs), dtype=bool) for name in factories}

    for j, s in enumerate(times):
        due = pairs[pairs["refresh"] == j]
        if due.empty:
            continue
        state = pc.make_task(legs, s) if s > task.t else task
        users_due = np.unique(due["user"].to_numpy())
        rows = np.searchsorted(state.users, users_due)
        assert (state.users[rows] == users_due).all(), "every evaluated user is known at every later refresh"
        for name, factory in factories.items():
            scores = factory(state, tokens)(rows)
            recommended = pr.top_k(scores, k, exclude=state.past[rows])
            listed = {(u, int(state.tokens[c])) for u, row in zip(users_due, recommended) for c in row}
            hit[name][due.index] = [(u, i) in listed for u, i in zip(due["user"], due["token"])]

    results = {}
    for name in factories:
        scored = pairs.assign(hit=hit[name], cold_hit=hit[name] & pairs["cold"].to_numpy())
        per_user = scored.groupby("user").agg(
            positives=("hit", "size"),
            hits=("hit", "sum"),
            cold_positives=("cold", "sum"),
            cold_hits=("cold_hit", "sum"),
        )
        per_user["recall"] = per_user["hits"] / per_user["positives"]
        results[name] = per_user.reset_index()
    return results


def summarize_refreshed(name: str, hours: float, task: pc.Task, per_user: pd.DataFrame) -> dict:
    """One row per (model, refresh interval, point): user-averaged recall and the share of cold pairs reached."""
    return {
        "model": name,
        "refresh_hours": hours,
        "point": task.t,
        "part": pc.part_of(task.t),
        "users": len(per_user),
        "recall": per_user["recall"].mean(),
        "cold_pairs": int(per_user["cold_positives"].sum()),
        "cold_reached": per_user["cold_hits"].sum() / max(per_user["cold_positives"].sum(), 1),
    }


# --------------------------------------------------------------------------
# Scorers for the refreshed protocol
# --------------------------------------------------------------------------


def popularity_factory(state: pc.Task, tokens: pd.DataFrame) -> pc.Scorer:
    """The fixed baseline, rebuilt at each refresh: most users in the 14 days before s."""
    return pc.popularity_scorer(state)


def token_ages_hours(state: pc.Task, tokens: pd.DataFrame) -> np.ndarray:
    """Age in hours at the refresh time of every token of the state's universe.

    Uses the first trade of each token from the token table. This is not a
    leak: a token in the universe at s was first traded before s, so its
    first trade time is known at s.
    """
    first = tokens.set_index("token_id")["first_timestamp"].reindex(state.tokens)
    assert first.notna().all() and (first < state.t).all(), "every token of the universe was traded before s"
    return ((state.t - first).dt.total_seconds() / 3600).to_numpy()


def newest_popular_scores(state: pc.Task, tokens: pd.DataFrame, max_age_hours: float) -> np.ndarray:
    """The trivial heuristic: among tokens younger than max_age_hours, those with the most users.

    One score per token of the universe, the same for every user: the number
    of distinct users that traded it before s if it is young enough, else 0.
    """
    users_so_far = np.asarray(state.past.sum(axis=0)).ravel()
    young = token_ages_hours(state, tokens) < max_age_hours
    return np.where(young, users_so_far, 0.0)


def newest_popular_factory(max_age_hours: float) -> RefreshFactory:
    """Scorer factory for the newest-popular heuristic alone."""

    def factory(state, tokens):
        scores = newest_popular_scores(state, tokens, max_age_hours)
        return lambda rows: np.tile(scores, (len(rows), 1))

    return factory


def mixed_factory(base: RefreshFactory, max_age_hours: float, n_new: int) -> RefreshFactory:
    """A list of the base model's best 10 - n_new tokens plus the n_new best young tokens.

    The young tokens are chosen per user among those the user has not traded
    (so that exclusion of the past cannot remove them afterwards), and only
    tokens with a positive newest-popular score are used: if fewer young
    tokens exist, the base model fills the list.
    """

    def factory(state, tokens):
        base_scorer = base(state, tokens)
        young = newest_popular_scores(state, tokens, max_age_hours)

        def scorer(rows):
            scores = base_scorer(rows)
            scores = scores / np.maximum(scores.max(axis=1, keepdims=True), 1e-12)  # now at most 1
            young_scores = np.tile(young, (len(rows), 1))
            past = state.past[rows].tocoo()
            young_scores[past.row, past.col] = 0
            picks = pr.top_k(young_scores, n_new)
            for r, columns in enumerate(picks):
                for c in columns:
                    if young_scores[r, c] > 0:
                        scores[r, c] = 2 + young_scores[r, c]  # above every base score
            return scores

        return scorer

    return factory


def shared_across_models(factory: RefreshFactory) -> RefreshFactory:
    """The same factory, but fitted once per refresh time even when several models use it.

    The mixed lists of several settings all start from the same base model.
    Without this, refreshed_evaluate would refit the base model once per
    setting at every refresh. The scorer of the latest refresh time is kept
    and reused while the refresh time stays the same.
    """
    latest = {"t": None, "scorer": None}

    def shared(state, tokens):
        if latest["t"] != state.t:
            latest["t"], latest["scorer"] = state.t, factory(state, tokens)
        return latest["scorer"]

    return shared
