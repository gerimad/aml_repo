"""Error analysis of the scored test period (work package 7, notebook 15).

Nothing here chooses or changes a model. The functions take the lists a
frozen model produced at one point and describe which positive pairs they
found and which they missed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import cold_start as cs
import protocol as pc

AGE_BINS = [-np.inf, 1, 7, 28, np.inf]  # days, at the evaluation point
AGE_LABELS = ["under 1 day", "1 to 7 days", "7 to 28 days", "over 28 days"]


def pair_outcomes(legs: pd.DataFrame, task: pc.Task, lists: np.ndarray, tokens: pd.DataFrame) -> pd.DataFrame:
    """One row per positive pair of the evaluated users at one point, with whether the list found it.

    `lists` are the recommended token columns of protocol.evaluate_point, in
    the order of task.eval_rows. Columns of the result: point, user, token,
    cold (the token did not exist at t), hit, token_age (the token's age at t:
    one of AGE_LABELS, or "did not exist yet" for cold pairs), and
    window_users (accounts that traded the token in the 14 days before t).
    """
    pairs = cs.first_trades_in_target(legs, task)

    # The tokens on each user's list, as token ids.
    listed = pd.DataFrame({
        "user": np.repeat(task.users[task.eval_rows], lists.shape[1]),
        "token": task.tokens[lists.ravel()],
    })  # fmt: skip
    listed["hit"] = True
    pairs = pairs.merge(listed, on=["user", "token"], how="left")
    pairs["hit"] = pairs["hit"].fillna(False).astype(bool)
    assert not (pairs["hit"] & pairs["cold"]).any(), "a token that did not exist at t cannot be on a list made at t"

    first = tokens.set_index("token_id")["first_timestamp"]
    age_days = (task.t - pairs["token"].map(first)).dt.total_seconds() / 86400
    pairs["token_age"] = pd.cut(age_days, AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
    pairs.loc[pairs["cold"], "token_age"] = "did not exist yet"

    window_users = np.asarray(task.history(pc.REFERENCE_DAYS).sum(axis=0)).ravel()
    users_by_token = pd.Series(window_users, index=task.tokens)
    pairs["window_users"] = pairs["token"].map(users_by_token).fillna(0).astype(int)
    pairs.insert(0, "point", task.t)
    return pairs


def returning_share(legs: pd.DataFrame, task: pc.Task) -> dict:
    """How much the label depends on the length of the past (implementation plan, open point 11.2).

    The label counts a pair as new if the user never traded the token before
    t, as far back as the data goes. A "returning" pair is one the user
    trades in the target window and had traded before t, but not in the 14
    days before t: it would be new under a 14-day label, and is not new under
    the plan's. Early points see less past, so they count fewer returning
    pairs. Returns the point, the positive pairs of the evaluated users, the
    returning pairs of the same users, and the share returning / (positive + returning).
    """
    t = task.t
    target = legs[(legs["timestamp"] >= t) & (legs["timestamp"] < t + pc.HORIZON)]
    target = target[["user", "token"]].drop_duplicates()
    evaluated = pd.Index(task.users[task.eval_rows])
    target = target[target["user"].isin(evaluated)]

    ages = task.past_age_days[task.eval_rows].tocoo()
    old = ages.data > pc.REFERENCE_DAYS
    traded_only_before_window = pd.DataFrame({
        "user": task.users[task.eval_rows][ages.row[old]],
        "token": task.tokens[ages.col[old]],
    })  # fmt: skip
    returning = len(target.merge(traded_only_before_window, on=["user", "token"]))
    positives = int(task.new[task.eval_rows].sum() + task.cold_new[task.eval_rows].sum())
    return {
        "point": t,
        "positives": positives,
        "returning": returning,
        "returning_share": returning / (positives + returning) if positives + returning else 0.0,
    }


OUTCOMES = ["found", "candidate, ranked below 10", "not a candidate", "did not exist yet"]


def outcome_of_pairs(pairs: pd.DataFrame, candidate_rows: pd.DataFrame, task: pc.Task) -> pd.Series:
    """Why each positive pair was found or missed, one of OUTCOMES.

    `pairs` comes from pair_outcomes; `candidate_rows` are the ranker's rows
    of the same point (ranker.point_rows), whose `user` and `col` give each
    user's candidate tokens. A missed pair either did not exist at t, was not
    among the user's candidates (the ceiling of the first stage), or was a
    candidate that the ranker did not put in the top 10.
    """
    candidates = pd.DataFrame({"user": candidate_rows["user"].to_numpy(), "token": task.tokens[candidate_rows["col"].to_numpy()]})
    candidates["candidate"] = True
    merged = pairs[["user", "token"]].merge(candidates, on=["user", "token"], how="left")
    is_candidate = merged["candidate"].fillna(False).astype(bool).to_numpy()
    outcome = np.select(
        [pairs["hit"].to_numpy(), pairs["cold"].to_numpy(), is_candidate],
        ["found", "did not exist yet", "candidate, ranked below 10"],
        "not a candidate",
    )
    return pd.Series(outcome, index=pairs.index, name="outcome")
