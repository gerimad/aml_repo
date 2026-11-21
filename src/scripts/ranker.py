"""The learned ranker: the second stage of the method (work package 6).

Stage 1 proposes candidate tokens for each evaluated user: the best tokens by
popularity and the best by neighbour score. Stage 2 is a model that orders
the candidates from features of the (user, token) pair. See the project
plan, section 6.4, and the implementation plan, section 8.

Pipeline at one evaluation point t:

    Task --candidates()--> one row per (user, candidate token)
         --features()-----> + feature columns, all computed from data before t
         --label----------> 1 if the user trades the token for the first time in [t, t + 2 days)

A trained model gives each row a score; ranker_scorer() turns these scores
into the scorer interface of protocol.py, so the ranker is evaluated by
exactly the same code as every other model.

The settings of the first stage come from work packages 3 and 4 and are
passed as a plain dict:

    settings = {
        "k": 50, "shrink": 10, "weight": 0.5,     # fixed blend of WP3
        "half_life": 7, "history_days": 28,       # time weighting of WP4 (None = off / all)
        "candidates": 200,                        # tokens per source
    }
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pool_recsys as pr
import protocol as pc
from token_data import KNOWN_SYMBOLS, PROJECT_DIR

FEATURE_GROUPS = {
    "popularity": ["pop_14d", "pop_1d", "pop_half_life_1d", "pop_half_life_7d", "pop_rank"],
    "neighbours": ["knn_score", "knn_score_time", "max_similarity", "n_pointing"],
    "user": ["user_window_tokens", "user_past_tokens", "user_days_since_last_trade", "user_share_buys"],
    "token": [
        "token_age_days", "token_censored", "token_users_first_hour", "token_users_first_day",
        "launch_weth", "launch_usdc", "launch_usdt",
    ],
}  # fmt: skip
ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]
NOT_A_CANDIDATE = -1e9  # score of tokens outside the candidate set: always ranked last


# --------------------------------------------------------------------------
# The first stage, fitted at one point
# --------------------------------------------------------------------------


class FirstStage:
    """Popularity and neighbour models of one evaluation point, fitted from its history only.

    Attributes used by candidates() and features():
      window          binary 14-day history (the baseline's)
      popularity      users per token in the 14-day window
      knn_plain       ItemKNN of WP3 on the 14-day window
      knn_time        ItemKNN on the time-weighted history of WP4
      profile_time    the time-weighted history (users x tokens), the users' profiles for knn_time
    """

    def __init__(self, task: pc.Task, settings: dict):
        self.task, self.settings = task, settings
        self.window = task.history(pc.REFERENCE_DAYS)
        self.popularity = np.asarray(self.window.sum(axis=0)).ravel()
        self.knn_plain = pr.ItemKNN(settings["k"], settings["shrink"]).fit(self.window)
        ages = task.history_ages(settings["history_days"])
        self.profile_time = pr.time_weights(ages, settings["half_life"])
        self.knn_time = pr.ItemKNN(settings["k"], settings["shrink"]).fit(self.profile_time)


def candidates(stage: FirstStage, rows: np.ndarray) -> pd.DataFrame:
    """The candidate tokens of each user row: union of the top n by popularity and the top n by neighbours.

    n = settings["candidates"]. The user's past tokens are never candidates.
    Neighbour candidates are only taken where the neighbour score is positive
    (a token no past token points to is not a neighbour). Returns columns
    row, col (the token's column in the task).
    """
    n = stage.settings["candidates"]
    past = stage.task.past[rows]
    popular = pr.top_k(np.tile(stage.popularity.astype(np.float32), (len(rows), 1)), n, exclude=past)
    neighbour_scores = np.asarray((stage.profile_time[rows] @ stage.knn_time.similarity).todense(), dtype=np.float32)
    nearest = pr.top_k(neighbour_scores, n, exclude=past)

    pairs = []
    for r, row in enumerate(rows):
        positive = nearest[r][neighbour_scores[r, nearest[r]] > 0]
        for col in np.union1d(popular[r], positive):
            pairs.append((row, col))
    frame = pd.DataFrame(pairs, columns=["row", "col"])
    assert stage.task.past[frame["row"].to_numpy(), frame["col"].to_numpy()].sum() == 0, "no past token is a candidate"
    return frame


def candidate_recall(task: pc.Task, pairs: pd.DataFrame) -> float:
    """Share of the positive pairs (cold ones included) of the evaluated users that are candidates.

    This is the ceiling of anything the ranker can reach.
    """
    found = task.new[pairs["row"].to_numpy(), pairs["col"].to_numpy()].sum()
    rows = task.eval_rows
    return float(found / (task.new[rows].sum() + task.cold_new[rows].sum()))


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


def token_context(legs: pd.DataFrame, tokens: pd.DataFrame, task: pc.Task) -> pd.DataFrame:
    """Token features at t, one row per column of the task (all from data before t).

    token_age_days           days since the token's first trade
    token_censored           first traded in the first day of the data (true age unknown)
    token_users_first_hour   distinct users in its first hour (only trades before t)
    token_users_first_day    distinct users in its first day (only trades before t)
    launch_weth/usdc/usdt    the other token of its launch pool is this base currency
    """
    t = task.t
    info = tokens.set_index("token_id").reindex(task.tokens)
    first = info["first_timestamp"]
    before = legs[(legs["timestamp"] < t) & legs["token"].isin(task.tokens)]
    since_launch = before["timestamp"] - before["token"].map(first)
    users_hour = before[since_launch < pd.Timedelta(hours=1)].groupby("token")["user"].nunique()
    users_day = before[since_launch < pd.Timedelta(days=1)].groupby("token")["user"].nunique()

    base_symbol = info["launch_base_id"].map(tokens.set_index("token_id")["address"]).map(KNOWN_SYMBOLS)
    context = pd.DataFrame({
        "token_age_days": ((t - first).dt.total_seconds() / 86400).to_numpy(),
        "token_censored": info["censored"].astype(float).to_numpy(),
        "token_users_first_hour": users_hour.reindex(task.tokens).fillna(0).to_numpy(),
        "token_users_first_day": users_day.reindex(task.tokens).fillna(0).to_numpy(),
        "launch_weth": (base_symbol == "WETH").astype(float).to_numpy(),
        "launch_usdc": (base_symbol == "USDC").astype(float).to_numpy(),
        "launch_usdt": (base_symbol == "USDT").astype(float).to_numpy(),
    })  # fmt: skip
    assert (context["token_age_days"] > 0).all(), "every token of the universe was traded before t"
    return context


def user_context(legs: pd.DataFrame, task: pc.Task, rows: np.ndarray) -> pd.DataFrame:
    """User features at t, one row per given user row (all from data before t)."""
    window_tokens = task.window_size[rows]
    past = task.past[rows]
    ages = task.history_ages()[rows]
    last_trade = np.array([ages[r].data.min() for r in range(len(rows))])
    user_ids = task.users[rows]
    before = legs[(legs["timestamp"] < task.t) & legs["user"].isin(user_ids)]
    share_buys = (before["side"] == "buy").groupby(before["user"]).mean().reindex(user_ids).to_numpy()
    return pd.DataFrame({
        "user_window_tokens": window_tokens,
        "user_past_tokens": np.asarray(past.sum(axis=1)).ravel(),
        "user_days_since_last_trade": last_trade,
        "user_share_buys": share_buys,
    })  # fmt: skip


def features(stage: FirstStage, pairs: pd.DataFrame, legs: pd.DataFrame, tokens: pd.DataFrame) -> pd.DataFrame:
    """All features of the candidate pairs (FEATURE_GROUPS), plus the label.

    Every feature is computed from data before t; a test deletes every leg at
    or after t and checks that no feature changes.
    """
    task = stage.task
    rows, cols = pairs["row"].to_numpy(), pairs["col"].to_numpy()
    out = pairs.copy()

    # Popularity group
    recent = task.history(1)
    all_ages = task.history_ages()
    pop = {
        "pop_14d": stage.popularity,
        "pop_1d": np.asarray(recent.sum(axis=0)).ravel(),
        "pop_half_life_1d": np.asarray(pr.time_weights(all_ages, 1).sum(axis=0)).ravel(),
        "pop_half_life_7d": np.asarray(pr.time_weights(all_ages, 7).sum(axis=0)).ravel(),
    }
    rank = np.empty(len(task.tokens))
    rank[np.argsort(-stage.popularity, kind="stable")] = np.arange(1, len(task.tokens) + 1)
    pop["pop_rank"] = rank
    for name, values in pop.items():
        out[name] = values[cols]

    # Neighbour group, user by user (each user has few past tokens)
    window, similarity = stage.window, stage.knn_plain.similarity
    knn_score, knn_time, max_sim, n_pointing = (np.zeros(len(pairs)) for _ in range(4))
    for row, index in pairs.groupby("row").indices.items():
        c = cols[index]
        own = window[row].indices
        block = similarity[own][:, c].toarray() if len(own) else np.zeros((0, len(c)))
        knn_score[index] = block.sum(axis=0)
        max_sim[index] = block.max(axis=0) if len(own) else 0
        n_pointing[index] = (block > 0).sum(axis=0)
        profile = stage.profile_time[row]
        knn_time[index] = np.asarray((profile @ stage.knn_time.similarity[:, c]).todense()).ravel()
    out["knn_score"], out["knn_score_time"] = knn_score, knn_time
    out["max_similarity"], out["n_pointing"] = max_sim, n_pointing

    # User group
    unique_rows = np.unique(rows)
    users = user_context(legs, task, unique_rows).set_index(unique_rows)
    out = out.join(users, on="row")

    # Token group
    context = token_context(legs, tokens, task)
    for name in context.columns:
        out[name] = context[name].to_numpy()[cols]

    out["label"] = np.asarray(task.new[rows, cols]).ravel().astype(int)
    out.insert(0, "point", task.t)
    out.insert(1, "user", task.users[rows])
    return out


def point_rows(legs, tokens, task: pc.Task, settings: dict, cache_dir: Path | None = PROJECT_DIR / "ranker") -> pd.DataFrame:
    """Candidates with features and labels for every evaluated user of one point, cached as Parquet."""
    if cache_dir is not None:
        path = cache_dir / f"rows_{task.t:%Y%m%d}.parquet"
        if path.exists():
            return pd.read_parquet(path)
    stage = FirstStage(task, settings)
    table = features(stage, candidates(stage, task.eval_rows), legs, tokens)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path)
    return table


def sample_negatives(rows: pd.DataFrame, per_user: int = 50, seed: int = 0) -> pd.DataFrame:
    """Training rows: all positives, and at most `per_user` random negatives of each (point, user)."""
    rng = np.random.default_rng(seed)
    keep = rows["label"].to_numpy() == 1
    for _, index in rows.groupby(["point", "user"]).indices.items():
        negatives = index[rows["label"].to_numpy()[index] == 0]
        if len(negatives) > per_user:
            negatives = rng.choice(negatives, per_user, replace=False)
        keep[negatives] = True
    return rows[keep].reset_index(drop=True)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


def fit_model(kind: str, train: pd.DataFrame, feature_names: list[str], params: dict, n_trees: int | None = None):
    """Fit one ranker. kind is "logistic", "gbdt" (logistic loss) or "lambdamart".

    Rows must be sorted by (point, user) for "lambdamart", whose groups are
    the rows of one user at one point.
    """
    x, y = train[feature_names].to_numpy(np.float64), train["label"].to_numpy()
    x = np.nan_to_num(x)
    if kind == "logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(C=params["C"], max_iter=2000))
        return model.fit(x, y)
    base = {
        "verbose": -1,
        "seed": 0,
        "num_leaves": params["num_leaves"],
        "learning_rate": params["learning_rate"],
        "min_data_in_leaf": params["min_data_in_leaf"],
    }
    if kind == "gbdt":
        data = lgb.Dataset(x, label=y)
        return lgb.train({**base, "objective": "binary"}, data, num_boost_round=n_trees or 200)
    if kind == "lambdamart":
        groups = train.groupby(["point", "user"], sort=False).size().to_numpy()
        data = lgb.Dataset(x, label=y, group=groups)
        params_rank = {**base, "objective": "lambdarank", "lambdarank_truncation_level": pc.K, "eval_at": [pc.K]}
        return lgb.train(params_rank, data, num_boost_round=n_trees or 200)
    raise ValueError(f"unknown model kind {kind!r}")


def predict(model, rows: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Scores of a fitted ranker; higher means recommend first."""
    x = np.nan_to_num(rows[feature_names].to_numpy(np.float64))
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.predict(x)


def recall_at_k(rows: pd.DataFrame, scores: np.ndarray, positives: pd.Series, k: int = pc.K) -> float:
    """Mean recall@k over users when each user's top k candidates by score are recommended.

    `positives` holds the number of positive pairs per (point, user),
    including the cold pairs and the pairs that are not candidates: those
    count as misses, as in the main protocol.
    """
    ranked = rows[["point", "user", "label"]].assign(score=scores)
    ranked = ranked.sort_values(["point", "user", "score"], ascending=[True, True, False])
    top = ranked.groupby(["point", "user"]).head(k)
    hits = top.groupby(["point", "user"])["label"].sum()
    recall = hits.reindex(positives.index, fill_value=0) / positives
    return float(recall.mean())


def positives_per_user(tasks: dict[pd.Timestamp, pc.Task]) -> pd.Series:
    """Number of positive pairs of every evaluated (point, user), cold pairs included."""
    parts = []
    for t, task in tasks.items():
        rows = task.eval_rows
        counts = np.asarray(task.new[rows].sum(axis=1)).ravel() + task.cold_new[rows]
        index = pd.MultiIndex.from_arrays([[t] * len(rows), task.users[rows]], names=["point", "user"])
        parts.append(pd.Series(counts, index=index))
    return pd.concat(parts)


def point_folds(points: list[pd.Timestamp], points_per_fold: int = 2) -> list[list[pd.Timestamp]]:
    """Consecutive points grouped into validation folds (5 folds of 2 for the 10 Dev A points)."""
    return [points[i : i + points_per_fold] for i in range(0, len(points), points_per_fold)]


def cv_over_points(
    kind: str,
    params: dict,
    rows: pd.DataFrame,
    train_rows: pd.DataFrame,
    positives: pd.Series,
    feature_names: list[str],
    n_trees: int | None = None,
) -> float:
    """Mean recall@10 on held-out points, holding out whole points (section 8.4).

    `rows` are the full candidate rows (used for validation), `train_rows`
    the subsampled ones (used for fitting). A point is never in both the
    training and the validation part of a fold.
    """
    points = sorted(rows["point"].unique())
    scores = []
    for held_out in point_folds(points):
        train = train_rows[~train_rows["point"].isin(held_out)]
        valid = rows[rows["point"].isin(held_out)]
        assert not set(train["point"]) & set(valid["point"]), "a point is in training and validation"
        model = fit_model(kind, train, feature_names, params, n_trees)
        held_positives = positives[positives.index.get_level_values("point").isin(held_out)]
        scores.append(recall_at_k(valid, predict(model, valid, feature_names), held_positives))
    return float(np.mean(scores))


def ranker_scorer(model, feature_names: list[str], task: pc.Task, rows: pd.DataFrame) -> pc.Scorer:
    """The protocol's scorer interface for a trained ranker at one point.

    Candidates get the model's score, every other token NOT_A_CANDIDATE, so
    the protocol's top 10 is the ranker's top 10 among the candidates.
    """
    scores = predict(model, rows, feature_names)
    by_row = {row: (rows["col"].to_numpy()[index], scores[index]) for row, index in rows.groupby("row").indices.items()}

    def scorer(user_rows):
        dense = np.full((len(user_rows), len(task.tokens)), NOT_A_CANDIDATE, dtype=np.float32)
        for r, row in enumerate(user_rows):
            cols, values = by_row.get(row, (np.array([], int), np.array([])))
            dense[r, cols] = values
        return dense

    return scorer


# --------------------------------------------------------------------------
# Saving and loading a trained ranker
# --------------------------------------------------------------------------


def save_model(model, path: Path) -> None:
    """Save a ranker: LightGBM boosters as their text format, the logistic pipeline with pickle."""
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, lgb.Booster):
        model.save_model(str(path))
    else:
        with open(path, "wb") as f:
            pickle.dump(model, f)


def load_model(path: Path, kind: str):
    """Load a ranker saved by save_model; `kind` says which format it is."""
    import pickle

    if kind in ("gbdt", "lambdamart"):
        return lgb.Booster(model_file=str(path))
    with open(path, "rb") as f:
        return pickle.load(f)
