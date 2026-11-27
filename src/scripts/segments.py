"""Account segmentation (work package 11, implementation_plan_applications.md section 11).

Trader types from behaviour, and whether the recommender should treat them
differently. Every feature of an account at a point t uses only legs before t.

Features (account_features), for the evaluated accounts of a task:

  trades_14d            legs in the 14 days before t
  tokens_14d            tokens in the 14-day window
  days_active_14d       distinct days with a trade in those 14 days
  account_age_days      days since the account's first trade in the data
  head_share            share of its window tokens in the popular head
  token_age_days        median age of the tokens it traded, at its first trade of each
  young_share           share of its 14-day trades in tokens younger than a day
  hold_hours            median hours from its first buy of a token to its first sell
                        after it; tokens never sold count as 336 (14 days), and
                        every value is capped there
  buy_share             share of its 14-day trades that are buys
  trades_per_day        trades_14d / days_active_14d
  prev_log_rank         median log(1 + rank) of its pick-ups at the previous
                        point (WP9); missing when it was not evaluated there.
                        The rank is used instead of the plan's rank percentile,
                        which WP9 found compressed toward 0 (notebook 20).

The model uses standardized features (log(1 + x) for counts and durations);
a missing prev_log_rank is filled with the median of the fitting data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import expectedness as ex
import pool_recsys as pr
import protocol as pc
import tuning

FEATURES = ["trades_14d", "tokens_14d", "days_active_14d", "account_age_days", "head_share", "token_age_days",
            "young_share", "hold_hours", "buy_share", "trades_per_day", "prev_log_rank"]  # fmt: skip
LOGGED = ["trades_14d", "tokens_14d", "days_active_14d", "account_age_days", "token_age_days", "hold_hours", "trades_per_day"]
MAX_HOLD_HOURS = 336.0
WEIGHTS = [round(w, 1) for w in np.arange(0, 1.01, 0.1)]


def account_features(legs: pd.DataFrame, task: pc.Task, tokens: pd.DataFrame, pairs_previous: pd.DataFrame | None = None) -> pd.DataFrame:
    """The features of the module docstring for the task's evaluated accounts (index: user)."""
    t = task.t
    users = task.users[task.eval_rows]
    past = legs[(legs["timestamp"] < t) & legs["user"].isin(users)]
    recent = past[past["timestamp"] >= t - pd.Timedelta(days=pc.REFERENCE_DAYS)]
    born = tokens.set_index("token_id")["first_timestamp"]
    out = pd.DataFrame(index=pd.Index(users, name="user"))

    out["trades_14d"] = recent.groupby("user").size().reindex(users).fillna(0)
    out["tokens_14d"] = task.window_size[task.eval_rows]
    out["days_active_14d"] = recent.groupby("user")["timestamp"].agg(lambda s: s.dt.floor("D").nunique()).reindex(users).fillna(0)
    out["account_age_days"] = ((t - past.groupby("user")["timestamp"].min()).dt.total_seconds() / 86400).reindex(users)

    head = set(task.tokens[pc.head_tokens(task.history(pc.REFERENCE_DAYS))].tolist())
    window_pairs = recent[["user", "token"]].drop_duplicates()
    out["head_share"] = window_pairs.assign(h=window_pairs["token"].isin(head)).groupby("user")["h"].mean().reindex(users)

    first = past.groupby(["user", "token"], observed=True)["timestamp"].min().reset_index()
    first["age"] = (first["timestamp"] - first["token"].map(born)).dt.total_seconds() / 86400
    out["token_age_days"] = first.groupby("user")["age"].median().reindex(users)

    trade_age = (recent["timestamp"] - recent["token"].map(born)).dt.total_seconds() / 86400
    out["young_share"] = (trade_age < 1).groupby(recent["user"]).mean().reindex(users)

    buys = past[past["side"] == "buy"].groupby(["user", "token"], observed=True)["timestamp"].min().rename("first_buy")
    sells = past[past["side"] == "sell"][["user", "token", "timestamp"]].merge(buys.reset_index(), on=["user", "token"])
    first_sell = sells[sells["timestamp"] > sells["first_buy"]].groupby(["user", "token"], observed=True)["timestamp"].min().rename("first_sell")
    held = buys.to_frame().join(first_sell)
    held["hours"] = ((held["first_sell"] - held["first_buy"]).dt.total_seconds() / 3600).fillna(MAX_HOLD_HOURS).clip(upper=MAX_HOLD_HOURS)
    out["hold_hours"] = held.groupby(level="user")["hours"].median().reindex(users).fillna(MAX_HOLD_HOURS)

    out["buy_share"] = (recent["side"] == "buy").groupby(recent["user"]).mean().reindex(users)
    out["trades_per_day"] = out["trades_14d"] / out["days_active_14d"].clip(lower=1)

    out["prev_log_rank"] = np.nan
    if pairs_previous is not None and len(pairs_previous):
        warm = pairs_previous[~pairs_previous["cold"]]
        out["prev_log_rank"] = np.log1p(warm["rank"]).groupby(warm["user"]).median().reindex(users)
    return out


@dataclass
class Model:
    """A fitted segmentation: the fill value, the scaler and the clustering."""

    fill: float
    scaler: StandardScaler
    cluster: object
    kind: str


def _matrix(features: pd.DataFrame, fill: float) -> np.ndarray:
    x = features[FEATURES].copy()
    x["prev_log_rank"] = x["prev_log_rank"].fillna(fill)
    x = x.fillna(x.median())
    for column in LOGGED:
        x[column] = np.log1p(x[column].clip(lower=0))
    return x.to_numpy(dtype=float)


def fit(features: pd.DataFrame, k: int, kind: str = "kmeans", seed: int = 0) -> Model:
    """Standardize and cluster: kind is "kmeans" or "mixture" (Gaussian, full covariance)."""
    fill = float(features["prev_log_rank"].median()) if features["prev_log_rank"].notna().any() else 0.0
    x = _matrix(features, fill)
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    if kind == "kmeans":
        cluster = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z)
    elif kind == "mixture":
        cluster = GaussianMixture(n_components=k, covariance_type="full", random_state=seed).fit(z)
    else:
        raise ValueError(kind)
    return Model(fill, scaler, cluster, kind)


def assign(model: Model, features: pd.DataFrame) -> np.ndarray:
    """The segment of each row of features."""
    return model.cluster.predict(model.scaler.transform(_matrix(features, model.fill)))


def split_stability(features: pd.DataFrame, k: int, kind: str = "kmeans", seed: int = 0) -> float:
    """Fit on two random halves; the adjusted Rand index of the two models' segments for all rows."""
    rng = np.random.default_rng(seed)
    half = rng.random(len(features)) < 0.5
    a = fit(features[half], k, kind, seed)
    b = fit(features[~half], k, kind, seed)
    return float(adjusted_rand_score(assign(a, features), assign(b, features)))


# --------------------------------------------------------------------------
# The blend per segment
# --------------------------------------------------------------------------


def recall_by_weight(task: pc.Task, weights=WEIGHTS, settings: dict = ex.FROZEN_BLEND) -> pd.DataFrame:
    """Recall@10 of every evaluated account under the fixed blend at each popularity weight: users x weights.

    An account's list depends only on its own weight, so a blend with one weight
    per segment is scored exactly by taking each account's column of its segment.
    """
    window = task.history(pc.REFERENCE_DAYS)
    popularity = pr.Popularity().fit(window).item_scores
    popularity_scaled = popularity / max(popularity.max(), 1e-12)
    similarity = pr.ItemKNN(settings["k"], settings["shrink"]).fit(window).similarity
    rows_all, past = task.eval_rows, task.past
    batch = max(64, 25_000_000 // max(len(task.tokens), 1))
    parts = {w: [] for w in weights}
    for start in range(0, len(rows_all), batch):
        rows = rows_all[start : start + batch]
        neighbours = tuning.scale_rows(np.asarray((window[rows] @ similarity).todense(), dtype=np.float32))
        for w in weights:
            recommended = pr.top_k(w * popularity_scaled[None, :] + (1 - w) * neighbours, pc.K, exclude=past[rows])
            parts[w].append(pr.ranking_metrics(recommended, task.new[rows], task.cold_new[rows])["recall"].to_numpy())
    return pd.DataFrame({w: np.concatenate(parts[w]) for w in weights}, index=pd.Index(task.users[rows_all], name="user"))


def per_segment_blend(task: pc.Task, segment_of_row: np.ndarray, weight_of_segment: dict, settings: dict = ex.FROZEN_BLEND) -> pc.Scorer:
    """A protocol scorer: the fixed blend with the popularity weight of each account's segment.

    segment_of_row: segment per task row (users of the task); weight_of_segment: segment -> w.
    """
    window = task.history(pc.REFERENCE_DAYS)
    popularity = pr.Popularity().fit(window)
    neighbours = pr.ItemKNN(settings["k"], settings["shrink"]).fit(window)

    def scorer(rows):
        pop = popularity.scores(window[rows])
        knn = neighbours.scores(window[rows])
        pop = pop / np.maximum(pop.max(axis=1, keepdims=True), 1e-12)
        knn = knn / np.maximum(knn.max(axis=1, keepdims=True), 1e-12)
        w = np.array([weight_of_segment[s] for s in segment_of_row[rows]], dtype=np.float32)[:, None]
        return w * pop + (1 - w) * knn

    return scorer
