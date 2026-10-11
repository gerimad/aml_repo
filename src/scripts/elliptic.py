"""Load, split and evaluate the Elliptic Bitcoin transaction dataset.

The dataset (Weber et al., 2019) has 203,769 Bitcoin transactions in 49 time
steps, 166 anonymized features per transaction, a payment-flow graph, and a
licit / illicit label for about a quarter of the transactions.

Download once, from the repository root (about 150 MB, kept in data/elliptic/):

    uv run python src/scripts/elliptic.py

Then, in a notebook in src/notebooks:

    import sys; sys.path.append("../scripts")
    import elliptic as el

    nodes, edges = el.load_elliptic()
    parts = el.temporal_split(nodes)                 # train / dev / test by time step
    train = el.labeled(parts["train"])
    X, y = train[el.feature_columns()], train["label"]

Labels are 1 for illicit, 0 for licit and -1 for unknown.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "elliptic"
# Public mirror used by PyTorch Geometric; the original release is on Kaggle
# (https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and needs an account.
MIRROR_URL = "https://data.pyg.org/datasets/elliptic"
RAW_FILES = ["elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"]

N_LOCAL, N_AGGREGATED = 93, 72  # plus the time step: the paper's 94 + 72 = 166 features
LOCAL_COLUMNS = [f"local_{i}" for i in range(1, N_LOCAL + 1)]
AGGREGATED_COLUMNS = [f"agg_{i}" for i in range(1, N_AGGREGATED + 1)]
LABEL_CODES = {"1": 1, "2": 0, "unknown": -1}  # the raw files code illicit as 1, licit as 2

TRAIN_END, DEV_END = 26, 34  # last time step of train and of dev; test is 35 to 49
SHUTDOWN_STEP = 43  # the dark market closure reported by Weber et al. (2019)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def download_raw(data_dir: Path = DATA_DIR, session=None) -> None:
    """Download the three raw CSV files into `data_dir`/raw, skipping those already there."""
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        if (raw_dir / name).exists():
            continue
        response = (session or requests).get(f"{MIRROR_URL}/{name}.zip", timeout=600)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            archive.extract(name, raw_dir)


def feature_columns(kind: str = "all") -> list[str]:
    """Names of the feature columns: "local" (93), "aggregated" (72) or "all" (165).

    The time step is not included: it identifies the period, and a model that
    uses it cannot be applied to a later period.
    """
    if kind == "local":
        return list(LOCAL_COLUMNS)
    if kind == "aggregated":
        return list(AGGREGATED_COLUMNS)
    if kind == "all":
        return LOCAL_COLUMNS + AGGREGATED_COLUMNS
    raise ValueError(f"unknown feature kind {kind!r}")


def parse_raw(features_csv, classes_csv, edges_csv) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn the three raw CSV files (paths or file objects) into (nodes, edges).

    nodes: one row per transaction with tx_id, time_step, label and the
           feature columns, in the order of the features file.
    edges: one row per payment flow with the transaction ids src and dst.
    """
    names = ["tx_id", "time_step"] + feature_columns()
    nodes = pd.read_csv(features_csv, header=None, names=names)
    nodes = nodes.astype({c: "float32" for c in feature_columns()})
    classes = pd.read_csv(classes_csv, dtype=str)
    labels = classes.set_index(classes["txId"].astype("int64"))["class"].map(LABEL_CODES)
    if labels.isna().any():
        raise ValueError("unexpected class code in the classes file")
    nodes.insert(2, "label", nodes["tx_id"].map(labels).astype("int8"))
    edges = pd.read_csv(edges_csv).rename(columns={"txId1": "src", "txId2": "dst"})
    return nodes, edges


def load_elliptic(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the dataset as (nodes, edges), downloading and caching it if needed.

    The first call parses the CSV files (slow) and writes Parquet files; later
    calls read those.
    """
    nodes_file, edges_file = data_dir / "nodes.parquet", data_dir / "edges.parquet"
    if not (nodes_file.exists() and edges_file.exists()):
        download_raw(data_dir)
        raw = data_dir / "raw"
        nodes, edges = parse_raw(*(raw / name for name in RAW_FILES))
        nodes.to_parquet(nodes_file)
        edges.to_parquet(edges_file)
    return pd.read_parquet(nodes_file), pd.read_parquet(edges_file)


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def temporal_split(
    nodes: pd.DataFrame, train_end: int = TRAIN_END, dev_end: int = DEV_END
) -> dict[str, pd.DataFrame]:
    """Split transactions by time step into train / dev / test, in order.

    Train is steps up to `train_end`, dev the steps after it up to `dev_end`,
    test everything later. With the defaults the test part (steps 35 to 49)
    is the one used by Weber et al. (2019) and most later work.
    """
    if not train_end < dev_end:
        raise ValueError("train_end must be before dev_end")
    step = nodes["time_step"]
    return {
        "train": nodes[step <= train_end],
        "dev": nodes[(step > train_end) & (step <= dev_end)],
        "test": nodes[step > dev_end],
    }


def labeled(nodes: pd.DataFrame) -> pd.DataFrame:
    """Keep only the transactions with a known label (licit or illicit)."""
    return nodes[nodes["label"] >= 0]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def illicit_scores(y_true, scores, threshold: float = 0.5) -> dict[str, float]:
    """Precision, recall and F1 of the illicit class when flagging scores >= threshold.

    Also returns the number of illicit transactions and of flagged ones.
    Precision is 0 when nothing is flagged, recall is NaN when nothing is illicit.
    """
    y_true = np.asarray(y_true).astype(bool)
    flagged = np.asarray(scores) >= threshold
    hits = int((flagged & y_true).sum())
    n_illicit, n_flagged = int(y_true.sum()), int(flagged.sum())
    precision = hits / n_flagged if n_flagged else 0.0
    recall = hits / n_illicit if n_illicit else float("nan")
    f1 = 2 * hits / (n_illicit + n_flagged) if (n_illicit + n_flagged) else float("nan")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_illicit": n_illicit,
        "n_flagged": n_flagged,
    }


def best_f1_threshold(y_true, scores) -> float:
    """The score threshold that maximizes illicit F1 on the given data.

    Use it on dev data only, then apply the threshold unchanged to test data.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def ranking_scores(y_true, scores) -> dict[str, float]:
    """Threshold-free scores: average precision (area under the PR curve) and ROC AUC.

    Both are NaN when only one class is present.
    """
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return {"average_precision": float("nan"), "roc_auc": float("nan")}
    return {
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }


def per_step_scores(time_steps, y_true, scores, threshold: float = 0.5) -> pd.DataFrame:
    """Illicit precision, recall, F1 and counts for each time step separately."""
    frame = pd.DataFrame({"step": np.asarray(time_steps), "y": np.asarray(y_true), "s": scores})
    rows = {
        int(step): {
            **illicit_scores(g["y"], g["s"], threshold),
            "n_labeled": len(g),
            "illicit_share": float(g["y"].mean()),
        }
        for step, g in frame.groupby("step")
    }
    return pd.DataFrame(rows).T.rename_axis("time_step")


def bootstrap_interval(
    y_true, scores, metric, n_boot: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap interval of `metric(y_true, scores)`, resampling transactions.

    Resamples that happen to contain a single class are skipped.
    """
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if 0 < y_true[idx].sum() < len(idx):
            values.append(metric(y_true[idx], scores[idx]))
    low, high = np.quantile(values, [(1 - level) / 2, (1 + level) / 2])
    return float(low), float(high)


# --------------------------------------------------------------------------
# Drift diagnostics
# --------------------------------------------------------------------------


def domain_auc(X_a, X_b, seed: int = 0, n_estimators: int = 100) -> float:
    """How well a classifier can tell two samples apart, as a cross-validated ROC AUC.

    0.5 means the two samples look alike (no shift in the features), 1.0
    means every row can be assigned to its sample. This is the classifier
    two-sample test for dataset shift (Rabanser et al., 2019).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X = np.vstack([np.asarray(X_a), np.asarray(X_b)])
    origin = np.r_[np.zeros(len(X_a), dtype=int), np.ones(len(X_b), dtype=int)]
    model = RandomForestClassifier(
        n_estimators=n_estimators, min_samples_leaf=5, n_jobs=-1, random_state=seed
    )
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    proba = cross_val_predict(model, X, origin, cv=folds, method="predict_proba")[:, 1]
    return float(roc_auc_score(origin, proba))


def edge_label_table(edges: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Count payment flows by the labels of their two ends.

    Returns a 3 x 3 table (source label by destination label) over the
    categories illicit, licit and unknown.
    """
    names = {1: "illicit", 0: "licit", -1: "unknown"}
    label = nodes.set_index("tx_id")["label"].map(names)
    table = pd.crosstab(edges["src"].map(label), edges["dst"].map(label))
    order = ["illicit", "licit", "unknown"]
    return table.reindex(index=order, columns=order, fill_value=0).rename_axis(
        index="source", columns="destination"
    )


# --------------------------------------------------------------------------
# Walk-forward evaluation
# --------------------------------------------------------------------------


def walk_forward_scores(
    nodes: pd.DataFrame,
    features: list[str],
    make_model,
    first_step: int,
    last_step: int,
    window: int | None = None,
    label_delay: int = 0,
) -> pd.Series:
    """Score each time step with a model trained only on earlier, labeled steps.

    For every step t from `first_step` to `last_step`, a fresh model from
    `make_model()` is fitted on the labeled transactions of steps before
    t - label_delay (the last `window` of them if given) and scores all
    labeled transactions of step t. `label_delay` models labels that only
    become known some steps after a transaction.

    Returns the illicit scores indexed like the scored rows of `nodes`.
    """
    known = labeled(nodes)
    out = []
    for step in range(first_step, last_step + 1):
        cutoff = step - label_delay  # train on steps strictly before the cutoff
        past = known[known["time_step"] < cutoff]
        if window is not None:
            past = past[past["time_step"] >= cutoff - window]
        current = known[known["time_step"] == step]
        if current.empty or past["label"].nunique() < 2:
            continue
        model = make_model().fit(past[features], past["label"])
        out.append(pd.Series(model.predict_proba(current[features])[:, 1], index=current.index))
    return pd.concat(out) if out else pd.Series(dtype=float)


if __name__ == "__main__":
    all_nodes, all_edges = load_elliptic()
    print(f"{len(all_nodes):,} transactions, {len(all_edges):,} payment flows in {DATA_DIR}")
