"""Tests for src/scripts/elliptic.py (no network access, synthetic data)."""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import elliptic as el


def raw_files(n=6):
    """Three tiny CSV files in the raw Elliptic layout."""
    rng = np.random.default_rng(0)
    values = rng.normal(size=(n, el.N_LOCAL + el.N_AGGREGATED))
    features = "\n".join(
        ",".join([str(100 + i), str(1 + i // 2)] + [repr(float(v)) for v in values[i]])
        for i in range(n)
    )
    classes = "txId,class\n" + "\n".join(
        f"{100 + i},{code}" for i, code in enumerate(["1", "2", "unknown", "2", "1", "unknown"][:n])
    )
    edges = "txId1,txId2\n100,101\n102,103\n104,105\n"
    return io.StringIO(features), io.StringIO(classes), io.StringIO(edges)


def toy_nodes(steps=range(1, 11), per_step=60, seed=0, flip_after=None):
    """Synthetic nodes: one informative feature, 20% illicit, 25% unlabeled.

    With `flip_after`, the feature's relation to the label reverses in later
    steps, which is a concept drift.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for step in steps:
        label = (rng.random(per_step) < 0.2).astype(int)
        sign = -1.0 if flip_after is not None and step > flip_after else 1.0
        x = rng.normal(sign * 2.0 * label, 1.0)
        observed = np.where(rng.random(per_step) < 0.25, -1, label)
        rows.append(pd.DataFrame({"time_step": step, "label": observed, "local_1": x, "local_2": rng.normal(size=per_step)}))
    nodes = pd.concat(rows, ignore_index=True)
    nodes.insert(0, "tx_id", np.arange(len(nodes)))
    return nodes


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self):
        self.urls = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        name = url.rsplit("/", 1)[1].removesuffix(".zip")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(name, "content of " + name)
        return FakeResponse(buffer.getvalue())


def test_download_raw_fetches_missing_files_only(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / el.RAW_FILES[0]).write_text("already here")
    session = FakeSession()
    el.download_raw(tmp_path, session)
    assert len(session.urls) == 2
    assert (tmp_path / "raw" / el.RAW_FILES[0]).read_text() == "already here"
    assert (tmp_path / "raw" / el.RAW_FILES[1]).read_text() == "content of " + el.RAW_FILES[1]


def test_feature_columns():
    assert len(el.feature_columns("local")) == 93
    assert len(el.feature_columns("aggregated")) == 72
    assert el.feature_columns() == el.feature_columns("local") + el.feature_columns("aggregated")
    assert "time_step" not in el.feature_columns()
    with pytest.raises(ValueError):
        el.feature_columns("other")


def test_parse_raw():
    nodes, edges = el.parse_raw(*raw_files())
    assert list(nodes.columns[:3]) == ["tx_id", "time_step", "label"]
    assert nodes.shape == (6, 3 + 165)
    # Raw class "1" is illicit, "2" is licit.
    assert nodes["label"].tolist() == [1, 0, -1, 0, 1, -1]
    assert nodes["time_step"].tolist() == [1, 1, 2, 2, 3, 3]
    assert nodes["local_1"].dtype == np.float32
    assert list(edges.columns) == ["src", "dst"] and len(edges) == 3


def test_parse_raw_rejects_unknown_class_code():
    features, _, edges = raw_files(2)
    with pytest.raises(ValueError):
        el.parse_raw(features, io.StringIO("txId,class\n100,3\n101,2\n"), edges)


def test_load_elliptic_caches_as_parquet(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name, content in zip(el.RAW_FILES, raw_files()):
        (raw / name).write_text(content.getvalue())
    nodes, edges = el.load_elliptic(tmp_path)
    assert (tmp_path / "nodes.parquet").exists() and (tmp_path / "edges.parquet").exists()
    # The second call must work from the cache alone.
    for name in el.RAW_FILES:
        (raw / name).unlink()
    again, edges_again = el.load_elliptic(tmp_path)
    pd.testing.assert_frame_equal(nodes, again)
    pd.testing.assert_frame_equal(edges, edges_again)


def test_temporal_split_is_ordered_and_complete():
    nodes = toy_nodes()
    parts = el.temporal_split(nodes, train_end=5, dev_end=8)
    assert sorted(parts["train"]["time_step"].unique()) == [1, 2, 3, 4, 5]
    assert sorted(parts["dev"]["time_step"].unique()) == [6, 7, 8]
    assert sorted(parts["test"]["time_step"].unique()) == [9, 10]
    assert sum(len(p) for p in parts.values()) == len(nodes)
    with pytest.raises(ValueError):
        el.temporal_split(nodes, train_end=8, dev_end=5)


def test_temporal_split_defaults_match_the_standard_test_period():
    nodes = toy_nodes(steps=range(1, 50), per_step=4)
    parts = el.temporal_split(nodes)
    assert parts["test"]["time_step"].min() == 35 and parts["dev"]["time_step"].min() == 27


def test_labeled_drops_unknown():
    nodes = toy_nodes()
    assert (el.labeled(nodes)["label"] >= 0).all()
    assert len(el.labeled(nodes)) == (nodes["label"] >= 0).sum()


def test_illicit_scores():
    y = [1, 1, 1, 0, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.7, 0.1, 0.1]
    result = el.illicit_scores(y, scores, threshold=0.5)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["n_illicit"] == 3 and result["n_flagged"] == 3


def test_illicit_scores_edge_cases():
    nothing_flagged = el.illicit_scores([1, 0], [0.1, 0.1])
    assert nothing_flagged["precision"] == 0.0 and nothing_flagged["recall"] == 0.0
    assert nothing_flagged["f1"] == 0.0
    no_illicit = el.illicit_scores([0, 0], [0.9, 0.1])
    assert np.isnan(no_illicit["recall"]) and no_illicit["precision"] == 0.0


def test_best_f1_threshold():
    y = np.array([0, 0, 0, 1, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9])
    threshold = el.best_f1_threshold(y, scores)
    # Flagging from 0.4 upwards gives 3 hits in 4 flags: F1 = 6/7, the best available.
    assert threshold == pytest.approx(0.4)
    assert el.illicit_scores(y, scores, threshold)["f1"] == pytest.approx(6 / 7)


def test_ranking_scores():
    perfect = el.ranking_scores([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert perfect == {"average_precision": 1.0, "roc_auc": 1.0}
    assert np.isnan(el.ranking_scores([0, 0], [0.1, 0.2])["average_precision"])


def test_per_step_scores():
    table = el.per_step_scores([1, 1, 2, 2], [1, 0, 1, 0], [0.9, 0.1, 0.2, 0.8])
    assert table.loc[1, "f1"] == pytest.approx(1.0)
    assert table.loc[2, "f1"] == pytest.approx(0.0)
    assert table.loc[1, "n_labeled"] == 2 and table.loc[2, "illicit_share"] == pytest.approx(0.5)


def test_bootstrap_interval_contains_the_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.2).astype(int)
    scores = y * 0.3 + rng.random(400) * 0.5

    def f1(a, b):
        return el.illicit_scores(a, b, 0.5)["f1"]

    low, high = el.bootstrap_interval(y, scores, f1, n_boot=200, seed=3)
    assert low < f1(y, scores) < high
    assert (low, high) == el.bootstrap_interval(y, scores, f1, n_boot=200, seed=3)


def test_domain_auc_separates_shifted_samples_only():
    rng = np.random.default_rng(0)
    same = el.domain_auc(rng.normal(size=(300, 4)), rng.normal(size=(300, 4)), n_estimators=30)
    shifted = el.domain_auc(rng.normal(size=(300, 4)), rng.normal(1.5, 1, size=(300, 4)), n_estimators=30)
    assert 0.4 < same < 0.6
    assert shifted > 0.9


def test_edge_label_table():
    nodes = pd.DataFrame({"tx_id": [1, 2, 3, 4], "label": [1, 0, -1, 1]})
    edges = pd.DataFrame({"src": [1, 1, 2, 3], "dst": [4, 2, 3, 1]})
    table = el.edge_label_table(edges, nodes)
    assert table.loc["illicit", "illicit"] == 1
    assert table.loc["illicit", "licit"] == 1
    assert table.loc["licit", "unknown"] == 1
    assert table.loc["unknown", "illicit"] == 1
    assert table.to_numpy().sum() == 4


def test_walk_forward_scores_uses_only_the_past():
    nodes = toy_nodes()
    features = ["local_1", "local_2"]
    scores = el.walk_forward_scores(nodes, features, LogisticRegression, first_step=6, last_step=10)
    scored = nodes.loc[scores.index]
    assert sorted(scored["time_step"].unique()) == [6, 7, 8, 9, 10]
    assert (scored["label"] >= 0).all()
    assert el.ranking_scores(scored["label"], scores)["roc_auc"] > 0.85

    # Changing labels in the future must not change earlier scores.
    changed = nodes.copy()
    changed.loc[changed["time_step"] >= 9, "label"] = changed.loc[changed["time_step"] >= 9, "label"].map({1: 0, 0: 1, -1: -1})
    again = el.walk_forward_scores(changed, features, LogisticRegression, first_step=6, last_step=10)
    early = scored.index[scored["time_step"] <= 9]
    assert again.loc[early].to_numpy() == pytest.approx(scores.loc[early].to_numpy())


def test_walk_forward_adapts_to_drift_and_respects_window_and_delay():
    nodes = toy_nodes(steps=range(1, 21), flip_after=10)
    features = ["local_1", "local_2"]
    late = nodes[(nodes["time_step"] >= 16) & (nodes["label"] >= 0)]

    def auc(**kwargs):
        s = el.walk_forward_scores(nodes, features, LogisticRegression, 16, 20, **kwargs)
        return el.ranking_scores(late["label"], s.loc[late.index])["roc_auc"]

    # A short window has forgotten the old concept; a long history is still confused by it.
    assert auc(window=3) > 0.85
    assert auc(window=3) > auc()
    # With a long label delay the window reaches back into the old concept.
    assert auc(window=3, label_delay=6) < 0.5


def test_walk_forward_skips_steps_without_usable_history():
    nodes = toy_nodes()
    scores = el.walk_forward_scores(nodes, ["local_1"], LogisticRegression, first_step=1, last_step=2)
    assert set(nodes.loc[scores.index, "time_step"]) == {2}
