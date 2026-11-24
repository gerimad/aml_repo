"""Tests for src/scripts/segments.py (WP11)."""

import numpy as np
import pandas as pd

import expectedness as ex
import protocol as pc
import segments as sg

T = pd.Timestamp("2026-07-10", tz="UTC")
DAY = pd.Timedelta(days=1)


def legs(rows):
    """Legs from (time, user, token, side) tuples."""
    return pd.DataFrame([{"timestamp": when, "user": u, "token": i, "side": s} for when, u, i, s in rows])


def market(seed=0, users=40, tokens=15, trades=500):
    rng = np.random.default_rng(seed)
    rows = [(T - DAY * float(rng.uniform(0.1, 13)), int(u), int(i), "buy" if rng.random() < 0.6 else "sell")
            for u, i in zip(rng.integers(0, users, trades), rng.integers(0, tokens, trades))]  # fmt: skip
    rows += [(T + DAY * float(rng.uniform(0, 1.9)), int(u), int(i), "buy") for u, i in zip(rng.integers(0, users, 120), rng.integers(0, tokens, 120))]
    return legs(rows)


def tokens_of(m):
    born = m.groupby("token")["timestamp"].min()
    return pd.DataFrame({"token_id": born.index, "first_timestamp": born.to_numpy()})


def test_features_use_only_legs_before_t():
    m = market()
    later = pd.concat([m, legs([(T + 3 * DAY, u, 3, "sell") for u in range(40)])], ignore_index=True)
    a = sg.account_features(m, pc.make_task(m, T), tokens_of(m))
    b = sg.account_features(later, pc.make_task(later, T), tokens_of(m))
    pd.testing.assert_frame_equal(a, b)


def test_hold_time_is_first_buy_to_first_later_sell():
    rows = [(T - 5 * DAY, 1, 10, "sell"), (T - 4 * DAY, 1, 10, "buy"), (T - 4 * DAY + pd.Timedelta(hours=6), 1, 10, "sell"),
            (T - 3 * DAY, 1, 11, "buy"), (T + DAY, 1, 12, "buy"), (T - DAY, 2, 12, "buy")]  # fmt: skip
    m = legs(rows)
    features = sg.account_features(m, pc.make_task(m, T), tokens_of(m)).loc[1]
    assert features["hold_hours"] == np.median([6.0, sg.MAX_HOLD_HOURS]), "token 10 held 6 h; token 11 never sold"


def test_two_separated_clouds_give_two_segments_and_stability_1():
    rng = np.random.default_rng(0)
    base = {f: rng.normal(size=400) * 0.01 + 1 for f in sg.FEATURES}
    features = pd.DataFrame(base)
    features.loc[200:, sg.FEATURES] += 5  # the second cloud is far away in every feature
    for kind in ("kmeans", "mixture"):
        labels = sg.assign(sg.fit(features, 2, kind), features)
        assert len(set(labels[:200])) == 1 and len(set(labels[200:])) == 1 and labels[0] != labels[-1]
        assert sg.split_stability(features, 2, kind) == 1.0


def test_one_weight_everywhere_equals_the_plain_blend():
    m = market()
    task = pc.make_task(m, T)
    settings = {"k": 5, "shrink": 1.0, "weight": 0.3}
    plain, _, _ = ex.blend_model(task, settings)
    segment_of_row = np.arange(len(task.users)) % 3
    per_segment = sg.per_segment_blend(task, segment_of_row, {0: 0.3, 1: 0.3, 2: 0.3}, settings)
    rows = task.eval_rows
    np.testing.assert_allclose(per_segment(rows), plain(rows), atol=1e-6)


def test_recall_by_weight_matches_the_protocol():
    m = market()
    task = pc.make_task(m, T)
    settings = {"k": 5, "shrink": 1.0, "weight": 0.5}
    table = sg.recall_by_weight(task, [0.5], settings)
    scorer, _, _ = ex.blend_model(task, settings)
    per_user, _ = pc.evaluate_point(scorer, task)
    np.testing.assert_allclose(table[0.5].to_numpy(), per_user["recall"].to_numpy())
