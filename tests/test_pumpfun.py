import numpy as np
import pandas as pd
import pytest

import pumpfun as pf


def launches(**columns) -> pd.DataFrame:
    """A small launch table with sensible defaults for the columns not given."""
    n = len(next(iter(columns.values())))
    base = {
        "mint": [f"m{i}" for i in range(n)],
        "creator": ["c"] * n,
        "t0": np.arange(n, dtype=float) * 100.0,
        "name": ["x"] * n,
        "mayhem": [0] * n,
        "quote_asset": ["sol"] * n,
        "decoder_complete": [1] * n,
        "dev_buy_sol": [1.0] * n,
        "graduated": [0] * n,
        "migrate_dt_s": [np.nan] * n,
        "rugged": [0] * n,
        "rug_dt_s": [np.nan] * n,
        "rug_branch_inferred": [None] * n,
        "dev_first_sell_dt_s": [np.nan] * n,
        "f10_bundle_wallets": [0.0] * n,
        "f10_dev_share": [0.01] * n,
        "f10_rs": [1.0] * n,
        "f10_age": [2.0] * n,
    }
    base.update(columns)
    df = pd.DataFrame(base)
    df["time"] = pd.to_datetime(df["t0"], unit="s", utc=True)
    return df


class FakeResponse:
    content = b"data"

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self):
        self.urls = []

    def get(self, url, timeout):
        self.urls.append(url)
        return FakeResponse()


def test_subset_days_covers_both_ends():
    days = pf.subset_days("launches")
    assert days[0] == "2026-09-27" and days[-1] == "2026-10-03" and len(days) == 7
    assert len(pf.subset_days("launches_legacy")) == 27


def test_day_url_pins_the_revision():
    url = pf.day_url("launches", "2026-09-28")
    assert url.endswith("/data/launches/2026-09-28.parquet") and pf.REVISION in url


def test_download_fetches_missing_files_only(tmp_path):
    (tmp_path / "launches").mkdir()
    (tmp_path / "launches" / "2026-09-27.parquet").write_bytes(b"old")
    session = FakeSession()
    pf.download(tmp_path, session=session)
    assert len(session.urls) == 2 + 34 - 1
    assert (tmp_path / "launches" / "2026-09-27.parquet").read_bytes() == b"old"
    assert (tmp_path / "launches_legacy" / "2026-08-31.parquet").read_bytes() == b"data"


def test_load_launches_sorts_and_adds_time(tmp_path):
    (tmp_path / "launches").mkdir()
    (tmp_path / "launches_legacy").mkdir()
    pd.DataFrame({"t0": [1790467300.0, 1790467200.0]}).to_parquet(tmp_path / "launches" / "a.parquet")
    pd.DataFrame({"t0": [1790380800.0]}).to_parquet(tmp_path / "launches_legacy" / "b.parquet")
    new = pf.load_launches("launches", tmp_path)
    assert new["t0"].tolist() == [1790467200.0, 1790467300.0]
    assert new["day"].tolist() == ["2026-09-27", "2026-09-27"]
    assert len(pf.load_launches("all", tmp_path)) == 3
    with pytest.raises(FileNotFoundError):
        pf.load_launches("launches", tmp_path / "missing")


def test_study_population_filters_each_condition():
    window = pd.Timestamp("2026-09-30 01:00", tz="UTC").timestamp()
    df = launches(
        mayhem=[0, 1, 0, 0, 0],
        quote_asset=["sol", "sol", "usdc", "sol", "sol"],
        decoder_complete=[1, 1, 1, 0, 1],
        t0=[0.0, 1.0, 2.0, 3.0, window],
    )
    assert pf.study_population(df)["mint"].tolist() == ["m0"]


def test_outcome_precedence():
    df = launches(
        dev_buy_sol=[85.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        graduated=[1, 1, 0, 0, 0, 0],
        rugged=pd.array([0, 1, 1, 1, 0, None], dtype="Int64"),
        rug_branch_inferred=[None, "dev_dump_or_recovered", "drain", "dev_dump_or_recovered", None, None],
    )
    assert pf.outcome(df).tolist() == ["creator_buyout", "graduated", "rug_drain", "rug_dump", "faded", "unresolved"]


def test_eligible_at_snapshot_excludes_decided_and_unknown():
    df = launches(
        dev_buy_sol=[1.0, 85.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        f10_rs=[1.0, 85.0, 84.0, 1.0, 1.0, 1.0, 1.0],
        graduated=[0, 1, 0, 1, 0, 0, 1],
        migrate_dt_s=[np.nan, 0.1, np.nan, 1.0, np.nan, np.nan, 50.0],
        rugged=pd.array([0, 0, 0, 0, 1, None, 0], dtype="Int64"),
        rug_dt_s=[np.nan, np.nan, np.nan, np.nan, 2.5, np.nan, np.nan],
        f10_bundle_wallets=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
    )
    assert pf.eligible_at_snapshot(df).tolist() == [True, False, False, False, False, False, True]
    # A rug after the snapshot (age 2 s) but before the prediction moment (3 s) is already decided.
    assert pf.eligible_at_snapshot(df, moment=2.2).tolist()[4]
    df.loc[0, "f10_bundle_wallets"] = np.nan
    df.loc[6, "f10_age"] = 5.0
    assert pf.eligible_at_snapshot(df).tolist() == [False] * 7


def test_add_targets_looks_only_after_the_snapshot():
    df = launches(
        f10_dev_share=[0.1, 0.1, 0.0, 0.1, 0.1],
        dev_first_sell_dt_s=[30.0, 2.5, np.nan, 500.0, 63.5],
        graduated=[1, 0, 0, 0, 0],
        rugged=[0, 1, 0, 0, 0],
    )
    out = pf.add_targets(df, moment=3.0, horizon=60.0)
    assert out["y_grad"].tolist() == [1, 0, 0, 0, 0] and out["y_rug"].tolist() == [0, 1, 0, 0, 0]
    # Sold before the moment, or holding nothing at the snapshot: not a holder.
    assert out["dev_holds"].tolist() == [True, False, False, True, True]
    assert out["y_dev_exit"].tolist() == [1, 1, 0, 0, 0]


def test_normalize_name_merges_spelling_variants():
    assert pf.normalize_name("Dog Coin") == pf.normalize_name("  dog-COIN!! ") == "dogcoin"
    assert pf.normalize_name("ＤＯＧ") == "dog"  # full-width letters
    assert pf.normalize_name("狗") == "狗"
    assert pf.normalize_name(" $$$ ") == "$$$"
    assert pf.normalize_name(None) == ""


def test_count_before_is_strict_and_per_group():
    counts = pf.count_before(["a", "a", "b", "a"], [10.0, 20.0, 20.0, 5.0], ["a", "a", "b", "a"], [5.0, 20.0, np.nan, 12.0])
    assert counts.tolist() == [1, 2, 0, 0]


def test_count_before_handles_groups_without_events():
    assert pf.count_before(["z"], [1.0], ["a"], [0.0]).tolist() == [0]
    assert pf.count_before(["a", "b"], [1.0, 1.0], [], []).tolist() == [0, 0]


def test_creator_history_uses_only_what_had_happened():
    df = launches(
        creator=["a", "a", "a", "b"],
        t0=[0.0, 100.0, 4000.0, 50.0],
        rugged=[1, 1, 0, 0],
        rug_dt_s=[500.0, 20.0, np.nan, np.nan],
        graduated=[0, 0, 1, 0],
        migrate_dt_s=[np.nan, np.nan, 5.0, np.nan],
    )
    out = pf.add_creator_history(df)
    assert out["cr_n_prior"].tolist() == [0, 1, 2, 0]
    # The first launch is rugged at t=500, so the second (t=100) cannot know it.
    assert out["cr_n_rug"].tolist() == [0, 0, 2, 0]
    assert out["cr_n_grad"].tolist() == [0, 0, 0, 0]
    assert out["cr_n_1h"].tolist() == [0, 1, 0, 0]
    assert np.isnan(out["cr_rug_rate"][0]) and out["cr_rug_rate"][2] == 1.0
    assert out["cr_gap_s"].tolist()[1:3] == [100.0, 3900.0] and np.isnan(out["cr_gap_s"][3])


def test_name_history_counts_copies_and_own_copies():
    df = launches(
        creator=["a", "b", "a", "c"],
        t0=[0.0, 10.0, 20.0, 30.0],
        graduated=[1, 0, 0, 0],
        migrate_dt_s=[15.0, np.nan, np.nan, np.nan],
    )
    df["name_key"] = ["dog", "dog", "dog", "cat"]
    out = pf.add_name_history(df)
    assert out["nm_n_prior"].tolist() == [0, 1, 2, 0]
    assert out["nm_n_grad"].tolist() == [0, 0, 1, 0]
    assert out["nm_n_own"].tolist() == [0, 0, 1, 0]
    assert out["nm_n_1h"].tolist() == [0, 1, 2, 0]
    assert out["nm_grad_rate"].tolist()[1:3] == [0.0, 0.5]
    assert np.isnan(out["nm_age_s"][0]) and out["nm_age_s"][2] == 20.0


def test_time_split_is_chronological_and_complete():
    days = pd.to_datetime(["2026-09-30 23:59", "2026-10-01 00:00", "2026-10-02 00:00"], utc=True)
    df = launches(t0=[d.timestamp() for d in days])
    parts = pf.time_split(df, "2026-10-01", "2026-10-02")
    assert [len(parts[k]) for k in ("train", "dev", "test")] == [1, 1, 1]
    assert parts["train"]["time"].max() < parts["dev"]["time"].min() < parts["test"]["time"].min()


def test_event_share_by_counts_missing_as_never():
    share = pf.event_share_by([1.0, 5.0, np.nan, np.nan], [0.5, 1.0, 10.0])
    assert share.tolist() == [0.0, 0.25, 0.5]


def test_top_rate_takes_the_highest_scores():
    y, score = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert pf.top_rate(y, score, 0.2) == 0.5
    assert pf.top_rate(y, score, 0.1) == 1.0


def test_concordance_of_perfect_reversed_and_constant_scores():
    time = np.arange(1.0, 51.0)
    assert pf.concordance(time, -time) == 1.0
    assert pf.concordance(time, time) == 0.0
    assert pf.concordance(time, np.zeros(50)) == 0.5


def test_cluster_bootstrap_is_wider_when_clusters_are_large():
    rng = np.random.default_rng(0)
    values = np.repeat(rng.normal(size=40), 25)  # 40 clusters of 25 identical rows
    alone = pf.cluster_bootstrap(np.mean, np.arange(1000), values, n_boot=300)
    grouped = pf.cluster_bootstrap(np.mean, np.repeat(np.arange(40), 25), values, n_boot=300)
    assert alone[0] == grouped[0] == pytest.approx(values.mean())
    assert grouped[1] <= grouped[0] <= grouped[2]
    assert (grouped[2] - grouped[1]) > 3 * (alone[2] - alone[1])


def test_cluster_bootstrap_skips_failing_resamples():
    def needs_both(y):
        if len(set(y)) < 2:
            raise ValueError("one class")
        return float(np.mean(y))

    point, lo, hi = pf.cluster_bootstrap(needs_both, ["a", "b", "c"], np.array([0, 1, 1]), n_boot=100)
    assert point == pytest.approx(2 / 3) and 0 < lo <= hi < 1
