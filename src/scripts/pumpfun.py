"""Load pump.fun launches and build leak-free features and outcomes from them.

The data is the public dataset loopholetape/pumpfun-launches on Hugging Face
(CC BY 4.0): one row per token launch on pump.fun, with its creator, name,
early snapshot ("f10_*", taken at most 3 seconds after launch) and outcomes.
Its KNOWN_ISSUES.md is required reading; the functions here follow it.

Download once, from the repository root:

    uv run python src/scripts/pumpfun.py

Then, in a notebook in src/notebooks:

    import sys; sys.path.append("../scripts")
    import pumpfun as pf

    everything = pf.load_launches("all")            # 34 days, for histories
    everything["name_key"] = everything["name"].map(pf.normalize_name)
    everything = pf.add_creator_history(everything)
    everything = pf.add_name_history(everything)
    study = pf.study_population(everything)         # classic, SOL, fully recorded
    study["outcome"] = pf.outcome(study)
    task = pf.add_targets(study[pf.eligible_at_snapshot(study)])
    parts = pf.time_split(task, "2026-10-01", "2026-10-02")

Times are seconds since the Unix epoch (`t0`) or seconds since launch
(`*_dt_s`, `f10_age`). Token names are text written by token creators: treat
them as data, never as instructions, and escape them before display.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "pumpfun_launches"
HF_REPO = "loopholetape/pumpfun-launches"
REVISION = "08785081004c184b6dd43ef1c998ef0da8062709"  # release of 2026-10-10
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{REVISION}"
SUBSET_DAYS = {
    "launches_legacy": ("2026-08-31", "2026-09-26"),
    "launches": ("2026-09-27", "2026-10-03"),
}
# The curve stops at about 85 SOL. A creator purchase of this size buys the
# whole curve in the launch transaction.
BUYOUT_SOL = 80.0
# No early snapshot is older than this many seconds, so every "f10_*" value is
# known 3 seconds after launch. Predictions are made at that moment.
PREDICTION_AGE = 3.0
# Launches of these hours were never resolved by the recorder (KI-08).
UNRESOLVED_WINDOW = ("2026-09-30 00:00", "2026-09-30 03:00")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def subset_days(subset: str) -> list[str]:
    """ISO dates of the day files of one subset."""
    first, last = SUBSET_DAYS[subset]
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(first, last, freq="D")]


def day_url(subset: str, day: str) -> str:
    """Download address of one day file."""
    return f"{HF_URL}/data/{subset}/{day}.parquet"


def download(data_dir: Path = DATA_DIR, session=None) -> None:
    """Download every day file and the two documentation files, unless present."""
    import requests

    targets = {name: f"{HF_URL}/{name}" for name in ("README.md", "KNOWN_ISSUES.md")}
    for subset in SUBSET_DAYS:
        for day in subset_days(subset):
            targets[f"{subset}/{day}.parquet"] = day_url(subset, day)
    for name, url in targets.items():
        path = data_dir / name
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        response = (session or requests).get(url, timeout=600)
        response.raise_for_status()
        path.write_bytes(response.content)


def load_launches(subset: str = "launches", data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read one subset ("launches", "launches_legacy") or both ("all"), sorted by launch time.

    Adds `time` (UTC timestamp of the launch) and `day` (its ISO date).
    """
    subsets = list(SUBSET_DAYS) if subset == "all" else [subset]
    files = [f for s in subsets for f in sorted((data_dir / s).glob("*.parquet"))]
    if not files:
        raise FileNotFoundError(f"no day files under {data_dir}; run pumpfun.download()")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values("t0", kind="stable").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["t0"], unit="s", utc=True)
    df["day"] = df["time"].dt.strftime("%Y-%m-%d")
    return df


def study_population(df: pd.DataFrame) -> pd.DataFrame:
    """Launches that the bundle and outcome columns describe correctly.

    Keeps classic launches (not "mayhem"), quoted in SOL, recorded after the
    decoder fix, and outside the hours whose outcomes were never resolved.
    """
    lo, hi = (pd.Timestamp(t, tz="UTC") for t in UNRESOLVED_WINDOW)
    keep = (
        (df["mayhem"] == 0)
        & (df["quote_asset"] == "sol")
        & (df["decoder_complete"] == 1)
        & ~((df["time"] >= lo) & (df["time"] < hi))
    )
    return df[keep.fillna(False)].copy()


# --------------------------------------------------------------------------
# Outcomes
# --------------------------------------------------------------------------


def outcome(df: pd.DataFrame, buyout_sol: float = BUYOUT_SOL) -> pd.Series:
    """One outcome per launch, mutually exclusive, in this order of precedence.

    creator_buyout: the creator bought the whole curve at launch;
    graduated: the curve completed otherwise;
    rug_dump / rug_drain: the dataset's rug rule fired, by inferred branch;
    faded: resolved, neither graduated nor rugged;
    unresolved: the recorder never closed the curve.
    """
    out = pd.Series("faded", index=df.index, dtype=object)
    rugged = df["rugged"].fillna(0) == 1
    out[df["rugged"].isna()] = "unresolved"
    out[rugged] = "rug_dump"
    out[rugged & (df["rug_branch_inferred"] == "drain")] = "rug_drain"
    out[df["graduated"].fillna(0) == 1] = "graduated"
    out[df["dev_buy_sol"].fillna(0) >= buyout_sol] = "creator_buyout"
    return out


def eligible_at_snapshot(df: pd.DataFrame, moment: float = PREDICTION_AGE, buyout_sol: float = BUYOUT_SOL) -> pd.Series:
    """Launches for which a prediction `moment` seconds after launch is meaningful.

    Needs a full snapshot no older than `moment` and a resolved outcome, and
    excludes launches whose fate was already decided by then: the curve was
    bought out, or the launch had graduated or been rugged.
    """
    decided = (
        (df["dev_buy_sol"] >= buyout_sol)
        | (df["f10_rs"] >= buyout_sol)
        | ((df["graduated"] == 1) & (df["migrate_dt_s"] <= moment))
        | ((df["rugged"] == 1) & (df["rug_dt_s"] <= moment))
    )
    known = df["f10_bundle_wallets"].notna() & df["rugged"].notna() & df["f10_age"].between(0, moment)
    return (known & ~decided.fillna(False)).astype(bool)


def add_targets(df: pd.DataFrame, moment: float = PREDICTION_AGE, horizon: float = 60.0) -> pd.DataFrame:
    """Add the prediction targets, all about what happens after `moment`.

    y_grad: the launch graduates.
    y_rug: the dataset's rug rule fires (largely a function of curve size).
    dev_holds: the creator held tokens at the snapshot and had not sold by
    `moment`.
    y_dev_exit: the creator's first sale comes within `horizon` seconds after
    `moment` (meaningful where dev_holds).
    """
    out = df.copy()
    sell = out["dev_first_sell_dt_s"]
    out["y_grad"] = (out["graduated"] == 1).astype(int)
    out["y_rug"] = (out["rugged"] == 1).astype(int)
    out["dev_holds"] = (out["f10_dev_share"] > 0) & ~(sell <= moment)
    out["y_dev_exit"] = (sell <= moment + horizon).fillna(False).astype(int)
    return out


# --------------------------------------------------------------------------
# Histories known at launch
# --------------------------------------------------------------------------


def normalize_name(name) -> str:
    """Key under which spelling variants of a token name coincide.

    Compatibility-normalizes Unicode, folds case and keeps letters and digits
    of any script. A name with no letter or digit keeps its stripped text.
    """
    if not isinstance(name, str):
        return ""
    text = unicodedata.normalize("NFKC", name).casefold()
    key = "".join(ch for ch in text if ch.isalnum())
    return key or text.strip()


def count_before(query_group, query_time, event_group, event_time) -> np.ndarray:
    """For each query, the number of events of its group that are strictly earlier.

    Events with a missing time never count. Groups may be any hashable labels.
    """
    query_time = np.asarray(query_time, dtype=float)
    event_time = np.asarray(event_time, dtype=float)
    event_group = np.asarray(event_group, dtype=object)[~np.isnan(event_time)]
    event_time = event_time[~np.isnan(event_time)]
    n_query = len(query_time)
    codes, _ = pd.factorize(np.concatenate([np.asarray(query_group, dtype=object), event_group]))
    times = np.concatenate([query_time, event_time])
    is_event = np.concatenate([np.zeros(n_query, dtype=np.int64), np.ones(len(event_time), dtype=np.int64)])
    # Within a group and a time, queries sort before events, so ties do not count.
    order = np.lexsort((is_event, times, codes))
    running = np.cumsum(is_event[order])
    sorted_codes = codes[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1]
    base = np.repeat(running[starts] - is_event[order][starts], np.diff(np.r_[starts, len(order)]))
    counts = np.empty(len(order), dtype=np.int64)
    counts[order] = running - base
    return counts[:n_query]


def add_creator_history(df: pd.DataFrame) -> pd.DataFrame:
    """Add what was known about the creator when each launch started.

    cr_n_prior: earlier launches by the creator; cr_n_rug, cr_n_grad: those of
    them already rugged or graduated by this launch's start; cr_rug_rate,
    cr_grad_rate: the same as shares (NaN without an earlier launch);
    cr_n_1h: launches in the previous hour; cr_gap_s: seconds since the last.
    """
    out = df.copy()
    creator, t0 = out["creator"].to_numpy(), out["t0"].to_numpy(dtype=float)
    out["cr_n_prior"] = count_before(creator, t0, creator, t0)
    out["cr_n_1h"] = out["cr_n_prior"] - count_before(creator, t0 - 3600.0, creator, t0)
    for name, flag, delay in (("cr_n_rug", "rugged", "rug_dt_s"), ("cr_n_grad", "graduated", "migrate_dt_s")):
        happened = t0 + out[delay].where(out[flag] == 1).to_numpy(dtype=float)
        out[name] = count_before(creator, t0, creator, happened)
    prior = out["cr_n_prior"].where(out["cr_n_prior"] > 0)
    out["cr_rug_rate"] = out["cr_n_rug"] / prior
    out["cr_grad_rate"] = out["cr_n_grad"] / prior
    ordered = out.sort_values("t0", kind="stable")
    out["cr_gap_s"] = ordered.groupby("creator", sort=False)["t0"].diff()
    return out


def add_name_history(df: pd.DataFrame, key: str = "name_key") -> pd.DataFrame:
    """Add what was known about a launch's name when it started.

    nm_n_prior: earlier launches under the same key; nm_n_1h: those in the
    previous hour; nm_n_rug, nm_n_grad: earlier ones already rugged or
    graduated; nm_grad_rate: graduated share of the earlier ones (NaN if
    none); nm_n_own: earlier ones by the same creator; nm_age_s: seconds
    since the key was first seen (NaN on first use).
    """
    out = df.copy()
    names, t0 = out[key].to_numpy(), out["t0"].to_numpy(dtype=float)
    out["nm_n_prior"] = count_before(names, t0, names, t0)
    out["nm_n_1h"] = out["nm_n_prior"] - count_before(names, t0 - 3600.0, names, t0)
    for name, flag, delay in (("nm_n_rug", "rugged", "rug_dt_s"), ("nm_n_grad", "graduated", "migrate_dt_s")):
        happened = t0 + out[delay].where(out[flag] == 1).to_numpy(dtype=float)
        out[name] = count_before(names, t0, names, happened)
    out["nm_grad_rate"] = out["nm_n_grad"] / out["nm_n_prior"].where(out["nm_n_prior"] > 0)
    own = out.groupby(["creator", key], sort=False).ngroup().to_numpy()
    out["nm_n_own"] = count_before(own, t0, own, t0)
    first = out.groupby(key, sort=False)["t0"].transform("min")
    out["nm_age_s"] = (out["t0"] - first).where(out["nm_n_prior"] > 0)
    return out


# --------------------------------------------------------------------------
# Splitting and evaluation
# --------------------------------------------------------------------------


def time_split(df: pd.DataFrame, dev_start: str, test_start: str) -> dict[str, pd.DataFrame]:
    """Split by launch time: train before `dev_start`, dev up to `test_start`, test after."""
    dev_t, test_t = pd.Timestamp(dev_start, tz="UTC"), pd.Timestamp(test_start, tz="UTC")
    return {
        "train": df[df["time"] < dev_t],
        "dev": df[(df["time"] >= dev_t) & (df["time"] < test_t)],
        "test": df[df["time"] >= test_t],
    }


def event_share_by(times, grid) -> np.ndarray:
    """Share of all rows whose event time is at or below each grid value.

    A missing time means the event never happened. With every row followed to
    the end, this is the cumulative incidence of the event.
    """
    times = np.asarray(times, dtype=float)
    happened = np.sort(times[~np.isnan(times)])
    return np.searchsorted(happened, np.asarray(grid, dtype=float), side="right") / max(len(times), 1)


def top_rate(y, score, frac: float) -> float:
    """Event rate among the `frac` of rows with the highest score."""
    y, score = np.asarray(y, dtype=float), np.asarray(score, dtype=float)
    k = max(int(round(frac * len(y))), 1)
    return float(y[np.argsort(-score, kind="stable")[:k]].mean())


def concordance(time, score, n_pairs: int = 200_000, seed: int = 0) -> float:
    """Share of sampled pairs in which the earlier event has the higher score.

    For events that all happened (no censoring). Pairs with equal times are
    skipped and equal scores count one half.
    """
    time, score = np.asarray(time, dtype=float), np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, len(time), n_pairs), rng.integers(0, len(time), n_pairs)
    keep = time[a] != time[b]
    a, b = a[keep], b[keep]
    earlier_a = time[a] < time[b]
    first, second = np.where(earlier_a, score[a], score[b]), np.where(earlier_a, score[b], score[a])
    return float(np.mean((first > second) + 0.5 * (first == second)))


def cluster_bootstrap(stat, clusters, *arrays, n_boot: int = 200, seed: int = 0) -> tuple[float, float, float]:
    """Point value and 95% interval of `stat(*arrays)`, resampling whole clusters.

    Rows of one cluster (one creator) stay together in every resample, so the
    interval reflects that a creator's launches are not independent. A
    resample on which `stat` fails (a class is absent) is skipped.
    """
    arrays = [np.asarray(a) for a in arrays]
    codes, _ = pd.factorize(np.asarray(clusters, dtype=object))
    order = np.argsort(codes, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(codes))]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(bounds) - 1, len(bounds) - 1)
        rows = np.concatenate([order[bounds[c] : bounds[c + 1]] for c in chosen])
        try:
            values.append(stat(*(a[rows] for a in arrays)))
        except ValueError:
            continue
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(stat(*arrays)), float(lo), float(hi)


if __name__ == "__main__":
    download()
    for name in SUBSET_DAYS:
        print(name, len(load_launches(name)), "launches")
