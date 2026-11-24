"""Token windows with features and collapse labels at all 35 evaluation points (work package 13).

Writes results/wp13/tokens.parquet: one row per scored (window, token) with the
features of scam_push.window_features, the token's address, the label
`collapsed` and the part (dev_a, dev_b, test).

Usage, from the repository root:

    uv run python src/scripts/build_scam_push.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

import farms as fm
import prices as px
import protocol as pc
import scam_push as sp
import token_data as td

OUT = pc.RESULTS_DIR / "wp13"


def token_swaps(addresses: set[str], start=None, end=None) -> pd.DataFrame:
    """Priced swaps of the given token addresses (only the columns the label needs), optionally in [start, end)."""
    dataset = pads.dataset(sorted(px.PRICES_DIR.glob("prices_*.parquet")), format="parquet")
    condition = pads.field("token").isin(sorted(addresses))
    if start is not None:
        condition = condition & (pads.field("timestamp") >= start) & (pads.field("timestamp") < end)
    table = dataset.to_table(columns=["timestamp", "pool", "token", "price_eth", "volume_eth"], filter=condition)
    return table.to_pandas()


def windows_with_features(legs: pd.DataFrame, tokens: pd.DataFrame) -> pd.DataFrame:
    """The scored token windows and their features at every point (no label yet)."""
    first = legs.groupby(["user", "token"], observed=True)["timestamp"].min().rename("first_time").reset_index()
    rows = []
    for part, points in pc.POINTS.items():
        pairs = pd.read_parquet(pc.RESULTS_DIR / "wp9" / f"pairs_{part}.parquet")
        for t in points:
            task = pc.load_task(legs, t)
            farm_members = set(fm.groups(fm.coaction_edges(fm.first_trades(legs, t - pd.Timedelta(days=14), t), 5, 5)).index)
            at_point = pairs[pairs["point"] == t]
            for s in (t, t + sp.WINDOW):
                table = sp.window_features(at_point, legs, first, task, s, farm_members)
                rows.append(table.assign(part=part))
                print(f"{part} window {s:%d %b %H:%M}: {len(table)} tokens", flush=True)
    windows = pd.concat(rows, ignore_index=True)
    windows["address"] = windows["token"].map(tokens.set_index("token_id")["address"])
    return windows


def late_trades(windows: pd.DataFrame) -> list[int]:
    """Each window's token trades (any pool, project and extension legs) in the last 24 hours of its label window."""
    legs = td.load_all_legs()
    legs = legs[legs["token"].isin(set(windows["token"]))][["token", "timestamp"]]
    by_token = {tok: np.sort(part["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[ns]")) for tok, part in legs.groupby("token")}
    out = []
    for tok, d in zip(windows["token"], windows["decision"]):
        times = by_token.get(tok, np.array([], dtype="datetime64[ns]"))
        end = (d + pd.Timedelta(hours=sp.LABEL_HOURS)).tz_convert("UTC").tz_localize(None).to_datetime64().astype("datetime64[ns]")
        start = end - np.timedelta64(24, "h")
        out.append(int(np.searchsorted(times, end) - np.searchsorted(times, start)))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features_path = OUT / "token_features.parquet"
    if features_path.exists():
        windows = pd.read_parquet(features_path)
    else:
        windows = windows_with_features(td.load_legs(), td.load_tokens())
        windows.to_parquet(features_path)

    windows["late_trades"] = late_trades(windows)
    swaps = token_swaps(set(windows["address"]))
    print(f"{len(swaps):,} priced swaps of {windows['address'].nunique():,} tokens", flush=True)
    by_token = dict(tuple(swaps.groupby("token")))
    empty = swaps.iloc[:0]
    windows["collapsed"] = [sp.collapse_label(by_token.get(a, empty), d, late)
                            for a, d, late in zip(windows["address"], windows["decision"], windows["late_trades"])]  # fmt: skip
    windows.to_parquet(OUT / "tokens.parquet")
    print(f"wrote tokens.parquet: {len(windows):,} token windows, {windows['collapsed'].mean():.1%} collapsed", flush=True)


if __name__ == "__main__":
    main()
