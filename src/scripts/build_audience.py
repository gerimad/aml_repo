"""Score every method of audience.py at all 35 evaluation points (work package 15).

Writes results/wp15/tokens_{part}.parquet (one row per point, token and
method) for dev_a, dev_b and test, and tokens_{part}_without_farms.parquet for
dev_b and test, with the WP10 farm groups (delta 5 minutes, m 5; built on Dev A
for Dev B and on Dev A + Dev B for test) left out of the audience.

Usage, from the repository root:

    uv run python src/scripts/build_audience.py
"""

from __future__ import annotations

import pandas as pd

import audience as au
import farms as fm
import protocol as pc
import token_data as td

OUT = pc.RESULTS_DIR / "wp15"
FARM_SETTING = {"delta_minutes": 5, "min_shared": 5}
BUILD_WINDOWS = {"dev_b": (pc.POINTS["dev_a"][0], pc.DEV_B_START), "test": (pc.POINTS["dev_a"][0], pc.TEST_START)}


def farm_members(legs: pd.DataFrame, part: str) -> pd.Index:
    """The accounts in WP10's groups built before the part."""
    start, end = BUILD_WINDOWS[part]
    first = fm.first_trades(legs, start, end)
    return fm.groups(fm.coaction_edges(first, FARM_SETTING["delta_minutes"], FARM_SETTING["min_shared"])).index


def main() -> None:
    legs = td.load_legs()
    OUT.mkdir(parents=True, exist_ok=True)
    for part in ("dev_a", "dev_b", "test"):
        versions = {"": ()}
        if part != "dev_a":
            versions["_without_farms"] = farm_members(legs, part)
        for suffix, excluded in versions.items():
            path = OUT / f"tokens_{part}{suffix}.parquet"
            if path.exists():
                print(f"{path.name} exists, skipped", flush=True)
                continue
            tables = []
            for t in pc.POINTS[part]:
                tables.append(au.point_table(pc.load_task(legs, t), exclude_users=excluded))
                print(f"{part}{suffix} {t:%d %b}: {tables[-1]['token'].nunique()} tokens", flush=True)
            pd.concat(tables, ignore_index=True).to_parquet(path)
            print(f"wrote {path.name}", flush=True)


if __name__ == "__main__":
    main()
