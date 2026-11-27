"""Score every method of new_token_audience.py at all 35 evaluation points (work package 19).

Writes results/wp19/tokens_{part}.parquet (one row per point, new token and
method) for dev_a, dev_b and test, and tokens_{part}_without_farms.parquet for
dev_b and test, with WP10's farm groups left out of the audience (as WP15).
One file per point in results/wp19/points/, so a stopped run resumes.

Usage, from the repository root:

    uv run python src/scripts/build_new_token_audience.py [part ...]
"""

from __future__ import annotations

import sys

import pandas as pd

import build_audience as ba
import new_token_audience as nt
import protocol as pc
import token_data as td

OUT = pc.RESULTS_DIR / "wp19"


def main(parts: list[str]) -> None:
    legs = td.load_legs()
    launches = nt.launch_times(legs)  # first trade in the data; every point is at least 14 days after its start
    (OUT / "points").mkdir(parents=True, exist_ok=True)
    for part in parts:
        versions = {"": ()}
        if part != "dev_a":
            versions["_without_farms"] = ba.farm_members(legs, part)
        for suffix, excluded in versions.items():
            tables = []
            for t in pc.POINTS[part]:
                path = OUT / "points" / f"{part}{suffix}_{t:%Y-%m-%d}.parquet"
                if not path.exists():
                    table = nt.point_table(pc.load_task(legs, t), legs, launches, exclude_users=excluded)
                    table.to_parquet(path)
                    print(f"{part}{suffix} {t:%d %b}: {table['token'].nunique() if len(table) else 0} new tokens", flush=True)
                tables.append(pd.read_parquet(path))
            pd.concat(tables, ignore_index=True).to_parquet(OUT / f"tokens_{part}{suffix}.parquet")
            print(f"wrote tokens_{part}{suffix}.parquet", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["dev_a", "dev_b", "test"])
