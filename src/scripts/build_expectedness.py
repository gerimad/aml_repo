"""Pair tables of WP9 for every evaluation point (expectedness.pair_table), one Parquet file per part.

Usage, from src/notebooks:

    uv run python -u ../scripts/build_expectedness.py

Writes results/wp9/pairs_dev_a.parquet, pairs_dev_b.parquet and
pairs_test.parquet (about 300k rows each; local, not committed). Each point
is also kept in results/wp9/points/ so that a stopped run resumes there.
"""

from __future__ import annotations

import time

import pandas as pd

import expectedness as ex
import protocol as pc
import token_data as td

OUT_DIR = pc.RESULTS_DIR / "wp9"


def main() -> None:
    legs = td.load_legs()
    tokens = td.load_tokens()
    points_dir = OUT_DIR / "points"
    points_dir.mkdir(parents=True, exist_ok=True)
    for part, points in pc.POINTS.items():
        for t in points:
            path = points_dir / f"pairs_{t:%Y%m%d}.parquet"
            if path.exists():
                continue
            start = time.time()
            table = ex.pair_table(legs, pc.load_task(legs, t), tokens)
            table.to_parquet(path)
            print(f"  {part} {t:%d %b}: {len(table):,} pairs ({table['cold'].mean():.0%} cold), {time.time() - start:.0f} s", flush=True)
        files = sorted(points_dir.glob("pairs_*.parquet"))
        wanted = {f"pairs_{t:%Y%m%d}.parquet" for t in points}
        table = pd.concat([pd.read_parquet(f) for f in files if f.name in wanted], ignore_index=True)
        table.to_parquet(OUT_DIR / f"pairs_{part}.parquet")
        print(f"Written pairs_{part}.parquet: {len(table):,} pairs from {len(wanted)} points", flush=True)


if __name__ == "__main__":
    main()
