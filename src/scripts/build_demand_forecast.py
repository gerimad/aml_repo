"""Build the WP17 token tables at all 35 evaluation points (demand_forecast.point_table).

Writes results/wp17/points/{yyyymmdd}.parquet per point (resumable) and
results/wp17/tokens.parquet with all points and a column `part`.

Usage, from the repository root:

    uv run python src/scripts/build_demand_forecast.py
"""

from __future__ import annotations

import pandas as pd

import demand_forecast as dfc
import protocol as pc
import token_data as td

OUT = pc.RESULTS_DIR / "wp17"


def main() -> None:
    legs, tokens = td.load_legs(), td.load_tokens()
    first = dfc.first_ever(legs)
    (OUT / "points").mkdir(parents=True, exist_ok=True)
    tables = []
    for part, points in pc.POINTS.items():
        for t in points:
            path = OUT / "points" / f"{t:%Y%m%d}.parquet"
            if not path.exists():
                dfc.point_table(legs, first, pc.load_task(legs, t), tokens).to_parquet(path)
            tables.append(pd.read_parquet(path).assign(part=part))
            print(f"{part} {t:%d %b}: {len(tables[-1]):,} tokens", flush=True)
    pd.concat(tables, ignore_index=True).to_parquet(OUT / "tokens.parquet")
    print("wrote tokens.parquet", flush=True)


if __name__ == "__main__":
    main()
