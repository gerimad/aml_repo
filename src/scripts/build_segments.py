"""Features and recall by popularity weight at all 35 evaluation points (work package 11).

Writes, per point, data/project/wp11/features_{yyyymmdd}.parquet (segments.account_features
of the evaluated accounts) and recall_{yyyymmdd}.parquet (segments.recall_by_weight).
Existing files are kept, so the script can be resumed.

Usage, from the repository root:

    uv run python src/scripts/build_segments.py
"""

from __future__ import annotations

import pandas as pd

import protocol as pc
import segments as sg
import token_data as td

OUT = td.PROJECT_DIR / "wp11"


def main() -> None:
    legs, tokens = td.load_legs(), td.load_tokens()
    pairs = pd.concat([pd.read_parquet(pc.RESULTS_DIR / "wp9" / f"pairs_{part}.parquet") for part in pc.POINTS])
    OUT.mkdir(parents=True, exist_ok=True)
    for part, points in pc.POINTS.items():
        for t in points:
            features_path, recall_path = OUT / f"features_{t:%Y%m%d}.parquet", OUT / f"recall_{t:%Y%m%d}.parquet"
            if features_path.exists() and recall_path.exists():
                continue
            task = pc.load_task(legs, t)
            previous = pairs[pairs["point"] == t - pc.HORIZON]
            sg.account_features(legs, task, tokens, previous).to_parquet(features_path)
            table = sg.recall_by_weight(task)
            table.columns = [str(c) for c in table.columns]
            table.to_parquet(recall_path)
            print(f"{part} {t:%d %b}: {len(table):,} accounts", flush=True)


if __name__ == "__main__":
    main()
