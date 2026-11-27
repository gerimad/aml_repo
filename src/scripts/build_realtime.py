"""Real-time lists on the test (work package 18, implementation_plan_applications.md section 18).

The frozen time-weighted blend and the popularity baseline, rebuilt every hour
inside the two target days (the protocol of notebook 13, cold_start.py) and
built once (48 hours, the main protocol), at the 14 test points. Nothing is
tuned: the blend's settings are the frozen ones of configs/frozen.json.

Writes one file per point and refresh interval to results/wp18/points/, so a
stopped run resumes, and results/wp18/per_point.csv at the end.

Usage, from the repository root:

    uv run python src/scripts/build_realtime.py [part]     (default: test)
"""

from __future__ import annotations

import json
import sys

import pandas as pd

import cold_start as cs
import protocol as pc
import token_data as td
import tuning

OUT = pc.RESULTS_DIR / "wp18"
REFRESH_HOURS = [48, 1]


def frozen_time_weighted() -> dict:
    """The settings of the frozen time-weighted blend."""
    frozen = json.loads((pc.RESULTS_DIR.parent / "configs" / "frozen.json").read_text())["models"]["time-weighted blend"]["settings"]
    return {k: frozen[k] for k in ("k", "shrink", "weight", "half_life", "history_days")}


def factories() -> dict:
    settings = frozen_time_weighted()
    return {
        "baseline": cs.popularity_factory,
        "time-weighted blend": cs.shared_across_models(lambda state, tokens: tuning.fixed_blend_scorer(state, settings)),
    }


def main(part: str) -> None:
    legs = td.load_legs()
    tokens = td.load_tokens()
    (OUT / "points").mkdir(parents=True, exist_ok=True)
    rows = []
    for t in pc.POINTS[part]:
        task = None
        for hours in REFRESH_HOURS:
            path = OUT / "points" / f"{part}_every_{hours}h_{t:%Y-%m-%d}.csv"
            if not path.exists():
                task = task or pc.load_task(legs, t)
                results = cs.refreshed_evaluate(factories(), legs, task, tokens, hours)
                pd.DataFrame([cs.summarize_refreshed(name, hours, task, per_user) for name, per_user in results.items()]).to_csv(path, index=False)
                print(f"{part} {t:%d %b}, every {hours} h: done", flush=True)
            rows.append(pd.read_csv(path, parse_dates=["point"]))
    pd.concat(rows, ignore_index=True).to_csv(OUT / f"per_point_{part}.csv", index=False)
    print("wrote", OUT / f"per_point_{part}.csv", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
