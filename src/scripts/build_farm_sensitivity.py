"""Per-account recall of the project's models at the Dev B and test points (WP10 sensitivity, section 10.3).

The project's result tables keep only means per point. The sensitivity
analysis removes farm members from the evaluated accounts, so it needs each
account's recall. This script scores the baseline, the fixed blend and the
LambdaMART ranker (configs/frozen.json) at the 11 Dev B and 14 test points with
protocol.run and writes one file per (model, part):

    data/project/wp10/per_user_{model}_{part}.parquet

A file that exists is not rebuilt, so the script can be resumed. The scores
are the same as those already reported: same models, same points, same code.

Usage, from the repository root:

    uv run python src/scripts/build_farm_sensitivity.py
"""

from __future__ import annotations

import json

import protocol as pc
import run_test
import token_data as td

OUT = td.PROJECT_DIR / "wp10"
MODELS = {
    "baseline": {"kind": "baseline"},
    "fixed_blend": "fixed blend",
    "ranker": "ranker: lambdamart",
}


def main() -> None:
    with open(run_test.FROZEN) as f:
        frozen = json.load(f)
    legs, tokens = td.load_legs(), td.load_tokens()
    OUT.mkdir(parents=True, exist_ok=True)
    for part in ("dev_b", "test"):
        for name, spec in MODELS.items():
            path = OUT / f"per_user_{name}_{part}.parquet"
            if path.exists():
                print(f"{path.name} exists, skipped")
                continue
            spec = frozen["models"][spec] if isinstance(spec, str) else spec
            result = pc.run(name, run_test.scorer_factory(spec, legs, tokens), legs, pc.POINTS[part])
            result.per_user.to_parquet(path)
            print(f"wrote {path.name}: {len(result.per_user):,} rows")


if __name__ == "__main__":
    main()
