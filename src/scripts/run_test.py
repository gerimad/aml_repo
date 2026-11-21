"""Score the frozen models on the 14 test points, once (work package 7).

The test period (13 August to 10 September 2026) has never been looked at
with a model. This script is the only code that does so, and it refuses to
run unless:

  1. the working tree has no uncommitted changes to tracked files,
  2. HEAD is the commit tagged `test-freeze`, and
  3. results/test/ does not exist yet (the test is scored once).

It reads configs/frozen.json, which names one primary model (the one the
success criterion is about) and the secondary models, scores all of them
and the fixed baseline at every test point, and writes results/test/:

    per_point.csv     one row per (model, test point)
    compare.csv       the paired comparison with the baseline
    by_group.csv      recall by the size of the account's history
    run.json          commit, start and end time, and the success verdict

Usage, from the repository root:

    uv run python src/scripts/run_test.py

configs/frozen.json has the form:

    {
      "primary": "fixed blend",
      "models": {
        "fixed blend": {"kind": "fixed_blend", "settings": {"k": 50, "shrink": 10, "weight": 0.5}},
        "ranker": {"kind": "ranker", "model": "lambdamart", "file": "data/project/models/lambdamart.txt",
                   "features": [...], "settings": {...first-stage settings...}}
      }
    }
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import protocol as pc
import ranker as rk
import stats
import token_data as td
import tuning

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "configs" / "frozen.json"
OUT = pc.RESULTS_DIR / "test"
BASELINE = "popularity (baseline)"


def git(*args: str) -> str:
    """Run a git command in the repository and return its output."""
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def check_frozen(out_dir: Path = OUT) -> str:
    """Stop unless the three conditions of the module docstring hold. Returns the commit hash."""
    changed = [line for line in git("status", "--porcelain").splitlines() if not line.startswith("??")]
    if changed:
        raise SystemExit("Uncommitted changes to tracked files:\n" + "\n".join(changed))
    head = git("rev-parse", "HEAD")
    try:
        tagged = git("rev-parse", "test-freeze^{commit}")
    except subprocess.CalledProcessError:
        raise SystemExit("No tag `test-freeze`. Freeze the models, commit, and tag the commit first.")
    if head != tagged:
        raise SystemExit(f"HEAD ({head[:8]}) is not the commit tagged test-freeze ({tagged[:8]}).")
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists: the test period has been scored. It is scored once.")
    return head


def scorer_factory(spec: dict, legs: pd.DataFrame, tokens: pd.DataFrame):
    """A protocol scorer factory for a model described in frozen.json."""
    if spec["kind"] == "baseline":
        return pc.popularity_scorer
    if spec["kind"] == "fixed_blend":
        return lambda task: tuning.fixed_blend_scorer(task, spec["settings"])
    if spec["kind"] == "ranker":
        model = rk.load_model(REPO / spec["file"], spec["model"])

        def factory(task):
            rows = rk.point_rows(legs, tokens, task, spec["settings"])
            return rk.ranker_scorer(model, spec["features"], task, rows)

        return factory
    raise ValueError(f"unknown model kind {spec['kind']!r}")


def main() -> None:
    commit = check_frozen()
    with open(FROZEN) as f:
        frozen = json.load(f)
    started = datetime.now(timezone.utc)
    print(f"Scoring the test period from commit {commit[:8]}. Primary model: {frozen['primary']}.")

    legs, tokens = td.load_legs(), td.load_tokens()
    models = {BASELINE: {"kind": "baseline"}, **frozen["models"]}
    results = {}
    for name, spec in models.items():
        results[name] = pc.run(name, scorer_factory(spec, legs, tokens), legs, pc.POINTS["test"])

    OUT.mkdir(parents=True)
    pd.concat([r.per_point for r in results.values()]).to_csv(OUT / "per_point.csv", index=False)
    compare = stats.compare({n: r.per_point for n, r in results.items()}, baseline=BASELINE)
    compare.to_csv(OUT / "compare.csv", index=False)
    by_group = pd.concat(
        [r.per_user.groupby("group")["recall"].mean().rename(n) for n, r in results.items()], axis=1
    )
    by_group.to_csv(OUT / "by_group.csv")

    verdict = stats.success(results[frozen["primary"]].per_point, results[BASELINE].per_point)
    with open(OUT / "run.json", "w") as f:
        json.dump(
            {
                "commit": commit,
                "started": started.isoformat(),
                "finished": datetime.now(timezone.utc).isoformat(),
                "primary": frozen["primary"],
                "mean_difference": verdict["mean_difference"],
                "interval": verdict["interval"],
                "success": verdict["success"],
            },
            f,
            indent=2,
        )
    print(compare.to_string(index=False))
    low, high = verdict["interval"]
    print(
        f"\nPrimary model {frozen['primary']!r}: mean difference {verdict['mean_difference']:+.4f}, "
        f"95% interval [{low:+.4f}, {high:+.4f}] -> success criterion {'MET' if verdict['success'] else 'NOT met'}."
    )


if __name__ == "__main__":
    main()
