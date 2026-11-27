"""The safety trade-off of the fixed blend's lists at the Dev B and test points (work package 16).

For each point: the blend's first 50 tokens per evaluated account, each listed
token's collapse risk at t (WP13 baseline model, fitted on WP13's Dev A windows)
and its collapse label over the 72 hours after t; then safer.tradeoff for every
lambda. Writes results/wp16/tradeoff.csv and results/wp16/list_tokens.parquet.

Usage, from the repository root:

    uv run python src/scripts/build_safer.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import build_scam_push as bsp
import protocol as pc
import safer as sf
import scam_push as sp
import token_data as td

OUT = pc.RESULTS_DIR / "wp16"


def risk_model():
    """The WP13 baseline model fitted on WP13's Dev A windows, with the medians used to fill missing values."""
    windows = pd.read_parquet(pc.RESULTS_DIR / "wp13" / "tokens.parquet")
    train = windows[windows["part"] == "dev_a"]
    medians = train[sp.BASELINE_FEATURES].median()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[sp.BASELINE_FEATURES].fillna(medians), train["collapsed"])
    return model, medians


def main() -> None:
    legs, tokens = td.load_legs(), td.load_tokens()
    first = legs.groupby(["user", "token"], observed=True)["timestamp"].min().rename("first_time").reset_index()
    model, medians = risk_model()
    address = tokens.set_index("token_id")["address"]

    lists, token_rows = {}, []
    for part in ("dev_b", "test"):
        for t in pc.POINTS[part]:
            task = pc.load_task(legs, t)
            columns, scores = sf.candidate_lists(task)
            listed = task.tokens[np.unique(columns)]
            features = sp.baseline_features(legs, first, listed, t)
            features["risk"] = model.predict_proba(features[sp.BASELINE_FEATURES].fillna(medians))[:, 1]
            token_rows.append(features.reset_index().assign(point=t, part=part, decision=t))
            lists[t] = (part, columns, scores)
            print(f"{part} {t:%d %b}: {len(listed):,} tokens on the candidate lists", flush=True)
    listed = pd.concat(token_rows, ignore_index=True)
    listed["address"] = listed["token"].map(address)

    listed["late_trades"] = bsp.late_trades(listed)
    labels = []
    for t, at in listed.groupby("point", sort=False):
        # Only this point's label window is read, to keep memory low (8 GB machine).
        swaps = bsp.token_swaps(set(at["address"]), t, t + pd.Timedelta(hours=sp.LABEL_HOURS))
        by_token = dict(tuple(swaps.groupby("token")))
        empty = swaps.iloc[:0]
        labels.append(pd.Series([sp.collapse_label(by_token.get(a, empty), t, late) for a, late in zip(at["address"], at["late_trades"])], index=at.index))
    listed["collapsed"] = pd.concat(labels)
    print(f"labelled {len(listed):,} listed (token, point) pairs: {listed['collapsed'].mean():.2%} collapse", flush=True)

    tables = []
    for t, (part, columns, scores) in lists.items():
        task = pc.load_task(legs, t)
        at = listed[listed["point"] == t].set_index("token")
        position = pd.Series(np.arange(len(task.tokens)), index=task.tokens)
        risk = np.zeros(len(task.tokens))
        collapsed = np.zeros(len(task.tokens), dtype=bool)
        risk[position[at.index].to_numpy()] = at["risk"].to_numpy()
        collapsed[position[at.index].to_numpy()] = at["collapsed"].to_numpy()
        tables.append(sf.tradeoff(task, columns, scores, risk, collapsed).assign(part=part))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(OUT / "tradeoff.csv", index=False)
    listed.drop(columns=["address"]).to_parquet(OUT / "list_tokens.parquet")
    print("wrote tradeoff.csv and list_tokens.parquet", flush=True)


if __name__ == "__main__":
    main()
