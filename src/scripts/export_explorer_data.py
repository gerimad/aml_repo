"""Data for the interactive explorer of the recommender and the strategy signal.

Usage, from the repository root:

    uv run python src/scripts/export_explorer_data.py <output folder>

Writes three JSON files, all from the frozen models of configs/frozen.json
and the saved results (no model is chosen or changed here):

    summary.json      per-point recall of every model on the test, and the
                      error analysis of notebook 15
    accounts.json     sample accounts at three test points: their recent
                      tokens, what they picked up, and the lists of the
                      baseline, the fixed blend and the primary ranker
    strategy.json     the demand signal of trading_strategy_backtest.md at
                      six test points: for tail tokens, predicted new
                      accounts (lists containing the token), accounts that
                      bought it in the last 24 hours, and accounts that
                      bought it for the first time in the next 48 hours

Accounts appear only as shortened addresses. Token names are their on-chain
symbols where readable, else shortened addresses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import dex_swaps as ds
import pool_recsys as pr
import protocol as pc
import run_test
import token_data as td

ACCOUNT_POINTS = ["2026-08-25", "2026-08-29", "2026-09-06"]
STRATEGY_POINTS = ["2026-08-13", "2026-08-19", "2026-08-25", "2026-08-31", "2026-09-04", "2026-09-08"]
PER_GROUP = {"1": 8, "2-4": 6, "5+": 6}  # sample accounts per history group and point
TAIL_KEEP = 250  # tail tokens kept per point and ranking (by predicted, by visible demand)


def short(address: str) -> str:
    """0x1234…abcd: enough to tell addresses apart on a page."""
    return f"{address[:6]}…{address[-4:]}"


class Names:
    """Readable names of tokens by id: the on-chain symbol where readable, else the shortened address."""

    def __init__(self, tokens: pd.DataFrame):
        self.address = tokens.set_index("token_id")["address"]
        self.symbol: dict[str, str] = {}

    def load(self, token_ids) -> None:
        addresses = [self.address[i] for i in set(int(t) for t in token_ids)]
        meta = ds.load_token_metadata(addresses, data_dir=td.PROJECT_DIR)
        self.symbol.update({a: s for a, s in zip(meta["address"], meta["symbol"]) if isinstance(s, str)})

    def __call__(self, token_id) -> str:
        address = self.address[int(token_id)]
        return self.symbol.get(address, short(address))


def fixed_blend_parts(task: pc.Task, settings: dict):
    """The untimed fixed blend of the frozen config, with its parts kept so that each list can be explained."""
    window = task.history(pc.REFERENCE_DAYS)
    popularity = pr.Popularity().fit(window)
    neighbours = pr.ItemKNN(settings["k"], settings["shrink"]).fit(window)
    blend = pr.Blend(popularity, neighbours, settings["weight"])
    return window, neighbours, blend


def summary(out: Path) -> None:
    per_point = pd.read_csv(pc.RESULTS_DIR / "test" / "per_point.csv", parse_dates=["point"])
    by_age = pd.read_csv(pc.RESULTS_DIR / "wp7" / "test_outcome_by_token_age.csv", index_col=0)
    by_history = pd.read_csv(pc.RESULTS_DIR / "wp7" / "test_recall_by_history.csv")
    data = {
        "points": [t.strftime("%Y-%m-%d") for t in sorted(per_point["point"].unique())],
        "models": {
            model: part.sort_values("point")[["recall", "recall_no_head", "coverage"]].round(4).to_dict("list")
            for model, part in per_point.groupby("model")
        },
        "outcome_by_age": {
            age: {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in row.items()}
            for age, row in by_age.iterrows()
        },
        "recall_by_history": by_history.round(4).to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(data))


def accounts(out: Path, legs, tokens, names: Names, frozen: dict) -> None:
    rng = np.random.default_rng(0)
    blend_settings = frozen["models"]["fixed blend"]["settings"]
    ranker_factory = run_test.scorer_factory(frozen["models"][frozen["primary"]], legs, tokens)
    points = []
    for day in ACCOUNT_POINTS:
        task = pc.load_task(legs, pd.Timestamp(day, tz="UTC"))
        groups = pc.history_group(task.window_size[task.eval_rows])
        chosen = np.concatenate([
            rng.choice(task.eval_rows[groups == g], size=n, replace=False) for g, n in PER_GROUP.items()
        ])  # fmt: skip
        window, neighbours, blend = fixed_blend_parts(task, blend_settings)
        lists = {
            "baseline": pr.top_k(pc.popularity_scorer(task)(chosen), pc.K, exclude=task.past[chosen]),
            "blend": pr.top_k(blend.scores(window[chosen]), pc.K, exclude=task.past[chosen]),
            "ranker": pr.top_k(ranker_factory(task)(chosen), pc.K, exclude=task.past[chosen]),
        }
        popularity = np.asarray(window.sum(axis=0)).ravel()
        ages = task.past_age_days
        target = legs[(legs["timestamp"] >= task.t) & (legs["timestamp"] < task.t + pc.HORIZON)]

        names.load(task.tokens[np.unique(np.concatenate([v.ravel() for v in lists.values()]))])
        rows = []
        for r, row in enumerate(chosen):
            history_cols = window[row].indices
            history = sorted(
                ({"token": int(task.tokens[c]), "days_ago": round(float(ages[row, c]), 1)} for c in history_cols),
                key=lambda item: item["days_ago"],
            )
            new_cols = set(task.new[row].indices)
            user = task.users[row]
            picked = target[target["user"] == user]["token"].unique()
            past_tokens = set(task.tokens[task.past[row].indices])
            picked = [int(t) for t in picked if t not in past_tokens]
            account_lists = {}
            for model, table in lists.items():
                items = []
                for c in table[r]:
                    item = {"token": int(task.tokens[c]), "hit": bool(c in new_cols), "users": int(popularity[c])}
                    if model == "blend" and len(history_cols):
                        similarities = np.asarray(neighbours.similarity[history_cols, c].todense()).ravel()
                        if similarities.max() > 0:
                            item["because"] = int(task.tokens[history_cols[similarities.argmax()]])
                    items.append(item)
                account_lists[model] = items
            names.load([h["token"] for h in history] + picked + [i.get("because", i["token"]) for i in account_lists["blend"]])
            rows.append({
                "account": short(str(user_address(user))),
                "group": str(groups[list(task.eval_rows).index(row)]),
                "history": history[:12],
                "history_size": len(history),
                "picked": picked,
                "cold": int(task.cold_new[row]),
                "lists": account_lists,
            })  # fmt: skip
        points.append({"point": day, "accounts": rows})
        print(f"  accounts: {day} done", flush=True)

    used = {h["token"] for p in points for a in p["accounts"] for h in a["history"]}
    used |= {t for p in points for a in p["accounts"] for t in a["picked"]}
    used |= {i[k] for p in points for a in p["accounts"] for items in a["lists"].values() for i in items for k in ("token", "because") if k in i}
    data = {"points": points, "names": {str(t): names(t) for t in used}}
    (out / "accounts.json").write_text(json.dumps(data))


_USERS = None


def user_address(user_id) -> str:
    """The address of an account id (accounts are numbered in data/project/users.parquet)."""
    global _USERS
    if _USERS is None:
        _USERS = pd.read_parquet(td.PROJECT_DIR / "users.parquet").set_index("user_id")["address"]
    return _USERS[user_id]


def strategy(out: Path, legs, tokens, names: Names, frozen: dict) -> None:
    blend_settings = frozen["models"]["fixed blend"]["settings"]
    first_seen = tokens.set_index("token_id")["first_timestamp"]
    points = []
    for day in STRATEGY_POINTS:
        t = pd.Timestamp(day, tz="UTC")
        task = pc.load_task(legs, t)
        window, _, blend = fixed_blend_parts(task, blend_settings)
        # Active accounts: a token in the 14-day window and a trade in the last 7 days. Uses the past only.
        recent = task.history(7)
        active = np.flatnonzero((task.window_size >= 1) & (np.asarray(recent.sum(axis=1)).ravel() > 0))
        head = set(pc.head_tokens(window))
        predicted = np.zeros(len(task.tokens))
        batch = max(64, 25_000_000 // len(task.tokens))
        for start in range(0, len(active), batch):
            rows = active[start : start + batch]
            top = pr.top_k(blend.scores(window[rows]), pc.K, exclude=task.past[rows])
            np.add.at(predicted, top.ravel(), 1)
        buys = legs[legs["side"] == "buy"]
        before = legs[legs["timestamp"] < t][["user", "token"]].drop_duplicates()
        last_day = buys[(buys["timestamp"] >= t - pd.Timedelta(hours=24)) & (buys["timestamp"] < t)]
        visible = last_day.groupby("token")["user"].nunique()
        future = buys[(buys["timestamp"] >= t) & (buys["timestamp"] < t + pc.HORIZON)][["user", "token"]].drop_duplicates()
        future = future.merge(before, on=["user", "token"], how="left", indicator=True)
        realized = future[future["_merge"] == "left_only"].groupby("token")["user"].nunique()

        table = pd.DataFrame({"token": task.tokens, "predicted": predicted})
        table = table[~np.isin(np.arange(len(task.tokens)), list(head))]
        table["visible"] = table["token"].map(visible).fillna(0).astype(int)
        table["realized"] = table["token"].map(realized).fillna(0).astype(int)
        table["age_days"] = ((t - table["token"].map(first_seen)).dt.total_seconds() / 86400).round(1)
        keep = set(table.nlargest(TAIL_KEEP, "predicted")["token"]) | set(table.nlargest(TAIL_KEEP, "visible")["token"])
        kept = table[table["token"].isin(keep) & ((table["predicted"] > 0) | (table["visible"] > 0))]
        names.load(kept["token"])
        points.append({
            "point": day,
            "active_accounts": int(len(active)),
            "head_size": len(head),
            "tail_tokens_with_signal": int((table["predicted"] > 0).sum()),
            "tokens": [
                {"token": int(r.token), "name": names(r.token), "predicted": int(r.predicted), "visible": int(r.visible),
                 "realized": int(r.realized), "age_days": float(r.age_days)}
                for r in kept.itertuples()
            ],
        })  # fmt: skip
        print(f"  strategy: {day} done ({len(active):,} active accounts, {len(kept)} tail tokens kept)", flush=True)
    (out / "strategy.json").write_text(json.dumps({"points": points}))


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    legs, tokens = td.load_legs(), td.load_tokens()
    frozen = json.loads((pc.RESULTS_DIR.parent / "configs" / "frozen.json").read_text())
    names = Names(tokens)
    summary(out)
    print("  summary done", flush=True)
    accounts(out, legs, tokens, names, frozen)
    strategy(out, legs, tokens, names, frozen)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
