"""Narrative shares at all 35 evaluation points (work package 20).

Clusters the tokens that carry the recommender's lists (the N_TOKENS tokens
with the most list slots over all points; the rest are "other") on the
whole-history similarity, at every resolution of narratives.RESOLUTIONS, and
writes to results/wp20/:

  clusters.parquet   token, symbol, total list slots, cluster at each resolution
  table.parquet      point, part, token and the four counts of narratives.COLUMNS

Usage, from the repository root:

    uv run python src/scripts/build_narratives.py
"""

from __future__ import annotations

import pandas as pd

import dex_swaps as ds
import narratives as nr
import protocol as pc
import token_data as td
import token_map as tm

OUT = pc.RESULTS_DIR / "wp20"
N_TOKENS = 3000


def main() -> None:
    legs = td.load_legs()
    table = pd.read_parquet(pc.RESULTS_DIR / "wp17" / "tokens.parquet")[["point", "part", "token", "list_count", "n_known", "persistence"]]
    popularity = []
    for t in sorted(table["point"].unique()):
        p = nr.popularity_14d(legs, t).rename("popularity").reset_index()
        popularity.append(p.assign(point=t))
    table = table.merge(pd.concat(popularity), on=["point", "token"], how="left").fillna({"popularity": 0})
    print(f"{len(table):,} rows, {table['point'].nunique()} points", flush=True)

    totals = table.groupby("token")["list_count"].sum().sort_values(ascending=False)
    keep = totals.index[:N_TOKENS]
    print(f"the {N_TOKENS} most-listed tokens hold {totals[keep].sum() / totals.sum():.1%} of all list slots", flush=True)

    task = pc.load_task(legs, pc.POINTS["test"][-1])
    graph = nr.whole_history_graph(task)
    inside = pd.Index(graph.tokens).get_indexer(keep)
    inside = inside[inside >= 0]
    graph = tm.Graph(graph.tokens[inside], graph.matrix[inside][:, inside].tocsr(), graph.accounts[inside])
    print(f"graph: {len(graph.tokens):,} tokens, {graph.matrix.nnz:,} links", flush=True)

    addresses = td.load_tokens().set_index("token_id")["address"].reindex(graph.tokens)
    # On-chain symbol() from the local metadata cache (dex_swaps), fetched where missing; the address if unreadable.
    onchain = ds.load_token_metadata(addresses.tolist()).drop_duplicates("address").set_index("address")["symbol"]
    symbols = addresses.map(onchain).fillna(addresses.map(td.symbol))
    out = pd.DataFrame({"token": graph.tokens, "symbol": symbols.to_numpy(), "list_slots": totals.reindex(graph.tokens).to_numpy()})
    for resolution in nr.RESOLUTIONS:
        out[f"cluster_{resolution:g}"] = out["token"].map(tm.clusters(graph, resolution))
        print(f"resolution {resolution:g}: {out[f'cluster_{resolution:g}'].nunique()} clusters", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "clusters.parquet")
    table.to_parquet(OUT / "table.parquet")
    print("wrote clusters.parquet and table.parquet", flush=True)


if __name__ == "__main__":
    main()
