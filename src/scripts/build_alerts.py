"""The alert table of work package 14 at the Dev A and Dev B points.

One row per (account, 6-hour block) with both ranked pick-ups (for the
deviation) and sales of its window tokens (for the liquidation), for accounts
with at least 5 tokens before the point: deviation, liquidation, exit (USD sent
to new addresses), silence, and the day. Writes data/project/wp14/blocks.parquet
(it holds account ids, so it stays local), and the eligible account-days per
part for the alert rate.

Transfers come from data/transfers/dex_chunks/ (transfers.py). Their times are
interpolated from the legs' block times; WETH amounts are converted to USD at
the day's mean ETH price from the price tables.

Usage, from the repository root:

    uv run python src/scripts/build_alerts.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

import alerts as al
import prices as px
import protocol as pc
import token_data as td
import transfers as tr

OUT = td.PROJECT_DIR / "wp14"
PARTS = ("dev_a", "dev_b")


def transfers_with_usd(legs: pd.DataFrame) -> pd.DataFrame:
    """The downloaded transfers with a timestamp and an amount in USD."""
    table = tr.load_transfers()
    times = legs.drop_duplicates("block")[["block", "timestamp"]].sort_values("block")
    seconds = (times["timestamp"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    table["timestamp"] = pd.to_datetime(np.interp(table["block"], times["block"], seconds), unit="s", utc=True)
    eth = pads.dataset(sorted(px.PRICES_DIR.glob("prices_*.parquet"))).to_table(columns=["timestamp", "eth_usd"]).to_pandas()
    eth_per_day = eth.groupby(eth["timestamp"].dt.floor("D"))["eth_usd"].mean()
    usd_per_token = np.where(table["token"] == "WETH", table["timestamp"].dt.floor("D").map(eth_per_day).to_numpy(), 1.0)
    table["amount_usd"] = table["amount"] * usd_per_token
    return table[["timestamp", "sender", "recipient", "amount_usd"]]


def main() -> None:
    legs = td.load_legs()
    addresses = pd.read_parquet(td.PROJECT_DIR / "users.parquet").set_index("user_id")["address"]
    pools = set(pd.read_parquet(td.PROJECT_DIR / "pools.parquet")["address"])
    sent = transfers_with_usd(legs)
    print(f"{len(sent):,} transfers from {sent['timestamp'].min():%d %b} to {sent['timestamp'].max():%d %b}", flush=True)
    pairs = pd.concat([pd.read_parquet(pc.RESULTS_DIR / "wp9" / f"pairs_{part}.parquet") for part in PARTS])
    times_by_user = {u: np.sort(part["timestamp"].dt.tz_localize(None).to_numpy().astype("datetime64[ns]")) for u, part in legs.groupby("user")}

    rows, account_days = [], {}
    for part in PARTS:
        days = set()
        for t in pc.POINTS[part]:
            task = pc.load_task(legs, t)
            eligible = task.users[np.asarray(task.past.sum(axis=1)).ravel() >= al.MIN_TOKENS]
            target = legs[(legs["timestamp"] >= t) & (legs["timestamp"] < t + pc.HORIZON) & legs["user"].isin(eligible)]
            days |= set(zip(target["user"], target["timestamp"].dt.floor("D")))
            dev = al.deviation(pairs[pairs["point"] == t], pairs[pairs["point"] < t])
            liq = al.liquidation(legs, t, eligible)
            table = dev.merge(liq, on=["user", "block"])
            table["exit_usd"] = al.exits(table, sent, addresses, pools).to_numpy()
            table["silent"] = al.silent(times_by_user, table["user"].to_numpy(), table["block"] + al.BLOCK)
            rows.append(table.assign(point=t, part=part, day=table["block"].dt.floor("D")))
            print(f"{part} {t:%d %b}: {len(table):,} account-blocks", flush=True)
        account_days[part] = len(days)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_parquet(OUT / "blocks.parquet")
    pd.Series(account_days, name="active_account_days").to_csv(OUT / "account_days.csv")
    print("wrote blocks.parquet and account_days.csv", flush=True)


if __name__ == "__main__":
    main()
