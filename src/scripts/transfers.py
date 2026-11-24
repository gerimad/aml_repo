"""ERC-20 Transfer logs of WETH, USDC and USDT (work package 14, implementation_plan_applications.md 14.1).

WP14 needs to know where an account's proceeds went after it sold: swap
recipients are mostly routers, so this cannot be read from the swaps. The
Transfer events of the three base tokens (where proceeds usually are) show it.
Plain ETH transfers need traces, which the free endpoints do not serve; they
are not covered.

Downloaded like the swaps (dex_swaps.py): eth_getLogs over block chunks, one
Parquet file per chunk in data/transfers/chunks/, safe to restart.

Usage, from the repository root (first block, number of blocks):

    uv run python src/scripts/transfers.py 25600000 50400
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

import dex_swaps as ds

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "transfers"
TOPIC_TRANSFER = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
CONTRACTS = {
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "WETH",
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "USDC",
    "0xdac17f958d2ee523a2206206994597c13d831ec7": "USDT",
}
DECIMALS = {"WETH": 18, "USDC": 6, "USDT": 6}
COLUMNS = ["block", "log_index", "tx_hash", "token", "sender", "recipient", "amount"]


def decode_transfer_log(log: dict) -> dict | None:
    """One Transfer log as a row (amount in whole tokens); None if it is not a standard Transfer of a base token."""
    topics = log.get("topics", [])
    token = CONTRACTS.get(log["address"].lower())
    if token is None or len(topics) != 3 or topics[0] != TOPIC_TRANSFER:
        return None
    return {
        "block": int(log["blockNumber"], 16),
        "log_index": int(log["logIndex"], 16),
        "tx_hash": log["transactionHash"],
        "token": token,
        "sender": ds.topic_address(topics[1]),
        "recipient": ds.topic_address(topics[2]),
        "amount": int(log["data"], 16) / 10 ** DECIMALS[token],
    }


def fetch_transfer_logs(start: int, end: int, url: str = ds.LOGS_RPC, session=None, step: int = 10) -> pd.DataFrame:
    """All base-token Transfer logs in blocks start..end (inclusive). Small steps: these tokens move often."""
    records = []
    for first in range(start, end + 1, step):
        last = min(first + step - 1, end)
        query = {"fromBlock": hex(first), "toBlock": hex(last), "address": list(CONTRACTS), "topics": [TOPIC_TRANSFER]}
        payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_getLogs", "params": [query]}
        for log in ds.rpc(url, payload, session):
            record = decode_transfer_log(log)
            if record is not None:
                records.append(record)
    return pd.DataFrame(records, columns=COLUMNS)


def download_chunk(start: int, end: int, out_dir: Path = DATA_DIR) -> Path:
    """Download blocks start..end into one Parquet file, unless it already exists."""
    path = out_dir / "chunks" / f"transfers_{start}_{end}.parquet"
    if path.exists():
        return path
    table = fetch_transfer_logs(start, end, session=requests.Session())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    table.to_parquet(temporary)
    temporary.rename(path)
    return path


def download_window(start: int, n_blocks: int, out_dir: Path = DATA_DIR, workers: int = 1) -> None:
    """Download a range of blocks in chunks of 1,000. Safe to restart. One worker by default: the free node rate-limits
    parallel log queries (found on the first sizing run)."""
    ranges = ds.chunk_ranges(start, n_blocks)
    with ThreadPoolExecutor(workers) as pool:
        for i, path in enumerate(pool.map(lambda r: download_chunk(*r, out_dir), ranges), 1):
            print(f"{i}/{len(ranges)} {path.name}", flush=True)


def load_transfers(out_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Every downloaded chunk, in chain order."""
    parts = [pd.read_parquet(p) for p in sorted((out_dir / "chunks").glob("transfers_*.parquet"))]
    return pd.concat(parts, ignore_index=True).sort_values(["block", "log_index"], ignore_index=True)


if __name__ == "__main__":
    download_window(int(sys.argv[1]), int(sys.argv[2]), workers=int(sys.argv[3]) if len(sys.argv) > 3 else 1)
