"""Download Uniswap-style swap events from Ethereum through public RPC endpoints.

Every swap on a Uniswap V2 or V3 pool (and on forks that keep the same event,
such as SushiSwap) emits a `Swap` log. This script collects all of them for a
range of blocks, decodes them, and looks up who sent each transaction.

Usage, from the repository root (no API key needed):

    uv run python src/scripts/dex_swaps.py --start 26100000 --blocks 28800

This writes one Parquet file per 1,000 blocks to data/dex_swaps/chunks/ and can
be interrupted and restarted. Load everything in a notebook with:

    from dex_swaps import load_swaps
    swaps = load_swaps()

One row per swap, with:
  block, tx_index, log_index   position on the chain (sort by these for time order)
  timestamp                    block time, UTC
  tx_hash, tx_from, tx_to      the transaction, its signer and the contract it called
  pool, version                the pool contract and its type ("v2" or "v3")
  sender                       who called the pool: a router or a trader's own contract
  recipient                    who received the output tokens
  amount0, amount1             token amounts from the pool's side: positive = pool received
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "dex_swaps"
LOGS_RPC = "https://eth.drpc.org"  # serves eth_getLogs for old blocks, 100 blocks per call
TX_RPC = "https://ethereum-rpc.publicnode.com"  # accepts batches of transaction lookups

# keccak256 of the event signatures.
TOPIC_V2 = "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"
TOPIC_V3 = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
CHUNK_BLOCKS = 1000
SWAP_COLUMNS = [
    "block", "tx_index", "log_index", "timestamp", "tx_hash", "pool", "version",
    "sender", "recipient", "amount0", "amount1",
]  # fmt: skip


def rpc(url: str, payload, session=None, retries: int = 6, timeout: int = 90):
    """POST a JSON-RPC request (one call or a batch), retrying on errors with backoff.

    Returns the "result" of a single call, or the list of results of a batch
    in request order. Raises RuntimeError if the node keeps answering with an error.
    """
    last_error = None
    for attempt in range(retries):
        try:
            response = (session or requests).post(url, json=payload, timeout=timeout)
            answer = response.json()
            if isinstance(payload, list):
                if isinstance(answer, list) and all("result" in a for a in answer):
                    by_id = {a["id"]: a["result"] for a in answer}
                    return [by_id[p["id"]] for p in payload]
            elif "result" in answer:
                return answer["result"]
            last_error = str(answer)[:300]
        except (requests.RequestException, ValueError) as error:
            last_error = repr(error)
        time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"RPC call to {url} failed: {last_error}")


def to_signed(word: str) -> int:
    """Interpret a 64-digit hex word as a signed 256-bit integer (two's complement)."""
    value = int(word, 16)
    return value - (1 << 256) if value >= (1 << 255) else value


def topic_address(topic: str) -> str:
    """The address stored in a 32-byte log topic (its last 20 bytes), lower case."""
    return "0x" + topic[-40:].lower()


def decode_swap_log(log: dict) -> dict | None:
    """Decode one raw `Swap` log into a flat record, or None if it is not a V2 / V3 swap.

    Amounts are from the pool's side: positive means the pool received that
    token, negative that it paid it out. They are stored as floats, which is
    enough for direction and relative size.
    """
    topics, data = log["topics"], log["data"][2:]
    words = [data[i : i + 64] for i in range(0, len(data), 64)]
    if len(topics) != 3:
        return None
    if topics[0] == TOPIC_V2 and len(words) == 4:
        in0, in1, out0, out1 = (int(w, 16) for w in words)
        version, amount0, amount1 = "v2", in0 - out0, in1 - out1
    elif topics[0] == TOPIC_V3 and len(words) == 5:
        version, amount0, amount1 = "v3", to_signed(words[0]), to_signed(words[1])
    else:
        return None
    return {
        "block": int(log["blockNumber"], 16),
        "tx_index": int(log["transactionIndex"], 16),
        "log_index": int(log["logIndex"], 16),
        "timestamp": int(log["blockTimestamp"], 16) if "blockTimestamp" in log else None,
        "tx_hash": log["transactionHash"],
        "pool": log["address"].lower(),
        "version": version,
        "sender": topic_address(topics[1]),
        "recipient": topic_address(topics[2]),
        "amount0": float(amount0),
        "amount1": float(amount1),
    }


def fetch_swap_logs(start: int, end: int, url: str = LOGS_RPC, session=None, step: int = 100):
    """All V2 and V3 swap logs in blocks start..end (inclusive), decoded, as a DataFrame."""
    records = []
    for first in range(start, end + 1, step):
        last = min(first + step - 1, end)
        query = {"fromBlock": hex(first), "toBlock": hex(last), "topics": [[TOPIC_V2, TOPIC_V3]]}
        payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_getLogs", "params": [query]}
        for log in rpc(url, payload, session):
            record = decode_swap_log(log)
            if record is not None:
                records.append(record)
    return pd.DataFrame(records, columns=SWAP_COLUMNS)


def fetch_transactions(hashes, url: str = TX_RPC, session=None, batch: int = 100) -> pd.DataFrame:
    """Signer and called contract of each transaction: columns tx_hash, tx_from, tx_to."""
    hashes = list(dict.fromkeys(hashes))
    rows = []
    for i in range(0, len(hashes), batch):
        part = hashes[i : i + batch]
        payload = [
            {"jsonrpc": "2.0", "id": j, "method": "eth_getTransactionByHash", "params": [h]}
            for j, h in enumerate(part)
        ]
        for tx_hash, tx in zip(part, rpc(url, payload, session)):
            if tx is None:
                raise RuntimeError(f"transaction {tx_hash} not found")
            rows.append(
                {"tx_hash": tx_hash, "tx_from": tx["from"].lower(), "tx_to": (tx["to"] or "").lower()}
            )
    return pd.DataFrame(rows, columns=["tx_hash", "tx_from", "tx_to"])


def download_chunk(start: int, end: int, out_dir: Path = DATA_DIR) -> Path:
    """Download blocks start..end into one Parquet file, unless it already exists."""
    path = out_dir / "chunks" / f"swaps_{start}_{end}.parquet"
    if path.exists():
        return path
    session = requests.Session()
    swaps = fetch_swap_logs(start, end, session=session)
    transactions = fetch_transactions(swaps["tx_hash"], session=session)
    swaps = swaps.merge(transactions, on="tx_hash", how="left")
    swaps["timestamp"] = pd.to_datetime(swaps["timestamp"], unit="s", utc=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    swaps.to_parquet(temporary)
    temporary.rename(path)  # only complete files get the final name
    return path


def chunk_ranges(start: int, n_blocks: int, size: int = CHUNK_BLOCKS) -> list[tuple[int, int]]:
    """Split n_blocks blocks from `start` into consecutive (first, last) ranges of `size`."""
    end = start + n_blocks - 1
    return [(first, min(first + size - 1, end)) for first in range(start, end + 1, size)]


def download_window(start: int, n_blocks: int, out_dir: Path = DATA_DIR, workers: int = 4) -> None:
    """Download a range of blocks in chunks, a few at a time. Safe to restart."""
    ranges = chunk_ranges(start, n_blocks)
    with ThreadPoolExecutor(workers) as pool:
        for i, path in enumerate(pool.map(lambda r: download_chunk(*r, out_dir), ranges), 1):
            print(f"{i}/{len(ranges)} {path.name}", flush=True)


def load_swaps(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load all downloaded chunks as one frame, in chain order."""
    files = sorted((data_dir / "chunks").glob("swaps_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no swap chunks in {data_dir}; run dex_swaps.py first")
    swaps = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return swaps.sort_values(["block", "tx_index", "log_index"], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", type=int, required=True, help="first block")
    parser.add_argument("--blocks", type=int, required=True, help="number of blocks")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    download_window(args.start, args.blocks, workers=args.workers)


if __name__ == "__main__":
    main()
