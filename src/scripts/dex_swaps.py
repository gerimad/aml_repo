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
import sys
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
        wait = min(2**attempt, 20)
        print(f"  retry {attempt + 1}/{retries} in {wait}s ({url}): {last_error[:120]}", file=sys.stderr, flush=True)
        time.sleep(wait)
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


def fetch_transactions(
    hashes, url: str = TX_RPC, session=None, batch: int = 25, rounds: int = 8, pause: float = 1.0
) -> pd.DataFrame:
    """Signer and called contract of each transaction: columns tx_hash, tx_from, tx_to.

    `batch` lookups are sent per request. Small batches waste less work when
    a rate-limited endpoint answers part of a batch with errors (the whole
    batch is then retried by rpc).

    The public endpoint spreads requests over several nodes, and some of them
    answer null for old transactions. The transactions certainly exist (their
    swap logs were just read), so the ones answered null are collected and
    asked for again together, after `pause` seconds, up to `rounds` times.
    Each such round is logged. Raises RuntimeError if any are still missing.
    """
    hashes = list(dict.fromkeys(hashes))
    found = {}
    for i in range(0, len(hashes), batch):
        part = hashes[i : i + batch]
        for tx_hash, tx in zip(part, _lookup_transactions(part, url, session)):
            if tx is not None:
                found[tx_hash] = tx

    missing = [h for h in hashes if h not in found]
    for attempt in range(1, rounds + 1):
        if not missing:
            break
        print(f"  {len(missing)} transaction(s) answered null, asking again (round {attempt}/{rounds})", file=sys.stderr, flush=True)
        time.sleep(pause)
        for i in range(0, len(missing), batch):
            part = missing[i : i + batch]
            for tx_hash, tx in zip(part, _lookup_transactions(part, url, session)):
                if tx is not None:
                    found[tx_hash] = tx
        missing = [h for h in missing if h not in found]
    if missing:
        raise RuntimeError(f"{len(missing)} transaction(s) still not found after {rounds} rounds, e.g. {missing[0]}")

    rows = [
        {"tx_hash": h, "tx_from": found[h]["from"].lower(), "tx_to": (found[h]["to"] or "").lower()} for h in hashes
    ]
    return pd.DataFrame(rows, columns=["tx_hash", "tx_from", "tx_to"])


def _lookup_transactions(part: list[str], url: str, session=None) -> list:
    """One batch request of eth_getTransactionByHash; the answers in the order of `part` (None if not found)."""
    payload = [
        {"jsonrpc": "2.0", "id": j, "method": "eth_getTransactionByHash", "params": [h]} for j, h in enumerate(part)
    ]
    return rpc(url, payload, session)


SELECTOR_TOKEN0, SELECTOR_TOKEN1 = "0x0dfe1681", "0xd21220a7"  # token0() and token1()


def fetch_pool_tokens(
    pools, url: str = LOGS_RPC, session=None, batch: int = 1, pause: float = 0.0
) -> pd.DataFrame:
    """The two tokens of each pool, read from the pool contracts: columns pool, token0, token1.

    Calls token0() and token1() on every pool, `batch` pools per request with
    `pause` seconds in between. The defaults suit the free endpoint used: it
    accepts at most three calls per request, so one pool (two calls) at a
    time. A pool that does not answer (not a standard pool, or destroyed)
    gets None in both columns.
    """
    pools = list(dict.fromkeys(pools))
    rows = []
    for i in range(0, len(pools), batch):
        part = pools[i : i + batch]
        payload = [
            {
                "jsonrpc": "2.0",
                "id": 2 * j + k,
                "method": "eth_call",
                "params": [{"to": pool, "data": selector}, "latest"],
            }
            for j, pool in enumerate(part)
            for k, selector in enumerate((SELECTOR_TOKEN0, SELECTOR_TOKEN1))
        ]
        answers = _batch_allowing_reverts(url, payload, session)
        for j, pool in enumerate(part):
            words = [answers.get(2 * j + k) for k in (0, 1)]
            ok = all(isinstance(w, str) and len(w) == 66 for w in words)
            rows.append(
                {
                    "pool": pool,
                    "token0": topic_address(words[0]) if ok else None,
                    "token1": topic_address(words[1]) if ok else None,
                }
            )
        time.sleep(pause)
    return pd.DataFrame(rows, columns=["pool", "token0", "token1"])


def _batch_allowing_reverts(url: str, payload: list, session=None, retries: int = 8) -> dict:
    """Send a batch of calls, some of which may revert; return {id: result or None}.

    A call that reverts is a real answer ("this contract has no such
    function") and maps to None. Any other error, such as a rate limit, means
    the batch was not served: it is retried with a growing wait, and never
    mistaken for a revert.
    """
    last_error = None
    for attempt in range(retries):
        try:
            answer = (session or requests).post(url, json=payload, timeout=90).json()
            if isinstance(answer, list) and len(answer) == len(payload):
                failed = [a for a in answer if "result" not in a and not _is_revert(a)]
                if not failed:
                    return {a["id"]: a.get("result") for a in answer}
                last_error = str(failed[0])[:300]
            else:
                last_error = str(answer)[:300]
        except (requests.RequestException, ValueError) as error:
            last_error = repr(error)
        time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"RPC batch to {url} failed: {last_error}")


def _is_revert(answer: dict) -> bool:
    """Whether a JSON-RPC error answer reports that the called contract reverted."""
    error = answer.get("error") or {}
    return error.get("code") == 3 or "revert" in str(error.get("message", "")).lower()


def load_pool_tokens(
    pools, data_dir: Path = DATA_DIR, block_size: int = 800, workers: int = 8
) -> pd.DataFrame:
    """Tokens of the given pools, from a local cache that is extended as needed.

    Missing pools are fetched `block_size` at a time by `workers` threads,
    and the cache is saved after each block, so an interrupted run loses little.
    """
    path = data_dir / "pool_tokens.parquet"
    columns = ["pool", "token0", "token1"]
    known = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=columns)
    have = set(known["pool"])
    missing = [pool for pool in dict.fromkeys(pools) if pool not in have]
    for i in range(0, len(missing), block_size):
        block = missing[i : i + block_size]
        shares = [block[j::workers] for j in range(workers) if block[j::workers]]
        with ThreadPoolExecutor(workers) as pool_of_threads:
            fetched = list(pool_of_threads.map(fetch_pool_tokens, shares))
        known = pd.concat([known, *fetched], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        known.to_parquet(path)
    return known[known["pool"].isin(set(pools))].reset_index(drop=True)


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


def chunk_files(
    data_dir: Path = DATA_DIR, first_block: int | None = None, last_block: int | None = None
) -> list[Path]:
    """The chunk files that lie inside blocks first_block..last_block, in block order.

    The range of a chunk is read from its file name (swaps_<first>_<last>.parquet),
    so no file is opened. With both bounds given, the chunks must cover the
    range exactly, without gaps; otherwise a FileNotFoundError lists what is
    missing. With no bounds, every chunk is returned.

    Assumption: chunk files are never partial. download_chunk only gives a
    file its final name after it is completely written.
    """
    files = list((data_dir / "chunks").glob("swaps_*.parquet"))
    ranges = []
    for path in files:
        _, first, last = path.stem.split("_")
        ranges.append((int(first), int(last), path))
    ranges.sort()

    if first_block is None and last_block is None:
        selected = ranges
    else:
        if first_block is None or last_block is None:
            raise ValueError("give both first_block and last_block, or neither")
        selected = [r for r in ranges if r[0] >= first_block and r[1] <= last_block]
        missing = missing_ranges([(f, l) for f, l, _ in selected], first_block, last_block)
        if missing:
            shown = ", ".join(f"{f:,}-{l:,}" for f, l in missing[:10])
            raise FileNotFoundError(
                f"blocks {first_block:,} to {last_block:,} are not fully downloaded; "
                f"{len(missing)} missing range(s): {shown}"
            )
    if not selected:
        raise FileNotFoundError(f"no swap chunks in {data_dir}; run dex_swaps.py first")
    return [path for _, _, path in selected]


def missing_ranges(ranges: list[tuple[int, int]], first_block: int, last_block: int) -> list[tuple[int, int]]:
    """Block ranges inside first_block..last_block that none of the given (first, last) ranges cover."""
    missing = []
    expected_next = first_block
    for first, last in sorted(ranges):
        if first > expected_next:
            missing.append((expected_next, first - 1))
        expected_next = max(expected_next, last + 1)
    if expected_next <= last_block:
        missing.append((expected_next, last_block))
    return missing


def load_swaps(
    data_dir: Path = DATA_DIR, first_block: int | None = None, last_block: int | None = None
) -> pd.DataFrame:
    """Load the downloaded chunks as one frame, in chain order.

    Without bounds every chunk is loaded (this is what notebooks 05 and 08
    do). The project always passes its block range, so that the exploration
    data is never loaded by accident; see chunk_files for the gap check.
    """
    files = chunk_files(data_dir, first_block, last_block)
    swaps = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return swaps.sort_values(["block", "tx_index", "log_index"], ignore_index=True)


def validate_swaps(swaps: pd.DataFrame, pool_tokens: pd.DataFrame) -> pd.DataFrame:
    """Data-quality checks per calendar week (ISO weeks, UTC). One row per week.

    Columns:
      week                       Monday of the week
      days                       number of days of the week present in the data
      swaps, users               number of swaps and of distinct signing accounts
      blocks_with_swap           share of the blocks of the week (from its first to
                                 its last block) that contain at least one swap
      duplicate_events           swaps whose (tx_hash, log_index) occurs more than once
      missing_values             number of empty cells in the core columns
      swaps_with_known_tokens    share of swaps on a pool whose two tokens are known
      swaps_per_day_min / _max   over the full days of the week
      users_per_day_min / _max   over the full days of the week
      low_days                   full days with fewer than half the week's median swaps
      passed                     blocks_with_swap >= 0.98, no duplicates, no missing
                                 values, swaps_with_known_tokens >= 0.999

    Daily volumes are reported, not checked against a threshold, because
    trading activity really does change from week to week. A "full day" is a
    day whose first and last hour both contain swaps; the first and last day
    of a download are usually partial.
    """
    core = ["block", "tx_index", "log_index", "timestamp", "tx_hash", "tx_from", "pool", "amount0", "amount1"]
    known_pools = set(pool_tokens.dropna()["pool"])
    frame = swaps[core].copy()
    frame["week"] = frame["timestamp"].dt.tz_localize(None).dt.to_period("W-SUN").dt.start_time
    frame["day"] = frame["timestamp"].dt.floor("D")
    frame["hour"] = frame["timestamp"].dt.hour

    rows = []
    for week, part in frame.groupby("week", sort=True):
        per_day = part.groupby("day").agg(
            swaps=("block", "size"), users=("tx_from", "nunique"), first_hour=("hour", "min"), last_hour=("hour", "max")
        )
        full_days = per_day[(per_day["first_hour"] == 0) & (per_day["last_hour"] == 23)]
        n_blocks = part["block"].max() - part["block"].min() + 1
        median_swaps = full_days["swaps"].median() if len(full_days) else float("nan")
        row = {
            "week": week.date(),
            "days": len(per_day),
            "swaps": len(part),
            "users": part["tx_from"].nunique(),
            "blocks_with_swap": part["block"].nunique() / n_blocks,
            "duplicate_events": int(part.duplicated(["tx_hash", "log_index"], keep=False).sum()),
            "missing_values": int(part[core].isna().sum().sum()),
            "swaps_with_known_tokens": part["pool"].isin(known_pools).mean(),
            "swaps_per_day_min": full_days["swaps"].min(),
            "swaps_per_day_max": full_days["swaps"].max(),
            "users_per_day_min": full_days["users"].min(),
            "users_per_day_max": full_days["users"].max(),
            "low_days": int((full_days["swaps"] < 0.5 * median_swaps).sum()),
        }
        row["passed"] = bool(
            row["blocks_with_swap"] >= 0.98
            and row["duplicate_events"] == 0
            and row["missing_values"] == 0
            and row["swaps_with_known_tokens"] >= 0.999
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", type=int, required=True, help="first block")
    parser.add_argument("--blocks", type=int, required=True, help="number of blocks")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    download_window(args.start, args.blocks, workers=args.workers)


if __name__ == "__main__":
    main()
