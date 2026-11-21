"""Tests for src/scripts/dex_swaps.py (no network access)."""

import pandas as pd
import pytest

import dex_swaps as ds


def word(value: int) -> str:
    return format(value % (1 << 256), "064x")


def topic(address: str) -> str:
    return "0x" + "0" * 24 + address[2:]


SENDER, RECIPIENT, POOL = "0x" + "a" * 40, "0x" + "b" * 40, "0x" + "C" * 40


def raw_log(version="v3", block=100, tx_index=2, log_index=7, tx_hash="0xabc"):
    if version == "v2":  # amount0In, amount1In, amount0Out, amount1Out
        signature, data = ds.TOPIC_V2, word(0) + word(500) + word(20) + word(0)
    else:  # amount0, amount1, sqrtPrice, liquidity, tick
        signature, data = ds.TOPIC_V3, word(-20) + word(500) + word(1) + word(2) + word(-3)
    return {
        "address": POOL, "topics": [signature, topic(SENDER), topic(RECIPIENT)], "data": "0x" + data,
        "blockNumber": hex(block), "transactionIndex": hex(tx_index), "logIndex": hex(log_index),
        "transactionHash": tx_hash, "blockTimestamp": hex(1_700_000_000),
    }  # fmt: skip


class FakeResponse:
    def __init__(self, answer):
        self.answer = answer

    def json(self):
        return self.answer


class FakeNode:
    """Answers eth_getLogs with one log per block and transaction lookups from a table."""

    def __init__(self, fail_first=0):
        self.calls, self.fail_first = [], fail_first

    def post(self, url, json=None, timeout=None):
        self.calls.append(json)
        if self.fail_first > 0:
            self.fail_first -= 1
            return FakeResponse({"error": {"message": "busy"}})
        if isinstance(json, list):
            return FakeResponse([
                {"id": p["id"], "result": {"from": "0xF" + "1" * 39, "to": "0xE" + "2" * 39}}
                for p in reversed(json)  # nodes may answer a batch out of order
            ])  # fmt: skip
        query = json["params"][0]
        first, last = int(query["fromBlock"], 16), int(query["toBlock"], 16)
        return FakeResponse({"result": [raw_log(block=b, tx_hash=f"0x{b:x}") for b in range(first, last + 1)]})


def test_to_signed():
    assert ds.to_signed(word(5)) == 5
    assert ds.to_signed(word(-5)) == -5
    assert ds.to_signed("f" * 64) == -1


def test_topic_address():
    assert ds.topic_address(topic("0x" + "AB" * 20)) == "0x" + "ab" * 20


def test_decode_swap_log_v3():
    record = ds.decode_swap_log(raw_log("v3"))
    assert record["version"] == "v3" and record["pool"] == POOL.lower()
    assert record["sender"] == SENDER and record["recipient"] == RECIPIENT
    assert (record["block"], record["tx_index"], record["log_index"]) == (100, 2, 7)
    # The pool paid out 20 of token0 and received 500 of token1.
    assert record["amount0"] == -20.0 and record["amount1"] == 500.0


def test_decode_swap_log_v2_matches_v3_sign_convention():
    record = ds.decode_swap_log(raw_log("v2"))
    assert record["version"] == "v2"
    assert record["amount0"] == -20.0 and record["amount1"] == 500.0


def test_decode_swap_log_rejects_other_layouts():
    log = raw_log("v3")
    log["data"] = log["data"][:-64]  # four words under the V3 signature
    assert ds.decode_swap_log(log) is None
    log = raw_log("v2")
    log["topics"] = log["topics"][:2]
    assert ds.decode_swap_log(log) is None


def test_rpc_retries_then_returns(monkeypatch):
    monkeypatch.setattr(ds.time, "sleep", lambda seconds: None)
    node = FakeNode(fail_first=2)
    payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_getLogs", "params": [{"fromBlock": "0x1", "toBlock": "0x1"}]}
    assert len(ds.rpc("http://node", payload, node)) == 1
    assert len(node.calls) == 3


def test_rpc_gives_up(monkeypatch):
    monkeypatch.setattr(ds.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError):
        ds.rpc("http://node", {"id": 1, "params": [{}]}, FakeNode(fail_first=99), retries=3)


def test_fetch_swap_logs_pages_through_the_range():
    node = FakeNode()
    swaps = ds.fetch_swap_logs(1000, 1249, session=node, step=100)
    assert len(node.calls) == 3
    assert list(swaps.columns) == ds.SWAP_COLUMNS
    assert swaps["block"].tolist() == list(range(1000, 1250))


def test_fetch_transactions_batches_and_keeps_order():
    node = FakeNode()
    hashes = [f"0x{i}" for i in range(5)] + ["0x0"]  # one duplicate
    transactions = ds.fetch_transactions(hashes, session=node, batch=2)
    assert len(node.calls) == 3
    assert transactions["tx_hash"].tolist() == [f"0x{i}" for i in range(5)]
    assert transactions["tx_from"].iloc[0] == "0xf" + "1" * 39  # lower case


class FakeTokenNode:
    """Answers token0() / token1() calls; the pool "0xdead" does not answer."""

    def __init__(self, rate_limited_first=0):
        self.calls, self.rate_limited_first = 0, rate_limited_first

    def post(self, url, json=None, timeout=None):
        self.calls += 1
        if self.rate_limited_first > 0:
            self.rate_limited_first -= 1
            return FakeResponse([{"id": c["id"], "error": {"code": -32005, "message": "Rate limit exceeded"}} for c in json])
        answers = []
        for call in json:
            pool, selector = call["params"][0]["to"], call["params"][0]["data"]
            if pool == "0xdead":
                answers.append({"id": call["id"], "error": {"message": "execution reverted"}})
            else:
                token = "0x" + ("1" if selector == ds.SELECTOR_TOKEN0 else "2") * 40
                answers.append({"id": call["id"], "result": topic(token)})
        return FakeResponse(answers)


def test_fetch_pool_tokens_retries_when_rate_limited(monkeypatch):
    monkeypatch.setattr(ds.time, "sleep", lambda seconds: None)
    node = FakeTokenNode(rate_limited_first=2)
    tokens = ds.fetch_pool_tokens(["0xaaa"], session=node)
    # A rate limit must never be recorded as "this pool has no tokens".
    assert tokens.loc[0, "token0"] == "0x" + "1" * 40 and node.calls == 3
    with pytest.raises(RuntimeError):
        ds._batch_allowing_reverts("http://node", [{"id": 0, "params": [{"to": "0xaaa", "data": ds.SELECTOR_TOKEN0}]}], FakeTokenNode(rate_limited_first=99), retries=2)


def test_fetch_pool_tokens_handles_pools_that_do_not_answer(monkeypatch):
    monkeypatch.setattr(ds.time, "sleep", lambda seconds: None)
    node = FakeTokenNode()
    tokens = ds.fetch_pool_tokens(["0xaaa", "0xdead", "0xbbb", "0xaaa"], session=node, batch=2)
    assert tokens["pool"].tolist() == ["0xaaa", "0xdead", "0xbbb"]
    assert tokens.loc[0, "token0"] == "0x" + "1" * 40 and tokens.loc[0, "token1"] == "0x" + "2" * 40
    assert tokens.loc[1, ["token0", "token1"]].isna().all()
    assert node.calls == 2


def test_load_pool_tokens_caches_and_extends(tmp_path, monkeypatch):
    monkeypatch.setattr(ds.time, "sleep", lambda seconds: None)
    node, real_fetch = FakeTokenNode(), ds.fetch_pool_tokens
    monkeypatch.setattr(ds, "fetch_pool_tokens", lambda pools: real_fetch(pools, session=node))
    first = ds.load_pool_tokens(["0xaaa", "0xbbb"], tmp_path)
    assert len(first) == 2 and node.calls == 2  # one request per pool
    again = ds.load_pool_tokens(["0xaaa"], tmp_path)
    assert len(again) == 1 and node.calls == 2  # served from the cache
    more = ds.load_pool_tokens(["0xaaa", "0xccc"], tmp_path)
    assert set(more["pool"]) == {"0xaaa", "0xccc"} and node.calls == 3
    assert more["token0"].notna().all()


def test_chunk_ranges():
    assert ds.chunk_ranges(1000, 2500, size=1000) == [(1000, 1999), (2000, 2999), (3000, 3499)]
    assert ds.chunk_ranges(5, 3, size=1000) == [(5, 7)]


def test_download_chunk_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr(ds.requests, "Session", FakeNode)
    path = ds.download_chunk(2000, 2049, tmp_path)
    assert path.name == "swaps_2000_2049.parquet"
    ds.download_chunk(2050, 2099, tmp_path)
    swaps = ds.load_swaps(tmp_path)
    assert len(swaps) == 100 and swaps["block"].is_monotonic_increasing
    assert {"tx_from", "tx_to", "timestamp"} <= set(swaps.columns)
    assert str(swaps["timestamp"].dt.tz) == "UTC"

    # An existing chunk is not downloaded again.
    monkeypatch.setattr(ds.requests, "Session", lambda: pytest.fail("should use the cached file"))
    assert ds.download_chunk(2000, 2049, tmp_path) == path


def test_download_window_and_main(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ds.requests, "Session", FakeNode)
    ds.download_window(3000, 150, tmp_path, workers=2)
    assert len(list((tmp_path / "chunks").glob("*.parquet"))) == 1
    monkeypatch.setattr(ds, "download_window", lambda start, blocks, workers: print("called", start, blocks, workers))
    monkeypatch.setattr("sys.argv", ["dex_swaps.py", "--start", "10", "--blocks", "20"])
    ds.main()
    assert "called 10 20 4" in capsys.readouterr().out


def test_load_swaps_without_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        ds.load_swaps(tmp_path)


# --------------------------------------------------------------------------
# Loading a block range and validating it (work package 1)
# --------------------------------------------------------------------------


def write_empty_chunk(directory, first, last):
    """Create a chunk file with the right name and one swap in its first block."""
    (directory / "chunks").mkdir(parents=True, exist_ok=True)
    row = {column: [0] for column in ds.SWAP_COLUMNS}
    row["block"] = [first]
    pd.DataFrame(row).to_parquet(directory / "chunks" / f"swaps_{first}_{last}.parquet")


def test_chunk_files_returns_only_the_requested_range_in_block_order(tmp_path):
    for first in (3000, 1000, 2000, 4000):
        write_empty_chunk(tmp_path, first, first + 999)
    files = ds.chunk_files(tmp_path, 1000, 2999)
    assert [f.name for f in files] == ["swaps_1000_1999.parquet", "swaps_2000_2999.parquet"]


def test_chunk_files_raises_and_names_the_gap(tmp_path):
    write_empty_chunk(tmp_path, 1000, 1999)
    write_empty_chunk(tmp_path, 3000, 3999)
    with pytest.raises(FileNotFoundError, match="2,000-2,999"):
        ds.chunk_files(tmp_path, 1000, 3999)


def test_chunk_files_needs_both_bounds_or_none(tmp_path):
    write_empty_chunk(tmp_path, 1000, 1999)
    with pytest.raises(ValueError):
        ds.chunk_files(tmp_path, first_block=1000)


def test_missing_ranges():
    assert ds.missing_ranges([(10, 19), (20, 29)], 10, 29) == []
    assert ds.missing_ranges([(10, 19), (30, 39)], 10, 39) == [(20, 29)]
    assert ds.missing_ranges([(20, 29)], 10, 35) == [(10, 19), (30, 35)]


def test_load_swaps_reads_only_the_requested_range(tmp_path):
    for first in (1000, 2000, 3000):
        write_empty_chunk(tmp_path, first, first + 999)
    swaps = ds.load_swaps(tmp_path, 2000, 3999)
    assert swaps["block"].tolist() == [2000, 3000]


def one_day_of_swaps(day: str, n_per_hour: int = 2):
    """A clean day: n_per_hour swaps in every hour, each in its own block, on one known pool."""
    times = pd.date_range(day, periods=24 * n_per_hour, freq=f"{60 // n_per_hour}min", tz="UTC")
    n = len(times)
    first_block = int(pd.Timestamp(day).timestamp()) // 12
    return pd.DataFrame({
        "block": range(first_block, first_block + n), "tx_index": 0, "log_index": 0, "timestamp": times,
        "tx_hash": [f"{day}-{i}" for i in range(n)], "tx_from": [f"user{i % 5}" for i in range(n)],
        "pool": "p", "amount0": 1.0, "amount1": -1.0,
    })  # fmt: skip


POOL_P = pd.DataFrame({"pool": ["p"], "token0": ["A"], "token1": ["B"]})


def test_validate_swaps_passes_clean_data():
    swaps = pd.concat([one_day_of_swaps("2026-07-06"), one_day_of_swaps("2026-07-07")], ignore_index=True)
    swaps["block"] = range(1000, 1000 + len(swaps))  # every block of the week has a swap
    report = ds.validate_swaps(swaps, POOL_P)
    assert len(report) == 1, "both days are in the same ISO week"
    week = report.iloc[0]
    assert week["swaps"] == 96 and week["duplicate_events"] == 0 and week["missing_values"] == 0
    assert week["swaps_per_day_min"] == 48 and week["users_per_day_max"] == 5
    assert bool(week["passed"])


def test_validate_swaps_counts_a_planted_duplicate_and_a_missing_value():
    swaps = one_day_of_swaps("2026-07-06")
    swaps = pd.concat([swaps, swaps.iloc[[3]]], ignore_index=True)  # the same event twice
    swaps.loc[10, "tx_from"] = None
    week = ds.validate_swaps(swaps, POOL_P).iloc[0]
    assert week["duplicate_events"] == 2, "both copies of the duplicated event are counted"
    assert week["missing_values"] == 1
    assert not week["passed"]


def test_validate_swaps_flags_unknown_pools():
    swaps = one_day_of_swaps("2026-07-06")
    swaps.loc[:4, "pool"] = "unknown"
    week = ds.validate_swaps(swaps, POOL_P).iloc[0]
    assert week["swaps_with_known_tokens"] < 0.999 and not week["passed"]


class LaggingNode:
    """Answers every transaction lookup, except that it first answers null `misses` times for "0xlate"."""

    def __init__(self, misses):
        self.misses = misses

    def post(self, url, json=None, timeout=None):
        calls = json if isinstance(json, list) else [json]
        answers = []
        for call in calls:
            tx_hash = call["params"][0]
            if tx_hash == "0xlate" and self.misses > 0:
                self.misses -= 1
                answers.append({"id": call["id"], "result": None})
            else:
                answers.append({"id": call["id"], "result": {"from": "0xAB", "to": "0xCD"}})
        return FakeResponse(answers if isinstance(json, list) else answers[0])


def test_fetch_transactions_asks_again_when_the_node_says_not_found(monkeypatch):
    monkeypatch.setattr(ds.time, "sleep", lambda seconds: None)
    transactions = ds.fetch_transactions(["0xa", "0xlate"], session=LaggingNode(misses=3))
    assert transactions["tx_from"].tolist() == ["0xab", "0xab"]


def test_fetch_transactions_gives_up_if_a_transaction_stays_missing(monkeypatch):
    monkeypatch.setattr(ds.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="not found after 5 more tries"):
        ds.fetch_transactions(["0xlate"], session=LaggingNode(misses=100))
