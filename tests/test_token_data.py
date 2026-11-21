"""Tests for src/scripts/token_data.py."""

import numpy as np
import pandas as pd

import token_data as td

T0 = pd.Timestamp("2026-06-18", tz="UTC")


def swap(block, tx, pool, amount0, signer, log_index=0):
    """One swap row as the downloader writes it. amount0 > 0: the pool received token0."""
    return {
        "block": block, "tx_index": 0, "log_index": log_index, "timestamp": T0 + pd.Timedelta(minutes=block),
        "tx_hash": tx, "tx_from": signer, "pool": pool, "amount0": amount0, "amount1": -amount0,
    }  # fmt: skip


POOLS = pd.DataFrame({
    "pool": ["eth_usdc", "usdc_tok", "eth_tok", "eth_tok_v3"],
    "token0": ["ETH", "USDC", "ETH", "ETH"],
    "token1": ["USDC", "TOK", "TOK", "TOK"],
})  # fmt: skip


def test_legs_carry_the_block_time_and_the_pool_where_the_token_was_traded():
    # ETH -> USDC -> TOK in one transaction: USDC is dropped, ETH was sold in eth_usdc, TOK bought in usdc_tok.
    swaps = pd.DataFrame([swap(5, "a", "eth_usdc", 1.0, "u", 0), swap(5, "a", "usdc_tok", 1.0, "u", 1)])
    legs = td.legs_with_time_and_pool(swaps, POOLS).set_index("token")
    assert set(legs.index) == {"ETH", "TOK"}
    assert legs.loc["ETH", "pool"] == "eth_usdc" and legs.loc["TOK", "pool"] == "usdc_tok"
    assert (legs["timestamp"] == T0 + pd.Timedelta(minutes=5)).all()


def test_split_route_keeps_the_pool_of_the_first_swap():
    # The trader buys TOK in two pools within one transaction; the first swap's pool is kept.
    swaps = pd.DataFrame([swap(5, "a", "eth_tok_v3", 1.0, "u", 0), swap(5, "a", "eth_tok", 1.0, "u", 1)])
    legs = td.legs_with_time_and_pool(swaps, POOLS).set_index("token")
    assert legs.loc["TOK", "pool"] == "eth_tok_v3"


def legs_frame(rows):
    """Legs from (minutes after T0, user, token) triples."""
    return pd.DataFrame([{"timestamp": T0 + pd.Timedelta(minutes=m), "user": u, "token": t} for m, u, t in rows])


def test_base_tokens_removes_only_tokens_above_the_threshold():
    # 10 users; WETH is traded by all of them, X by exactly one (10%), Y by two (20%).
    rows = [(1, f"u{i}", "WETH") for i in range(10)] + [(2, "u0", "X"), (3, "u0", "Y"), (3, "u1", "Y")]
    table = td.base_tokens(legs_frame(rows), T0, T0 + pd.Timedelta(days=1), threshold=0.10).set_index("token")
    assert table.loc["WETH", "is_base"] and table.loc["Y", "is_base"]
    assert not table.loc["X", "is_base"], "exactly 10% is not more than 10%"


def test_base_tokens_looks_only_at_the_given_period():
    rows = [(1, "u0", "A"), (1, "u1", "B"), (60 * 24 * 5, "u2", "LATE")]
    table = td.base_tokens(legs_frame(rows), T0, T0 + pd.Timedelta(days=1))
    assert "LATE" not in set(table["token"])


def test_id_map_is_stable_across_calls():
    ids = td.IdMap()
    assert ids.encode(pd.Series(["b", "a", "b"])).tolist() == [0, 1, 0]
    assert ids.encode(pd.Series(["c", "a"])).tolist() == [2, 1]
    assert ids.table("id")["address"].tolist() == ["b", "a", "c"]


def test_token_table_finds_first_trade_launch_pool_and_base_and_marks_censored_tokens():
    # Token 1 is first traded 1 hour into the data (censored), token 2 three days in (not censored).
    legs = pd.DataFrame({
        "block": [1, 2, 3],
        "timestamp": [T0 + pd.Timedelta(hours=1), T0 + pd.Timedelta(days=3), T0 + pd.Timedelta(days=4)],
        "token": [1, 2, 2],
        "pool": [10, 20, 21],
    })  # fmt: skip
    pools = pd.DataFrame({"pool_id": [10, 20, 21], "token0_id": [0, 2, 1], "token1_id": [1, 0, 2]})
    table = td.token_table(legs, pools, data_start=T0).set_index("token_id")
    assert table.loc[2, "first_block"] == 2 and table.loc[2, "launch_pool_id"] == 20
    assert table.loc[1, "launch_base_id"] == 0 and table.loc[2, "launch_base_id"] == 0
    assert bool(table.loc[1, "censored"]) and not bool(table.loc[2, "censored"])


def test_symbol_names_known_tokens_and_shortens_the_rest():
    assert td.symbol("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2") == "WETH"
    assert td.symbol("0x1234567890abcdef1234567890abcdef12345678") == "0x1234…5678"


def test_merge_week_rows_combines_two_halves_of_a_week():
    halves = pd.DataFrame({
        "week": ["2026-07-06", "2026-07-06"], "days": [4, 4], "swaps": [100, 300], "users": [10, 20],
        "blocks_with_swap": [1.0, 0.98], "duplicate_events": [0, 0], "missing_values": [0, 0],
        "swaps_with_known_tokens": [1.0, 1.0], "swaps_per_day_min": [20, 40], "swaps_per_day_max": [30, 80],
        "users_per_day_min": [3, 5], "users_per_day_max": [8, 9], "low_days": [0, 1], "passed": [True, True],
    })  # fmt: skip
    week = td.merge_week_rows(halves).iloc[0]
    assert week["days"] == 7 and week["swaps"] == 400
    assert np.isclose(week["blocks_with_swap"], 0.25 * 1.0 + 0.75 * 0.98)
    assert week["swaps_per_day_min"] == 20 and week["swaps_per_day_max"] == 80 and week["low_days"] == 1
