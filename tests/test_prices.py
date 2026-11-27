"""Tests for src/scripts/prices.py, on toy swaps written by hand."""

import numpy as np
import pandas as pd
import pytest

import prices as px

USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
PEPE = "0x00000000000000000000000000000000000000aa"
DECIMALS = pd.Series({px.WETH: 18, USDC: 6, PEPE: 9})


def swap(block, pool, token0, token1, amount0, amount1, version="v2"):
    return {"block": block, "tx_index": 0, "log_index": 0, "timestamp": pd.Timestamp("2026-07-02", tz="UTC"),
            "pool": pool, "version": version, "tx_from": "0xabc", "token0": token0, "token1": token1,
            "amount0": float(amount0), "amount1": float(amount1)}  # fmt: skip


def test_price_is_the_same_whichever_token_is_token0():
    # Buying 1,000 PEPE for 0.5 WETH: the pool receives 0.5 WETH and gives out 1,000 PEPE.
    a = swap(1, "p1", PEPE, px.WETH, -1000 * 10**9, 0.5 * 10**18)
    b = swap(1, "p2", px.WETH, PEPE, 0.5 * 10**18, -1000 * 10**9, version="v3")
    prices = px.swap_prices(pd.DataFrame([a, b]), DECIMALS)
    assert prices["price_base"].tolist() == pytest.approx([0.0005, 0.0005])
    assert prices["side"].tolist() == ["buy", "buy"]
    assert prices["token"].tolist() == [PEPE, PEPE] and prices["token_amount"].tolist() == pytest.approx([1000, 1000])


def test_a_sell_is_when_the_pool_receives_the_token():
    s = swap(1, "p1", PEPE, px.WETH, 2000 * 10**9, -0.8 * 10**18)
    prices = px.swap_prices(pd.DataFrame([s]), DECIMALS)
    assert prices["side"].iloc[0] == "sell" and prices["price_base"].iloc[0] == pytest.approx(0.0004)


def test_swaps_without_exactly_one_base_or_without_decimals_get_no_price():
    two_bases = swap(1, "p1", USDC, px.WETH, -3000 * 10**6, 1 * 10**18)
    no_base = swap(1, "p2", PEPE, "0xbb", -10, 10)
    unknown = swap(1, "p3", "0xcc", px.WETH, -10, 10)
    same_direction = swap(1, "p4", PEPE, px.WETH, 10, 10)
    assert px.swap_prices(pd.DataFrame([two_bases, no_base, unknown, same_direction]), DECIMALS).empty


def test_eth_price_is_volume_weighted_and_ignores_small_swaps():
    swaps = pd.DataFrame([
        swap(5, "e1", USDC, px.WETH, -3000 * 10**6, 1 * 10**18),      # 1 ETH at 3,000
        swap(5, "e2", px.WETH, USDC, 3 * 10**18, -9300 * 10**6),     # 3 ETH at 3,100
        swap(5, "e3", USDC, px.WETH, -1 * 10**6, 0.0002 * 10**18),   # tiny: ignored
    ])  # fmt: skip
    eth = px.eth_usd_per_block(swaps, DECIMALS)
    assert eth.loc[5] == pytest.approx((3000 + 9300) / 4)


def test_eth_prices_are_carried_forward_and_usdc_prices_converted():
    priced = pd.DataFrame({
        "block": [4, 6, 6], "base": [px.WETH, px.WETH, USDC], "price_base": [0.001, 0.001, 3.0], "base_amount": [1.0, 1.0, 300.0],
    })  # fmt: skip
    eth_usd = pd.Series({5: 3000.0})
    out = px.add_eth_prices(priced, eth_usd)
    assert out["block"].tolist() == [6, 6], "the swap before the first ETH price is dropped"
    assert out["price_eth"].tolist() == pytest.approx([0.001, 0.001])
    assert out["volume_eth"].tolist() == pytest.approx([1.0, 0.1])


def test_v2_pools_get_the_fixed_fee_without_a_lookup(tmp_path):
    swaps = pd.DataFrame([swap(1, "p1", PEPE, px.WETH, -1, 1)])
    fees = px.pool_fees(swaps, cache=tmp_path / "fees.parquet")
    assert fees["p1"] == px.V2_FEE and not (tmp_path / "fees.parquet").exists()
