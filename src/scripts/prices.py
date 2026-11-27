"""Prices from the swaps themselves (work package 8, implementation plan section 12.2).

Every swap against a base currency (WETH, USDC or USDT) gives a price: the
base amount paid or received per unit of the other token. Amounts in the
raw swaps are in the token's smallest unit and signed from the pool's side
(positive = the pool received it), the same for V2 and V3 pools
(dex_swaps.py). Dividing by 10^decimals gives whole tokens.

    price_base = |base amount| / |token amount|           (base per token)
    side       = "buy" if the pool received the base, else "sell"
    price_eth  = price_base                  if the base is WETH
               = price_base / ETH price in USD  if the base is USDC or USDT

The ETH price comes from swaps between WETH and USDC or USDT: per block, the
volume-weighted mean of USD per ETH over swaps of at least 0.1 ETH (smaller
swaps carry more rounding and price impact), carried forward to blocks
without such a swap.

Assumptions, stated where they are used:
  - USDC and USDT are worth one dollar each.
  - A swap whose two tokens are both bases is an ETH price swap, not a
    token price; a swap without a base gets no price (its tokens are priced
    by their own base pools).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from eth_abi import decode, encode

import dex_swaps as ds
import token_data as td

BASE_FILE = td.PROJECT_DIR / "base_tokens.json"
PRICES_DIR = td.PROJECT_DIR / "prices"
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
STABLES = {"0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", "0xdac17f958d2ee523a2206206994597c13d831ec7"}  # USDC, USDT
BASES = {WETH} | STABLES
MIN_ETH_FOR_ETH_PRICE = 0.1
V2_FEE = 0.003  # every V2-style pool charges 0.3% per swap
SELECTOR_FEE = "0xddca3f43"  # fee() of a V3 pool, in hundredths of a basis point (3000 = 0.3%)



def check_bases(base_file: Path = BASE_FILE) -> None:
    """The prices assume the project's base currencies are exactly WETH, USDC and USDT (notebook 09); stop if not."""
    removed = {item["address"] for item in json.loads(base_file.read_text())["removed"]}
    assert removed == BASES, f"the project's base currencies are {removed}, not WETH, USDC and USDT"


# --------------------------------------------------------------------------
# Token decimals and pool fees
# --------------------------------------------------------------------------


def token_decimals(addresses) -> pd.Series:
    """decimals() of each token address, from the cached on-chain lookup (dex_swaps.load_token_metadata).

    Tokens whose decimals() call failed are left out: without decimals a
    token cannot be priced. The caller counts how many that is.
    """
    meta = ds.load_token_metadata(list(addresses), data_dir=td.PROJECT_DIR)
    meta = meta.dropna(subset=["decimals"])
    return meta.set_index("address")["decimals"].astype(int)


def fetch_pool_fees_multicall(pools, url: str = ds.TX_RPC, session=None, per_request: int = 300) -> pd.DataFrame:
    """fee() of each V3 pool through Multicall3: columns pool, fee (a fraction, 0.003 = 0.3%); None if it fails."""
    pools = list(dict.fromkeys(pools))
    rows = []
    for i in range(0, len(pools), per_request):
        part = pools[i : i + per_request]
        calls = [(pool, True, bytes.fromhex(SELECTOR_FEE[2:])) for pool in part]
        data = ds.SELECTOR_AGGREGATE3 + encode(["(address,bool,bytes)[]"], [calls]).hex()
        payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_call", "params": [{"to": ds.MULTICALL3, "data": data}, "latest"]}
        answers = decode(["(bool,bytes)[]"], bytes.fromhex(ds.rpc(url, payload, session)[2:]))[0]
        for pool, (ok, word) in zip(part, answers):
            fee = int.from_bytes(word, "big") / 1_000_000 if ok and len(word) == 32 else None
            rows.append({"pool": pool, "fee": fee if fee is not None and fee < 0.1 else None})
    return pd.DataFrame(rows, columns=["pool", "fee"])


def pool_fees(swaps: pd.DataFrame, cache: Path = td.PROJECT_DIR / "pool_fees.parquet") -> pd.Series:
    """The fee per swap of every pool in the swaps: 0.3% for V2 pools, fee() for V3 pools (cached)."""
    v3 = swaps.loc[swaps["version"] == "v3", "pool"].unique()
    known = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["pool", "fee"])
    missing = [p for p in v3 if p not in set(known["pool"])]
    if missing:
        session = requests.Session()
        known = pd.concat([known, fetch_pool_fees_multicall(missing, session=session)], ignore_index=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        known.to_parquet(cache)
    fees = pd.Series(V2_FEE, index=swaps["pool"].unique(), dtype=float)
    in_v3 = fees.index.isin(v3)
    if in_v3.any():
        fees.loc[in_v3] = known.set_index("pool")["fee"].reindex(fees.index[in_v3]).to_numpy(dtype=float)
    return fees


# --------------------------------------------------------------------------
# Prices per swap
# --------------------------------------------------------------------------


def with_pool_tokens(swaps: pd.DataFrame, pool_tokens: pd.DataFrame) -> pd.DataFrame:
    """The swaps with their pool's token0 and token1; swaps of pools without known tokens are dropped."""
    out = swaps.merge(pool_tokens, on="pool", how="inner")
    return out.dropna(subset=["token0", "token1"])


def swap_prices(swaps: pd.DataFrame, decimals: pd.Series) -> pd.DataFrame:
    """One row per swap between a base and another token, with the price in the base.

    `swaps` must carry token0 and token1 (with_pool_tokens). Swaps whose
    other token has no known decimals are dropped. Columns: block, tx_index,
    log_index, timestamp, pool, version, tx_from, token, base, side,
    token_amount, base_amount, price_base.
    """
    base0 = swaps["token0"].isin(BASES)
    base1 = swaps["token1"].isin(BASES)
    one_base = swaps[base0 ^ base1].copy()
    token_is_0 = ~one_base["token0"].isin(BASES)

    one_base["token"] = np.where(token_is_0, one_base["token0"], one_base["token1"])
    one_base["base"] = np.where(token_is_0, one_base["token1"], one_base["token0"])
    raw_token = np.where(token_is_0, one_base["amount0"], one_base["amount1"])
    raw_base = np.where(token_is_0, one_base["amount1"], one_base["amount0"])

    token_decimals_ = one_base["token"].map(decimals)
    base_decimals = one_base["base"].map(decimals)
    known = token_decimals_.notna() & base_decimals.notna() & (raw_token != 0) & (raw_base != 0)
    # A real swap moves the two tokens in opposite directions; anything else is not a trade price.
    opposite = np.sign(raw_token) == -np.sign(raw_base)
    keep = known.to_numpy() & opposite
    one_base = one_base[keep]
    token_amount = np.abs(raw_token[keep]) / 10.0 ** token_decimals_[keep].to_numpy()
    base_amount = np.abs(raw_base[keep]) / 10.0 ** base_decimals[keep].to_numpy()

    out = one_base[["block", "tx_index", "log_index", "timestamp", "pool", "version", "tx_from", "token", "base"]].copy()
    out["side"] = np.where(raw_base[keep] > 0, "buy", "sell")
    out["token_amount"] = token_amount
    out["base_amount"] = base_amount
    out["price_base"] = base_amount / token_amount
    return out.reset_index(drop=True)


def eth_usd_per_block(swaps: pd.DataFrame, decimals: pd.Series) -> pd.Series:
    """USD per ETH in each block that has a WETH/USDC or WETH/USDT swap of at least MIN_ETH_FOR_ETH_PRICE.

    Volume-weighted over the block's swaps. `swaps` must carry token0 and token1.
    """
    pair = swaps[
        ((swaps["token0"] == WETH) & swaps["token1"].isin(STABLES)) | ((swaps["token1"] == WETH) & swaps["token0"].isin(STABLES))
    ]
    weth_is_0 = pair["token0"] == WETH
    stable = np.where(weth_is_0, pair["token1"], pair["token0"])
    eth = np.abs(np.where(weth_is_0, pair["amount0"], pair["amount1"])) / 10.0 ** decimals[WETH]
    usd = np.abs(np.where(weth_is_0, pair["amount1"], pair["amount0"])) / 10.0 ** pd.Series(stable).map(decimals).to_numpy()
    table = pd.DataFrame({"block": pair["block"].to_numpy(), "eth": eth, "usd": usd})
    table = table[table["eth"] >= MIN_ETH_FOR_ETH_PRICE]
    per_block = table.groupby("block")[["eth", "usd"]].sum()
    return (per_block["usd"] / per_block["eth"]).rename("eth_usd")


def add_eth_prices(prices: pd.DataFrame, eth_usd: pd.Series) -> pd.DataFrame:
    """price_eth and volume_eth for every priced swap, with the ETH price carried forward from earlier blocks.

    Swaps before the first ETH price are dropped (they cannot be converted).
    """
    eth = eth_usd.sort_index()
    position = np.searchsorted(eth.index.to_numpy(), prices["block"].to_numpy(), side="right") - 1
    known = position >= 0
    out = prices[known].copy()
    rate = eth.to_numpy()[position[known]]
    in_weth = (out["base"] == WETH).to_numpy()
    out["eth_usd"] = rate
    out["price_eth"] = np.where(in_weth, out["price_base"], out["price_base"] / rate)
    out["volume_eth"] = np.where(in_weth, out["base_amount"], out["base_amount"] / rate)
    return out


# --------------------------------------------------------------------------
# Building the price tables
# --------------------------------------------------------------------------


def build_prices(first_block: int, last_block: int, chunks_per_part: int = 50, out_dir: Path = PRICES_DIR, verbose: bool = True) -> list[Path]:
    """Price tables for a block range, written in parts of `chunks_per_part` chunk files.

    Each part is one Parquet file prices_<first>_<last>.parquet; an existing
    part is not rebuilt. The ETH price of a part is carried forward from the
    part before it, so that its first blocks are converted too.
    """
    check_bases()
    files = ds.chunk_files(ds.DATA_DIR, first_block, last_block)
    out_dir.mkdir(parents=True, exist_ok=True)
    written, last_eth = [], None
    for i in range(0, len(files), chunks_per_part):
        part_files = files[i : i + chunks_per_part]
        swaps = pd.concat([pd.read_parquet(f) for f in part_files], ignore_index=True)
        swaps = swaps[(swaps["block"] >= first_block) & (swaps["block"] <= last_block)]
        # Pools first seen in these blocks are looked up on-chain and added to the cache.
        swaps = with_pool_tokens(swaps, ds.load_pool_tokens(sorted(set(swaps["pool"]))))
        path = out_dir / f"prices_{int(swaps['block'].min())}_{int(swaps['block'].max())}.parquet"
        decimals = token_decimals(set(swaps["token0"]) | set(swaps["token1"]))
        eth_usd = eth_usd_per_block(swaps, decimals)
        if last_eth is not None:
            eth_usd = pd.concat([last_eth, eth_usd]).sort_index()
        last_eth = eth_usd.iloc[-1:]
        if not path.exists():
            prices = add_eth_prices(swap_prices(swaps, decimals), eth_usd)
            prices.to_parquet(path)
        written.append(path)
        if verbose:
            print(f"  {path.name}: done", flush=True)
    return written


def load_prices(first_block: int | None = None, last_block: int | None = None, out_dir: Path = PRICES_DIR) -> pd.DataFrame:
    """The price tables overlapping a block range, in chain order."""
    parts = []
    for path in sorted(out_dir.glob("prices_*.parquet")):
        start, end = (int(x) for x in path.stem.split("_")[1:])
        if (last_block is None or start <= last_block) and (first_block is None or end >= first_block):
            parts.append(pd.read_parquet(path))
    prices = pd.concat(parts, ignore_index=True)
    if first_block is not None:
        prices = prices[prices["block"] >= first_block]
    if last_block is not None:
        prices = prices[prices["block"] <= last_block]
    return prices.sort_values(["block", "tx_index", "log_index"], ignore_index=True)
