"""Load and score Polymarket's five-minute "Bitcoin Up or Down" markets.

The market data is the public dataset
kachoio/polymarket-5-minute-crypto-up-down-markets on Hugging Face: one row
per market and one row per second with the best bid and ask of the "Up" and
"Down" shares. Spot prices come from Binance one-second candles.

Download once, from the repository root:

    uv run python src/scripts/prediction_markets.py

Then, in a notebook in src/notebooks:

    import sys; sys.path.append("../scripts")
    import prediction_markets as pm

    markets, ticks = pm.load_markets(), pm.load_ticks()
    table = pm.quotes_at(markets, ticks, seconds=[0, 60, 299])   # mid and spread per market
    spot = pm.load_spot_seconds()                                # BTCUSDT close per second
    loss = pm.log_loss(table["target"], table["mid_60"])         # per market, in bits
    pm.reliability_table(table["mid_60"], table["target"])

A forecast is always the probability of "Up". Targets may be hard (0 or 1) or
soft (a probability), see `add_targets`.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import binance_data as bd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MARKET_DIR = DATA_DIR / "polymarket_5m_kachoio"
HF_REPO = "kachoio/polymarket-5-minute-crypto-up-down-markets"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
SPOT_FILE = DATA_DIR / "binance_spot_1s" / "BTCUSDT_1s.parquet"
MARKET_SECONDS = 300


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def download_markets(coin: str = "btc", data_dir: Path = MARKET_DIR, session=None) -> None:
    """Download the markets and ticks files of one coin from Hugging Face, unless present."""
    import requests

    data_dir.mkdir(parents=True, exist_ok=True)
    for name in (f"{coin}_markets.parquet", f"{coin}_ticks.parquet"):
        path = data_dir / name
        if path.exists():
            continue
        response = (session or requests).get(f"{HF_URL}/{name}", timeout=600)
        response.raise_for_status()
        path.write_bytes(response.content)


def load_markets(coin: str = "btc", data_dir: Path = MARKET_DIR) -> pd.DataFrame:
    """One row per market, in time order, indexed by condition_id.

    `outcome` is "Up", "Down" or missing. In this dataset it was inferred
    from the last recorded quote, not read from the on-chain resolution, and
    it is missing when that quote was not decisive.
    """
    markets = pd.read_parquet(data_dir / f"{coin}_markets.parquet")
    return markets.sort_values("market_start").set_index("condition_id")


def load_ticks(coin: str = "btc", data_dir: Path = MARKET_DIR) -> pd.DataFrame:
    """One row per second and market: best bid and ask of the Up and the Down share."""
    return pd.read_parquet(data_dir / f"{coin}_ticks.parquet")


def quotes_at(markets: pd.DataFrame, ticks: pd.DataFrame, seconds) -> pd.DataFrame:
    """Per market, the Up share's mid price and spread at given seconds into the market.

    Returns a frame indexed like `markets` with columns mid_<s>, spread_<s>,
    bid_<s> and ask_<s> for each requested second (0 = the market's first
    second, 299 = its last). The quote at second s is what was resting in the
    book at that moment, so it reflects information up to s and no later.
    A market without a tick at that second gets NaN.
    """
    start = markets["market_start"].map(lambda ts: int(ts.timestamp()))
    offset = ticks["t"].to_numpy() - ticks["condition_id"].map(start).to_numpy()
    out = pd.DataFrame(index=markets.index)
    for second in seconds:
        at = ticks[offset == second].set_index("condition_id")
        out[f"bid_{second}"] = at["bu"]
        out[f"ask_{second}"] = at["au"]
        out[f"mid_{second}"] = (at["bu"] + at["au"]) / 2
        out[f"spread_{second}"] = at["au"] - at["bu"]
    return out


def add_targets(markets: pd.DataFrame, final_mid: pd.Series) -> pd.DataFrame:
    """Add the forecasting targets to a markets frame.

    hard    1 for "Up", 0 for "Down", NaN where the outcome is not recorded
    target  the hard label where there is one; otherwise the Up share's mid
            price in the market's last second. One second before the end that
            price is the best available estimate of the outcome, and scoring
            against it (a "soft label") keeps the undecided markets in the
            sample instead of dropping exactly the close calls.
    """
    out = markets.copy()
    out["hard"] = out["outcome"].map({"Up": 1.0, "Down": 0.0})
    out["target"] = out["hard"].fillna(final_mid)
    return out


# --------------------------------------------------------------------------
# Spot prices at one-second resolution
# --------------------------------------------------------------------------


def download_spot_seconds(start: date, end: date, symbol: str = "BTCUSDT", path: Path = SPOT_FILE):
    """Download Binance one-second candles for start..end (inclusive) into one Parquet file.

    Keeps the close and the traded volume of every second. Uses the daily
    files of data.binance.vision, one request per day.
    """
    import requests

    session = requests.Session()
    frames = []
    day = start
    while day <= end:
        raw = bd.fetch_zip(bd.daily_url(symbol, "1s", day), session)
        if raw is not None:
            frames.append(bd.parse_kline_csv(raw)[["close", "volume"]].astype("float32"))
        day += timedelta(days=1)
    if not frames:
        raise ValueError(f"no one-second data published for {symbol} between {start} and {end}")
    spot = pd.concat(frames).sort_index()
    spot = spot[~spot.index.duplicated()]
    path.parent.mkdir(parents=True, exist_ok=True)
    spot.to_parquet(path)
    return spot


def load_spot_seconds(path: Path = SPOT_FILE) -> pd.Series:
    """Spot close per second as a gap-free series indexed by unix time.

    Seconds without a trade carry the previous close forward. The value at
    unix second s is the last trade price of the second s (that is, up to
    just before s + 1).
    """
    spot = pd.read_parquet(path)["close"].astype("float64")
    seconds = np.asarray(spot.index.as_unit("s").asi8)
    series = pd.Series(spot.to_numpy(), index=seconds)
    full = np.arange(seconds[0], seconds[-1] + 1)
    return series.reindex(full).ffill()


def spot_before(spot: pd.Series, times) -> np.ndarray:
    """The last spot price known strictly before each unix time (NaN if out of range).

    The close of second s - 1 is complete at time s, so this is the newest
    price a forecast made at time s may use.
    """
    return spot.reindex(np.asarray(times, dtype="int64") - 1).to_numpy()


def trailing_volatility(spot: pd.Series, times, window: int = 3600, step: int = 60) -> np.ndarray:
    """Standard deviation of `step`-second log returns over the `window` seconds before each time.

    Uses prices strictly before the time only. Returned per sqrt(second), so
    that volatility over h seconds is the value times sqrt(h).
    """
    times = np.asarray(times, dtype="int64")
    n_steps = window // step
    offsets = np.arange(n_steps + 1) * step  # 0, step, ..., window seconds back
    grid = times[:, None] - 1 - offsets[None, :]
    prices = spot.reindex(grid.ravel()).to_numpy().reshape(grid.shape)
    returns = np.diff(np.log(prices[:, ::-1]), axis=1)
    return returns.std(axis=1, ddof=1) / np.sqrt(step)


# --------------------------------------------------------------------------
# Scoring probability forecasts
# --------------------------------------------------------------------------


def log_loss(target, forecast, eps: float = 1e-4) -> np.ndarray:
    """Per-event log loss in bits: -(t log2 p + (1 - t) log2 (1 - p)).

    `target` may be 0 / 1 or a probability (soft label). Forecasts are
    clipped to [eps, 1 - eps] so that a single certain-and-wrong forecast
    cannot make the mean infinite. A forecast of 0.5 costs exactly 1 bit.
    """
    t = np.asarray(target, float)
    p = np.clip(np.asarray(forecast, float), eps, 1 - eps)
    return -(t * np.log2(p) + (1 - t) * np.log2(1 - p))


def brier(target, forecast) -> np.ndarray:
    """Per-event Brier score: (forecast - target) squared. A forecast of 0.5 costs 0.25."""
    return (np.asarray(forecast, float) - np.asarray(target, float)) ** 2


def reliability_table(forecast, target, edges=None) -> pd.DataFrame:
    """Group forecasts into bins and compare the mean forecast with what happened.

    One row per non-empty bin: n, mean_forecast, observed (mean target),
    gap (observed - mean_forecast), and se, the standard error of the
    observed frequency. A calibrated forecaster has gaps within about two se.
    """
    forecast, target = np.asarray(forecast, float), np.asarray(target, float)
    edges = np.linspace(0, 1, 11) if edges is None else np.asarray(edges, float)
    which = np.clip(np.digitize(forecast, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for i in range(len(edges) - 1):
        inside = which == i
        if not inside.any():
            continue
        observed, n = target[inside].mean(), int(inside.sum())
        rows.append({
            "bin": f"{edges[i]:.2f}-{edges[i + 1]:.2f}", "n": n,
            "mean_forecast": forecast[inside].mean(), "observed": observed,
            "gap": observed - forecast[inside].mean(),
            "se": np.sqrt(max(observed * (1 - observed), 1e-12) / n),
        })  # fmt: skip
    return pd.DataFrame(rows)


def brier_decomposition(forecast, target, edges=None) -> dict[str, float]:
    """Murphy's decomposition of the mean Brier score over forecast bins.

    brier is approximately reliability - resolution + uncertainty, where
    reliability is the miscalibration (0 is best), resolution is how much the
    forecasts separate the outcomes (higher is better), and uncertainty is
    the variance of the outcome, which no forecaster controls.
    """
    table = reliability_table(forecast, target, edges)
    weights = table["n"] / table["n"].sum()
    base = float(np.mean(target))
    return {
        "brier": float(np.mean(brier(target, forecast))),
        "reliability": float((weights * table["gap"] ** 2).sum()),
        "resolution": float((weights * (table["observed"] - base) ** 2).sum()),
        "uncertainty": base * (1 - base),
    }


def logit(p, eps: float = 1e-4) -> np.ndarray:
    """Log odds of a probability, clipped away from 0 and 1."""
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def chrono_split(markets: pd.DataFrame, train_end: str, dev_end: str) -> pd.Series:
    """Label each market "train", "dev" or "test" by its start time.

    `train_end` and `dev_end` are the first instants (UTC) of dev and of
    test. Markets are never shuffled.
    """
    start = markets["market_start"]
    first_dev, first_test = pd.Timestamp(train_end, tz="UTC"), pd.Timestamp(dev_end, tz="UTC")
    if not first_dev < first_test:
        raise ValueError("train_end must be before dev_end")
    labels = np.select([start < first_dev, start < first_test], ["train", "dev"], "test")
    return pd.Series(labels, index=markets.index, name="split")


if __name__ == "__main__":
    download_markets()
    all_markets = load_markets()
    first, last = all_markets["market_start"].min().date(), all_markets["market_end"].max().date()
    if not SPOT_FILE.exists():
        download_spot_seconds(first - timedelta(days=1), last)
    print(f"{len(all_markets):,} markets from {first} to {last}; spot seconds in {SPOT_FILE}")
