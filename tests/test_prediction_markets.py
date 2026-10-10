"""Tests for src/scripts/prediction_markets.py (no network access)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

import prediction_markets as pm

T0 = 1_774_390_200  # a market start, unix seconds


def toy_markets():
    starts = pd.to_datetime([T0, T0 + 300, T0 + 600], unit="s", utc=True)
    return pd.DataFrame({
        "condition_id": ["c", "a", "b"], "market_start": [starts[2], starts[0], starts[1]],
        "market_end": [starts[2] + pd.Timedelta(minutes=5), starts[0] + pd.Timedelta(minutes=5), starts[1] + pd.Timedelta(minutes=5)],
        "outcome": [None, "Up", "Down"],
    })  # fmt: skip


def toy_ticks():
    rows = []
    for market, start in [("a", T0), ("b", T0 + 300), ("c", T0 + 600)]:
        for second in (0, 60, 299):
            bid = {"a": 0.50, "b": 0.40, "c": 0.60}[market] + (0.2 if second == 299 and market == "a" else 0)
            rows.append({"condition_id": market, "t": start + second, "bu": bid, "au": bid + 0.02})
    return pd.DataFrame(rows).iloc[:-1]  # market c has no tick at second 299


class FakeResponse:
    content = b"parquet bytes"

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self):
        self.urls = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        return FakeResponse()


def test_download_markets_skips_existing_files(tmp_path):
    (tmp_path / "btc_markets.parquet").write_bytes(b"already here")
    session = FakeSession()
    pm.download_markets("btc", tmp_path, session)
    assert session.urls == [f"{pm.HF_URL}/btc_ticks.parquet"]
    assert (tmp_path / "btc_ticks.parquet").read_bytes() == b"parquet bytes"
    assert (tmp_path / "btc_markets.parquet").read_bytes() == b"already here"


def test_load_markets_sorts_by_time_and_load_ticks(tmp_path):
    toy_markets().to_parquet(tmp_path / "btc_markets.parquet")
    toy_ticks().to_parquet(tmp_path / "btc_ticks.parquet")
    markets = pm.load_markets("btc", tmp_path)
    assert markets.index.tolist() == ["a", "b", "c"] and markets.index.name == "condition_id"
    assert len(pm.load_ticks("btc", tmp_path)) == 8


def test_quotes_at():
    markets = toy_markets().sort_values("market_start").set_index("condition_id")
    table = pm.quotes_at(markets, toy_ticks(), [0, 299])
    assert table.loc["a", "mid_0"] == pytest.approx(0.51) and table.loc["a", "spread_0"] == pytest.approx(0.02)
    assert table.loc["a", "bid_299"] == pytest.approx(0.70) and table.loc["a", "ask_299"] == pytest.approx(0.72)
    assert table.loc["b", "mid_0"] == pytest.approx(0.41)
    assert np.isnan(table.loc["c", "mid_299"])  # no tick at that second
    assert table.index.equals(markets.index)


def test_add_targets_uses_final_mid_only_where_outcome_is_missing():
    markets = toy_markets().sort_values("market_start").set_index("condition_id")
    final_mid = pd.Series({"a": 0.99, "b": 0.01, "c": 0.62})
    out = pm.add_targets(markets, final_mid)
    assert out["hard"].tolist()[:2] == [1.0, 0.0] and np.isnan(out.loc["c", "hard"])
    assert out["target"].tolist() == [1.0, 0.0, 0.62]


def write_spot(path, seconds, prices):
    index = pd.to_datetime(seconds, unit="s", utc=True)
    pd.DataFrame({"close": np.asarray(prices, "float32"), "volume": 1.0}, index=index).to_parquet(path)


def test_load_spot_seconds_fills_gaps_forward(tmp_path):
    write_spot(tmp_path / "spot.parquet", [100, 101, 104], [10.0, 11.0, 14.0])
    spot = pm.load_spot_seconds(tmp_path / "spot.parquet")
    assert spot.index.tolist() == [100, 101, 102, 103, 104]
    assert spot.tolist() == [10.0, 11.0, 11.0, 11.0, 14.0]


def test_spot_before_never_uses_the_current_second():
    spot = pd.Series([10.0, 11.0, 12.0], index=[100, 101, 102])
    assert pm.spot_before(spot, [101, 102, 103]).tolist() == [10.0, 11.0, 12.0]
    assert np.isnan(pm.spot_before(spot, [100])[0])  # nothing is known before the first second


def test_trailing_volatility_recovers_known_volatility_and_is_causal():
    rng = np.random.default_rng(0)
    per_second = 0.001
    log_price = np.cumsum(rng.normal(0, per_second, 40_000))
    spot = pd.Series(np.exp(log_price), index=np.arange(40_000))
    estimate = pm.trailing_volatility(spot, [30_000, 35_000], window=20_000, step=50)
    assert estimate == pytest.approx([per_second, per_second], rel=0.15)
    changed = spot.copy()
    changed.loc[30_000:] = 1e6  # the future must not matter
    assert pm.trailing_volatility(changed, [30_000], window=20_000, step=50)[0] == pytest.approx(estimate[0])


def test_download_spot_seconds(tmp_path, monkeypatch):
    def fake_fetch(url, session=None):
        if "2026-03-25" in url:
            return None  # a day that is not published
        day = 0 if "2026-03-24" in url else 1
        first = 1_774_310_400_000 + day * 2 * 86_400_000
        return f"{first},1,1,1,{100 + day},5,0,0,0,0,0,0\n{first + 1000},1,1,1,{200 + day},7,0,0,0,0,0,0".encode()

    monkeypatch.setattr(pm.bd, "fetch_zip", fake_fetch)
    path = tmp_path / "spot" / "BTCUSDT_1s.parquet"
    spot = pm.download_spot_seconds(date(2026, 3, 24), date(2026, 3, 26), path=path)
    assert spot["close"].tolist() == [100.0, 200.0, 101.0, 201.0] and path.exists()
    assert list(spot.columns) == ["close", "volume"]
    monkeypatch.setattr(pm.bd, "fetch_zip", lambda url, session=None: None)
    with pytest.raises(ValueError):
        pm.download_spot_seconds(date(2026, 3, 24), date(2026, 3, 24), path=path)


def test_log_loss():
    assert pm.log_loss([1, 0], [0.5, 0.5]).tolist() == [1.0, 1.0]
    assert pm.log_loss([1], [0.25])[0] == pytest.approx(2.0)
    # A soft target is scored by its expectation, and is minimized by forecasting the target itself.
    assert pm.log_loss([0.7], [0.7])[0] < pm.log_loss([0.7], [0.6])[0]
    assert pm.log_loss([0.7], [0.7])[0] < pm.log_loss([0.7], [0.8])[0]
    assert np.isfinite(pm.log_loss([1], [0.0])[0])  # clipped, not infinite


def test_brier():
    assert pm.brier([1, 0, 1], [0.5, 0.5, 1.0]).tolist() == [0.25, 0.25, 0.0]


def test_reliability_table():
    forecast = np.array([0.1] * 100 + [0.9] * 100)
    target = np.array([0] * 90 + [1] * 10 + [1] * 70 + [0] * 30)
    table = pm.reliability_table(forecast, target, edges=[0, 0.5, 1])
    assert table["n"].tolist() == [100, 100]
    assert table["observed"].tolist() == pytest.approx([0.1, 0.7])
    assert table["gap"].tolist() == pytest.approx([0.0, -0.2])
    assert table.loc[1, "se"] == pytest.approx(np.sqrt(0.7 * 0.3 / 100))
    # Empty bins are left out, and a forecast of exactly 1 lands in the last bin.
    assert len(pm.reliability_table([0.05, 1.0], [0, 1])) == 2


def test_brier_decomposition_adds_up_for_binned_forecasts():
    rng = np.random.default_rng(0)
    forecast = rng.choice([0.2, 0.5, 0.8], 5000)
    target = (rng.random(5000) < forecast).astype(float)
    parts = pm.brier_decomposition(forecast, target, edges=[0, 0.3, 0.6, 1])
    assert parts["brier"] == pytest.approx(parts["reliability"] - parts["resolution"] + parts["uncertainty"], abs=1e-9)
    assert parts["reliability"] < 0.001  # the forecasts are calibrated by construction
    assert parts["resolution"] > 0.04


def test_logit():
    assert pm.logit([0.5])[0] == pytest.approx(0.0)
    assert pm.logit([0.9])[0] == pytest.approx(-pm.logit([0.1])[0])
    assert np.isfinite(pm.logit([0.0, 1.0])).all()


def test_chrono_split():
    markets = pd.DataFrame({"market_start": pd.to_datetime(["2026-04-01", "2026-04-10", "2026-04-20"], utc=True)})
    assert pm.chrono_split(markets, "2026-04-10", "2026-04-15").tolist() == ["train", "dev", "test"]
    with pytest.raises(ValueError):
        pm.chrono_split(markets, "2026-04-15", "2026-04-10")
