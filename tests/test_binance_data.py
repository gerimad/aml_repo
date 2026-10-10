"""Tests for src/scripts/binance_data.py (no network access needed)."""

import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import binance_data as bd

ROW_MS = "1502942400000,4261.48,4313.62,4261.32,4308.83,47.18,1502945999999,202366.13,171,35.16,150952.47,0"
ROW_US = "1788220800000000,78581.3,78900.01,78562.0,78652.0,522.7,1788224399999999,41160252.3,90560,336.2,26474473.1,0"
HEADER = ",".join(bd.KLINE_COLUMNS)


class FakeResponse:
    def __init__(self, status_code, content=b""):
        self.status_code, self.content = status_code, content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeBucket:
    """Serves S3-style XML listings, two entries per page, to exercise pagination."""

    def __init__(self, entries):
        self.entries = entries  # prefix -> list of ("Prefix" | "Key", value)
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        items = sorted(e for e in self.entries[params["prefix"]] if e[1] > params["marker"])
        page, more = items[:2], len(items) > 2
        body = f"<Prefix>{params['prefix']}</Prefix>"
        body += "".join(
            f"<CommonPrefixes><Prefix>{v}</Prefix></CommonPrefixes>" if kind == "Prefix" else f"<Contents><Key>{v}</Key></Contents>"
            for kind, v in page
        )
        body += f"<IsTruncated>{'true' if more else 'false'}</IsTruncated>"
        if more:
            body += f"<NextMarker>{page[-1][1]}</NextMarker>"
        response = FakeResponse(200)
        response.text = body
        return response


ROOT = "data/spot/monthly/klines/"


def test_list_bucket_follows_pagination():
    bucket = FakeBucket({ROOT: [("Prefix", f"{ROOT}{s}/") for s in ["AAAUSDT", "BBBBTC", "CCCUSDT"]]})
    assert bd.list_bucket(ROOT, bucket) == [f"{ROOT}AAAUSDT/", f"{ROOT}BBBBTC/", f"{ROOT}CCCUSDT/"]
    assert bucket.calls == 2


def test_list_symbols_filters_by_quote():
    bucket = FakeBucket({ROOT: [("Prefix", f"{ROOT}{s}/") for s in ["CCCUSDT", "AAAUSDT", "BBBBTC"]]})
    assert bd.list_symbols("USDT", bucket) == ["AAAUSDT", "CCCUSDT"]
    assert bd.list_symbols("BTC", bucket) == ["BBBBTC"]


def test_list_months_parses_file_names():
    prefix = f"{ROOT}AAAUSDT/1h/"
    keys = [f"{prefix}AAAUSDT-1h-2020-12.zip", f"{prefix}AAAUSDT-1h-2020-12.zip.CHECKSUM", f"{prefix}AAAUSDT-1h-2021-01.zip"]
    bucket = FakeBucket({prefix: [("Key", k) for k in keys]})
    assert bd.list_months("AAAUSDT", "1h", bucket) == [(2020, 12), (2021, 1)]


class FakeSession:
    """Serves zipped CSVs from a dict of url -> csv text, 404 for the rest."""

    def __init__(self, pages):
        self.pages = pages

    def get(self, url, timeout=None):
        if url not in self.pages:
            return FakeResponse(404)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("data.csv", self.pages[url])
        return FakeResponse(200, buffer.getvalue())


def test_month_range_crosses_year_boundary():
    assert bd.month_range(date(2023, 11, 15), date(2024, 2, 1)) == [
        (2023, 11),
        (2023, 12),
        (2024, 1),
        (2024, 2),
    ]


def test_month_range_single_month():
    assert bd.month_range(date(2024, 5, 1), date(2024, 5, 31)) == [(2024, 5)]


def test_monthly_url():
    assert bd.monthly_url("BTCUSDT", "1h", 2017, 8) == (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2017-08.zip"
    )


def test_daily_url():
    assert bd.daily_url("ETHUSDT", "1d", date(2026, 10, 9)) == (
        "https://data.binance.vision/data/spot/daily/klines/ETHUSDT/1d/ETHUSDT-1d-2026-10-09.zip"
    )


def test_exclusion_reason():
    assert bd.exclusion_reason("BTCUSDT") is None
    assert bd.exclusion_reason("SUPERUSDT") is None  # ends in "UP" only by accident of spelling
    assert bd.exclusion_reason("JUPUSDT") is None
    assert bd.exclusion_reason("USDCUSDT") == "stablecoin or fiat"
    assert bd.exclusion_reason("EURUSDT") == "stablecoin or fiat"
    assert bd.exclusion_reason("BTCUPUSDT") == "leveraged token"
    assert bd.exclusion_reason("ETHBEARUSDT") == "leveraged token"
    assert bd.exclusion_reason("BEARUSDT") == "leveraged token"
    assert bd.exclusion_reason("PAXGUSDT") == "pegged to another asset"
    assert bd.exclusion_reason("LUNAUSDT") == "ticker reused for a different asset"


def test_parse_kline_csv_milliseconds():
    df = bd.parse_kline_csv(ROW_MS.encode())
    assert df.index[0] == pd.Timestamp("2017-08-17 04:00", tz="UTC")
    assert df["close"].iloc[0] == pytest.approx(4308.83)
    assert list(df.columns) == bd.KEEP_COLUMNS


def test_parse_kline_csv_microseconds():
    df = bd.parse_kline_csv(ROW_US.encode())
    assert df.index[0] == pd.Timestamp("2026-09-01 00:00", tz="UTC")


def test_parse_kline_csv_skips_header():
    df = bd.parse_kline_csv(f"{HEADER}\n{ROW_MS}\n".encode())
    assert len(df) == 1
    assert df["open"].iloc[0] == pytest.approx(4261.48)


def test_fetch_zip_returns_csv_and_none_for_missing():
    session = FakeSession({"https://x/a.zip": ROW_MS})
    assert bd.fetch_zip("https://x/a.zip", session) == ROW_MS.encode()
    assert bd.fetch_zip("https://x/missing.zip", session) is None


def test_download_klines_joins_monthly_and_daily(monkeypatch):
    pages = {
        bd.monthly_url("BTCUSDT", "1h", 2017, 8): ROW_MS,
        bd.daily_url("BTCUSDT", "1h", date(2017, 9, 1)): ROW_MS.replace(
            "1502942400000", "1504224000000"
        ),
    }
    monkeypatch.setattr(bd.requests, "Session", lambda: FakeSession(pages))
    df = bd.download_klines("BTCUSDT", "1h", date(2017, 7, 1), date(2017, 9, 2), workers=2)
    assert list(df.index) == [
        pd.Timestamp("2017-08-17 04:00", tz="UTC"),
        pd.Timestamp("2017-09-01 00:00", tz="UTC"),
    ]


def test_download_klines_raises_when_nothing_published(monkeypatch):
    monkeypatch.setattr(bd.requests, "Session", lambda: FakeSession({}))
    with pytest.raises(ValueError):
        bd.download_klines("NOPE", "1h", date(2020, 1, 1), date(2020, 1, 3), workers=2)


def test_klines_path_and_load_roundtrip(tmp_path: Path):
    df = bd.parse_kline_csv(ROW_MS.encode())
    path = bd.klines_path("BTCUSDT", "1h", tmp_path)
    assert path == tmp_path / "binance_spot_1h" / "BTCUSDT_1h.parquet"
    path.parent.mkdir(parents=True)
    df.to_parquet(path)
    pd.testing.assert_frame_equal(bd.load_klines("BTCUSDT", "1h", tmp_path), df, check_freq=False)


def test_main_writes_one_file_per_symbol(monkeypatch, tmp_path: Path, capsys):
    df = bd.parse_kline_csv(ROW_MS.encode())
    monkeypatch.setattr(bd, "download_klines", lambda *args, **kwargs: df)
    monkeypatch.setattr(bd, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        bd, "klines_path", lambda s, i: tmp_path / f"binance_spot_{i}" / f"{s}_{i}.parquet"
    )
    monkeypatch.setattr("sys.argv", ["binance_data.py", "--symbols", "AAA", "BBB"])
    bd.main()
    assert sorted(p.name for p in (tmp_path / "binance_spot_1h").iterdir()) == [
        "AAA_1h.parquet",
        "BBB_1h.parquet",
    ]
    assert "AAA: 1 rows" in capsys.readouterr().out
