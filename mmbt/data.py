"""Download Binance USD-M futures daily files and turn them into two compact tables per symbol-day.

Source: https://data.binance.vision (public, no key). Per symbol and day:
  bookTicker  every best bid/ask change    -> grid: best bid/ask as of every 100 ms (last update at or before t)
  aggTrades   every aggressive trade        -> trades: time, price, quantity, taker side

Each zip is verified against Binance's published SHA-256 checksum before use.
Futures bookTicker files are published up to 2024-03-30, so the study uses the 7 days 2024-03-24 to 2024-03-30.

    python -m mmbt.data            # downloads about 1.2 GB, writes data/parquet/*.parquet and reports/data_manifest.csv
"""
from __future__ import annotations

import hashlib
import io
import os
import time
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import polars as pl

BASE = "https://data.binance.vision/data/futures/um/daily"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DAYS = [str(date(2024, 3, 24) + timedelta(days=i)) for i in range(7)]
GRID_MS = 100
ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("MMBT_DATA", ROOT / "data")) / "raw"
PQ = Path(os.environ.get("MMBT_DATA", ROOT / "data")) / "parquet"


def fetch(kind: str, sym: str, day: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    name = f"{sym}-{kind}-{day}.zip"
    dest = RAW / name
    url = f"{BASE}/{kind}/{sym}/{name}"
    want = urllib.request.urlopen(url + ".CHECKSUM").read().decode().split()[0]
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == want:
        return dest
    for attempt in range(3):
        data = urllib.request.urlopen(url, timeout=300).read()
        if hashlib.sha256(data).hexdigest() == want:
            dest.write_bytes(data)
            return dest
        time.sleep(2)
    raise IOError(f"checksum mismatch for {name}")


def read_zip_csv(path: Path) -> pl.DataFrame:
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    return pl.read_csv(io.BytesIO(raw))


def build(sym: str, day: str) -> dict:
    PQ.mkdir(parents=True, exist_ok=True)
    out_grid, out_tr = PQ / f"{sym}-{day}-grid.parquet", PQ / f"{sym}-{day}-trades.parquet"
    bt = read_zip_csv(fetch("bookTicker", sym, day)).select(
        pl.col("transaction_time").alias("t"), pl.col("best_bid_price").alias("bid"),
        pl.col("best_ask_price").alias("ask")).sort("t")
    n_quotes = bt.height
    day0 = int(pl.Series([day]).str.to_datetime("%Y-%m-%d").dt.epoch("ms")[0])
    grid = pl.DataFrame({"t": pl.int_range(day0 + GRID_MS, day0 + 86_400_000 + 1, GRID_MS, eager=True)})
    grid = grid.join_asof(bt, on="t", strategy="backward").drop_nulls()
    grid = grid.filter(pl.col("ask") > pl.col("bid"))
    grid.write_parquet(out_grid)
    ag = read_zip_csv(fetch("aggTrades", sym, day))
    n_trades = int((ag["last_trade_id"] - ag["first_trade_id"] + 1).sum())
    tr = ag.select(pl.col("transact_time").alias("t"), pl.col("price"), pl.col("quantity").alias("qty"),
                   # is_buyer_maker = the taker sold, so the trade hit bids: side -1
                   pl.when(pl.col("is_buyer_maker")).then(-1).otherwise(1).cast(pl.Int8).alias("side")
                   ).filter((pl.col("t") >= day0) & (pl.col("t") < day0 + 86_400_000)).sort("t", maintain_order=True)
    tr.write_parquet(out_tr)
    return {"symbol": sym, "day": day, "book_updates": n_quotes, "agg_trades": ag.height, "trades": n_trades,
            "grid_rows": grid.height, "notional_usd": float((tr["price"] * tr["qty"]).sum())}


def load(sym: str, day: str):
    g = pl.read_parquet(PQ / f"{sym}-{day}-grid.parquet")
    t = pl.read_parquet(PQ / f"{sym}-{day}-trades.parquet")
    return g, t


if __name__ == "__main__":
    rows = []
    t0 = time.time()
    for sym in SYMBOLS:
        for day in DAYS:
            r = build(sym, day)
            rows.append(r)
            print(r, flush=True)
    m = pl.DataFrame(rows)
    (ROOT / "reports").mkdir(exist_ok=True)
    m.write_csv(ROOT / "reports" / "data_manifest.csv")
    print(m.select(pl.sum("book_updates", "agg_trades", "trades")), f"{time.time() - t0:.0f} s")
