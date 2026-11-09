"""Walk-forward study: tune on days 1-4, report days 5-7 only.

    python -m mmbt.experiment           # writes reports/*.csv, reports/*.json, reports/timeseries/*.parquet

Strategies (see mmbt/sim.py): S0 naive symmetric, S1 Avellaneda-Stoikov with constant sigma, S2 S1 + live
volatility and hard inventory limits. Every strategy, the baseline included, gets its parameters from the same
grid search on the training days. Objective: the highest net PnL after fees among settings that meet a volume
obligation of $5M traded per symbol per day on average, the way an exchange market-maker program requires
participation. Without it the "best" quoter is one so wide it never trades. Quote size 1 lot = $1,000; maker fee
0.02% (Binance USD-M futures, regular tier); quote latency 50 ms unless stated; each day starts flat and the
end-of-day inventory is marked to mid, not liquidated.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from .analytics import combine, summarize
from .data import DAYS, SYMBOLS, load
from .sim import S0, S1, S2, ewma_sigma, run

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
TRAIN, TEST = DAYS[:4], DAYS[4:]
TICK = {"BTCUSDT": 0.1, "ETHUSDT": 0.01}
LOT_USD, FEE, TAU_S, HALFLIFE_TICKS, LATENCY = 1000.0, 0.0002, 30.0, 600, 50
VOL_OBLIGATION_PER_DAY = 5e6
ALT_FEES = {"fee_0bp": 0.0, "rebate_0.5bp": -0.00005}      # what-if fee tiers, applied to the same fills
NAMES = {S0: "S0 naive", S1: "S1 Avellaneda-Stoikov", S2: "S2 AS + live vol + limits"}
GRID = {
    S0: [dict(half_bps=h) for h in (0.5, 1, 2, 3, 5, 8, 12)],
    S1: [dict(gamma=g, kappa=k) for g, k in itertools.product((0.01, 0.03, 0.1, 0.3, 1.0), (0.1, 0.3, 1.0, 3.0))],
    S2: [dict(gamma=g, kappa=k, max_lots=m) for g, k, m in
         itertools.product((0.01, 0.03, 0.1, 0.3, 1.0), (0.1, 0.3, 1.0, 3.0), (3, 10))],
}
_cache: dict = {}


def day_arrays(sym, day):
    key = (sym, day)
    if key not in _cache:
        g, t = load(sym, day)
        d = {"g_t": g["t"].to_numpy(), "bid": g["bid"].to_numpy(), "ask": g["ask"].to_numpy(),
             "tr_t": t["t"].to_numpy(), "tr_p": t["price"].to_numpy(), "tr_q": t["qty"].to_numpy(),
             "tr_side": t["side"].to_numpy()}
        d["mid"] = 0.5 * (d["bid"] + d["ask"])
        r = np.diff(np.log(d["mid"]))
        d["day_sigma"] = float(np.sqrt(np.mean(r * r) * 10) * 1e4)      # per-sqrt-second, bps
        _cache[key] = d
    return _cache[key]


def train_sigma(sym):
    """S1's constant sigma: median daily realized vol over the training days only."""
    return float(np.median([day_arrays(sym, d)["day_sigma"] for d in TRAIN]))


def run_day(sym, day, strat, p, latency=LATENCY):
    d = day_arrays(sym, day)
    if strat == S1:
        sig = np.full(len(d["g_t"]), train_sigma(sym))
    elif strat == S2:
        # warm-start the EWMA with the previous training estimate (no look-ahead into this day)
        warm = (train_sigma(sym) / 1e4) ** 2 / 10
        sig = ewma_sigma(d["mid"], HALFLIFE_TICKS, 10.0, warm)
    else:
        sig = np.zeros(len(d["g_t"]))
    res = run(d["g_t"], d["bid"], d["ask"], sig, d["tr_t"], d["tr_p"], d["tr_q"], d["tr_side"],
              strat, p.get("half_bps", 0.0), p.get("gamma", 1.0), p.get("kappa", 1.0), TAU_S,
              p.get("max_lots", 1e9), LOT_USD, TICK[sym], latency, FEE)
    return summarize(d["g_t"], d["mid"], res, LOT_USD), res


def timeseries(sym, day, strat, res):
    """1-minute series for the figures: equity, inventory, cumulative spread capture and fees."""
    d = day_arrays(sym, day)
    f_t, f_side, f_px, f_qty, f_mid, inv, cash, fees = res
    g_t, mid = d["g_t"], d["mid"]
    idx = np.arange(599, len(g_t), 600)
    edge = np.cumsum(f_side * (f_mid - f_px) * f_qty)
    fee = np.cumsum(f_px * f_qty * FEE)
    k = np.searchsorted(f_t, g_t[idx], side="right") - 1
    cum_edge = np.where(k >= 0, edge[np.clip(k, 0, None)] if len(edge) else 0.0, 0.0)
    cum_fee = np.where(k >= 0, fee[np.clip(k, 0, None)] if len(fee) else 0.0, 0.0)
    eq = cash[idx] + inv[idx] * mid[idx]
    return pl.DataFrame({"t": g_t[idx], "symbol": sym, "strategy": NAMES[strat], "mid": mid[idx],
                         "inventory_usd": inv[idx] * mid[idx], "net_pnl": eq, "spread_capture": cum_edge,
                         "fees": cum_fee, "inventory_pnl": eq + cum_fee - cum_edge})


def main():
    t0 = time.time()
    REP.mkdir(exist_ok=True)
    (REP / "timeseries").mkdir(exist_ok=True)
    tuning, best = [], {}
    for sym in SYMBOLS:
        for strat in (S0, S1, S2):
            scores = []
            for p in GRID[strat]:
                rows = [run_day(sym, d, strat, p)[0] for d in TRAIN]
                c = combine(rows)
                tuning.append({"symbol": sym, "strategy": NAMES[strat], **p,
                               **{k: v for k, v in c.items() if not isinstance(v, np.ndarray)}})
                scores.append((c["volume_usd"] / len(TRAIN) >= VOL_OBLIGATION_PER_DAY, c["net_pnl"],
                               c["volume_usd"], p))
            ok = [x for x in scores if x[0]]
            best[(sym, strat)] = (max(ok, key=lambda x: x[1]) if ok else max(scores, key=lambda x: x[2]))[3]
            print(sym, NAMES[strat], "best on train:", best[(sym, strat)], f"{time.time() - t0:.0f}s", flush=True)
    pl.DataFrame(tuning).write_csv(REP / "tuning_train.csv")

    # out-of-sample, latency 50 ms
    results, per_day, ts = [], [], []
    for sym in SYMBOLS:
        for strat in (S0, S1, S2):
            rows = []
            for d in TEST:
                s, res = run_day(sym, d, strat, best[(sym, strat)])
                rows.append(s)
                per_day.append({"symbol": sym, "day": d, "strategy": NAMES[strat],
                                **{k: v for k, v in s.items() if not isinstance(v, np.ndarray)}})
                ts.append(timeseries(sym, d, strat, res))
            c = combine(rows)
            c["gross_pnl"] = c["net_pnl"] + c["fees"]
            for name, f in ALT_FEES.items():
                c[f"net_pnl_{name}"] = c["gross_pnl"] - f * c["volume_usd"]
            results.append({"symbol": sym, "strategy": NAMES[strat], "params": json.dumps(best[(sym, strat)]),
                            **{k: v for k, v in c.items() if not isinstance(v, np.ndarray)}})
    pl.DataFrame(results).write_csv(REP / "results_test.csv")
    pl.DataFrame(per_day).write_csv(REP / "results_test_by_day.csv")
    pl.concat(ts).write_parquet(REP / "timeseries" / "test_1min.parquet")

    # latency sensitivity on the test days
    lat = []
    for sym in SYMBOLS:
        for strat in (S0, S1, S2):
            for L in (0, 50, 200):
                c = combine([run_day(sym, d, strat, best[(sym, strat)], L)[0] for d in TEST])
                lat.append({"symbol": sym, "strategy": NAMES[strat], "latency_ms": L,
                            **{k: v for k, v in c.items() if not isinstance(v, np.ndarray)}})
    pl.DataFrame(lat).write_csv(REP / "latency_test.csv")

    # robustness view (not used for selection): every grid setting on the test days
    front = []
    for sym in SYMBOLS:
        for strat in (S0, S1, S2):
            for p in GRID[strat]:
                c = combine([run_day(sym, d, strat, p)[0] for d in TEST])
                front.append({"symbol": sym, "strategy": NAMES[strat], **p,
                              **{k: v for k, v in c.items() if not isinstance(v, np.ndarray)}})
    pl.DataFrame(front).write_csv(REP / "frontier_test_all_settings.csv")

    meta = {"train_days": TRAIN, "test_days": TEST, "lot_usd": LOT_USD, "maker_fee": FEE, "tau_s": TAU_S,
            "latency_ms": LATENCY, "volume_obligation_usd_per_day": VOL_OBLIGATION_PER_DAY, "ewma_halflife_s": HALFLIFE_TICKS / 10,
            "train_sigma_bps_per_sqrt_s": {s: train_sigma(s) for s in SYMBOLS},
            "best_params": {f"{s} {NAMES[k]}": v for (s, k), v in best.items()},
            "configs_tuned": len(tuning), "seconds": round(time.time() - t0, 1)}
    (REP / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(pl.DataFrame(results).select("symbol", "strategy", "fills", "net_pnl", "spread_capture", "inventory_pnl",
                                       "fees", "max_abs_inv_usd", "markout_5s_bps", "adverse_share_5s",
                                       "hourly_sharpe"))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
