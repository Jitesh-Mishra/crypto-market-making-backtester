"""Where the naive quoter's PnL comes from: hour by hour over all 7 days, with each strategy's tuned parameters.

Descriptive, not a test: days 1-4 are the tuning days. Writes reports/hourly_all_days.csv and
reports/regime.json (worst S0 hour per symbol, and how much of S0's hourly PnL is explained by holding
inventory into a price move).

    python -m mmbt.regime
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from .data import DAYS, SYMBOLS
from .experiment import NAMES, REP, TRAIN, day_arrays, run_day
from .sim import S0, S1, S2


def main():
    meta = json.loads((REP / "run_meta.json").read_text())
    best = meta["best_params"]
    rows = []
    for sym in SYMBOLS:
        for day in DAYS:
            d = day_arrays(sym, day)
            hour_idx = np.arange(35_999, len(d["g_t"]), 36_000)          # last grid tick of each hour
            mid_h = np.r_[d["mid"][0], d["mid"][hour_idx]]
            for strat in (S0, S1, S2):
                _, res = run_day(sym, day, strat, best[f"{sym} {NAMES[strat]}"])
                inv, cash = res[5], res[6]
                eq = np.r_[0.0, cash[hour_idx] + inv[hour_idx] * d["mid"][hour_idx]]
                inv_h = np.r_[0.0, inv[hour_idx]]
                for h in range(len(hour_idx)):
                    rows.append({"symbol": sym, "day": day, "split": "train" if day in TRAIN else "test",
                                 "hour": h, "strategy": NAMES[strat], "pnl": eq[h + 1] - eq[h],
                                 "inv_start_usd": inv_h[h] * mid_h[h],
                                 "mid_return_bps": (mid_h[h + 1] / mid_h[h] - 1) * 1e4,
                                 "carry_pnl": inv_h[h] * (mid_h[h + 1] - mid_h[h])})
    df = pl.DataFrame(rows)
    df.write_csv(REP / "hourly_all_days.csv")
    out = {}
    for sym in SYMBOLS:
        s0 = df.filter((pl.col("symbol") == sym) & (pl.col("strategy") == NAMES[S0]))
        x, y = s0["carry_pnl"].to_numpy(), s0["pnl"].to_numpy()
        r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
        w = s0.sort("pnl").row(0, named=True)
        same = df.filter((pl.col("symbol") == sym) & (pl.col("day") == w["day"]) & (pl.col("hour") == w["hour"]))
        by_split = {sp: {NAMES[k]: float(df.filter((pl.col("symbol") == sym) & (pl.col("split") == sp) &
                                                   (pl.col("strategy") == NAMES[k]))["pnl"].sum())
                         for k in (S0, S1, S2)} for sp in ("train", "test")}
        out[sym] = {"s0_hourly_pnl_r2_vs_inventory_carry": round(r2, 3), "hours": len(y),
                    "worst_s0_hour": {"day": w["day"], "hour_utc": w["hour"], "split": w["split"],
                                      "mid_move_bps": round(w["mid_return_bps"], 1),
                                      "s0_inventory_at_start_usd": round(w["inv_start_usd"]),
                                      "pnl_by_strategy": {r["strategy"]: round(r["pnl"], 1) for r in same.iter_rows(named=True)}},
                    "net_pnl_by_split": {sp: {k: round(v) for k, v in d.items()} for sp, d in by_split.items()}}
    (REP / "regime.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
