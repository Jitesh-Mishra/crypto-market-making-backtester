"""Per-run metrics: PnL decomposition, inventory risk, markouts, adverse selection."""
from __future__ import annotations

import numpy as np

MARKOUTS_S = (1, 5, 30, 60)


def mid_at(g_t, g_mid, t):
    """Mid as of time t (last grid point at or before t)."""
    i = np.searchsorted(g_t, t, side="right") - 1
    return g_mid[np.clip(i, 0, len(g_mid) - 1)]


def summarize(g_t, g_mid, res, lot_usd):
    f_t, f_side, f_px, f_qty, f_mid, inv_path, cash_path, fees = res
    notional = f_px * f_qty
    end_mid = g_mid[-1]
    gross = cash_path[-1] + fees + inv_path[-1] * end_mid           # before fees
    spread = float(np.sum(f_side * (f_mid - f_px) * f_qty))        # edge vs mid at the fill
    out = {
        "fills": int(len(f_t)),
        "volume_usd": float(notional.sum()),
        "net_pnl": float(gross - fees),
        "spread_capture": spread,
        "inventory_pnl": float(gross - spread),
        "fees": float(fees),
    }
    inv_usd = inv_path * g_mid
    out["max_abs_inv_usd"] = float(np.max(np.abs(inv_usd)))
    out["mean_abs_inv_usd"] = float(np.mean(np.abs(inv_usd)))
    out["end_inv_usd"] = float(inv_usd[-1])
    eq = cash_path + inv_path * g_mid
    hour = (g_t - g_t[0]) // 3_600_000
    last = np.r_[np.nonzero(np.diff(hour))[0], len(hour) - 1]
    out["hourly_equity"] = np.r_[0.0, eq[last]]
    w = notional / notional.sum() if len(f_t) else notional
    for h in MARKOUTS_S:
        m = mid_at(g_t, g_mid, f_t + h * 1000)
        mo = f_side * (m - f_px) / f_px * 1e4
        out[f"markout_{h}s_bps"] = float(np.sum(w * mo)) if len(f_t) else np.nan
        if h == 5:
            moved = f_side * (m - f_mid)
            out["adverse_share_5s"] = float(np.mean(moved < 0)) if len(f_t) else np.nan
    return out


def combine(rows):
    """Sum a list of per-day summaries (one strategy, one symbol) into one row."""
    keys_sum = ["fills", "volume_usd", "net_pnl", "spread_capture", "inventory_pnl", "fees"]
    out = {k: sum(r[k] for r in rows) for k in keys_sum}
    out["max_abs_inv_usd"] = max(r["max_abs_inv_usd"] for r in rows)
    out["mean_abs_inv_usd"] = float(np.mean([r["mean_abs_inv_usd"] for r in rows]))
    hourly = np.concatenate([np.diff(r["hourly_equity"]) for r in rows])
    out["hours"] = len(hourly)
    out["hourly_sharpe"] = float(hourly.mean() / hourly.std()) if hourly.std() > 0 else np.nan
    out["pct_hours_positive"] = float(np.mean(hourly > 0))
    out["pnl_per_1m_volume"] = out["net_pnl"] / out["volume_usd"] * 1e6 if out["volume_usd"] else np.nan
    tot = out["volume_usd"]
    for h in MARKOUTS_S:
        out[f"markout_{h}s_bps"] = (float(sum(r[f"markout_{h}s_bps"] * r["volume_usd"] for r in rows if r["fills"]) / tot)
                                    if tot else np.nan)
    fills = out["fills"]
    out["adverse_share_5s"] = (float(sum(r["adverse_share_5s"] * r["fills"] for r in rows if r["fills"]) / fills)
                               if fills else np.nan)
    return out
