"""Event-driven market-making simulator (numba).

Events, in time order:
  decision ticks   every 100 ms: the strategy reads the best bid/ask and its inventory, and posts one bid and one
                   ask. A posted quote goes live `latency_ms` later and replaces the previous one.
  trades           Binance aggTrades. A live bid fills only when a seller-initiated trade prints STRICTLY BELOW
                   it (the whole level at our price must have traded through, so no queue position is assumed);
                   a live ask, when a buyer-initiated trade prints strictly above it. Fill size is capped by the
                   trade's size and by what is left of our quote. We get our own price (maker).
At equal timestamps trades are processed before the decision tick, so a quote never trades against a print it
has already seen.

Quote prices (bps around mid, per strategy):
  S0 naive        bid = mid - h, ask = mid + h
  S1 Avellaneda-Stoikov, constant sigma (estimated on training days):
                  reservation r = mid * (1 - q * gamma * sigma^2 * tau / 1e4)          q in lots
                  half spread   d = 0.5 * gamma * sigma^2 * tau + (1/gamma) ln(1 + gamma/kappa)   (bps)
  S2 S1 with sigma from a causal EWMA of 100 ms mid returns, and hard inventory limits (stop quoting the side
     that would grow |q| past max_lots)
All prices are rounded away from mid to the tick, and a bid is never placed at or above the best ask
(post-only), nor an ask at or below the best bid.
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np

S0, S1, S2 = 0, 1, 2


@nb.njit(cache=True)
def ewma_sigma(mid, halflife_ticks, ticks_per_sec, warm):
    """Causal EWMA of squared 100 ms log returns -> per-sqrt-second vol in bps, using data up to each tick."""
    n = mid.shape[0]
    out = np.empty(n)
    a = 1.0 - math.exp(math.log(0.5) / halflife_ticks)
    v = warm
    out[0] = math.sqrt(v * ticks_per_sec) * 1e4
    for i in range(1, n):
        r = math.log(mid[i] / mid[i - 1])
        v = (1 - a) * v + a * r * r
        out[i] = math.sqrt(v * ticks_per_sec) * 1e4
    return out


@nb.njit(cache=True)
def run(g_t, g_bid, g_ask, sigma_bps, tr_t, tr_p, tr_q, tr_side,
        strategy, half_bps, gamma, kappa, tau_s, max_lots, lot_usd, tick, latency_ms, fee_rate):
    n_g = g_t.shape[0]
    n_tr = tr_t.shape[0]
    # decisions
    d_bid = np.full(n_g, np.nan)
    d_ask = np.full(n_g, np.nan)
    d_size = np.zeros(n_g)
    # fills
    cap = 1 << 16
    f_t = np.empty(cap, np.int64)
    f_side = np.empty(cap, np.int8)
    f_px = np.empty(cap)
    f_qty = np.empty(cap)
    f_mid = np.empty(cap)
    nf = 0
    inv = 0.0           # base-asset units
    cash = 0.0          # quote currency, fees included
    fees = 0.0
    inv_path = np.empty(n_g)
    cash_path = np.empty(n_g)
    live = -1           # index of the live decision
    rem_bid = 0.0
    rem_ask = 0.0
    j = 0               # next trade
    for k in range(n_g):
        tk = g_t[k]
        # trades up to and including tk, against quotes live at the trade time
        while j < n_tr and tr_t[j] <= tk:
            tt = tr_t[j]
            while live + 1 < k and g_t[live + 1] + latency_ms <= tt:
                live += 1
                rem_bid = d_size[live]
                rem_ask = d_size[live]
            if live >= 0:
                if tr_side[j] < 0 and rem_bid > 0 and not math.isnan(d_bid[live]) and tr_p[j] < d_bid[live]:
                    q = min(rem_bid, tr_q[j])
                    px = d_bid[live]
                    rem_bid -= q
                    inv += q
                    fee = px * q * fee_rate
                    cash -= px * q + fee
                    fees += fee
                    if nf == f_t.shape[0]:
                        f_t = np.concatenate((f_t, np.empty(cap, np.int64)))
                        f_side = np.concatenate((f_side, np.empty(cap, np.int8)))
                        f_px = np.concatenate((f_px, np.empty(cap)))
                        f_qty = np.concatenate((f_qty, np.empty(cap)))
                        f_mid = np.concatenate((f_mid, np.empty(cap)))
                    f_t[nf] = tt; f_side[nf] = 1; f_px[nf] = px; f_qty[nf] = q
                    f_mid[nf] = 0.5 * (g_bid[k - 1] + g_ask[k - 1]) if k > 0 else px
                    nf += 1
                elif tr_side[j] > 0 and rem_ask > 0 and not math.isnan(d_ask[live]) and tr_p[j] > d_ask[live]:
                    q = min(rem_ask, tr_q[j])
                    px = d_ask[live]
                    rem_ask -= q
                    inv -= q
                    fee = px * q * fee_rate
                    cash += px * q - fee
                    fees += fee
                    if nf == f_t.shape[0]:
                        f_t = np.concatenate((f_t, np.empty(cap, np.int64)))
                        f_side = np.concatenate((f_side, np.empty(cap, np.int8)))
                        f_px = np.concatenate((f_px, np.empty(cap)))
                        f_qty = np.concatenate((f_qty, np.empty(cap)))
                        f_mid = np.concatenate((f_mid, np.empty(cap)))
                    f_t[nf] = tt; f_side[nf] = -1; f_px[nf] = px; f_qty[nf] = q
                    f_mid[nf] = 0.5 * (g_bid[k - 1] + g_ask[k - 1]) if k > 0 else px
                    nf += 1
            j += 1
        # decision at tk
        bid, ask = g_bid[k], g_ask[k]
        mid = 0.5 * (bid + ask)
        size = lot_usd / mid
        q_lots = inv * mid / lot_usd
        if strategy == S0:
            r = mid
            h = half_bps
        else:
            s2 = sigma_bps[k] * sigma_bps[k]
            r = mid * (1.0 - q_lots * gamma * s2 * tau_s / 1e4)
            h = 0.5 * gamma * s2 * tau_s + math.log(1.0 + gamma / kappa) / gamma + half_bps
        qb = math.floor(r * (1.0 - h / 1e4) / tick) * tick
        qa = math.ceil(r * (1.0 + h / 1e4) / tick) * tick
        if qb >= ask:
            qb = ask - tick
        if qa <= bid:
            qa = bid + tick
        if strategy == S2:
            if q_lots >= max_lots:
                qb = np.nan
            if q_lots <= -max_lots:
                qa = np.nan
        d_bid[k] = qb
        d_ask[k] = qa
        d_size[k] = size
        inv_path[k] = inv
        cash_path[k] = cash
    return (f_t[:nf], f_side[:nf], f_px[:nf], f_qty[:nf], f_mid[:nf], inv_path, cash_path, fees)
