"""Simulator unit tests on hand-built event streams, where the right answer is known."""
import numpy as np

from mmbt.analytics import summarize
from mmbt.sim import S0, S1, S2, ewma_sigma, run


def grid(n=50, bid=100.0, ask=100.1, step_ms=100):
    t = np.arange(1, n + 1, dtype=np.int64) * step_ms
    return t, np.full(n, bid), np.full(n, ask)


def call(g_t, b, a, tr, strategy=S0, half_bps=10.0, latency=0, fee=0.0, max_lots=1e9, gamma=1.0, kappa=1.0, sig=None):
    tr_t = np.array([x[0] for x in tr], dtype=np.int64)
    tr_p = np.array([x[1] for x in tr], dtype=float)
    tr_q = np.array([x[2] for x in tr], dtype=float)
    tr_s = np.array([x[3] for x in tr], dtype=np.int8)
    sig = np.zeros(len(g_t)) if sig is None else sig
    return run(g_t, b, a, sig, tr_t, tr_p, tr_q, tr_s, strategy, half_bps, gamma, kappa, 30.0, max_lots,
               1000.0, 0.01, latency, fee)


def test_fill_only_when_trade_prints_through():
    g_t, b, a = grid()
    mid = 100.05
    bid_q = np.floor(mid * (1 - 10 / 1e4) / 0.01) * 0.01           # 99.94
    tr = [(250, bid_q, 5.0, -1),            # at our price: no fill (no queue position)
          (350, bid_q - 0.01, 5.0, 1),      # through, but buyer-initiated: cannot hit our bid
          (450, bid_q - 0.01, 3.0, -1)]     # through, seller-initiated: fills
    f_t, f_side, f_px, f_qty, *_ = call(g_t, b, a, tr)
    assert list(f_t) == [450] and f_side[0] == 1
    assert abs(f_px[0] - bid_q) < 1e-9                              # we get our own price
    assert abs(f_qty[0] - min(3.0, 1000 / mid)) < 1e-9              # capped by quote size (about 9.99)


def test_latency_delays_quotes():
    g_t, b, a = grid()
    tr = [(140, 90.0, 100.0, -1)]                                   # 40 ms after the first quote is posted
    assert len(call(g_t, b, a, tr, latency=0)[0]) == 1
    assert len(call(g_t, b, a, tr, latency=50)[0]) == 0             # quote not live yet


def test_same_timestamp_trade_does_not_see_new_quote():
    g_t, b, a = grid()
    tr = [(100, 90.0, 100.0, -1)]                                   # same ms as the first decision tick
    assert len(call(g_t, b, a, tr)[0]) == 0


def test_partial_fills_and_pnl_accounting():
    g_t, b, a = grid()
    tr = [(150, 99.0, 4.0, -1), (160, 99.0, 4.0, -1), (170, 99.0, 4.0, -1), (250, 101.0, 100.0, 1)]
    res = call(g_t, b, a, tr, fee=0.0002)
    f_t, f_side, f_px, f_qty, f_mid, inv, cash, fees = res
    size = 1000 / 100.05
    assert np.allclose(f_qty[:3], [4.0, 4.0, size - 8.0])           # third fill is what is left of the quote
    s = summarize(g_t, 0.5 * (b + a), res, 1000.0)
    assert abs(s["net_pnl"] - (s["spread_capture"] + s["inventory_pnl"] - s["fees"])) < 1e-9
    assert abs(s["fees"] - np.sum(f_px * f_qty) * 0.0002) < 1e-9


def test_avellaneda_stoikov_skews_against_inventory():
    g_t, b, a = grid(n=30)
    sig = np.full(len(g_t), 1.0)
    tr = [(150, 90.0, 10.0, -1)]                                    # we buy one lot at t=150
    res = call(g_t, b, a, tr, strategy=S1, gamma=0.5, kappa=1.0, sig=sig, half_bps=0.0)
    # after buying, the next quotes must shift down: compare fills of a probe sell trade at the old ask level
    assert res[5][-1] > 0                                           # long inventory
    g2 = grid(n=30)
    tr2 = [(150, 90.0, 10.0, -1), (950, 100.05 * (1 + 3 / 1e4), 10.0, 1)]
    long_fill = call(*g2, tr2, strategy=S1, gamma=0.5, kappa=1.0, sig=sig, half_bps=0.0)
    flat_fill = call(*g2, tr2[1:], strategy=S1, gamma=0.5, kappa=1.0, sig=sig, half_bps=0.0)
    assert len(long_fill[0]) == 2 and len(flat_fill[0]) == 0        # long book lowers its ask enough to sell


def test_hard_inventory_limit_stops_buying():
    g_t, b, a = grid(n=80)
    tr = [(150 + 100 * i, 90.0, 100.0, -1) for i in range(30)]      # relentless selling through our bid
    res = call(g_t, b, a, tr, strategy=S2, gamma=0.01, kappa=1.0, max_lots=3, half_bps=0.0, sig=np.full(len(g_t), 1.0))
    inv_lots = res[5] * 100.05 / 1000
    assert inv_lots.max() <= 4.0 + 1e-9                             # stops at the limit (plus the lot that crossed it)
    s0 = call(g_t, b, a, tr, strategy=S0, half_bps=1)
    assert s0[5].max() * 100.05 / 1000 > 20                         # the naive quoter keeps buying


def test_ewma_sigma_is_causal():
    mid = 100 * np.exp(np.cumsum(np.r_[0, np.full(999, 1e-4)]))
    a = ewma_sigma(mid, 50, 10.0, 1e-8)
    mid2 = mid.copy()
    mid2[600:] *= 1.05                                              # change the future
    b = ewma_sigma(mid2, 50, 10.0, 1e-8)
    assert np.allclose(a[:600], b[:600])
