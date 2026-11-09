"""Architecture diagram as SVG, light and dark (docs/img/architecture-*.svg)."""
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
T = {"light": dict(bg="#ffffff", ink="#1f2328", ink2="#59636e", box="#f6f8fa", line="#8c959f",
                   a="#2a78d6", b="#eb6834", c="#1baf7a"),
     "dark": dict(bg="#0d1117", ink="#e6edf3", ink2="#9198a1", box="#161b22", line="#6e7681",
                  a="#3987e5", b="#d95926", c="#199e70")}


def box(x, y, w, h, title, lines, col, t):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{t["box"]}" stroke="{col}" stroke-width="1.8"/>'
    s += f'<text x="{x + 12}" y="{y + 22}" font-weight="600" fill="{t["ink"]}">{title}</text>'
    for i, ln in enumerate(lines):
        s += f'<text x="{x + 12}" y="{y + 42 + 16 * i}" font-size="12" fill="{t["ink2"]}">{ln}</text>'
    return s


def arrow(x1, y1, x2, y2, t):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{t["line"]}" stroke-width="1.6" '
            f'marker-end="url(#ah)"/>')


for name, t in T.items():
    W, H = 1000, 330
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif" font-size="14">',
         f'<rect width="{W}" height="{H}" fill="{t["bg"]}"/>',
         f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{t["line"]}"/></marker></defs>']
    s.append(box(20, 20, 220, 120, "Binance public data", ["USD-M futures, BTC and ETH", "bookTicker: 158.8M updates",
                                                           "aggTrades: 19.5M (52.2M trades)", "SHA-256 verified"], t["ink2"], t))
    s.append(box(280, 20, 210, 120, "Loader (Polars)", ["best bid/ask as of each 100 ms", "trades with taker side",
                                                         "Parquet per symbol-day"], t["a"], t))
    s.append(box(530, 20, 450, 120, "Event loop (numba)", ["every 100 ms: strategy posts a bid and an ask",
                                                            "quotes go live after 0 / 50 / 200 ms latency",
                                                            "fill only if a trade prints THROUGH our price",
                                                            "maker fee 0.02% on every fill"], t["b"], t))
    s.append(arrow(240, 80, 278, 80, t))
    s.append(arrow(490, 80, 528, 80, t))
    s.append(box(20, 190, 300, 120, "Strategies", ["S0 naive: fixed half-spread around mid",
                                                    "S1 Avellaneda-Stoikov: skew by inventory,",
                                                    "     spread from volatility, constant sigma",
                                                    "S2 S1 + live EWMA vol + inventory limits"], t["c"], t))
    s.append(box(360, 190, 290, 120, "Analytics", ["PnL = spread capture + inventory - fees",
                                                    "markouts 1 / 5 / 30 / 60 s",
                                                    "inventory risk, hourly PnL, Sharpe",
                                                    "hourly carry regression"], t["c"], t))
    s.append(box(690, 190, 290, 120, "Walk-forward", ["tune on Mar 24-27 (grid search,",
                                                       "     $5M/day volume obligation)",
                                                       "report Mar 28-30 only",
                                                       "latency + fee sensitivity"], t["c"], t))
    s.append(arrow(640, 140, 260, 188, t))
    s.append(arrow(320, 250, 358, 250, t))
    s.append(arrow(650, 250, 688, 250, t))
    s.append("</svg>")
    (OUT / f"architecture-{name}.svg").write_text("\n".join(s))
print("ok")
