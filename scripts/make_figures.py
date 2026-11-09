"""README figures, light and dark, from reports/ (and one replay of the tuned strategies over all 7 BTC days)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmbt.data import DAYS  # noqa: E402
from mmbt.experiment import NAMES, TRAIN, day_arrays, run_day  # noqa: E402
from mmbt.sim import S0, S1, S2  # noqa: E402

REP, OUT = ROOT / "reports", ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
THEMES = {"light": dict(bg="#ffffff", ink="#1f2328", ink2="#59636e", grid="#e6e9ed", shade="#f3f5f7",
                        cols=["#2a78d6", "#eb6834", "#1baf7a"]),
          "dark": dict(bg="#0d1117", ink="#e6edf3", ink2="#9198a1", grid="#21262d", shade="#161b22",
                       cols=["#3987e5", "#d95926", "#199e70"])}
SHORT = {NAMES[S0]: "S0 naive", NAMES[S1]: "S1 Avellaneda-Stoikov", NAMES[S2]: "S2 AS + vol + limits"}
best = json.loads((REP / "run_meta.json").read_text())["best_params"]

# one replay of all 7 BTC days at 1-minute resolution, days chained
series = {k: {"t": [], "pnl": [], "inv": []} for k in (S0, S1, S2)}
mids, ts = [], []
for i, day in enumerate(DAYS):
    d = day_arrays("BTCUSDT", day)
    idx = np.arange(599, len(d["g_t"]), 600)
    ts.append(i + (d["g_t"][idx] - d["g_t"][0]) / 86_400_000)
    mids.append(d["mid"][idx])
    for k in (S0, S1, S2):
        _, res = run_day("BTCUSDT", day, k, best[f"BTCUSDT {NAMES[k]}"])
        inv, cash = res[5], res[6]
        base = series[k]["pnl"][-1][-1] if series[k]["pnl"] else 0.0
        series[k]["pnl"].append(base + cash[idx] + inv[idx] * d["mid"][idx])
        series[k]["inv"].append(inv[idx] * d["mid"][idx])
t = np.concatenate(ts)
mid = np.concatenate(mids)
res = pl.read_csv(REP / "results_test.csv")
lat = pl.read_csv(REP / "latency_test.csv")


def style(ax, th, grid="y"):
    ax.set_facecolor(th["bg"])
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color=th["grid"], lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


for theme, th in THEMES.items():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": th["ink2"], "axes.labelcolor": th["ink"],
                         "xtick.color": th["ink2"], "ytick.color": th["ink2"], "text.color": th["ink"]})
    # 1. seven days: price, cumulative PnL, inventory
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.4), sharex=True, facecolor=th["bg"],
                             gridspec_kw={"height_ratios": [1, 1.4, 1.2]})
    for ax in axes:
        style(ax, th)
        ax.axvspan(4, 7, color=th["shade"], zorder=0)
    axes[0].plot(t, mid, color=th["ink2"], lw=1.2)
    axes[0].set_ylabel("BTC mid ($)")
    axes[0].text(2, axes[0].get_ylim()[1], "train: tune parameters (days 1-4)", ha="center", va="top", fontsize=9,
                 color=th["ink2"])
    axes[0].text(5.5, axes[0].get_ylim()[1], "test: out-of-sample (days 5-7)", ha="center", va="top", fontsize=9,
                 color=th["ink2"])
    for j, k in enumerate((S0, S1, S2)):
        p = np.concatenate(series[k]["pnl"]) / 1000
        v = np.concatenate(series[k]["inv"]) / 1000
        axes[1].plot(t, p, color=th["cols"][j], lw=2, label=SHORT[NAMES[k]])
        nudge = {S0: 0, S1: -1.2, S2: 1.2}[k]
        axes[1].text(7.03, p[-1] + nudge, f"{SHORT[NAMES[k]]}", va="center", fontsize=8.5, color=th["ink"])
        axes[2].plot(t, v, color=th["cols"][j], lw=1.6 if k == S0 else 2, label=SHORT[NAMES[k]])
    axes[1].set_ylabel("cumulative net PnL ($K)")
    axes[2].set_ylabel("inventory ($K)")
    axes[2].axhline(0, color=th["ink2"], lw=0.8)
    axes[1].legend(frameon=False, loc="lower left", fontsize=9)
    wi = 1 + 14.5 / 24
    axes[1].annotate("Mar 25, 14:00 UTC: BTC +2.0% in an hour,\nnaive quoter short \\$232K, loses \\$7.6K",
                     xy=(wi, np.interp(wi, t, np.concatenate(series[S0]["pnl"]) / 1000)), xytext=(2.15, -29),
                     fontsize=8.5, color=th["ink2"], arrowprops=dict(arrowstyle="-", color=th["ink2"], lw=0.8))
    axes[2].set_xticks(range(8))
    axes[2].set_xticklabels([d[5:] for d in DAYS] + [""], fontsize=9)
    axes[2].set_xlim(0, 7.6)
    fig.suptitle("BTCUSDT perpetual, 7 days of real Binance order flow: the naive quoter's PnL is its inventory",
                 x=0.01, ha="left", fontsize=11, color=th["ink"])
    fig.tight_layout()
    fig.savefig(OUT / f"week-{theme}.png", dpi=150, facecolor=th["bg"])
    plt.close(fig)

    # 2. PnL decomposition (test days)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), facecolor=th["bg"], sharey=True)
    comps = [("spread_capture", "spread capture"), ("inventory_pnl", "inventory\n(incl. adverse selection)"),
             ("fees", "fees"), ("net_pnl", "net PnL")]
    for ax, sym in zip(axes, ("BTCUSDT", "ETHUSDT")):
        r = res.filter(pl.col("symbol") == sym)
        x = np.arange(len(comps))
        for j, k in enumerate((S0, S1, S2)):
            row = r.filter(pl.col("strategy") == NAMES[k]).row(0, named=True)
            vals = [row[c] * (-1 if c == "fees" else 1) / 1000 for c, _ in comps]
            ax.bar(x + (j - 1) * 0.27, vals, 0.25, color=th["cols"][j], label=SHORT[NAMES[k]])
        ax.axhline(0, color=th["ink2"], lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([c[1] for c in comps], fontsize=9)
        ax.set_title(f"{sym}, test days", loc="left", fontsize=10.5)
        style(ax, th)
    axes[0].set_ylabel("$K over 3 days")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle("Where the PnL went, out-of-sample (fees at 0.02% maker)", x=0.01, ha="left", fontsize=11,
                 color=th["ink"])
    fig.tight_layout()
    fig.savefig(OUT / f"decomposition-{theme}.png", dpi=150, facecolor=th["bg"])
    plt.close(fig)

    # 3. markouts by horizon, and the 5 s markout by quote latency
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), facecolor=th["bg"])
    hz = [1, 5, 30, 60]
    for ax, sym in zip(axes[:2], ("BTCUSDT", "ETHUSDT")):
        r = res.filter(pl.col("symbol") == sym)
        for j, k in enumerate((S0, S1, S2)):
            row = r.filter(pl.col("strategy") == NAMES[k]).row(0, named=True)
            ax.plot(hz, [row[f"markout_{h}s_bps"] for h in hz], "o-", color=th["cols"][j], lw=2, ms=6,
                    label=SHORT[NAMES[k]])
        ax.axhline(0, color=th["ink2"], lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(hz)
        ax.set_xticklabels([f"{h}s" for h in hz])
        ax.minorticks_off()
        ax.set_title(f"{sym} markout after a fill", loc="left", fontsize=10.5)
        ax.set_ylabel("bps vs fill price")
        style(ax, th)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    ax = axes[2]
    for j, k in enumerate((S0, S1, S2)):
        rr = lat.filter((pl.col("symbol") == "BTCUSDT") & (pl.col("strategy") == NAMES[k])).sort("latency_ms")
        ax.plot(rr["latency_ms"], rr["markout_5s_bps"], "o-", color=th["cols"][j], lw=2, ms=6)
    ax.axhline(0, color=th["ink2"], lw=0.8)
    ax.set_xticks([0, 50, 200])
    ax.set_xlabel("quote latency (ms)")
    ax.set_ylabel("5 s markout (bps)")
    ax.set_title("BTCUSDT: slower quotes get picked off", loc="left", fontsize=10.5)
    style(ax, th)
    fig.suptitle("Adverse selection, out-of-sample. Below zero = the market moved against the fill",
                 x=0.01, ha="left", fontsize=11, color=th["ink"])
    fig.tight_layout()
    fig.savefig(OUT / f"markouts-{theme}.png", dpi=150, facecolor=th["bg"])
    plt.close(fig)
print("ok")
