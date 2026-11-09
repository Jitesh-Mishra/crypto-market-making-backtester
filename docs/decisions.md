# Decisions

**D1 Data window: Mar 24 to 30, 2024.** Binance publishes USD-M futures `bookTicker` daily files only up to
2024-03-30 (checked: 404 from 2024-03-31 on). The brief asked for futures best bid/ask plus trades, so the 7
days ending there are the only real option. Recent data would need spot (no book files) or a live recorder.

**D2 aggTrades, not trades.** An aggTrade is consecutive fills by one taker at one price. For a fill rule that
asks "did a trade print through our price, and how much", that is equivalent, and 2.7x smaller (19.5M rows vs
52.2M trades).

**D3 100 ms decision grid.** Book updates arrive about 160 per second for BTC. Requoting every 100 ms on the
as-of best bid/ask keeps the loop at 0.3 s per symbol-day in numba, so the 1,010 simulator runs in the
study finish in about 5 minutes. Cost: real makers requote on every event, so this understates how fast we can step away.

**D4 Trade-through fill rule.** A fill needs a print strictly through our price. With no queue model, this is the
conservative choice: it never credits a fill we may not have got. It also biases every fill toward adverse
selection, which the README states up front. The comparison between strategies is fair because they share it.

**D5 Tuning objective with a volume obligation.** Maximising net PnL alone picks the widest quoter, which barely
trades (12 bps half-spread, $19K volume over 3 days). That isn't market making. Requiring $5M/day of train volume,
as an exchange program would, makes every strategy actually provide liquidity. S1 and S2 met it on train
($5.5M to $9.9M/day) but traded $3.2M to $6.1M/day on the quieter test days; that is reported, not hidden.

**D6 Baseline tuned too.** S0's half-spread came from the same grid search as S1/S2's parameters, so the
comparison isn't against a straw man.

**D7 Days independent.** Each day starts flat and is marked to mid at the end. That keeps days comparable and
stops one big overnight position dominating the test.

**D8 Fee what-ifs on the same fills.** Fills don't depend on the fee, so net PnL at 0% or a -0.005% rebate is
gross PnL minus fee times volume. No rerun is needed, and it isolates the fee effect.
