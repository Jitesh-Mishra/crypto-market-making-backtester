# Probe prep

Run `pytest -q`, read `mmbt/sim.py` top to bottom, and reread the README's "What I learned" before any interview.

## How the code works (5 lines)
1. `mmbt/data.py` downloads Binance futures `bookTicker` and `aggTrades` zips, checks their SHA-256, and uses Polars to make a 100 ms best bid/ask grid (as-of join, never ahead) plus a trade table with the taker side.
2. `mmbt/sim.py: run` is one numba loop. Before each 100 ms decision tick it replays the trades since the last tick against the quote that was live at each trade's time (latency applied). Then it asks the strategy for a new bid and ask.
3. A bid fills only on a seller-initiated print strictly below it; fills are capped by trade size and remaining quote size, at our price, minus a 0.02% fee.
4. `mmbt/analytics.py` splits PnL into spread capture (fill vs mid), inventory (everything else before fees) and fees. It measures markouts at 1/5/30/60 s, inventory, and hourly PnL.
5. `mmbt/experiment.py` grid-searches each strategy on Mar 24 to 27 under a $5M/day volume obligation, runs the winners once on Mar 28 to 30, then runs latency and fee sensitivity.

## Trading intuition

**Why did the naive market maker lose money?**
- 15 s: Its inventory is a bet against the trend. When BTC rallied 2% in an hour on Mar 25 it was short $232K and lost $7.6K in that hour.
- 60 s: A symmetric quoter gets hit on the side the market is moving toward, so in a trend it accumulates the losing side. Over all 168 hours, `inventory × price move` explains 93% of its hourly PnL variance on BTC.
- Spread capture was positive (+$1.6K on the BTC test days); the fee (-$4.0K) and inventory swamped it.
- Evidence: `reports/regime.json`, `reports/results_test.csv`.

**Then why did it beat Avellaneda-Stoikov on the BTC test days?**
- Luck of regime. Test days were ranging (BTC +0.2%), so its $328K positions mean-reverted (+$418 inventory PnL).
- On the trending train days the same quoter lost $37.9K vs $11.3K. A strategy whose ranking flips with the regime is a directional bet, not market making.

**What does Avellaneda-Stoikov actually do?**
- It shifts the reservation price against inventory, `r = mid(1 - q·γ·σ²·τ)`, so a long book quotes lower on both sides: it sells more easily and buys less.
- The spread widens with volatility and risk aversion.
- Result: max inventory $3.8K vs $328K, hourly PnL std $49 vs $536.
- Cost: it pays to get flat, so it lost in every test hour, but by small amounts.

**Why are all markouts negative?**
- The fill rule only fills when a trade prints through our price, so we only trade after the market has moved past us. This is the pessimistic case.
- At the front of the queue a maker gets filled on prints at its price, without the move. That is the edge this backtest can't credit.
- So the absolute PnL is a lower bound; the comparison between strategies is the point.

**What does latency do?** Stale quotes get picked off. From 0 to 200 ms, BTC fills nearly tripled (25,018 to 71,415 for S0) and the 5 s markout worsened from -0.57 to -0.87 bps (S0) and -0.58 to -1.08 (S1).

**Could this be profitable?** Not at retail fees: 2 bps of fee vs about 0.8 bps of spread captured per dollar. On the same fills, at 0% fee the naive quoter is +$2.0K on BTC and with a 0.5 bp rebate +$3.0K. Real makers live on rebate tiers, queue priority and short-horizon signals.

**What would you add first?** A queue-position model (fills at our level once the displayed size ahead of us has traded), then an order-flow imbalance signal to skew quotes before the move rather than after.

## Methodology

**How did you avoid overfitting?**
- Parameters for all three strategies came from the same grid on 4 days. The 3 test days ran once with no retuning.
- S1's sigma is the training median.
- S2's EWMA is causal: a unit test changes the future and checks the past is unchanged.

**Why a volume obligation in the objective?** Without it the best "market maker" quotes 12 bps wide and trades $19K in 3 days. Exchange market-maker programs require participation; $5M/day makes the strategies actually provide liquidity.

**Isn't one week too short?** Yes. It's what Binance publishes for futures `bookTicker` (it stops at Mar 30, 2024). It does contain a trend and a range, which is why I report both periods.

**Look-ahead checks?**
- The book grid is as-of (last update at or before t).
- Trades at the same millisecond as a decision are processed before it, so a quote never fills against a print it saw.
- Quotes go live only after the latency.

**What does the decomposition's "inventory" term contain?** Everything before fees except spread capture. For S1/S2 that is mostly adverse selection, since inventory stays near zero. For S0 it is mostly directional carry.

## Weak spots to own
- No queue position, no market impact, a 100 ms grid, one week of one month.
- Avellaneda-Stoikov didn't reduce adverse selection. S2's vol scaling made it worse on BTC (85% adversely selected vs 75%).
- The brief's stack listed pandas and SciPy; the code uses neither, so they aren't claimed.
