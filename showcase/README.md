# Meliona: the engine, short version

The core math behind [Meliona](https://summer-2026-quant-project-risk-engine-blueprint-production.up.railway.app), a portfolio risk engine on live Yahoo Finance data.
About 330 lines, no website code. The full app is in the folders above.

## Run it

```bash
pip install -r requirements.txt
python run.py                  # big tech basket
python run.py AAPL JPM XOM GLD # or your own
```

## What's inside

| File | What it does |
|---|---|
| `data.py` | Gets real daily prices and volume from Yahoo |
| `risk.py` | VaR, CVaR, a walk-forward VaR backtest (Kupiec), and two 10,000-path Monte Carlo simulators: block bootstrap (Politis & Romano 1994) and Merton jump-diffusion (1976) |
| `portfolio.py` | Covariance, who brings the risk, risk parity (equal risk per stock), volatility targeting, days to exit a position |
| `stress.py` | Replays 10 real crashes (2000 to 2023) on your basket, plus the Grit score: who recovers after a fall |
| `run.py` | Runs all of it and prints one report |
| `test_showcase.py` | Proves every function here gives the same numbers as the full engine in `src/` |

## Rules it follows

- **Real data only.** Every market number comes from Yahoo at run time. Nothing is typed in or invented.
- **The headline is CVaR:** the average loss in the worst 5% of simulated years.
- **Honest gaps.** A stock that didn't trade during a crash sits that crash out, and that is reported. Missing volume shows as "can't exit", never as a made-up number.
