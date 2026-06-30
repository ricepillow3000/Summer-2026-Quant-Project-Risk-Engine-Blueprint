# Institutional-Grade Portfolio Optimization & Risk Engine

A production-ready quantitative risk dashboard targeting the QQQ universe and the "Magnificent Seven" (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA). Built by a 2nd-year Data Analytics student to recruiter-ready, hedge-fund standards.

**Live demo:** _coming August 2026_

---

## What This Is

Most student finance projects pull data and plot a chart. This is not that.

This engine is being built to the standard a quant desk would actually deploy: modular Python, statistically rigorous math, and a live Streamlit dashboard a recruiter or PM can open from a resume link and stress-test in real time — no cold starts, no slow math.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | `yfinance`, `pandas`, custom caching layer |
| Math | `numpy`, `scipy` (eigen-decomposition, CVaR, Monte Carlo) |
| Optimization | Mean-Variance, Black-Litterman |
| Dashboard | `streamlit`, `plotly` |
| Deployment | Railway / Render, custom domain + SSL |

---

## Project Structure

```
portfolio_engine/
├── .streamlit/
│   └── config.toml       # Dark-mode theme, no footer
├── data/                 # Cached price data
├── src/
│   ├── ingestion.py      # DataEngine with rate-limit protection
│   ├── analytics.py      # Covariance, eigen-decomposition
│   └── risk.py           # CVaR, Monte Carlo, tail risk
├── requirements.txt
├── Procfile              # Railway/Heroku deployment config
└── main.py               # Streamlit entry point
```

---

## Build Roadmap (Summer 2026)

| Phase | Weeks | What Gets Built | What I Learn |
|---|---|---|---|
| I | 1–2 | DataEngine with caching | Normality & stationarity testing (Shapiro-Wilk) |
| II | 3 | Covariance mapping, eigen-decomposition | Multicollinearity & concentration risk |
| III | 4–5 | Mean-Variance & Black-Litterman optimizer | Bayesian math, the Oracle Problem |
| IV | 6 | CVaR & Monte Carlo risk guardrails | Tail risk, Black Swan visualization |
| V | 7 | Streamlit dashboard (session state, fragments) | Active Share: skill vs. luck in the Mag 7 |
| VI | 8 | Cloud deploy, custom domain, SSL | The Pitch: walking an MD through the live link |

**Target ship date: August 23, 2026**

---

## Design Principles

- **No retail logic.** Every architectural choice is defensible in a quant interview.
- **Recruiter-ready at all times.** The live link must handle a PM clicking it cold.
- **Teach as we build.** Every module comes with a "Quant Deep Dive" explaining the math behind it.

---

## Status

> Week 1 — Project initialized. Data engineering sprint starting now.
