import sys

import numpy as np

import data
import portfolio as pf
import risk
import stress

# default basket: big tech. or pass your own: python run.py AAPL MSFT JPM
TICKERS = sys.argv[1:] or ["QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]


def main():
    close, dollar_vol = data.download(TICKERS, "2y")
    r = data.returns(close)
    w = np.ones(r.shape[1]) / r.shape[1]  # equal money in every stock
    day = risk.book(r, w)
    print(f"Meliona risk report: {', '.join(r.columns)} ({len(r)} days, equal weight)\n")

    # 1. the headline: how bad can a normal year get
    boot, jump = risk.monte_carlo(r, w), risk.jump_mc(r, w)
    print(f"1-year CVaR 95%: {boot['cvar']:.1%} (replayed history), {jump['cvar']:.1%} (jump model)")
    print(f"chance of losing money over a year: {boot['prob_loss']:.0%}")

    # 2. a single bad day, and is that line actually honest
    bt = risk.backtest(day)
    print(f"1-day VaR 95%: {risk.var(day):.2%}, CVaR 95%: {risk.cvar(day):.2%}")
    print(f"VaR backtest: {bt['breaches']} breaches in {bt['n']} days -> "
          f"{'pass' if bt['passed'] else 'FAIL' if bt['passed'] is False else 'not enough days'}\n")

    # 3. who brings the risk, and the fair-share fix
    c = pf.cov(r)
    rp = pf.risk_parity(c)
    print("risk share now vs risk-parity weight:")
    for t, share, fair in zip(r.columns, pf.risk_share(w, c), rp):
        print(f"  {t:6s} {share:6.1%} of the risk -> hold {fair:6.1%}")
    print(f"yearly wiggle {pf.vol(w, c):.1%}; to hit 10% use {pf.vol_target(w, c)['leverage']:.2f}x\n")

    # 4. how fast can a $1M book get out (last ~3 months of volume)
    exit_days = pf.days_to_exit(w, dollar_vol[r.columns].tail(63).mean().fillna(0.0))
    print(f"days to sell $1M at 20% of volume: slowest {exit_days.max():.2f} ({exit_days.idxmax()})\n")

    # 5. replay real crashes on today's basket
    full, _ = data.download(TICKERS, "max")
    print("real crash replays (equal weight, names that traded then):")
    for name, (a, b) in stress.CRISES.items():
        cr = stress.replay(full, a, b)
        if cr.empty:
            continue
        path = (1 + cr.mean(axis=1)).cumprod()
        print(f"  {name:32s} {path.iloc[-1] - 1:+7.1%}  worst point {(path / path.cummax() - 1).min():+7.1%}")

    # 6. grit: who gets back up after getting knocked down
    print("\ngrit score (0-100, vs this basket):")
    for t, g in stress.grit(full[r.columns])["grit"].items():
        print(f"  {t:6s} {g:5.1f}")


if __name__ == "__main__":
    main()
