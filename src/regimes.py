"""market regimes by clustering whole return distributions - Wasserstein k-means.

how the math works
------------------
copies Horvath, Issa & Muguruza (2021), "Clustering Market Regimes using the
Wasserstein Distance". instead of clustering a few summary numbers (mean,
vol), cluster the WHOLE return distribution of each rolling window. two
stretches can have the same vol but totally different skew and tails - only
a distribution-level clusterer can tell them apart.

in 1-d the optimal transport stuff gets really simple: for two samples of
the same size, the Wasserstein distance is just the gap between their
SORTED vectors (their quantile functions). so:

  * distance   W2(a, b) = sqrt(mean((sort(a) - sort(b))^2))
  * cluster center = element-wise mean of the members' sorted vectors
    (an average of sorted vectors is still sorted, so it's a valid quantile
    vector)
  * Wasserstein k-means = plain Lloyd's algorithm on sorted windows

honest limits: you pick k, the labels are in-sample clusters (not causal
market states), and window/step size shape what it can see. an educational
copy of published research, not investment advice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.covariance import TRADING_DAYS


def rolling_windows(returns: pd.Series, window: int = 20, step: int = 5):
    """rolling windows as sorted quantile vectors -> (Q, end_dates).
    Q[i] = the `window` daily returns ending on end_dates[i], sorted ascending.
    steps `step` days at a time; partial windows are dropped."""
    r = returns.dropna()
    vals = r.to_numpy(dtype=float)
    idx = r.index
    rows, ends = [], []
    for end in range(window, len(vals) + 1, step):
        rows.append(np.sort(vals[end - window:end]))
        ends.append(idx[end - 1])
    if not rows:
        return np.empty((0, window)), pd.DatetimeIndex([])
    return np.vstack(rows), pd.DatetimeIndex(ends)


def wasserstein_kmeans(Q: np.ndarray, k: int = 3, n_init: int = 8,
                       max_iter: int = 100, seed: int = 42):
    """Lloyd's k-means where each row of Q is a sorted quantile vector.
    distance = 1-d W2, center = mean of sorted members (still sorted, so still
    a quantile function). keeps the best of `n_init` seeded restarts by
    inertia, so the same seed always gives the same answer."""
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    if n < k:
        raise ValueError(f"need at least k={k} windows, got {n}")
    rng = np.random.default_rng(seed)
    best_labels, best_centers, best_inertia = None, None, np.inf
    for _ in range(n_init):
        centers = Q[rng.choice(n, size=k, replace=False)].copy()
        labels = np.zeros(n, dtype=int)
        for it in range(max_iter):
            # squared W2 = mean squared gap between quantile vectors
            d2 = ((Q[:, None, :] - centers[None, :, :]) ** 2).mean(axis=2)
            new_labels = d2.argmin(axis=1)
            if it > 0 and (new_labels == labels).all():
                break
            labels = new_labels
            for j in range(k):
                members = Q[labels == j]
                if len(members):
                    centers[j] = members.mean(axis=0)   # barycenter
                else:                                    # empty cluster: reseed
                    centers[j] = Q[rng.integers(n)]
        inertia = float(d2[np.arange(n), labels].sum())
        if inertia < best_inertia:
            best_inertia, best_labels, best_centers = inertia, labels, centers
    return best_labels, best_centers


def _vol_order(Q: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """renumber clusters by member vol, 0 = calmest."""
    vols = []
    for j in range(k):
        members = Q[labels == j]
        vols.append(members.std() if len(members) else np.inf)
    order = np.argsort(vols)
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    return remap[labels]


def regime_stats(Q: np.ndarray, labels: np.ndarray) -> list[dict]:
    """stats per regime (0 = calmest), pooling every member window's returns.
    cvar_95 = average of the worst 5% pooled daily returns, as a positive loss."""
    k = int(labels.max()) + 1
    labels = _vol_order(Q, labels, k)
    out = []
    for j in range(k):
        pooled = Q[labels == j].ravel()
        if not len(pooled):
            continue
        tail = np.sort(pooled)[:max(1, int(np.ceil(len(pooled) * 0.05)))]
        sd = pooled.std(ddof=1) if len(pooled) > 1 else 0.0
        m = pooled.mean()
        skew = (float(np.mean((pooled - m) ** 3)) / sd ** 3) if sd > 0 else 0.0
        out.append({
            "label": j,
            "n_windows": int((labels == j).sum()),
            "ann_vol": float(sd * np.sqrt(TRADING_DAYS)),
            "mean_daily": float(m),
            "skew": skew,
            "cvar_95": float(-tail.mean()),
        })
    return out


def vol_ordered_labels(Q: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """public version of _vol_order: labels renumbered so 0 = calmest."""
    return _vol_order(Q, labels, int(labels.max()) + 1)


def transition_matrix(labels: np.ndarray, k: int) -> np.ndarray:
    """P(next regime | this regime), counted from back-to-back windows."""
    T = np.zeros((k, k))
    for a, b in zip(labels[:-1], labels[1:]):
        T[a, b] += 1
    sums = T.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(sums > 0, T / sums, 0.0)
    return P


if __name__ == "__main__":
    # live-data smoke test, like the other src modules
    from src.ingestion import DataEngine, get_returns
    eng = DataEngine()
    prices = eng.fetch_prices(["SPY", "TLT", "GLD"], period="2y")
    port = get_returns(prices).mean(axis=1)
    Q, ends = rolling_windows(port)
    labels, _ = wasserstein_kmeans(Q, k=3)
    labels = vol_ordered_labels(Q, labels)
    for s in regime_stats(Q, labels):
        print(s)
    print(transition_matrix(labels, 3).round(2))
