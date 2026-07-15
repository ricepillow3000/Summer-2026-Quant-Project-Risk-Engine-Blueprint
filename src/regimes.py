"""Market regime clustering on FULL return distributions - Wasserstein k-means.

Quant Deep Dive
---------------
Reproduces the method of Horvath, Issa & Muguruza (2021), "Clustering Market
Regimes using the Wasserstein Distance": instead of clustering summary
features (mean, volatility), cluster the ENTIRE empirical return distribution
of each rolling window. Two regimes can share volatility yet differ violently
in skew and tail mass - a summary-feature clusterer cannot see that; a
distributional one can.

The optimal-transport machinery collapses beautifully in one dimension: the
p-Wasserstein distance between two empirical distributions with the same
number of equally-weighted samples is just the L_p distance between their
SORTED sample vectors (their empirical quantile functions). So:

  * distance   W2(a, b) = sqrt(mean((sort(a) - sort(b))^2))
  * barycenter of a cluster = element-wise mean of member quantile vectors
    (an average of sorted vectors is itself sorted, so it stays a valid
    quantile vector)
  * Wasserstein k-means = Lloyd's algorithm on sorted-window vectors

Honest limits: k is a user choice, labels are in-sample statistical clusters
(not causal market states), and window/step sizes shape what the clusterer
can resolve. Educational reproduction of published research - not investment
advice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_windows(returns: pd.Series, window: int = 20, step: int = 5):
    """Rolling return windows as sorted quantile vectors.

    Returns (Q, end_dates): Q[i] is the ascending-sorted vector of the
    `window` daily returns ending at end_dates[i]. Windows advance by `step`
    days; incomplete windows are dropped.
    """
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


def wasserstein_distance_1d(a: np.ndarray, b: np.ndarray) -> float:
    """W2 between two equal-length SORTED sample vectors (1-D closed form)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"quantile vectors must match: {a.shape} vs {b.shape}")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def wasserstein_kmeans(Q: np.ndarray, k: int = 3, n_init: int = 8,
                       max_iter: int = 100, seed: int = 42):
    """Lloyd's k-means where rows of Q are sorted quantile vectors.

    Distances are 1-D W2; centers are Wasserstein barycenters (element-wise
    means of sorted member rows - the mean of sorted vectors is sorted, so
    every center remains a valid quantile function). Best of `n_init`
    seeded restarts by inertia. Deterministic for a given seed.
    """
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
            # assign: squared W2 = mean squared gap between quantile vectors
            d2 = ((Q[:, None, :] - centers[None, :, :]) ** 2).mean(axis=2)
            new_labels = d2.argmin(axis=1)
            if it > 0 and (new_labels == labels).all():
                break
            labels = new_labels
            for j in range(k):
                members = Q[labels == j]
                if len(members):
                    centers[j] = members.mean(axis=0)   # barycenter
                else:                                    # dead center: reseed
                    centers[j] = Q[rng.integers(n)]
        inertia = float(d2[np.arange(n), labels].sum())
        if inertia < best_inertia:
            best_inertia, best_labels, best_centers = inertia, labels, centers
    return best_labels, best_centers


def _vol_order(Q: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Relabel clusters 0..k-1 by ascending member volatility (0 = calmest)."""
    vols = []
    for j in range(k):
        members = Q[labels == j]
        vols.append(members.std() if len(members) else np.inf)
    order = np.argsort(vols)
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    return remap[labels]


def regime_stats(Q: np.ndarray, labels: np.ndarray) -> list[dict]:
    """Per-regime distribution stats, vol-ordered (label 0 = calmest).

    Pools every member window's returns. cvar_95 = mean of the pooled worst
    5% of daily returns (reported as a loss, positive number).
    """
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
            "ann_vol": float(sd * np.sqrt(252)),
            "mean_daily": float(m),
            "skew": skew,
            "cvar_95": float(-tail.mean()),
        })
    return out


def vol_ordered_labels(Q: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Public wrapper: labels relabeled so 0 = calmest regime."""
    return _vol_order(Q, labels, int(labels.max()) + 1)


def transition_matrix(labels: np.ndarray, k: int) -> np.ndarray:
    """Empirical P(next regime | current regime) from consecutive windows."""
    T = np.zeros((k, k))
    for a, b in zip(labels[:-1], labels[1:]):
        T[a, b] += 1
    sums = T.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(sums > 0, T / sums, 0.0)
    return P


if __name__ == "__main__":
    # smoke test on live data, same pattern as the other src modules
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
