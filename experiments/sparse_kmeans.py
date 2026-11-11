"""Sparse K-Means (Witten & Tibshirani, 2010) e Gap Statistic (Tibshirani et al., 2001) — tesi, par. 2.1.

Implementazione in numpy/scikit-learn, senza dipendenza da `pysparcl`.
Il vincolo L1 è ||w||_1 <= s con ||w||_2 <= 1, w >= 0.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def _soft(x: np.ndarray, delta: float) -> np.ndarray:
    return np.maximum(x - delta, 0.0)


def update_weights(bcss: np.ndarray, s: float, tol: float = 1e-8) -> np.ndarray:
    """Passo B: soft-thresholding di a_j (BCSS per variabile) con Δ trovato per bisezione."""
    a = np.maximum(bcss, 0.0)
    if np.linalg.norm(a) == 0:
        return np.full(len(a), 1 / np.sqrt(len(a)))
    w = a / np.linalg.norm(a)
    if w.sum() <= s:
        return w
    lo, hi = 0.0, a.max()
    for _ in range(200):
        delta = (lo + hi) / 2
        t = _soft(a, delta)
        nrm = np.linalg.norm(t)
        w = t / nrm if nrm > 0 else t
        if w.sum() > s:
            lo = delta
        else:
            hi = delta
        if hi - lo < tol:
            break
    t = _soft(a, hi)
    return t / np.linalg.norm(t)


def _bcss(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """BCSS per colonna = TSS_j - WSS_j."""
    tss = ((X - X.mean(axis=0)) ** 2).sum(axis=0)
    wss = np.zeros(X.shape[1])
    for k in np.unique(labels):
        Xk = X[labels == k]
        wss += ((Xk - Xk.mean(axis=0)) ** 2).sum(axis=0)
    return tss - wss


def sparse_kmeans(X: np.ndarray, k: int, s: float, n_start: int = 20, max_iter: int = 6,
                  tol: float = 1e-4, random_state: int = 42):
    """Coordinate descent alternato: (A) K-Means su X*sqrt(w); (B) aggiornamento di w.

    Ritorna (w, labels). `X` deve essere già standardizzata (Z-score).
    """
    p = X.shape[1]
    w = np.full(p, 1 / np.sqrt(p))
    labels = None
    for _ in range(max_iter):
        km = KMeans(k, n_init=n_start, random_state=random_state).fit(X * np.sqrt(w))
        labels = km.labels_
        w_new = update_weights(_bcss(X, labels), s)
        converged = np.abs(w_new - w).sum() / max(np.abs(w).sum(), 1e-12) < tol
        w = w_new
        if converged:
            break
    return w, labels


def _log_wcss(X: np.ndarray, k: int, s: float, random_state: int) -> float:
    if k == 1:
        w = np.full(X.shape[1], s / np.sqrt(X.shape[1]))
        return float(np.log(((X - X.mean(0)) ** 2).sum(0) @ w + 1e-10))
    w, labels = sparse_kmeans(X, k, s, n_start=5, random_state=random_state)
    wcss = 0.0
    for c in np.unique(labels):
        Xc = X[labels == c]
        wcss += (((Xc - Xc.mean(0)) ** 2).sum(0) * w).sum()
    return float(np.log(wcss + 1e-10))


def gap_statistic(X: np.ndarray, s: float, k_max: int = 5, n_ref: int = 5, random_state: int = 42):
    """Gap Statistic con regola dell'1-standard-error; riferimento nullo uniforme nel bounding box
    dei dati ruotati sulle componenti principali (par. 2.1.3). Ritorna (K ottimo, gap, s_k)."""
    rng = np.random.default_rng(random_state)
    Xc = X - X.mean(0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    Xp = Xc @ vt.T
    lo, hi = Xp.min(0), Xp.max(0)

    gaps, sks = [], []
    for k in range(1, k_max + 1):
        real = _log_wcss(X, k, s, random_state)
        refs = []
        for _ in range(n_ref):
            ref = (rng.uniform(lo, hi, size=Xp.shape) @ vt) + X.mean(0)
            refs.append(_log_wcss(ref, k, s, random_state))
        gaps.append(np.mean(refs) - real)
        sks.append(np.std(refs) * np.sqrt(1 + 1 / n_ref))
    k_hat = k_max
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i + 1] - sks[i + 1]:
            k_hat = i + 1
            break
    return k_hat, gaps, sks
