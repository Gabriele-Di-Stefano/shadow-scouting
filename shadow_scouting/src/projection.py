"""Weighted KNN pseudo-supervisionato (tesi, par. 2.2.2).

Le pseudo-label dei Top (cluster dello Sparse K-Means) sono il training set; ogni Under-22 è
proiettato nello spazio d'élite con la distanza euclidea ponderata (pesi Lasso) e riceve
l'archetipo con voto maggioritario pesato 1/d sui M Top più vicini.
"""
from __future__ import annotations

import numpy as np

DEFAULT_M = 1   # M = 1: ogni giovane è mappato sul singolo Top più simile (come nel codice originale)


def weighted_distances(Zy: np.ndarray, Zx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Matrice (n_giovani x n_top) di d_w(y_i, x_r) = sqrt(sum_j w_j (y_ij - x_rj)^2)."""
    sw = np.sqrt(w)
    A, B = Zy * sw, Zx * sw
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.sqrt(np.maximum(sq, 0.0))


def wknn_assign(D: np.ndarray, top_labels: np.ndarray, m: int = DEFAULT_M, eps: float = 1e-9) -> np.ndarray:
    """Archetipo di ciascun giovane: argmax_k sum_{r in N_M} 1[C_r = k] / d(y, x_r)."""
    m = min(m, D.shape[1])
    idx = np.argsort(D, axis=1)[:, :m]
    dist = np.take_along_axis(D, idx, axis=1) + eps
    ks = np.unique(top_labels)
    votes = np.stack([((top_labels[idx] == k) / dist).sum(axis=1) for k in ks], axis=1)
    return ks[votes.argmax(axis=1)]
