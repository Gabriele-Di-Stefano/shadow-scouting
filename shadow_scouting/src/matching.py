"""Matching chirurgico 1-to-1 (tesi, par. 2.3 e 3.3) e identificazione dei "Calciatori Unicorno"."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .projection import weighted_distances


def shadow_matches(top_space, u22: pd.DataFrame, u22_labels: np.ndarray, top_n: int = 50) -> pd.DataFrame:
    """Per ogni Top: i giovani dello STESSO cluster ordinati per distanza ponderata (Rank 1 = Shadow Match)."""
    Zy = top_space.transform(u22)
    D = weighted_distances(Zy, top_space.Z, top_space.weights.values)   # (n_u22, n_top)
    rows = []
    for r, top in top_space.top.iterrows():
        cluster = top_space.labels[r]
        cand = np.where(u22_labels == cluster)[0]
        if len(cand) == 0:
            continue
        order = cand[np.argsort(D[cand, r])][:top_n]
        for rank, i in enumerate(order, start=1):
            rows.append({
                "position": top_space.position, "top_id": top["player_id"], "top_player": top["player"],
                "cluster": int(cluster), "archetype": top_space.archetype_name(cluster),
                "rank": rank, "u22_id": u22.loc[i, "player_id"], "u22_player": u22.loc[i, "player"],
                "distance": float(D[i, r]),
            })
    return pd.DataFrame(rows)


def flag_unicorns(matches: pd.DataFrame) -> pd.DataFrame:
    """Unicorno: distanza di Rank 1 oltre i baffi superiori del boxplot del cluster (Q3 + 1.5*IQR)."""
    r1 = matches[matches["rank"] == 1].copy()
    g = r1.groupby(["position", "cluster"])["distance"]
    q1, q3 = g.transform(lambda s: s.quantile(0.25)), g.transform(lambda s: s.quantile(0.75))
    r1["unicorn"] = r1["distance"] > q3 + 1.5 * (q3 - q1)
    return r1[["top_id", "unicorn"]]
