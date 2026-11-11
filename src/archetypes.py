"""Archetipi d'élite (tesi, par. 4.1.2) e assegnazione dei cluster dei Top Player.

I pesi Lasso w_j vengono da Sparse K-Means (Fase 1, pysparcl) e sono letti da `artifacts/`.
I cluster dei Top si ricostruiscono con un K-Means sullo spazio pesato X*sqrt(w) (Fase A di
Sparse K-Means a pesi fissati). Se esiste `artifacts/top_clusters_original.csv`
(player_id, cluster) si usano invece le etichette originali della tesi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .config import ARTIFACTS_DIR, N_CLUSTERS, WEIGHTS_FILES

# Nome dell'archetipo, nella numerazione dei cluster della tesi (par. 4.1.2)
ARCHETYPE_NAMES = {
    "DF": {0: "Terzini d'Assalto", 1: "Registi Arretrati", 2: "Mastini della Marcatura"},
    "MF": {0: "Incursori Goleador", 1: "Cuori Pulsanti", 2: "Pendolari del Cross"},
    "FW": {0: "Rapaci d'Area", 1: "Registi Offensivi", 2: "Guerrieri del Pressing", 3: "Funamboli Dribblatori"},
}

# Giocatori-àncora citati nella tesi: servono a numerare i cluster come nella tesi
ANCHORS = {
    "DF": {0: "Giovanni Di Lorenzo", 1: "Virgil van Dijk", 2: "Alessandro Bastoni"},
    "MF": {0: "Jude Bellingham", 1: "Rodri", 2: "Federico Dimarco"},
    "FW": {0: "Erling Haaland", 1: "Paulo Dybala", 2: "Khvicha Kvaratskhelia", 3: "Rafael Leão"},
}


def load_weights(position: str, artifacts_dir=ARTIFACTS_DIR) -> pd.Series:
    w = pd.read_csv(artifacts_dir / WEIGHTS_FILES[position], index_col=0)["Lasso_Weight"]
    return w[w > 0].sort_values(ascending=False)


class TopSpace:
    """Spazio d'élite di un reparto: feature pesate, scaler dei Top, cluster e centroidi."""

    def __init__(self, position: str, top_df: pd.DataFrame, artifacts_dir=ARTIFACTS_DIR):
        self.position = position
        self.weights = load_weights(position, artifacts_dir)
        self.features = list(self.weights.index)
        X = top_df[self.features].astype(float)
        self.mean_ = X.mean()
        self.std_ = X.std(ddof=0)                      # Z-score a varianza unitaria (StandardScaler)
        self.Z = ((X - self.mean_) / self.std_).values  # Top standardizzati
        self.top = top_df.reset_index(drop=True)
        self.labels = self._cluster_labels(artifacts_dir)

    def _cluster_labels(self, artifacts_dir) -> np.ndarray:
        k = N_CLUSTERS[self.position]
        orig = artifacts_dir / "top_clusters_original.csv"
        if orig.exists():
            lab = pd.read_csv(orig).set_index("player_id")["cluster"]
            return self.top["player_id"].map(lab).astype(int).values
        Zw = self.Z * np.sqrt(self.weights.values)
        best = min((KMeans(k, n_init=1, random_state=s).fit(Zw) for s in range(200)),
                   key=lambda km: km.inertia_)
        return self._renumber(best.labels_)

    def _renumber(self, raw: np.ndarray) -> np.ndarray:
        """Rinumera i cluster come nella tesi, usando i giocatori-àncora."""
        names = self.top["player"].values
        mapping = {}
        for new_id, anchor in ANCHORS[self.position].items():
            hit = np.where(names == anchor)[0]
            if len(hit):
                mapping[int(raw[hit[0]])] = new_id
        return np.array([mapping.get(int(r), int(r)) for r in raw])

    def set_scaling(self, mode: str, u22_df: pd.DataFrame | None = None) -> None:
        """Scelta della standardizzazione usata per le distanze.

        "top"      : media e dev. std dei soli Top (come descritto nel par. 3.3 della tesi);
        "combined" : media e dev. std calcolate su Top + Under-22 insieme (come fa find_under22_shadows
                     nel notebook originale: con questa scelta si riproducono i boxplot del cap. 4.3).
        """
        if mode == "top":
            return
        if mode != "combined" or u22_df is None:
            raise ValueError("mode deve essere 'top' o 'combined' (con u22_df)")
        U = u22_df[self.features].astype(float)
        U = U.fillna(U.mean())
        T = self.top[self.features].astype(float)
        both = pd.concat([T, U])
        self.mean_, self.std_ = both.mean(), both.std(ddof=0)
        self.Z = ((T - self.mean_) / self.std_).values

    def assignment_matrices(self, u22_df: pd.DataFrame, mode: str = "own"):
        """Matrici standardizzate (Top, Under-22) per la Fase 2 (assegnazione agli archetipi).

        "own"      : Top e Under-22 standardizzati ciascuno con le proprie media e dev. std
                     (riproduce l'assegnazione del notebook originale);
        "combined" : media e dev. std su Top + Under-22 insieme;
        "top"      : media e dev. std dei soli Top.
        """
        T = self.top[self.features].astype(float)
        U = u22_df[self.features].astype(float)
        U = U.fillna(U.mean())
        if mode == "own":
            return ((T - T.mean()) / T.std(ddof=0)).values, ((U - U.mean()) / U.std(ddof=0)).values
        if mode == "combined":
            both = pd.concat([T, U])
            mu, sd = both.mean(), both.std(ddof=0)
        elif mode == "top":
            mu, sd = T.mean(), T.std(ddof=0)
        else:
            raise ValueError("mode deve essere 'own', 'combined' o 'top'")
        return ((T - mu) / sd).values, ((U - mu) / sd).values

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Standardizza un DataFrame con media e dev. std correnti (di default quelle dei Top, par. 3.3).
        I valori mancanti sono imputati con la media del DataFrame stesso."""
        X = df[self.features].astype(float)
        X = X.fillna(X.mean())
        return ((X - self.mean_) / self.std_).values

    def representatives(self, universe: list[str]) -> np.ndarray:
        """Indice (in self.top) del Top più vicino al centroide di ciascun cluster, nello spazio standardizzato
        di tutte le metriche senza valori mancanti (come nel notebook)."""
        cols = [c for c in universe if self.top[c].notna().all()]
        X = self.top[cols].astype(float)
        Z = ((X - X.mean()) / X.std(ddof=0)).dropna(axis=1).values
        reps = []
        for k in np.unique(self.labels):
            idx = np.where(self.labels == k)[0]
            d = np.linalg.norm(Z[idx] - Z[idx].mean(axis=0), axis=1)
            reps.append(idx[d.argmin()])
        return np.array(reps)

    def assign_original(self, u22_df: pd.DataFrame, universe: list[str]) -> np.ndarray:
        """Assegnazione della coorte U22 come nel notebook originale (`assign_players_knn`, lasso, k=1):
        distanza euclidea NON pesata sulle feature Lasso, 1-NN verso i soli Top rappresentativi dei cluster;
        Top (rappresentativi) e U22 standardizzati ciascuno con le proprie media e dev. std."""
        reps = self.representatives(universe)
        R = self.top.iloc[reps][self.features].astype(float)
        U = u22_df[self.features].astype(float)
        U = U.fillna(U.mean())
        Zr = ((R - R.mean()) / R.std(ddof=0)).values
        Zu = ((U - U.mean()) / U.std(ddof=0)).values
        d = ((Zu[:, None, :] - Zr[None, :, :]) ** 2).sum(axis=2)
        return self.labels[reps][d.argmin(axis=1)]

    def archetype_name(self, cluster: int) -> str:
        return ARCHETYPE_NAMES[self.position].get(int(cluster), f"Cluster {cluster}")
