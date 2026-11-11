"""Esegue la pipeline della tesi una sola volta e salva i file leggeri letti dall'app Streamlit.

    python -m scripts.build_artifacts            # dalla radice del progetto
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.archetypes import TopSpace
from src.cohorts import cohort_summary, feature_universe, get_top, get_u22
from src.config import ARTIFACTS_DIR, ASSIGN_METHOD, ASSIGN_SCALING, MATCH_SCALING, POSITIONS
from src.data_prep import build_master
from src.matching import flag_unicorns, shadow_matches
from src.projection import DEFAULT_M, weighted_distances, wknn_assign

N_RADAR = 8          # feature con peso Lasso più alto mostrate nel radar
TOP_N = 150          # giovani salvati per ciascun Top
INFO = ["player_id", "player", "age", "club", "league", "nationality", "minutes_played"]


def main(m_neighbors: int = DEFAULT_M, assign_method: str = ASSIGN_METHOD, assign_scaling: str = ASSIGN_SCALING,
         match_scaling: str = MATCH_SCALING) -> None:
    master = build_master()
    universe = feature_universe(master)
    print(cohort_summary(master).to_string(index=False))

    all_matches, top_rows, radar_rows = [], [], []
    for pos in POSITIONS:
        space = TopSpace(pos, get_top(master, pos))
        u22 = get_u22(master, pos)

        # Fase 2: proiezione WKNN della coorte Under-22 negli archetipi d'élite
        if assign_method == "original":
            u22_labels = space.assign_original(u22, universe)
        else:
            Zt_a, Zu_a = space.assignment_matrices(u22, assign_scaling)
            D = weighted_distances(Zu_a, Zt_a, space.weights.values)
            u22_labels = wknn_assign(D, space.labels, m=m_neighbors)

        space.set_scaling(match_scaling, u22)   # standardizzazione per le distanze della Fase 3

        # Fase 3: matching intra-cluster con distanza euclidea ponderata
        matches = shadow_matches(space, u22, u22_labels, top_n=TOP_N)
        all_matches.append(matches)

        top = space.top[INFO].copy()
        top["position"], top["cluster"] = pos, space.labels
        top["archetype"] = [space.archetype_name(c) for c in space.labels]
        top_rows.append(top.rename(columns={"player_id": "top_id"}))

        # z-score (scaler dei Top) delle feature principali, per il radar Top vs giovane
        feats = space.features[:N_RADAR]
        used = pd.Index(matches["u22_id"].unique())
        Zu = pd.DataFrame(space.transform(u22), columns=space.features)[feats]
        Zu["player_id"] = u22["player_id"].values
        Zu = Zu[Zu["player_id"].isin(used)]
        Zt = pd.DataFrame(space.Z, columns=space.features)[feats]
        Zt["player_id"] = space.top["player_id"].values
        for df in (Zt, Zu):
            radar_rows.append(df.melt(id_vars="player_id", var_name="feature", value_name="z")
                              .assign(position=pos))

    matches = pd.concat(all_matches, ignore_index=True)
    tops = pd.concat(top_rows, ignore_index=True)
    r1 = matches[matches["rank"] == 1][["top_id", "distance"]].rename(columns={"distance": "rank1_distance"})
    tops = tops.merge(r1, on="top_id", how="left").merge(flag_unicorns(matches), on="top_id", how="left")

    u_info = master[INFO].rename(columns={"player_id": "u22_id", "player": "_p", "age": "u22_age",
                                          "club": "u22_club", "league": "u22_league",
                                          "nationality": "u22_nationality", "minutes_played": "u22_minutes"})
    matches = matches.merge(u_info.drop(columns="_p"), on="u22_id", how="left")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    tops.to_csv(ARTIFACTS_DIR / "top_players.csv", index=False)
    matches.to_csv(ARTIFACTS_DIR / "matches.csv", index=False)
    pd.concat(radar_rows, ignore_index=True).drop_duplicates(["player_id", "feature", "position"]) \
        .to_csv(ARTIFACTS_DIR / "radar.csv", index=False)
    print(f"salvati: {len(tops)} Top, {len(matches)} abbinamenti")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=DEFAULT_M, help="vicini M del WKNN")
    ap.add_argument("--assign-method", choices=["original", "wknn"], default=ASSIGN_METHOD)
    ap.add_argument("--assign-scaling", choices=["own", "combined", "top"], default=ASSIGN_SCALING)
    ap.add_argument("--match-scaling", choices=["combined", "top"], default=MATCH_SCALING)
    a = ap.parse_args()
    main(a.m, a.assign_method, a.assign_scaling, a.match_scaling)
