"""Coorti della tesi: 75 Top Player (par. 3.1) e coorte Under-22 outfield (par. 3.2)."""
from __future__ import annotations

import pandas as pd

from .config import ID_COLS, POSITIONS, TOP_IDS, U22_MAX_AGE

ALL_TOP_IDS = {pid for ids in TOP_IDS.values() for pid in ids}


def get_top(master: pd.DataFrame, position: str) -> pd.DataFrame:
    """Top Player del reparto `position`, nell'ordine di TOP_IDS."""
    top = master[master["player_id"].isin(TOP_IDS[position])]
    return top.set_index("player_id").loc[TOP_IDS[position]].reset_index()


def get_u22(master: pd.DataFrame, position: str) -> pd.DataFrame:
    """Under-22 del reparto `position` (ruolo Fbref esatto), SENZA alcun Top Player di nessun reparto.

    Nessun filtro su minutaggio o campionato (par. 3.2). I Top con età <= 22 vanno esclusi
    per tutti i reparti: altrimenti verrebbero abbinati a sé stessi con distanza 0.
    """
    mask = (master["age"] <= U22_MAX_AGE) & (master["position"] == position)
    mask &= ~master["player_id"].isin(ALL_TOP_IDS)
    return master[mask].reset_index(drop=True)


def cohort_summary(master: pd.DataFrame) -> pd.DataFrame:
    rows = [{"position": p, "top": len(get_top(master, p)), "u22": len(get_u22(master, p))} for p in POSITIONS]
    return pd.DataFrame(rows)


def feature_universe(master: pd.DataFrame) -> list[str]:
    """Metriche numeriche usate nel clustering: colonne senza valori mancanti tra i difensori U21
    (come nel notebook, da cui nasce l'elenco delle statistiche `stats_non_gk`)."""
    num = [c for c in master.columns if c not in ID_COLS and pd.api.types.is_numeric_dtype(master[c])]
    ref = master[(master["age"] <= 21) & (master["position"] == "DF")]
    return [c for c in num if ref[c].notna().all()]
