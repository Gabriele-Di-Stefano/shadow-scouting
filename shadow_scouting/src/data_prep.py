"""Costruzione del master dataset da Fbref/Opta 2023/24 (tesi, cap. 3: igiene strutturale del dato).

Operazioni (come da tesi):
 1. rimozione colonne duplicate tra tabelle (Pass_types.passes/passes_comp, Misc.crosses/tklw);
 2. rimozione di Defense.intr (correlazione 1 con Misc.interceptions);
 3. conversione dei minuti da stringa con separatore delle migliaia a numerico;
 4. outer join delle tabelle prestazionali su player_id e left merge dei dati di contesto.
"""
from __future__ import annotations

import pandas as pd

from .config import RAW_DIR

PERF_TABLES = ["Playing_Time", "Shooting", "Passing", "Pass_types", "GSC", "Defense", "Possession", "Misc"]


def _read(name: str, raw_dir=RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "normalized" / f"{name}.csv")


def _add_shots(shooting: pd.DataFrame, raw_dir=RAW_DIR) -> pd.DataFrame:
    """Aggiunge `shots` (assente nella tabella normalizzata) dal file denormalizzato.

    Il join NON può essere per posizione di riga: nel file denormalizzato gli omonimi compaiono
    in ordine diverso (es. due "Aldo Cruz"), quindi si allinea con una chiave fatta di statistiche.
    """
    den = pd.read_csv(raw_dir / "Shooting.csv")
    players = pd.read_csv(raw_dir / "normalized" / "t_player.csv")[["player_id", "player", "age"]]
    keys = ["player", "age", "goals", "SOTs", "xg", "shots_per90", "avg_shot_dist", "shots_fk"]
    base = shooting.merge(players, on="player_id", how="left")
    den = den[keys + ["shots"]].drop_duplicates(subset=keys, keep=False)  # solo chiavi univoche
    out = base.merge(den, on=keys, how="left")
    return out.drop(columns=["player", "age"])


def build_master(raw_dir=RAW_DIR) -> pd.DataFrame:
    perf = {n: _read(n, raw_dir) for n in PERF_TABLES}

    perf["Pass_types"] = perf["Pass_types"].drop(columns=["passes", "passes_comp"], errors="ignore")
    perf["Misc"] = perf["Misc"].drop(columns=["crosses", "tklw"], errors="ignore")
    perf["Defense"] = perf["Defense"].drop(columns=["intr"], errors="ignore")
    perf["Shooting"] = _add_shots(perf["Shooting"], raw_dir)

    pt = perf["Playing_Time"].copy()
    pt["minutes"] = pd.to_numeric(pt["minutes"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    perf["Playing_Time"] = pt.rename(columns={"games": "matches_played", "minutes": "minutes_played"})

    master = perf["Playing_Time"]
    for name in PERF_TABLES[1:]:
        dup = (set(master.columns) & set(perf[name].columns)) - {"player_id"}
        assert not dup, f"colonne duplicate tra tabelle ({name}): {dup}"
        master = master.merge(perf[name], on="player_id", how="outer")

    master = _read("t_player", raw_dir).merge(master, on="player_id", how="left")
    master = (master
              .merge(_read("t_position", raw_dir), left_on="position_id", right_on="pos_id", how="left")
              .merge(_read("t_nationality", raw_dir), left_on="nationality_id", right_on="nat_id", how="left")
              .merge(_read("t_club", raw_dir), on="club_id", how="left")
              .merge(_read("t_leagues", raw_dir), on="league_id", how="left")
              .drop(columns=["position_id", "pos_id", "nationality_id", "nat_id", "club_id", "league_id"])
              .rename(columns={"continent": "league_continent"}))
    return master
