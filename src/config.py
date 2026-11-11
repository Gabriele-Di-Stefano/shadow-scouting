"""Costanti del progetto: percorsi, coorti Top Player e parametri (tesi, cap. 3-4)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"          # CSV Kaggle: normalized/*.csv + Shooting.csv
ARTIFACTS_DIR = ROOT / "artifacts"

POSITIONS = ["DF", "MF", "FW"]            # reparti di movimento (portieri: sviluppo futuro)
POSITION_LABELS = {"DF": "Difensori", "MF": "Centrocampisti", "FW": "Attaccanti"}
U22_MAX_AGE = 22

# 75 Top Player (5 per reparto per ciascuno dei Big 5), selezionati via rating Sofascore (tesi, par. 3.1)
TOP_IDS = {
    "DF": [8, 70, 11, 263, 129, 3598, 3548, 3589, 3651, 3938, 2778, 2421, 2395, 2818, 2419,
           1446, 1470, 1396, 1368, 1377, 3006, 3032, 3007, 3059, 3093],
    "MF": [5, 7, 184, 267, 66, 3539, 3540, 3744, 3681, 3566, 2394, 2396, 2560, 2418, 2448,
           1366, 1445, 1418, 1369, 1372, 3030, 3001, 3317, 3063, 3065],
    "FW": [6, 265, 157, 32, 4, 3568, 3652, 3712, 3545, 3542, 2748, 2474, 2596, 2398, 2393,
           1415, 1395, 1397, 1543, 1370, 3000, 3176, 3011, 3141, 3201],
}

# K ottimale per reparto (Gap Statistic, tesi par. 4.1.2)
N_CLUSTERS = {"DF": 3, "MF": 3, "FW": 4}

# Pesi Lasso estratti dallo Sparse K-Means (Fase 1), uno per reparto
WEIGHTS_FILES = {
    "DF": "variabili_importanti_difensori_top_df.csv",
    "MF": "variabili_importanti_centrocampisti_top_df.csv",
    "FW": "variabili_importanti_attaccanti_top_df.csv",
}

# Colonne identificative/di contesto (mai usate come feature)
ID_COLS = ["player_id", "player", "nationality", "position", "age", "club",
           "league", "division", "Continent", "no_clubs", "country", "league_continent"]

# Parametri della pipeline, scelti per riprodurre i risultati originali della tesi
# Fase 2, metodo di assegnazione degli Under-22 agli archetipi:
#   "original": come nel notebook (1-NN NON pesato sui Top rappresentativi dei cluster) -> riproduce i risultati della tesi
#   "wknn"    : WKNN descritto nel par. 2.2.2 (distanza pesata con i pesi Lasso su tutti i Top, voto 1/d sui M vicini)
ASSIGN_METHOD = "original"
WKNN_M = 1                    # vicini del WKNN: 1 = ogni giovane è mappato sul singolo Top più simile
ASSIGN_SCALING = "own"        # Fase 2: "own" | "combined" | "top"  (vedi TopSpace.assignment_matrices)
MATCH_SCALING = "combined"    # Fase 3: "combined" (notebook originale) | "top" (descrizione del par. 3.3)
