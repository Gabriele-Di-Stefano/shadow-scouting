"""Shadow Scouting nel Calcio Moderno — app Streamlit.

Scegli uno dei 75 Top Player della tesi e ottieni la classifica dei giovani Under-22 più vicini
(distanza euclidea ponderata con i pesi Lasso dello Sparse K-Means), nello stesso archetipo tattico.

    streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.plots import radar_compare

ART = Path(__file__).parent / "artifacts"
POS_LABEL = {"DF": "Difensori", "MF": "Centrocampisti", "FW": "Attaccanti"}

st.set_page_config(page_title="Shadow Scouting", page_icon="⚽", layout="wide")


@st.cache_data
def load():
    tops = pd.read_csv(ART / "top_players.csv")
    matches = pd.read_csv(ART / "matches.csv")
    radar = pd.read_csv(ART / "radar.csv")
    return tops, matches, radar


tops, matches, radar = load()

st.title("⚽ Shadow Scouting nel Calcio Moderno")
st.caption("Tesi di laurea di Gabriele Di Stefano · Università di Milano-Bicocca · dati Fbref/Opta, stagione 2023/24")

# ---------- barra laterale ----------
with st.sidebar:
    st.header("Ricerca")
    pos = st.radio("Reparto", ["Tutti"] + list(POS_LABEL), format_func=lambda x: "Tutti" if x == "Tutti" else POS_LABEL[x],
                   horizontal=False)
    pool = tops if pos == "Tutti" else tops[tops["position"] == pos]
    names = sorted(pool["player"])
    name = st.selectbox("Top Player (digita per cercare)", names, index=None, placeholder="es. Erling Haaland")

    st.header("Filtri sui giovani")
    n_show = st.slider("Quanti giovani mostrare", 5, 50, 10)
    max_age = st.slider("Età massima", 14, 22, 22)
    min_min = st.slider("Minuti minimi giocati", 0, 2500, 0, step=100,
                        help="La tesi non applica filtri sui minuti; qui è solo un aiuto per lo scouting.")
    leagues = sorted(matches["u22_league"].dropna().unique())
    chosen = st.multiselect("Campionati", leagues, placeholder="tutti")

with st.expander("Come funziona"):
    st.markdown(
        "1. **Fase 1** – Sparse K-Means con penalizzazione Lasso sui 75 Top Player (25 per reparto) identifica gli "
        "archetipi tattici e i pesi delle variabili discriminanti.\n"
        "2. **Fase 2** – I giovani Under-22 vengono proiettati negli archetipi con un Weighted KNN pseudo-supervisionato.\n"
        "3. **Fase 3** – Dentro lo stesso archetipo, la **distanza euclidea ponderata** (pesi Lasso) ordina i giovani: "
        "il Rank 1 è lo *Shadow Match*. Un Top è un **Unicorno** se la distanza del suo Rank 1 supera Q3 + 1.5·IQR "
        "dei Rank 1 del suo archetipo."
    )

if not name:
    st.info("Scegli un Top Player nella barra laterale per vedere i giovani più simili.")
    st.stop()

top = tops[tops["player"] == name].iloc[0]

# ---------- scheda del Top ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Top Player", top["player"])
c2.metric("Club", top["club"], top["league"], delta_color="off")
c3.metric("Archetipo", top["archetype"], POS_LABEL[top["position"]], delta_color="off")
c4.metric("Distanza Rank 1", f"{top['rank1_distance']:.2f}")
if top["unicorn"]:
    st.warning("🦄 **Calciatore Unicorno**: il suo profilo è difficilmente replicabile nella coorte Under-22 "
               "(distanza di Rank 1 oltre i baffi superiori del boxplot del suo archetipo).")

# ---------- classifica ----------
m = matches[matches["top_id"] == top["top_id"]].copy()
m = m[(m["u22_age"] <= max_age) & (m["u22_minutes"].fillna(0) >= min_min)]
if chosen:
    m = m[m["u22_league"].isin(chosen)]
m = m.sort_values("distance").head(n_show)

st.subheader(f"Giovani più vicini a {top['player']}")
if m.empty:
    st.write("Nessun giovane soddisfa i filtri scelti.")
    st.stop()

table = pd.DataFrame({
    "#": range(1, len(m) + 1),
    "Giocatore": m["u22_player"].values,
    "Età": m["u22_age"].values,
    "Club": m["u22_club"].values,
    "Campionato": m["u22_league"].values,
    "Nazionalità": m["u22_nationality"].values,
    "Minuti": m["u22_minutes"].values,
    "Distanza": m["distance"].round(2).values,
})
st.dataframe(table, hide_index=True, use_container_width=True)

# ---------- radar ----------
st.subheader("Confronto di profilo")
pick = st.selectbox("Giovane da confrontare", table["Giocatore"], index=0)
row = m[m["u22_player"] == pick].iloc[0]
fig = radar_compare(radar, int(top["top_id"]), int(row["u22_id"]), top["position"], top["player"], pick)
if fig is not None:
    left, _ = st.columns([1, 1])
    left.pyplot(fig)
    st.caption("Le otto variabili con peso Lasso più alto del reparto, in deviazioni standard rispetto ai Top "
               "dello stesso reparto (valori troncati a ±3σ).")
