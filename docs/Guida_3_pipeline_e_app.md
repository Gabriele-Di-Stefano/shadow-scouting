# Guida 3 — Pipeline e app: `build_artifacts.py`, `plots.py`, `app.py`

*Spiegato riga per riga, per chi non conosce Python*

Progetto: Shadow Scouting nel Calcio Moderno

## A cosa servono questi tre file

- `scripts/build_artifacts.py` è il **direttore d'orchestra**: chiama in ordine i moduli delle guide 1 e 2 (dati → coorti → archetipi → assegnazione → matching) e salva i risultati in tre file CSV leggeri nella cartella `artifacts/`. Si esegue **una volta sola** (o quando si cambia qualcosa).
- `src/plots.py` disegna il **grafico radar** che confronta un Top Player con un giovane.
- `app.py` è l'**applicazione web** Streamlit: non rifà nessun calcolo, legge i CSV di `artifacts/` e li mostra con menu, tabelle e grafici.

Questa separazione rende l'app veloce e semplice da pubblicare: online non servono i dati di Kaggle, bastano i tre CSV.

## Come si legge il codice: le regole di base

- Il programma si legge **dall'alto verso il basso**; `#` introduce un commento.
- Un **DataFrame** (`df`) è una tabella gestita da *pandas*.
- **Regola chiave di Streamlit:** ogni volta che l'utente tocca un controllo (menu, cursore, casella), l'intero file `app.py` viene **rieseguito da capo**. Per questo i dati si caricano con la *cache*, una memoria che evita di rileggere i file a ogni click.
- `st.stop()` interrompe l'esecuzione della pagina in quel punto: tutto ciò che segue non viene mostrato.

## Mappa dei file

| File | Righe | Cosa contiene |
| --- | --- | --- |
| `build_artifacts.py` | 1–90 | Importazioni, funzione `main` con le tre fasi, riga di comando |
| `plots.py` | 1–35 | Funzione `radar_compare` |
| `app.py` | 1–110 | Pagina, barra laterale, scheda del Top, classifica, radar |

---

# `scripts/build_artifacts.py`

## Documentazione e importazioni (righe 1–15)

```python
import numpy as np
import pandas as pd

from src.archetypes import TopSpace
from src.cohorts import cohort_summary, feature_universe, get_top, get_u22
from src.config import ARTIFACTS_DIR, ASSIGN_METHOD, ASSIGN_SCALING, MATCH_SCALING, POSITIONS
from src.data_prep import build_master
from src.matching import flag_unicorns, shadow_matches
from src.projection import DEFAULT_M, weighted_distances, wknn_assign
```

- **Righe 1–4**: la documentazione dice come si lancia: `python -m scripts.build_artifacts` dalla cartella principale.
- **Righe 7–8**: librerie numpy e pandas.
- **Righe 10–15**: importa i «mattoni» dagli altri file del progetto: la classe `TopSpace`, le funzioni delle coorti, le impostazioni, la costruzione del master, il matching e le distanze.

## Costanti (righe 17–19)

```python
N_RADAR = 8
TOP_N = 150
INFO = ["player_id", "player", "age", "club", "league", "nationality", "minutes_played"]
```

- **Riga 17**: quante variabili mostrare nel radar: le 8 con peso Lasso più alto.
- **Riga 18**: per ogni Top si salvano i **150 giovani più vicini** (il filtro dell'app poi ne mostra al massimo 50).
- **Riga 19**: le colonne anagrafiche da conservare per ogni giocatore: ID, nome, età, club, campionato, nazionalità, minuti giocati.

## `main`: definizione e primi passi (righe 22–26)

```python
def main(m_neighbors=DEFAULT_M, assign_method=ASSIGN_METHOD, assign_scaling=ASSIGN_SCALING,
         match_scaling=MATCH_SCALING) -> None:
    master = build_master()
    universe = feature_universe(master)
    print(cohort_summary(master).to_string(index=False))
```

- **Righe 22–23**: la funzione `main` ha quattro parametri, ognuno con un valore predefinito che viene dal file `config.py`. `-> None` indica che non restituisce nulla: salva dei file.
- **Riga 24**: costruisce il master dataset (guida 1).
- **Riga 25**: ricava l'elenco delle ~141 metriche.
- **Riga 26**: stampa a schermo quanti Top e quanti Under-22 ci sono per reparto, come controllo veloce.

## Ciclo sui reparti: Fase 2 (righe 28–41)

```python
all_matches, top_rows, radar_rows = [], [], []
for pos in POSITIONS:
    space = TopSpace(pos, get_top(master, pos))
    u22 = get_u22(master, pos)

    if assign_method == "original":
        u22_labels = space.assign_original(u22, universe)
    else:
        Zt_a, Zu_a = space.assignment_matrices(u22, assign_scaling)
        D = weighted_distances(Zu_a, Zt_a, space.weights.values)
        u22_labels = wknn_assign(D, space.labels, m=m_neighbors)

    space.set_scaling(match_scaling, u22)
```

- **Riga 28**: tre liste vuote dove si raccoglieranno i risultati parziali dei tre reparti.
- **Riga 29**: **ciclo**: tutto ciò che è rientrato sotto si ripete tre volte, per `"DF"`, `"MF"` e `"FW"`.
- **Riga 30**: crea lo spazio d'élite del reparto con i suoi 25 Top (pesi, scaler, cluster).
- **Riga 31**: prende i giovani del reparto (senza i Top).
- **Riga 33**: commento: la Fase 2 assegna ogni giovane a un archetipo.
- **Righe 34–35**: **se** il metodo è `"original"`, usa 1-NN sui rappresentanti (la procedura del notebook).
- **Righe 36–39**: **altrimenti** (`else`) usa il WKNN: standardizza (riga 37), calcola le distanze ponderate di ogni giovane da ogni Top (riga 38) e assegna il cluster per voto sui `m` vicini (riga 39).
- **Riga 41**: **cambia** lo scaler dello spazio, per le distanze della Fase 3 (di default `"combined"`: media di Top e giovani insieme).
- Nota: l'ordine conta. L'assegnazione usa gli scaler suoi propri (righe 34–39), il matching usa quello impostato alla riga 41.

## Ciclo sui reparti: Fase 3 (righe 43–50)

```python
matches = shadow_matches(space, u22, u22_labels, top_n=TOP_N)
all_matches.append(matches)

top = space.top[INFO].copy()
top["position"], top["cluster"] = pos, space.labels
top["archetype"] = [space.archetype_name(c) for c in space.labels]
top_rows.append(top.rename(columns={"player_id": "top_id"}))
```

- **Riga 44**: esegue il matching (guida 2): per ogni Top, i 150 giovani più vicini nello stesso cluster.
- **Riga 45**: aggiunge il risultato alla lista `all_matches`.
- **Riga 47**: costruisce la tabella dei 25 Top, tenendo solo le colonne anagrafiche.
- **Riga 48**: aggiunge due colonne: il reparto (lo stesso valore su tutte le righe) e il numero del cluster di ciascun Top.
- **Riga 49**: aggiunge il **nome** dell'archetipo con una *list comprehension* (una lista costruita con un ciclo su una riga).
- **Riga 50**: rinomina `player_id` in `top_id` (per distinguerlo dagli ID dei giovani) e aggiunge la tabella alla lista.

## Ciclo sui reparti: dati per il radar (righe 52–62)

```python
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
```

- **Riga 53**: `[:N_RADAR]` prende le prime 8 variabili dell'elenco, che è ordinato per peso decrescente: sono le 8 più importanti.
- **Riga 54**: `used` = gli ID dei giovani che compaiono in almeno un abbinamento (`unique()` toglie le ripetizioni). Salvare il radar solo per loro tiene il file piccolo.
- **Riga 55**: standardizza **tutti** i giovani e ne fa una tabella con i nomi delle variabili come colonne; poi tiene solo le 8 del radar.
- **Riga 56**: aggiunge la colonna con l'ID del giocatore.
- **Riga 57**: tiene solo i giovani che servono (`isin(used)`).
- **Righe 58–59**: la stessa cosa per i Top.
- **Riga 60**: ciclo sulle due tabelle (Top e giovani).
- **Righe 61–62**: `melt` trasforma la tabella da «larga» (una colonna per variabile) a «lunga» (una riga per coppia giocatore-variabile, con colonne `feature` e `z`). `.assign(position=pos)` aggiunge il reparto. Le parentesi tonde e il punto a capo servono a scrivere la catena su due righe.

## Dopo il ciclo: unire i tre reparti (righe 64–72)

```python
matches = pd.concat(all_matches, ignore_index=True)
tops = pd.concat(top_rows, ignore_index=True)
r1 = matches[matches["rank"] == 1][["top_id", "distance"]].rename(columns={"distance": "rank1_distance"})
tops = tops.merge(r1, on="top_id", how="left").merge(flag_unicorns(matches), on="top_id", how="left")

u_info = master[INFO].rename(columns={...})
matches = matches.merge(u_info.drop(columns="_p"), on="u22_id", how="left")
```

- **Righe 64–65**: impila i risultati dei tre reparti in due tabelle: tutti gli abbinamenti e tutti i 75 Top.
- **Riga 66**: dagli abbinamenti tiene solo il Rank 1 e rinomina la sua distanza `rank1_distance`.
- **Riga 67**: aggiunge ai Top la distanza del loro Shadow Match e il flag «Unicorno».
- **Righe 69–71**: prepara le informazioni anagrafiche dei giocatori con nomi di colonna adatti ai giovani (`u22_age`, `u22_club`…).
- **Riga 72**: le aggiunge a ogni abbinamento, in modo che l'app possa mostrare club, età, campionato e minuti dei giovani. La colonna provvisoria `_p` (nome) si elimina perché il nome c'è già.

## Salvataggio (righe 74–79)

```python
ARTIFACTS_DIR.mkdir(exist_ok=True)
tops.to_csv(ARTIFACTS_DIR / "top_players.csv", index=False)
matches.to_csv(ARTIFACTS_DIR / "matches.csv", index=False)
pd.concat(radar_rows, ignore_index=True).drop_duplicates(["player_id", "feature", "position"]) \
    .to_csv(ARTIFACTS_DIR / "radar.csv", index=False)
print(f"salvati: {len(tops)} Top, {len(matches)} abbinamenti")
```

- **Riga 74**: crea la cartella `artifacts` se non esiste (`exist_ok=True` evita l'errore se c'è già).
- **Riga 75**: salva i 75 Top in `top_players.csv`. `index=False` evita di scrivere la numerazione di riga.
- **Riga 76**: salva gli abbinamenti in `matches.csv`.
- **Righe 77–78**: unisce i dati del radar, elimina eventuali duplicati (un giovane potrebbe comparire due volte) e li salva in `radar.csv`. Il simbolo `\` a fine riga dice «l'istruzione continua sotto».
- **Riga 79**: stampa un riepilogo.

## Avvio da riga di comando (righe 82–90)

```python
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=DEFAULT_M, help="vicini M del WKNN")
    ap.add_argument("--assign-method", choices=["original", "wknn"], default=ASSIGN_METHOD)
    ap.add_argument("--assign-scaling", choices=["own", "combined", "top"], default=ASSIGN_SCALING)
    ap.add_argument("--match-scaling", choices=["combined", "top"], default=MATCH_SCALING)
    a = ap.parse_args()
    main(a.m, a.assign_method, a.assign_scaling, a.match_scaling)
```

- **Riga 82**: `if __name__ == "__main__":` significa «esegui quanto segue **solo se il file viene lanciato direttamente**», non quando viene importato da un altro file.
- **Riga 83**: importa `argparse`, lo strumento di Python per leggere le opzioni scritte dopo il comando.
- **Riga 84**: crea il lettore di opzioni.
- **Righe 85–88**: dichiara le quattro opzioni. `--m` accetta un numero intero. Le altre accettano solo i valori elencati in `choices`; se non si scrive nulla vale il valore predefinito.
- **Riga 89**: legge le opzioni scritte dall'utente.
- **Riga 90**: lancia `main` con quei valori. Esempio: `python -m scripts.build_artifacts --assign-method wknn --m 1`.

---

# `src/plots.py`

## Importazioni (righe 1–8)

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

- **Riga 4–5**: importa *matplotlib* (la libreria dei grafici) e le dice di usare il motore **Agg**, che disegna in memoria senza aprire finestre. È quello adatto a un'app web.
- **Righe 6–8**: `plt` (comandi di disegno), numpy, pandas.

## `radar_compare`: preparare i dati (righe 11–22)

```python
def radar_compare(radar, top_id, u22_id, position, top_name, u22_name, clip=3.0):
    r = radar[radar["position"] == position]
    a = r[r["player_id"] == top_id].set_index("feature")["z"]
    b = r[r["player_id"] == u22_id].set_index("feature")["z"]
    feats = [f for f in a.index if f in b.index]
    if len(feats) < 3:
        return None
    a, b = a[feats].clip(-clip, clip), b[feats].clip(-clip, clip)
    ang = np.linspace(0, 2 * np.pi, len(feats), endpoint=False)
    ang = np.concatenate([ang, ang[:1]])
```

- **Righe 11–12**: la funzione riceve la tabella del radar, gli ID dei due giocatori, il reparto, i due nomi e `clip` (il limite dei valori, 3 deviazioni standard).
- **Riga 14**: tiene le righe del reparto.
- **Righe 15–16**: per il Top e per il giovane estrae le sue 8 variabili: `set_index("feature")` usa il nome della variabile come etichetta, `["z"]` prende il valore standardizzato.
- **Riga 17**: tiene le variabili presenti per entrambi.
- **Righe 18–19**: se ce ne sono meno di 3 un radar non ha senso: la funzione restituisce `None` (nessun grafico).
- **Riga 20**: `.clip(-3, 3)` **taglia** i valori estremi: sotto −3 diventa −3, sopra 3 diventa 3. Evita che un valore anomalo schiacci il grafico.
- **Riga 21**: `np.linspace` calcola gli **angoli** equidistanti, uno per variabile, su un cerchio di 360° (2π radianti). `endpoint=False` esclude l'ultimo punto, che coinciderebbe con il primo.
- **Riga 22**: aggiunge in fondo di nuovo il primo angolo, per **chiudere** la figura.

## `radar_compare`: disegnare (righe 23–35)

```python
fig, ax = plt.subplots(figsize=(5.2, 5.2), subplot_kw={"polar": True})
for vals, name, color in ((a, top_name, "#d62728"), (b, u22_name, "#1f77b4")):
    v = np.concatenate([vals.values, vals.values[:1]])
    ax.plot(ang, v, color=color, linewidth=2, label=name)
    ax.fill(ang, v, color=color, alpha=0.15)
ax.set_xticks(ang[:-1])
ax.set_xticklabels([f.replace("_", " ") for f in feats], fontsize=8)
ax.set_ylim(-clip, clip)
ax.set_yticks([-2, 0, 2])
ax.set_yticklabels(["-2σ", "media", "+2σ"], fontsize=7)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.12), fontsize=8)
fig.tight_layout()
return fig
```

- **Riga 23**: crea una figura quadrata di 5,2 pollici con un grafico **polare** (`polar=True`: gli assi sono angoli e distanza dal centro).
- **Riga 24**: ciclo sui due giocatori, ognuno con nome e colore: rosso (`#d62728`) per il Top, blu (`#1f77b4`) per il giovane (codici colore esadecimali).
- **Riga 25**: ripete il primo valore in fondo per chiudere la linea.
- **Riga 26**: traccia il bordo del poligono.
- **Riga 27**: colora l'interno con trasparenza `alpha=0.15` (15% di opacità).
- **Righe 28–29**: sistema le etichette dei raggi con i nomi delle variabili, sostituendo i trattini bassi con gli spazi.
- **Riga 30**: la scala va da −3 a +3.
- **Righe 31–32**: segna solo tre livelli: −2σ, la media e +2σ (σ = deviazione standard).
- **Riga 33**: legenda con i nomi dei giocatori, spostata fuori dal grafico.
- **Righe 34–35**: sistema i margini e restituisce la figura.

---

# `app.py`

## Intestazione e configurazione (righe 1–18)

```python
from pathlib import Path

import pandas as pd
import streamlit as st

from src.plots import radar_compare

ART = Path(__file__).parent / "artifacts"
POS_LABEL = {"DF": "Difensori", "MF": "Centrocampisti", "FW": "Attaccanti"}

st.set_page_config(page_title="Shadow Scouting", page_icon="⚽", layout="wide")
```

- **Righe 1–7**: documentazione con il comando di avvio: `streamlit run app.py`.
- **Righe 8–13**: importa gli strumenti: percorsi, pandas, Streamlit (soprannome `st`) e la funzione del radar.
- **Riga 15**: `ART` è il percorso della cartella `artifacts`, accanto a `app.py`.
- **Riga 16**: dizionario che traduce le sigle dei reparti.
- **Riga 18**: imposta titolo della scheda del browser, icona e larghezza (`wide` = tutto lo schermo). Deve essere il primo comando Streamlit.

## Caricamento dati con cache (righe 21–29)

```python
@st.cache_data
def load():
    tops = pd.read_csv(ART / "top_players.csv")
    matches = pd.read_csv(ART / "matches.csv")
    radar = pd.read_csv(ART / "radar.csv")
    return tops, matches, radar

tops, matches, radar = load()
```

- **Riga 21**: `@st.cache_data` è un **decoratore**: modifica la funzione sotto così che il risultato venga memorizzato. La prima volta legge i file, le volte successive (a ogni click dell'utente) restituisce subito la copia in memoria.
- **Righe 23–25**: legge i tre CSV.
- **Riga 26**: restituisce le tre tabelle insieme.
- **Riga 29**: chiama la funzione e dà un nome alle tre tabelle.

## Titolo (righe 31–32)

- **Riga 31**: `st.title` mostra il titolo grande.
- **Riga 32**: `st.caption` mostra una riga piccola con autore, università e fonte dei dati.

## Barra laterale (righe 34–49)

```python
with st.sidebar:
    st.header("Ricerca")
    pos = st.radio("Reparto", ["Tutti"] + list(POS_LABEL), format_func=..., horizontal=False)
    pool = tops if pos == "Tutti" else tops[tops["position"] == pos]
    names = sorted(pool["player"])
    name = st.selectbox("Top Player (digita per cercare)", names, index=None, placeholder="es. Erling Haaland")
```

- **Riga 35**: `with st.sidebar:` dice che tutto ciò che è rientrato va nella **barra laterale**.
- **Riga 36**: un'intestazione di sezione.
- **Righe 37–38**: `st.radio` mostra dei pulsanti a scelta singola: «Tutti» più i tre reparti (`list(POS_LABEL)` prende le sigle `DF`, `MF`, `FW`). `format_func` dice come **scrivere** ogni voce: «Tutti» resta così, le sigle diventano il nome esteso (con una `lambda`). Il valore scelto, la sigla, si salva in `pos`.
- **Riga 39**: `pool` è l'elenco dei Top tra cui cercare: tutti se è scelto «Tutti», altrimenti solo quelli del reparto. È un'espressione condizionale su una riga: `A if condizione else B`.
- **Riga 40**: `names` = i nomi in ordine alfabetico.
- **Riga 41**: `st.selectbox` è un menu a tendina con ricerca digitando. `index=None` lo lascia **vuoto** all'inizio; `placeholder` è il testo suggerito. Finché non si sceglie nulla, `name` vale `None`.

```python
    st.header("Filtri sui giovani")
    n_show = st.slider("Quanti giovani mostrare", 5, 50, 10)
    max_age = st.slider("Età massima", 14, 22, 22)
    min_min = st.slider("Minuti minimi giocati", 0, 2500, 0, step=100, help="...")
    leagues = sorted(matches["u22_league"].dropna().unique())
    chosen = st.multiselect("Campionati", leagues, placeholder="tutti")
```

- **Riga 44**: `st.slider` è un cursore: etichetta, minimo (5), massimo (50), valore iniziale (10). Il valore scelto va in `n_show`.
- **Riga 45**: età massima, da 14 a 22 (iniziale 22 = nessun limite).
- **Righe 46–47**: minuti minimi giocati, da 0 a 2500 a passi di 100. `help` mostra un suggerimento al passaggio del mouse: la tesi non applica filtri sui minuti, questo è solo un aiuto allo scouting.
- **Riga 48**: l'elenco dei campionati presenti: `dropna()` toglie i vuoti, `unique()` le ripetizioni, `sorted` mette in ordine.
- **Riga 49**: `st.multiselect` permette di scegliere **più** campionati. Vuoto = tutti.

## Spiegazione e arresto (righe 51–63)

```python
with st.expander("Come funziona"):
    st.markdown("1. **Fase 1** – ...")

if not name:
    st.info("Scegli un Top Player nella barra laterale per vedere i giovani più simili.")
    st.stop()
```

- **Riga 51**: `st.expander` è un riquadro che si apre e chiude con un clic.
- **Righe 52–59**: `st.markdown` mostra testo formattato: `**…**` è il grassetto, `*…*` il corsivo, `\n` va a capo. Le stringhe affiancate tra virgolette vengono unite in un unico testo. Descrive le tre fasi della tesi.
- **Riga 61**: `if not name:` significa «se non è stato scelto nessun Top» (`None` conta come «niente»).
- **Riga 62**: mostra un riquadro informativo azzurro.
- **Riga 63**: `st.stop()` ferma la pagina qui: finché non si sceglie un Top non si vede altro.

## Scheda del Top (righe 65–75)

```python
top = tops[tops["player"] == name].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Top Player", top["player"])
c2.metric("Club", top["club"], top["league"], delta_color="off")
c3.metric("Archetipo", top["archetype"], POS_LABEL[top["position"]], delta_color="off")
c4.metric("Distanza Rank 1", f"{top['rank1_distance']:.2f}")
if top["unicorn"]:
    st.warning("🦄 **Calciatore Unicorno**: ...")
```

- **Riga 65**: filtra la tabella dei Top sul nome scelto e prende la **prima riga** (`.iloc[0]`: un nome compare una sola volta).
- **Riga 68**: `st.columns(4)` divide la pagina in quattro colonne, assegnate a quattro variabili.
- **Righe 69–72**: `st.metric` mostra un «numero grande» con etichetta. Il terzo valore è un testo piccolo sotto (di solito una variazione): `delta_color="off"` lo lascia neutro invece di colorarlo di verde o rosso. Qui compaiono nome, club con campionato, archetipo con reparto e distanza del Rank 1 scritta con 2 decimali (`:.2f` in una f-string).
- **Righe 73–75**: **se** il Top è un Unicorno, mostra un avviso giallo con la spiegazione.

## Classifica (righe 77–99)

```python
m = matches[matches["top_id"] == top["top_id"]].copy()
m = m[(m["u22_age"] <= max_age) & (m["u22_minutes"].fillna(0) >= min_min)]
if chosen:
    m = m[m["u22_league"].isin(chosen)]
m = m.sort_values("distance").head(n_show)
```

- **Riga 78**: dagli abbinamenti tiene quelli del Top scelto. `.copy()` fa una copia per poterla modificare.
- **Riga 79**: applica **due filtri insieme**: età ≤ massima **e** (`&`) minuti ≥ minimo. `.fillna(0)` considera 0 i minuti mancanti.
- **Righe 80–81**: **se** sono stati scelti dei campionati, tiene solo i giovani di quei campionati (`isin`).
- **Riga 82**: ordina per distanza crescente (il più simile per primo) e tiene i primi `n_show`.

```python
st.subheader(f"Giovani più vicini a {top['player']}")
if m.empty:
    st.write("Nessun giovane soddisfa i filtri scelti.")
    st.stop()
```

- **Riga 84**: il sottotitolo con il nome del Top (f-string).
- **Righe 85–87**: se i filtri sono troppo severi e non resta nessuno (`m.empty`), mostra un messaggio e ferma la pagina.

```python
table = pd.DataFrame({
    "#": range(1, len(m) + 1),
    "Giocatore": m["u22_player"].values,
    ...
    "Distanza": m["distance"].round(2).values,
})
st.dataframe(table, hide_index=True)
```

- **Righe 89–98**: costruisce la tabella da mostrare scegliendo le colonne e dando loro **titoli leggibili**. `range(1, len(m) + 1)` genera la posizione 1, 2, 3… `.round(2)` arrotonda la distanza a 2 decimali.
- **Riga 99**: `st.dataframe` mostra la tabella interattiva (si può ordinare cliccando sulle intestazioni). `hide_index=True` nasconde la numerazione di riga di pandas.

## Radar (righe 101–110)

```python
st.subheader("Confronto di profilo")
pick = st.selectbox("Giovane da confrontare", table["Giocatore"], index=0)
row = m[m["u22_player"] == pick].iloc[0]
fig = radar_compare(radar, int(top["top_id"]), int(row["u22_id"]), top["position"], top["player"], pick)
if fig is not None:
    left, _ = st.columns([1, 1])
    left.pyplot(fig)
    st.caption("Le otto variabili con peso Lasso più alto del reparto, ...")
```

- **Riga 103**: un secondo menu a tendina per scegliere con quale giovane della tabella fare il confronto; `index=0` preseleziona il primo (il più simile).
- **Riga 104**: ricava la riga del giovane scelto, per conoscere il suo ID.
- **Riga 105**: chiama la funzione che disegna il radar (vedi `plots.py`). `int(...)` converte gli ID in numeri interi.
- **Riga 106**: se il grafico è stato creato (cioè non è `None`)…
- **Riga 107**: …divide la pagina in due colonne uguali e tiene la prima in `left`; il trattino basso `_` è il nome convenzionale di un valore che non serve.
- **Riga 108**: `left.pyplot(fig)` disegna il grafico nella colonna di sinistra.
- **Righe 109–110**: sotto, una nota che spiega cosa mostra il radar: le otto variabili con peso più alto, in deviazioni standard rispetto alla media di Top e Under-22, tagliate a ±3σ.

## Riepilogo: cosa succede quando si usa l'app

1. L'app legge una volta i tre CSV (`top_players`, `matches`, `radar`).
2. L'utente sceglie un reparto e un Top Player nella barra laterale.
3. Streamlit riesegue il file: mostra la scheda del Top, l'avviso Unicorno se serve e la classifica dei giovani più vicini.
4. I filtri (età, minuti, campionati, numero di risultati) riducono la classifica **senza rifare calcoli**: filtrano soltanto la tabella già pronta.
5. Scegliendo un giovane dal menu compare il radar di confronto.

---

## Mini-glossario

| Termine | Significato |
| --- | --- |
| Artefatto | File di risultati prodotto una volta dalla pipeline e letto dall'app |
| `main` | Funzione principale di un programma |
| Decoratore (`@`) | Riga che modifica il comportamento della funzione sotto |
| Cache | Memoria che conserva un risultato per non ricalcolarlo |
| `with` | Blocco in cui ciò che è rientrato è associato a un contenitore (es. la barra laterale) |
| `argparse` | Strumento per leggere le opzioni scritte dopo il comando |
| `melt` | Trasforma una tabella larga in lunga |
| `iloc[0]` | La prima riga di una tabella |
| `clip` | Taglia i valori entro un minimo e un massimo |
| f-string | Testo con la `f` davanti, in cui `{ }` viene sostituito da un valore |
| Radar | Grafico a «ragnatela» con un raggio per ogni variabile |
| σ (sigma) | Deviazione standard |
