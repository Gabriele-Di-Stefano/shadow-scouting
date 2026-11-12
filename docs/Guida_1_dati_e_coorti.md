# Guida 1 — Dati e coorti: `config.py`, `data_prep.py`, `cohorts.py`

*Spiegato riga per riga, per chi non conosce Python*

Progetto: Shadow Scouting nel Calcio Moderno

## A cosa servono questi tre file

Sono la **base** del progetto: stabiliscono *quali giocatori* si studiano e *con quali dati*. Non fanno ancora nessun calcolo statistico.

- `config.py` raccoglie le impostazioni fisse: dove stanno i file, chi sono i 75 Top Player, quanti archetipi ha ogni reparto.
- `data_prep.py` prende i 13 file CSV scaricati da Kaggle e li unisce in **una sola tabella** (il «master dataset»), una riga per giocatore.
- `cohorts.py` da quella tabella ritaglia le due popolazioni della tesi: i **Top Player** (l'élite di riferimento) e gli **Under-22** (i giovani da cercare).

## Come si legge il codice: le regole di base

- Il programma viene letto **dall'alto verso il basso**.
- Tutto ciò che segue `#` è un **commento**, ignorato dal computer.
- Una **variabile** è una scatola con un nome: `X = 5` significa «metti 5 in X». Per confrontare si usa `==`.
- Una **funzione** è una ricetta con un nome, definita con `def`. Si usa scrivendo il nome seguito da parentesi.
- Le righe **rientrate** appartengono al blocco della riga precedente che finisce con `:`.
- Un **DataFrame** (`df`) è una tabella come un foglio Excel, gestita dalla libreria *pandas*. Una colonna si indica `df["age"]`.
- Le **righe tra tre virgolette** `""" ... """` all'inizio di un file o di una funzione sono la *documentazione*: spiegano a cosa serve, ma non vengono eseguite.

## Mappa dei file

| File | Righe | Cosa contiene |
| --- | --- | --- |
| `config.py` | 1–43 | Percorsi, ID dei 75 Top, numero di cluster, parametri della pipeline |
| `data_prep.py` | 1–63 | Lettura dei CSV, pulizia, unione nel master dataset |
| `cohorts.py` | 1–38 | Coorte Top, coorte Under-22, elenco delle metriche |

---

# `src/config.py`

## Intestazione e percorsi (righe 1–6)

```python
"""Costanti del progetto: percorsi, coorti Top Player e parametri (tesi, cap. 3-4)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"          # CSV Kaggle: normalized/*.csv + Shooting.csv
ARTIFACTS_DIR = ROOT / "artifacts"
```

- **Riga 1**: descrizione del file (documentazione, non eseguita).
- **Riga 2**: `from pathlib import Path` porta lo strumento per lavorare con i percorsi delle cartelle in modo che funzioni su Windows, Mac e Linux.
- **Riga 4**: `__file__` è la posizione di questo file. `.resolve()` ne ricava il percorso completo, `.parents[1]` risale di **due** cartelle (da `src/config.py` si arriva alla cartella principale del progetto). Il risultato si chiama `ROOT`.
- **Riga 5**: `RAW_DIR` è la cartella dei dati grezzi, `ROOT/data/raw`. Il simbolo `/` tra percorsi serve a *comporli*. Qui vanno messi i CSV di Kaggle.
- **Riga 6**: `ARTIFACTS_DIR` è la cartella `artifacts`, dove la pipeline salva i risultati che l'app legge.

## Reparti e soglia d'età (righe 8–10)

```python
POSITIONS = ["DF", "MF", "FW"]
POSITION_LABELS = {"DF": "Difensori", "MF": "Centrocampisti", "FW": "Attaccanti"}
U22_MAX_AGE = 22
```

- **Riga 8**: una **lista** (elenco tra parentesi quadre) con le sigle dei tre reparti: difensori, centrocampisti, attaccanti. I portieri non ci sono: sono rimandati a uno sviluppo futuro.
- **Riga 9**: un **dizionario** (coppie «chiave: valore» tra graffe) che traduce ogni sigla nel nome esteso.
- **Riga 10**: l'età massima della coorte giovanile. Chi ha **22 anni o meno** è un Under-22.

## I 75 Top Player (righe 12–20)

```python
TOP_IDS = {
    "DF": [8, 70, 11, ...],
    "MF": [5, 7, 184, ...],
    "FW": [6, 265, 157, ...],
}
```

- **Riga 12**: commento. I Top sono stati scelti con i voti di Sofascore: i 5 migliori per ruolo in ciascuno dei 5 campionati principali.
- **Righe 13–20**: un dizionario che a ogni reparto associa la lista dei **25 numeri identificativi** (`player_id`) dei suoi Top Player. In ogni lista, 5 numeri per campionato: 5 campionati × 5 giocatori = 25. Tre reparti × 25 = 75 giocatori in tutto.
- Si usano i numeri e non i nomi perché i nomi possono ripetersi (omonimi), i numeri no.

## Numero di archetipi (righe 22–23)

```python
N_CLUSTERS = {"DF": 3, "MF": 3, "FW": 4}
```

- **Riga 23**: quanti **archetipi tattici** (gruppi) esistono in ciascun reparto: 3 per i difensori, 3 per i centrocampisti, 4 per gli attaccanti. Questi numeri sono il risultato della Gap Statistic nella tesi (par. 4.1.2).

## File dei pesi Lasso (righe 25–30)

```python
WEIGHTS_FILES = {
    "DF": "variabili_importanti_difensori_top_df.csv",
    "MF": "variabili_importanti_centrocampisti_top_df.csv",
    "FW": "variabili_importanti_attaccanti_top_df.csv",
}
```

- **Righe 26–30**: per ogni reparto, il nome del file CSV con i **pesi Lasso**. Ogni riga di quei file dice «quanto conta» una variabile (passaggi, tiri, contrasti…) per distinguere gli archetipi di quel reparto. Un peso a zero significa «variabile ignorata».

## Colonne di identità (righe 32–34)

```python
ID_COLS = ["player_id", "player", "nationality", "position", "age", "club",
           "league", "division", "Continent", "no_clubs", "country", "league_continent"]
```

- **Righe 33–34**: l'elenco delle colonne che descrivono *chi è* il giocatore (nome, club, età, campionato…) e non *come gioca*. Servono per escluderle quando si cercano le metriche di prestazione. Il rientro della seconda riga è solo estetico: la lista continua perché la parentesi quadra non è ancora chiusa.

## Parametri della pipeline (righe 36–43)

```python
ASSIGN_METHOD = "original"
WKNN_M = 1
ASSIGN_SCALING = "own"
MATCH_SCALING = "combined"
```

- **Righe 36–39**: commenti che spiegano le due possibilità per la Fase 2 (assegnare i giovani a un archetipo): `"original"` riproduce il metodo del notebook di tesi; `"wknn"` è il metodo descritto nel capitolo 2.2.2. Vedi il documento `Confronto_1NN_vs_WKNN.md`.
- **Riga 40**: `ASSIGN_METHOD` sceglie quale dei due usare. Predefinito: `"original"`.
- **Riga 41**: `WKNN_M`, il numero di «vicini» considerati dal metodo WKNN. Con 1 ogni giovane viene collegato al singolo Top più simile.
- **Riga 42**: `ASSIGN_SCALING`, come si **standardizzano** i dati nel metodo WKNN: `"own"` = Top e giovani ciascuno con le proprie medie.
- **Riga 43**: `MATCH_SCALING`, come si standardizzano i dati per calcolare le distanze finali: `"combined"` = con le medie di Top e giovani messi insieme (come nel notebook).

> **Standardizzare** (z-score) significa togliere a ogni variabile la sua media e dividerla per la sua deviazione standard, così che passaggi (centinaia) e percentuali (da 0 a 100) pesino allo stesso modo.

---

# `src/data_prep.py`

## Documentazione e importazioni (righe 1–15)

```python
from __future__ import annotations

import pandas as pd

from .config import RAW_DIR

PERF_TABLES = ["Playing_Time", "Shooting", "Passing", "Pass_types", "GSC", "Defense", "Possession", "Misc"]
```

- **Righe 1–8**: la documentazione elenca le quattro operazioni di pulizia descritte nel capitolo 3 della tesi.
- **Riga 9**: istruzione tecnica che rende più flessibile la scrittura dei tipi nelle funzioni. Non cambia i risultati.
- **Riga 11**: porta la libreria *pandas* (soprannome `pd`).
- **Riga 13**: `from .config import RAW_DIR` prende dal file `config.py` la variabile `RAW_DIR`. Il punto davanti a `config` significa «nella stessa cartella di questo file».
- **Riga 15**: l'elenco degli **8 file di prestazione**: tempo di gioco, tiri, passaggi, tipi di passaggio, azioni che creano tiri/gol, difesa, possesso, varie.

## `_read` (righe 18–19)

```python
def _read(name: str, raw_dir=RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "normalized" / f"{name}.csv")
```

- **Riga 18**: definisce la funzione `_read` che riceve il `name` di una tabella e la cartella `raw_dir` (se non specificata usa `RAW_DIR`). Il trattino basso iniziale è una convenzione: «funzione di uso interno».
- **Riga 19**: `pd.read_csv` legge un file CSV e lo trasforma in tabella. Il percorso è `raw_dir/normalized/<name>.csv`. La **f-string** `f"{name}.csv"` inserisce il valore di `name` al posto delle graffe: con `name = "Passing"` diventa `Passing.csv`. `return` restituisce la tabella a chi ha chiamato la funzione.

## `_add_shots` (righe 22–34)

Questa funzione aggiunge la colonna `shots` (tiri totali), che manca nei file «normalizzati» ma esiste nel file `Shooting.csv` «denormalizzato».

```python
def _add_shots(shooting: pd.DataFrame, raw_dir=RAW_DIR) -> pd.DataFrame:
    den = pd.read_csv(raw_dir / "Shooting.csv")
    players = pd.read_csv(raw_dir / "normalized" / "t_player.csv")[["player_id", "player", "age"]]
    keys = ["player", "age", "goals", "SOTs", "xg", "shots_per90", "avg_shot_dist", "shots_fk"]
    base = shooting.merge(players, on="player_id", how="left")
    den = den[keys + ["shots"]].drop_duplicates(subset=keys, keep=False)
    out = base.merge(den, on=keys, how="left")
    return out.drop(columns=["player", "age"])
```

- **Righe 23–27**: la documentazione spiega il problema: i due file hanno le righe in ordine diverso quando due giocatori hanno lo stesso nome (omonimi, per esempio due «Aldo Cruz»). Copiare la colonna *per posizione di riga* abbinerebbe i tiri al giocatore sbagliato in circa 140 righe.
- **Riga 28**: legge `Shooting.csv` (versione denormalizzata, con i nomi dei giocatori).
- **Riga 29**: legge la tabella dei giocatori e tiene solo tre colonne (le doppie parentesi quadre selezionano più colonne): numero, nome, età.
- **Riga 30**: `keys` è la **chiave di abbinamento**: otto colonne (nome, età, gol, tiri in porta, xG…) la cui combinazione identifica un giocatore in modo praticamente univoco. Due omonimi hanno quasi sempre statistiche diverse.
- **Riga 31**: aggiunge nome ed età alla tabella dei tiri, abbinando per `player_id`. `how="left"` significa «tieni tutte le righe della tabella di sinistra, anche se non c'è corrispondenza».
- **Riga 32**: dal file denormalizzato tiene le colonne della chiave più `shots`. `drop_duplicates(..., keep=False)` **elimina tutte le righe che hanno una chiave ripetuta**: se due giocatori fossero indistinguibili, meglio non assegnare nulla che assegnare il valore sbagliato.
- **Riga 33**: unisce le due tabelle usando la chiave. Dove non c'è corrispondenza `shots` resta vuota (sono 42 casi su oltre 10.000).
- **Riga 34**: toglie le colonne temporanee `player` e `age` e restituisce la tabella.

## `build_master` — parte 1: leggere e ripulire (righe 37–47)

```python
def build_master(raw_dir=RAW_DIR) -> pd.DataFrame:
    perf = {n: _read(n, raw_dir) for n in PERF_TABLES}

    perf["Pass_types"] = perf["Pass_types"].drop(columns=["passes", "passes_comp"], errors="ignore")
    perf["Misc"] = perf["Misc"].drop(columns=["crosses", "tklw"], errors="ignore")
    perf["Defense"] = perf["Defense"].drop(columns=["intr"], errors="ignore")
    perf["Shooting"] = _add_shots(perf["Shooting"], raw_dir)

    pt = perf["Playing_Time"].copy()
    pt["minutes"] = pd.to_numeric(pt["minutes"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    perf["Playing_Time"] = pt.rename(columns={"games": "matches_played", "minutes": "minutes_played"})
```

- **Riga 37**: definisce la funzione principale del file, che restituisce il master dataset.
- **Riga 38**: `perf` è un dizionario che contiene le 8 tabelle di prestazione. `{n: _read(n, raw_dir) for n in PERF_TABLES}` è una **dictionary comprehension**: per ogni nome `n` nell'elenco, legge la tabella e la salva con quel nome come chiave.
- **Riga 40**: dalla tabella `Pass_types` toglie `passes` e `passes_comp`, già presenti in `Passing` (stesse informazioni, quindi ridondanti). `errors="ignore"` evita errori se una colonna non esiste.
- **Riga 41**: da `Misc` toglie `crosses` e `tklw`, già presenti in altre tabelle.
- **Riga 42**: da `Defense` toglie `intr`: ha correlazione 1 con `interceptions` (sono la stessa misura) e la tesi ne tiene una sola.
- **Riga 43**: aggiunge la colonna `shots` con la funzione appena vista.
- **Riga 45**: `.copy()` fa una copia della tabella dei minuti, per poterla modificare senza toccare l'originale.
- **Riga 46**: i minuti sono scritti come testo con il separatore delle migliaia («2,345»). Questa riga: `astype(str)` li trasforma in testo, `.str.replace(",", "")` toglie la virgola, `pd.to_numeric(...)` li riconverte in numeri. `errors="coerce"` mette un valore vuoto dove la conversione non è possibile invece di fermarsi con un errore.
- **Riga 47**: rinomina due colonne con nomi più chiari: `games` → `matches_played` e `minutes` → `minutes_played`.

## `build_master` — parte 2: unire le tabelle (righe 49–53)

```python
    master = perf["Playing_Time"]
    for name in PERF_TABLES[1:]:
        dup = (set(master.columns) & set(perf[name].columns)) - {"player_id"}
        assert not dup, f"colonne duplicate tra tabelle ({name}): {dup}"
        master = master.merge(perf[name], on="player_id", how="outer")
```

- **Riga 49**: parte dalla tabella dei tempi di gioco.
- **Riga 50**: **ciclo** `for`: ripete le righe rientrate per ognuna delle altre 7 tabelle. `PERF_TABLES[1:]` significa «dall'elemento numero 1 in poi», cioè tutti tranne il primo.
- **Riga 51**: controlla se la nuova tabella ha colonne con lo stesso nome di quelle già presenti. Un **insieme** (`set`) è un elenco senza ripetizioni; `&` calcola l'intersezione (i nomi in comune); `- {"player_id"}` toglie `player_id`, che è *giusto* che sia in comune.
- **Riga 52**: `assert` è un controllo di sicurezza: se `dup` non è vuoto, il programma si ferma e scrive quali colonne sono duplicate. Serve a evitare unioni che creerebbero colonne con suffissi confusi.
- **Riga 53**: unisce la tabella al master usando `player_id`. `how="outer"` mantiene **tutti** i giocatori presenti in almeno una delle due tabelle.

## `build_master` — parte 3: aggiungere club, nazionalità, campionato (righe 55–63)

```python
    master = _read("t_player", raw_dir).merge(master, on="player_id", how="left")
    master = (master
              .merge(_read("t_position", raw_dir), left_on="position_id", right_on="pos_id", how="left")
              .merge(_read("t_nationality", raw_dir), left_on="nationality_id", right_on="nat_id", how="left")
              .merge(_read("t_club", raw_dir), on="club_id", how="left")
              .merge(_read("t_leagues", raw_dir), on="league_id", how="left")
              .drop(columns=["position_id", "pos_id", "nationality_id", "nat_id", "club_id", "league_id"])
              .rename(columns={"continent": "league_continent"}))
    return master
```

- **Riga 55**: parte dalla tabella anagrafica dei giocatori (nome, età…) e le attacca le statistiche. Così ogni giocatore ha una riga anche se mancano alcuni dati.
- **Righe 56–62**: una **catena** di operazioni racchiusa tra parentesi (così si può andare a capo). Ogni `.merge(...)` aggiunge informazioni di contesto:
  - riga 57: il **ruolo** (`DF`, `MF`, `FW`, `GK`), unendo `position_id` con `pos_id`;
  - riga 58: la **nazionalità** e il continente;
  - riga 59: il **club**;
  - riga 60: il **campionato** (lega, divisione, nazione).
- **Riga 61**: `.drop` elimina i codici numerici ormai inutili (gli «ID» di appoggio).
- **Riga 62**: `.rename` evita un conflitto di nomi: la colonna `continent` della lega diventa `league_continent`, per distinguerla dal continente della nazionalità del giocatore.
- **Riga 63**: restituisce il master dataset: circa **10.578 righe** (giocatori) e 154 colonne.

---

# `src/cohorts.py`

## Importazioni (righe 1–8)

```python
from __future__ import annotations

import pandas as pd

from .config import ID_COLS, POSITIONS, TOP_IDS, U22_MAX_AGE

ALL_TOP_IDS = {pid for ids in TOP_IDS.values() for pid in ids}
```

- **Riga 6**: importa da `config.py` le impostazioni che servono.
- **Riga 8**: costruisce un **insieme** con gli ID di **tutti** i 75 Top. `TOP_IDS.values()` dà le tre liste di ID; il doppio `for` le percorre una dopo l'altra e raccoglie ogni numero in un unico insieme. Servirà a escludere i Top dalla coorte dei giovani.

## `get_top` (righe 11–14)

```python
def get_top(master: pd.DataFrame, position: str) -> pd.DataFrame:
    top = master[master["player_id"].isin(TOP_IDS[position])]
    return top.set_index("player_id").loc[TOP_IDS[position]].reset_index()
```

- **Riga 11**: la funzione riceve il master e il reparto (`"DF"`, `"MF"` o `"FW"`).
- **Riga 13**: `master["player_id"].isin(...)` produce una colonna di Vero/Falso («questo giocatore è nella lista?»). Mettere questa colonna tra parentesi quadre **filtra** la tabella: restano solo le righe vere, cioè i 25 Top di quel reparto.
- **Riga 14**: li rimette **nell'ordine della lista `TOP_IDS`**. `set_index("player_id")` usa l'ID come etichetta di riga, `.loc[lista]` pesca le righe in quell'ordine, `reset_index()` riporta l'ID a colonna normale. L'ordine fisso rende ripetibili i risultati.

## `get_u22` (righe 17–25)

```python
def get_u22(master: pd.DataFrame, position: str) -> pd.DataFrame:
    mask = (master["age"] <= U22_MAX_AGE) & (master["position"] == position)
    mask &= ~master["player_id"].isin(ALL_TOP_IDS)
    return master[mask].reset_index(drop=True)
```

- **Righe 18–22**: documentazione. Il punto chiave: i Top con 22 anni o meno vanno **esclusi dai giovani** per tutti i reparti, altrimenti ciascuno verrebbe abbinato a sé stesso con distanza 0 (era il problema degli attaccanti nel notebook).
- **Riga 23**: crea una **maschera**, una colonna di Vero/Falso. È vera per chi ha età ≤ 22 **e** (`&`) ruolo esattamente uguale a quello richiesto. Le parentesi sono obbligatorie. Chi ha due ruoli (per esempio «DF,MF») è escluso, come nella tesi.
- **Riga 24**: `&=` aggiunge una condizione. `~` significa «NON»: quindi si tiene chi **non** è fra i Top.
- **Riga 25**: applica la maschera e rinumera le righe da 0 (`reset_index(drop=True)`). Nessun filtro su minuti o campionato: come nella tesi (par. 3.2).
- Risultato: 1070 difensori, 1229 centrocampisti, 1100 attaccanti (3399 in tutto; nella tesi 3400).

## `cohort_summary` (righe 28–30)

```python
def cohort_summary(master: pd.DataFrame) -> pd.DataFrame:
    rows = [{"position": p, "top": len(get_top(master, p)), "u22": len(get_u22(master, p))} for p in POSITIONS]
    return pd.DataFrame(rows)
```

- **Riga 29**: per ogni reparto costruisce un dizionario con tre valori: il reparto, quanti Top, quanti Under-22. `len(...)` conta le righe. Il risultato è una lista di tre dizionari.
- **Riga 30**: la trasforma in una piccola tabella, che la pipeline stampa a schermo come controllo.

## `feature_universe` (righe 33–38)

```python
def feature_universe(master: pd.DataFrame) -> list[str]:
    num = [c for c in master.columns if c not in ID_COLS and pd.api.types.is_numeric_dtype(master[c])]
    ref = master[(master["age"] <= 21) & (master["position"] == "DF")]
    return [c for c in num if ref[c].notna().all()]
```

- **Righe 34–35**: documentazione: è l'elenco delle metriche numeriche usate nell'analisi, costruito come nel notebook.
- **Riga 36**: `num` è la lista di tutte le colonne che **non** sono di identità e che contengono **numeri**. È una *list comprehension*: «per ogni colonna `c`, tienila se (condizione)».
- **Riga 37**: `ref` è un campione di riferimento: i difensori con 21 anni o meno.
- **Riga 38**: restituisce solo le metriche **senza nessun valore mancante** in quel campione (`notna()` = «non vuoto», `.all()` = «per tutte le righe»). Sono circa 141 metriche. Le metriche con dati mancanti cadono automaticamente.

---

## Mini-glossario

| Termine | Significato |
| --- | --- |
| Master dataset | La tabella unica con una riga per giocatore e tutte le metriche |
| Top Player | I 75 giocatori d'élite di riferimento (25 per reparto) |
| Under-22 | I giovani con 22 anni o meno, tra cui si cercano i «sostituti» |
| Archetipo / cluster | Gruppo di giocatori con lo stesso stile di gioco |
| `player_id` | Numero identificativo unico del giocatore |
| Merge | Unione di due tabelle usando una colonna in comune |
| `how="left"` / `"outer"` | Quali righe tenere nell'unione: tutte quelle della prima tabella / tutte quelle di entrambe |
| Maschera | Colonna di Vero/Falso usata per filtrare le righe |
| Insieme (`set`) | Elenco senza ripetizioni |
| List comprehension | Modo compatto per costruire una lista con un ciclo su una riga |
| Standardizzazione (z-score) | Sottrarre la media e dividere per la deviazione standard |
