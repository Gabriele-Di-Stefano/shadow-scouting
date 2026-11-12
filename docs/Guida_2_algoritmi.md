# Guida 2 — Gli algoritmi: `archetypes.py`, `projection.py`, `matching.py`

*Spiegato riga per riga, per chi non conosce Python*

Progetto: Shadow Scouting nel Calcio Moderno

## A cosa servono questi tre file

Qui c'è il cuore statistico della tesi, in tre passi:

1. `archetypes.py` descrive lo **spazio dei Top Player**: quali variabili contano (pesi Lasso), a quale archetipo appartiene ciascun Top, e come si **assegna un giovane a un archetipo** (Fase 2).
2. `projection.py` contiene gli strumenti matematici per misurare la **distanza ponderata** tra giocatori e il metodo **WKNN** descritto nella tesi.
3. `matching.py` esegue la Fase 3: dentro lo stesso archetipo, ordina i giovani per somiglianza a un Top (lo *Shadow Match*) e individua i «Calciatori Unicorno».

## Idee statistiche da conoscere

- **Standardizzare (z-score)**: per ogni variabile si toglie la media e si divide per la deviazione standard. Dopo, 0 significa «nella media» e +2 «due deviazioni standard sopra la media». Serve a confrontare grandezze con scale diverse.
- **Peso Lasso** `w`: numero tra 0 e 1 che dice quanto una variabile conta nel distinguere gli archetipi. Molte variabili hanno peso 0 e vengono ignorate.
- **Distanza euclidea ponderata** tra due giocatori: si prende la differenza su ogni variabile, la si eleva al quadrato, la si moltiplica per il peso della variabile, si sommano tutte e si fa la radice. Più è piccola, più i due giocatori sono simili *sulle variabili che contano*.
- **Cluster (archetipo)**: gruppo di giocatori simili, numerato 0, 1, 2… Esempio nei difensori: 0 = Terzini d'Assalto, 1 = Registi Arretrati, 2 = Mastini della Marcatura.
- **KNN (K-Nearest Neighbors)**: per classificare un giocatore nuovo si guardano i `k` giocatori già classificati più vicini e se ne copia il gruppo.

## Mappa dei file

| File | Righe | Cosa contiene |
| --- | --- | --- |
| `archetypes.py` | 1–145 | Nomi degli archetipi, pesi, classe `TopSpace` con scaler, cluster e assegnazione |
| `projection.py` | 1–29 | Distanze ponderate e voto WKNN |
| `matching.py` | 1–37 | Shadow Match e Unicorni |

---

# `src/archetypes.py`

## Documentazione e importazioni (righe 1–14)

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .config import ARTIFACTS_DIR, N_CLUSTERS, WEIGHTS_FILES
```

- **Righe 1–7**: documentazione. Dice che i pesi Lasso arrivano dalla Fase 1 (Sparse K-Means) e che i cluster dei Top si leggono dal file con le etichette originali; solo se manca, si ricostruiscono.
- **Riga 8**: istruzione tecnica sui tipi, senza effetti sui risultati.
- **Riga 10**: `numpy` (soprannome `np`) è la libreria per il calcolo su vettori e matrici di numeri.
- **Riga 11**: `pandas` per le tabelle.
- **Riga 12**: da *scikit-learn* prende `KMeans`, l'algoritmo di raggruppamento classico.
- **Riga 14**: importa le impostazioni da `config.py`.

## Nomi degli archetipi (righe 16–21)

```python
ARCHETYPE_NAMES = {
    "DF": {0: "Terzini d'Assalto", 1: "Registi Arretrati", 2: "Mastini della Marcatura"},
    "MF": {0: "Incursori Goleador", 1: "Cuori Pulsanti", 2: "Pendolari del Cross"},
    "FW": {0: "Rapaci d'Area", 1: "Registi Offensivi", 2: "Guerrieri del Pressing", 3: "Funamboli Dribblatori"},
}
```

- **Righe 17–21**: un **dizionario di dizionari**: dato il reparto e il numero del cluster, restituisce il nome dell'archetipo secondo il paragrafo 4.1.2 della tesi. La numerazione 0, 1, 2… è quella della tesi.

## Giocatori-àncora (righe 23–28)

```python
ANCHORS = {
    "DF": {0: "Giovanni Di Lorenzo", 1: "Virgil van Dijk", 2: "Alessandro Bastoni"},
    ...
}
```

- **Righe 24–28**: per ogni cluster, un giocatore che la tesi cita come esempio tipico. Servono **solo** quando i cluster si ricostruiscono da zero (non quando si usa il file con le etichette originali): K-Means numera i gruppi a caso, e con gli àncora si riportano ai numeri della tesi.

## `load_weights` (righe 31–33)

```python
def load_weights(position: str, artifacts_dir=ARTIFACTS_DIR) -> pd.Series:
    w = pd.read_csv(artifacts_dir / WEIGHTS_FILES[position], index_col=0)["Lasso_Weight"]
    return w[w > 0].sort_values(ascending=False)
```

- **Riga 31**: la funzione riceve il reparto e restituisce una `Series` (una colonna con etichette: il nome della variabile e il suo peso).
- **Riga 32**: legge il CSV dei pesi. `index_col=0` dice che la **prima colonna** contiene i nomi delle variabili e va usata come etichetta di riga. `["Lasso_Weight"]` seleziona la colonna dei pesi.
- **Riga 33**: `w[w > 0]` tiene solo i pesi maggiori di zero (le variabili che contano). `.sort_values(ascending=False)` li ordina dal più alto al più basso: così le prime variabili sono le più importanti (le userà il radar). Risultato: 44 variabili per i difensori, 58 per i centrocampisti, 46 per gli attaccanti.

## La classe `TopSpace` (riga 36)

```python
class TopSpace:
    """Spazio d'élite di un reparto: ..."""
```

- Una **classe** è uno «stampo» che raggruppa dati e funzioni sullo stesso argomento. Qui, tutto ciò che riguarda lo spazio d'élite di **un reparto**: i suoi Top, le variabili pesate, gli scaler, i cluster. Le funzioni dentro una classe si chiamano **metodi**; il primo parametro `self` indica «l'oggetto stesso» e permette ai metodi di condividere i dati.

## `__init__`: preparare lo spazio (righe 39–48)

```python
def __init__(self, position: str, top_df: pd.DataFrame, artifacts_dir=ARTIFACTS_DIR):
    self.position = position
    self.weights = load_weights(position, artifacts_dir)
    self.features = list(self.weights.index)
    X = top_df[self.features].astype(float)
    self.mean_ = X.mean()
    self.std_ = X.std(ddof=0)
    self.Z = ((X - self.mean_) / self.std_).values
    self.top = top_df.reset_index(drop=True)
    self.labels = self._cluster_labels(artifacts_dir)
```

- **Riga 39**: `__init__` è il metodo speciale eseguito quando si crea l'oggetto, per esempio `TopSpace("DF", dati_dei_top)`.
- **Riga 40**: memorizza il reparto.
- **Riga 41**: carica i pesi Lasso del reparto.
- **Riga 42**: `features` è l'elenco dei **nomi delle variabili** con peso positivo (l'etichetta di riga dei pesi).
- **Riga 43**: dalla tabella dei Top tiene solo quelle variabili e le converte in numeri decimali.
- **Riga 44**: `mean_` = media di ogni variabile sui Top.
- **Riga 45**: `std_` = deviazione standard. `ddof=0` indica la formula «sull'intera popolazione» (divisione per *n*), la stessa usata da `StandardScaler` di scikit-learn nel notebook.
- **Riga 46**: `Z` = i Top **standardizzati**: (valore − media) / deviazione standard. `.values` estrae la matrice di numeri (25 righe × numero di variabili).
- **Riga 47**: salva la tabella dei Top con le righe rinumerate da 0.
- **Riga 48**: calcola a quale cluster appartiene ciascun Top, con il metodo qui sotto.

## `_cluster_labels` (righe 50–59)

```python
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
```

- **Riga 51**: `k` = numero di archetipi di questo reparto.
- **Riga 52**: il percorso del file con le **etichette originali** dei cluster.
- **Righe 53–55**: **se il file esiste** (`if`), lo legge, usa `player_id` come etichetta di riga e prende la colonna `cluster`. Poi `.map(lab)` assegna a ogni Top il suo cluster cercandolo per ID, `.astype(int)` lo rende un numero intero, `.values` restituisce l'elenco. Il `return` termina la funzione: le righe sotto non vengono eseguite. È il percorso normale: riproduce esattamente la tesi.
- **Riga 56** (solo se il file manca): `Zw` = i Top standardizzati e moltiplicati per la radice dei pesi. In questo spazio una distanza *ordinaria* equivale alla distanza *ponderata*.
- **Righe 57–58**: lancia K-Means **200 volte** con inizializzazioni casuali diverse (`random_state=s` fissa il caso, così il risultato si ripete) e tiene la soluzione con `inertia_` più bassa (gruppi più compatti). Il `for s in range(200)` dentro le parentesi è un'espressione generatrice; `key=lambda km: km.inertia_` dice a `min` di confrontare le soluzioni sull'inerzia (`lambda` è una mini-funzione senza nome).
- **Riga 59**: rinumera i gruppi con gli àncora e restituisce le etichette.

## `_renumber` (righe 61–69)

```python
def _renumber(self, raw):
    names = self.top["player"].values
    mapping = {}
    for new_id, anchor in ANCHORS[self.position].items():
        hit = np.where(names == anchor)[0]
        if len(hit):
            mapping[int(raw[hit[0]])] = new_id
    return np.array([mapping.get(int(r), int(r)) for r in raw])
```

- **Riga 63**: l'elenco dei nomi dei Top.
- **Riga 64**: `mapping` è un dizionario vuoto: «numero dato da K-Means → numero della tesi».
- **Righe 65–68**: per ogni cluster della tesi e il suo giocatore-àncora: `np.where(...)` trova la posizione dell'àncora tra i Top; se lo trova (`if len(hit)`) guarda in quale gruppo K-Means l'ha messo e annota la corrispondenza.
- **Riga 69**: converte ogni etichetta secondo la mappa (`mapping.get(x, x)` restituisce `x` stesso se non c'è corrispondenza) e restituisce l'elenco.

## `set_scaling` (righe 71–87)

Sceglie **come standardizzare** i dati per le distanze della Fase 3.

```python
def set_scaling(self, mode, u22_df=None):
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
```

- **Righe 72–77**: documentazione: `"top"` usa solo le medie dei Top (come descrive il par. 3.3 della tesi); `"combined"` usa quelle di Top **e** Under-22 insieme (come fa il notebook originale, e come riproduce i boxplot del cap. 4.3).
- **Righe 78–79**: con `"top"` non c'è nulla da cambiare (media e deviazione dei Top sono già state calcolate in `__init__`) e la funzione termina.
- **Righe 80–81**: se il valore di `mode` non è valido, o manca la tabella dei giovani, solleva un errore con un messaggio chiaro (`raise`).
- **Riga 82**: variabili dei giovani come numeri decimali.
- **Riga 83**: i **valori mancanti** dei giovani si sostituiscono con la media della loro colonna (`fillna`).
- **Riga 84**: variabili dei Top.
- **Riga 85**: `pd.concat` impila le due tabelle, una sotto l'altra.
- **Riga 86**: **sostituisce** media e deviazione standard con quelle della tabella impilata.
- **Riga 87**: ricalcola i Top standardizzati con le nuove medie.

## `assignment_matrices` (righe 89–109)

Prepara le matrici standardizzate per la Fase 2 nel metodo WKNN.

```python
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
```

- **Righe 90–96**: documentazione delle tre possibilità: `"own"` ciascun gruppo con le proprie medie; `"combined"` medie di Top+giovani insieme; `"top"` medie dei soli Top.
- **Righe 97–99**: prepara le variabili di Top e giovani e sostituisce i valori mancanti.
- **Riga 100–101**: con `"own"` restituisce **due** matrici (una virgola separa i due valori restituiti): Top standardizzati con le loro medie, giovani con le loro.
- **Righe 102–106**: negli altri casi sceglie la media `mu` e la deviazione `sd` da usare per entrambi i gruppi.
- **Righe 107–108**: se `mode` non è tra i tre ammessi, errore.
- **Riga 109**: standardizza entrambi i gruppi con la stessa media e deviazione e li restituisce.

## `transform` (righe 111–116)

```python
def transform(self, df):
    X = df[self.features].astype(float)
    X = X.fillna(X.mean())
    return ((X - self.mean_) / self.std_).values
```

- **Righe 114–116**: dato un qualsiasi insieme di giocatori (di solito i giovani), tiene le variabili del reparto, riempie i valori mancanti con la media della tabella e li standardizza con **la media e la deviazione correnti** (`mean_` e `std_`, che `set_scaling` può aver modificato).

## `representatives` (righe 118–129)

Trova, per ogni cluster, il Top **più vicino al centro del cluster**: è il «rappresentante» dell'archetipo.

```python
def representatives(self, universe):
    cols = [c for c in universe if self.top[c].notna().all()]
    X = self.top[cols].astype(float)
    Z = ((X - X.mean()) / X.std(ddof=0)).dropna(axis=1).values
    reps = []
    for k in np.unique(self.labels):
        idx = np.where(self.labels == k)[0]
        d = np.linalg.norm(Z[idx] - Z[idx].mean(axis=0), axis=1)
        reps.append(idx[d.argmin()])
    return np.array(reps)
```

- **Riga 121**: dalle ~141 metriche dell'«universo» tiene quelle senza valori mancanti tra i Top.
- **Riga 122**: tabella dei Top con quelle metriche.
- **Riga 123**: standardizza; `.dropna(axis=1)` elimina le colonne che diventano vuote (metriche con deviazione standard 0, cioè costanti, per cui lo z-score non è definito).
- **Riga 124**: lista vuota dove raccogliere i rappresentanti.
- **Riga 125**: ciclo sui cluster. `np.unique` dà i numeri dei cluster senza ripetizioni.
- **Riga 126**: `idx` = posizioni dei Top appartenenti a quel cluster.
- **Riga 127**: calcola la distanza di ciascun membro dal **centroide** (la media del cluster, `.mean(axis=0)` media colonna per colonna). `np.linalg.norm(..., axis=1)` è la lunghezza di ogni vettore differenza, cioè la distanza euclidea. Qui **non ci sono pesi**: conta ogni metrica allo stesso modo.
- **Riga 128**: `d.argmin()` è la posizione del più piccolo; `idx[...]` la traduce nella posizione tra tutti i Top. Si aggiunge alla lista.
- **Riga 129**: restituisce le posizioni come vettore. Esempio reale: difensori → Cancelo, Koundé, Singo; centrocampisti → Wirtz, Vitinha, Luis Alberto; attaccanti → Guirassy, Kramarić, Kvaratskhelia, Rodrygo.

## `assign_original` (righe 131–142)

È l'assegnazione dei giovani agli archetipi **come nel notebook di tesi**: 1-NN verso i soli rappresentanti.

```python
reps = self.representatives(universe)
R = self.top.iloc[reps][self.features].astype(float)
U = u22_df[self.features].astype(float)
U = U.fillna(U.mean())
Zr = ((R - R.mean()) / R.std(ddof=0)).values
Zu = ((U - U.mean()) / U.std(ddof=0)).values
d = ((Zu[:, None, :] - Zr[None, :, :]) ** 2).sum(axis=2)
return self.labels[reps][d.argmin(axis=1)]
```

- **Riga 135**: trova i rappresentanti (3 o 4 giocatori).
- **Riga 136**: `iloc[reps]` seleziona le righe dei rappresentanti; poi tiene le variabili Lasso (quelle con peso > 0).
- **Righe 137–138**: variabili dei giovani, con i valori mancanti sostituiti dalla media.
- **Riga 139**: standardizza i rappresentanti **con le loro stesse medie** (calcolate solo su 3 o 4 giocatori).
- **Riga 140**: standardizza i giovani con le loro medie.
- **Riga 141**: calcola la distanza al quadrato di **ogni giovane da ogni rappresentante**. `Zu[:, None, :]` e `Zr[None, :, :]` aggiungono una dimensione in modo che numpy sottragga ogni coppia (tecnica detta *broadcasting*): il risultato è un blocco di numeri con tre dimensioni (giovani × rappresentanti × variabili). Si eleva al quadrato e `.sum(axis=2)` somma sulle variabili: resta una tabella giovani × rappresentanti. **Pesi 0/1**: le variabili scelte dal Lasso contano tutte uguale (peso 1), le altre sono già state escluse (peso 0). I pesi continui `w` del Lasso **non** vengono usati qui.
- **Riga 142**: `d.argmin(axis=1)` indica, per ogni giovane, **quale rappresentante è il più vicino**; `self.labels[reps]` sono i cluster dei rappresentanti, quindi il risultato è il cluster del rappresentante vincente.

## `archetype_name` (righe 144–145)

```python
def archetype_name(self, cluster):
    return ARCHETYPE_NAMES[self.position].get(int(cluster), f"Cluster {cluster}")
```

- **Riga 145**: cerca il nome dell'archetipo; se per qualche motivo manca, restituisce «Cluster N» invece di dare errore (`.get(chiave, valore_di_riserva)`).

---

# `src/projection.py`

## Documentazione e costante (righe 1–11)

```python
import numpy as np

DEFAULT_M = 1
```

- **Righe 1–6**: descrive il metodo WKNN del par. 2.2.2: i Top con la loro etichetta di cluster sono il «training set»; ogni giovane riceve il cluster con un **voto pesato** (1/distanza) tra i `M` Top più vicini.
- **Riga 11**: `DEFAULT_M = 1`: valore predefinito del numero di vicini. Con 1, il giovane prende il cluster del singolo Top più simile.

## `weighted_distances` (righe 14–19)

```python
def weighted_distances(Zy, Zx, w):
    sw = np.sqrt(w)
    A, B = Zy * sw, Zx * sw
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.sqrt(np.maximum(sq, 0.0))
```

- **Riga 14**: riceve `Zy` (giovani standardizzati, una riga per giovane), `Zx` (Top standardizzati) e `w` (i pesi).
- **Riga 15**: documentazione: la formula è la distanza ponderata d = √(Σ wⱼ · (yⱼ − xⱼ)²).
- **Riga 16**: `sw` = radice quadrata dei pesi.
- **Riga 17**: moltiplica ogni variabile per la radice del suo peso. Così la distanza *ordinaria* tra i vettori «scalati» coincide con la distanza *ponderata* originale: variabile con peso 0 → cancellata, peso alto → ingrandita.
- **Riga 18**: calcola **tutte le distanze al quadrato in un colpo solo** usando l'identità |a − b|² = |a|² + |b|² − 2·a·b. `(A ** 2).sum(1)` è |a|² per ogni giovane (`[:, None]` lo dispone in colonna), `(B ** 2).sum(1)` è |b|² per ogni Top (`[None, :]` in riga), `A @ B.T` è il prodotto scalare di ogni coppia (`@` è il prodotto tra matrici). È molto più veloce di un doppio ciclo.
- **Riga 19**: `np.maximum(sq, 0.0)` evita numeri leggermente negativi dovuti ad arrotondamenti del computer (una radice di un numero negativo darebbe errore); poi `np.sqrt` calcola la radice. Il risultato è una tabella **giovani × Top** di distanze.

## `wknn_assign` (righe 22–29)

```python
def wknn_assign(D, top_labels, m=DEFAULT_M, eps=1e-9):
    m = min(m, D.shape[1])
    idx = np.argsort(D, axis=1)[:, :m]
    dist = np.take_along_axis(D, idx, axis=1) + eps
    ks = np.unique(top_labels)
    votes = np.stack([((top_labels[idx] == k) / dist).sum(axis=1) for k in ks], axis=1)
    return ks[votes.argmax(axis=1)]
```

- **Riga 22**: `D` è la tabella delle distanze (giovani × Top), `top_labels` i cluster dei Top, `m` il numero di vicini, `eps` un numero minuscolo.
- **Riga 24**: `m` non può superare il numero di Top disponibili (`D.shape[1]` = numero di colonne).
- **Riga 25**: `np.argsort` ordina ogni riga dalla distanza più piccola alla più grande e restituisce le **posizioni**; `[:, :m]` tiene le prime `m` colonne: gli `m` Top più vicini di ogni giovane.
- **Riga 26**: le relative distanze. Si somma `eps` (0,000000001) per non dividere per zero.
- **Riga 27**: i numeri dei cluster esistenti.
- **Riga 28**: calcola i **voti**. Per ogni cluster `k`, ogni vicino che vi appartiene vota con peso `1/distanza`: più è vicino, più vale il suo voto. `(top_labels[idx] == k)` è Vero/Falso (1 o 0), diviso per la distanza e sommato sui vicini. `np.stack` mette i voti dei vari cluster in una tabella giovani × cluster.
- **Riga 29**: per ogni giovane `argmax` sceglie il cluster con più voti e `ks[...]` lo traduce nel numero del cluster.
- Con `m = 1` c'è un solo voto: vince il cluster del Top più vicino.

---

# `src/matching.py`

## Importazioni (righe 1–7)

- **Righe 4–5**: `numpy` e `pandas`.
- **Riga 7**: importa `weighted_distances` da `projection.py`: le distanze vanno calcolate allo stesso modo ovunque.

## `shadow_matches` (righe 10–28)

```python
def shadow_matches(top_space, u22, u22_labels, top_n=50):
    Zy = top_space.transform(u22)
    D = weighted_distances(Zy, top_space.Z, top_space.weights.values)
    rows = []
    for r, top in top_space.top.iterrows():
        cluster = top_space.labels[r]
        cand = np.where(u22_labels == cluster)[0]
        if len(cand) == 0:
            continue
        order = cand[np.argsort(D[cand, r])][:top_n]
        for rank, i in enumerate(order, start=1):
            rows.append({...})
    return pd.DataFrame(rows)
```

- **Riga 10**: riceve lo spazio d'élite (`top_space`), la tabella dei giovani, il cluster assegnato a ciascun giovane e quanti abbinamenti tenere per ogni Top (`top_n`).
- **Riga 12**: standardizza i giovani con lo scaler corrente.
- **Riga 13**: tabella di tutte le distanze ponderate (giovani × Top), con i pesi Lasso.
- **Riga 14**: lista vuota dove raccogliere le righe del risultato.
- **Riga 15**: ciclo su ogni Top. `.iterrows()` restituisce, riga per riga, la posizione `r` e i dati del Top.
- **Riga 16**: il cluster di questo Top.
- **Riga 17**: `cand` = posizioni dei **giovani assegnati allo stesso cluster**. Solo loro sono candidati: è il vincolo «intra-cluster» della tesi.
- **Righe 18–19**: se in quel cluster non c'è nessun giovane, passa al Top successivo (`continue`).
- **Riga 20**: ordina i candidati per distanza da questo Top (`D[cand, r]` sono le loro distanze) e tiene i primi `top_n`.
- **Riga 21**: percorre i selezionati dal più vicino. `enumerate(..., start=1)` numera 1, 2, 3… e `rank` è la **posizione in classifica**; il Rank 1 è lo *Shadow Match*.
- **Righe 22–27**: aggiunge una riga con reparto, ID e nome del Top, cluster e nome dell'archetipo, posizione in classifica, ID e nome del giovane e **distanza**.
- **Riga 28**: trasforma la lista in una tabella.

## `flag_unicorns` (righe 31–37)

```python
def flag_unicorns(matches):
    r1 = matches[matches["rank"] == 1].copy()
    g = r1.groupby(["position", "cluster"])["distance"]
    q1, q3 = g.transform(lambda s: s.quantile(0.25)), g.transform(lambda s: s.quantile(0.75))
    r1["unicorn"] = r1["distance"] > q3 + 1.5 * (q3 - q1)
    return r1[["top_id", "unicorn"]]
```

- **Riga 33**: tiene solo le righe di Rank 1: la distanza dello Shadow Match di ciascun Top.
- **Riga 34**: raggruppa per reparto e cluster: ogni archetipo ha il suo insieme di distanze.
- **Riga 35**: calcola per ogni riga il **primo e il terzo quartile** (il 25° e il 75° percentile) delle distanze del suo archetipo (`transform` ripete il valore del gruppo su ogni riga del gruppo).
- **Riga 36**: regola dei boxplot: un valore è un **outlier superiore** se supera Q3 + 1,5 × IQR, dove IQR = Q3 − Q1 (scarto interquartile). Un Top con Rank 1 oltre quella soglia è un **Unicorno**: nessun giovane gli somiglia abbastanza rispetto agli altri Top del suo archetipo. Il risultato è Vero/Falso in una nuova colonna.
- **Riga 37**: restituisce solo ID e flag.
- Con i dati attuali gli Unicorni sono Di Lorenzo (Terzini d'Assalto) e Rodri (Cuori Pulsanti).

---

## Mini-glossario

| Termine | Significato |
| --- | --- |
| Classe / metodo | Stampo che raggruppa dati e funzioni / funzione dentro una classe |
| `self` | L'oggetto stesso, dentro i suoi metodi |
| Matrice | Tabella di soli numeri; `numpy` ci fa i calcoli |
| Broadcasting | Regola con cui numpy combina tabelle di forme diverse |
| Centroide | Il «centro» (media) di un gruppo di giocatori |
| Rappresentante | Il Top più vicino al centroide del suo cluster |
| Peso Lasso | Importanza di una variabile; 0 = ignorata |
| Distanza ponderata | Distanza in cui ogni variabile conta in proporzione al suo peso |
| 1-NN | Copiare il gruppo del singolo giocatore più vicino |
| WKNN | KNN con voto pesato sull'inverso della distanza |
| Quartile / IQR | 25° e 75° percentile / loro differenza |
| Outlier | Valore anomalo, oltre i «baffi» del boxplot |
| `argmin` / `argmax` | Posizione del valore più piccolo / più grande |
