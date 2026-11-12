# Come cambia l'algoritmo: 1-NN del codice contro WKNN della tesi

*Spiegato per chi vuole capire cosa cambia davvero nei risultati*

Progetto: Shadow Scouting nel Calcio Moderno

## In breve

La Fase 2 della pipeline deve rispondere a una domanda: **«a quale archetipo tattico appartiene questo giovane?»**. Il codice del notebook e la tesi la risolvono in modo diverso.

| | **1-NN del codice** (`--assign-method original`) | **WKNN della tesi** (`--assign-method wknn`) |
| --- | --- | --- |
| Con chi si confronta il giovane | Con **3 o 4 Top «rappresentativi»**, uno per cluster | Con **tutti e 25** i Top del reparto |
| Come si misura la distanza | Euclidea sulle variabili scelte dal Lasso: **peso 1 se scelta, 0 se no** | Euclidea con i **pesi Lasso continui** `w` (tra 0 e 1) |
| Standardizzazione | Rappresentanti e giovani ciascuno con le **proprie** medie | Configurabile (`own`, `combined`, `top`) |
| Quanti vicini decidono | 1 | `M` (con `M = 1`, il singolo Top più vicino) |
| Come si decide l'archetipo | Cluster del rappresentante più vicino | Voto dei `M` vicini, ognuno con peso `1/distanza` |
| Cosa produce la tesi | I numeri di Tabella 4.6 e dei boxplot del cap. 4.3 | Descritto nel par. 2.2.2 e nelle slide 8–9, mai eseguito nel notebook |

**La Fase 3 (distanza finale e Shadow Match) è identica nei due casi.** Cambia solo *chi è candidato*: nella Fase 3 un Top viene confrontato soltanto con i giovani assegnati al **suo** archetipo, quindi cambiare l'assegnazione cambia l'elenco dei candidati e, di conseguenza, la classifica.

## Metodo 1 — 1-NN sui rappresentanti (il codice)

Passi, nell'ordine in cui il codice li esegue (`TopSpace.assign_original`):

1. Per ogni cluster dei Top si calcola il **centroide** (la media) in uno spazio con tutte le ~140 metriche standardizzate, senza pesi.
2. Si sceglie come **rappresentante** il Top più vicino al centroide. Un solo giocatore per cluster:
   - difensori: Cancelo, Koundé, Singo;
   - centrocampisti: Wirtz, Vitinha, Luis Alberto;
   - attaccanti: Guirassy, Kramarić, Kvaratskhelia, Rodrygo.
3. Si prendono le sole variabili con peso Lasso > 0 e si standardizzano i rappresentanti (con le medie di 3 o 4 giocatori) e i giovani (con le loro medie).
4. Per ogni giovane si calcola la distanza euclidea da ciascun rappresentante, con **pesi 0/1** (variabile scelta dal Lasso = 1, altrimenti 0).
5. Il giovane prende il cluster del rappresentante più vicino.

Punti da notare:

- Il confine tra due archetipi è deciso da **3 o 4 giocatori soltanto**: se un rappresentante è un po' atipico, sposta il confine.
- Il Lasso entra solo come **selezione**: le variabili scelte contano tutte uguale, i pesi continui `w` non vengono usati. (Nel notebook questa è la variante «lasso»/«pesata»; l'altra variante usa tutte le variabili.)
- Gli scaler sono diversi per rappresentanti e giovani: la scala dei Top rappresentativi è calcolata su 3 o 4 giocatori.

## Metodo 2 — WKNN (la tesi, par. 2.2.2)

Passi (`projection.weighted_distances` e `projection.wknn_assign`):

1. Si standardizzano Top e giovani (di default, ciascuno con le proprie medie).
2. Si calcola la distanza **ponderata** di ogni giovane da **tutti e 25** i Top:

   d(y, x) = √( Σⱼ wⱼ · (yⱼ − xⱼ)² )

   Una variabile con peso 0 non conta, una con peso alto conta molto.
3. Si prendono i `M` Top più vicini.
4. Ogni vicino vota per il proprio cluster con peso `1/d`: più è vicino, più il suo voto pesa.
5. Il giovane prende il cluster con più voti.

Con `M = 1` il voto è uno solo: il giovane prende il cluster **del singolo Top più simile tra tutti i 25**.

## Esempio con numeri inventati (due sole variabili)

Variabili standardizzate: (passaggi, contrasti). Pesi Lasso: passaggi 0,2 e contrasti 0,8 (i contrasti contano molto di più). Un giovane `y` ha profilo (0,9 ; 0,7).

| Top | Cluster | Profilo | Rappresentante? |
| --- | --- | --- | --- |
| A | 0 | (2 ; 0) | sì |
| B | 1 | (0 ; 2) | sì |
| C | 1 | (1 ; 1) | no |
| D | 0 | (2 ; 2) | no |

**1-NN sui rappresentanti** (pesi 0/1: entrambe le variabili sono scelte, quindi contano uguale; solo A e B):

- distanza da A = √(1,1² + 0,7²) = √1,70 ≈ **1,30**
- distanza da B = √(0,9² + 1,3²) = √2,50 ≈ 1,58

Vince A: il giovane va nel **cluster 0**.

**WKNN con M = 1** (distanze pesate, tutti i Top):

- da A = √(0,2·1,21 + 0,8·0,49) = √0,634 ≈ 0,80
- da B = √(0,2·0,81 + 0,8·1,69) = √1,514 ≈ 1,23
- da C = √(0,2·0,01 + 0,8·0,09) = √0,074 ≈ **0,27**
- da D = √(0,2·1,21 + 0,8·1,69) = √1,594 ≈ 1,26

Vince C: il giovane va nel **cluster 1**. Con M = 3 vincerebbe ancora il cluster 1 (voto 1/0,27 + 1/1,23 ≈ 4,5 contro 1/0,80 ≈ 1,3).

Lo stesso giovane finisce in due archetipi diversi perché il secondo metodo guarda **i contrasti** (che pesano molto) e vede che è quasi uguale a C; il primo metodo guarda soltanto i due rappresentanti e dà lo stesso peso a tutte le variabili.

## Che cosa distingue davvero i due metodi

I due metodi differiscono per **due scelte indipendenti**: (1) con chi si confronta il giovane, cioè i 3-4 rappresentanti oppure tutti i 25 Top; (2) come si pesano le variabili, cioè 0/1 oppure con i pesi continui `w`. Le ho separate, tenendo invariato tutto il resto (stessi cluster, stessi giovani, standardizzazione `own`). La tabella indica la quota di giovani assegnati a un archetipo **diverso** da quello del metodo del codice (A):

| Metodo | Confronto con | Pesi | Difensori | Centrocampisti | Attaccanti | Media |
| --- | --- | --- | --- | --- | --- | --- |
| **A** – 1-NN del codice | 3-4 rappresentanti | 0/1 | 0% | 0% | 0% | 0% |
| **B** | 3-4 rappresentanti | continui | 2% | 4% | 7% | 4% |
| **C** | tutti i 25 Top | 0/1 | 17% | 23% | 28% | 23% |
| **D** – WKNN, M = 1 | tutti i 25 Top | continui | 17% | 26% | 31% | 25% |
| **E** – WKNN, M = 3 | tutti i 25 Top | continui | 18% | 15% | 38% | 24% |

**Conclusione:** quasi tutta la differenza viene dalla scelta del *riferimento* (rappresentanti contro tutti i Top), non dai pesi. Passare da pesi 0/1 a pesi continui sposta solo il 4% dei giovani (A → B), mentre passare da 3-4 rappresentanti a tutti i Top ne sposta il 23% (A → C). Il WKNN di tesi (D) e il 1-NN su tutti i Top con pesi 0/1 (C) sono molto vicini tra loro.

### Sui Top (test leave-one-out)

Ogni Top è stato classificato usando gli altri 24 del reparto. Percentuale di archetipi ritrovati su 75 Top: 1-NN del codice (A) 93%, 1-NN sui rappresentanti con pesi continui (B) 95%, WKNN M = 1 (D) 95%, WKNN M = 3 96%, WKNN M = 5 89%. Differenze di 1 o 2 giocatori: i due metodi sono equivalenti su questo test. Le etichette dei Top nascono da un clustering con gli stessi pesi Lasso, che favorisce i metodi pesati, e il test non dice nulla sui giovani, per i quali non esiste un archetipo «vero».

## Cosa cambia sui dati reali

Confronto fatto con le stesse coorti (1070 difensori, 1229 centrocampisti, 1100 attaccanti), stessi pesi Lasso e stesse etichette originali dei cluster; per il WKNN, standardizzazione `own` (Top e giovani ciascuno con le proprie medie). Il confronto è con il metodo originale.

### Quanti giovani cambiano archetipo

| Reparto | WKNN M = 1 | WKNN M = 5 |
| --- | --- | --- |
| Difensori | 16,8% | 19,1% |
| Centrocampisti | 26,1% | 15,4% |
| Attaccanti | 30,7% | 40,1% |

### Quanti giovani finiscono in ciascun archetipo

| Reparto | 1-NN originale | WKNN M = 1 |
| --- | --- | --- |
| Difensori (Terzini, Registi, Mastini) | 238 · 240 · 592 | 234 · 110 · 726 |
| Centrocampisti (Incursori, Cuori, Pendolari) | 172 · 264 · 793 | 212 · 264 · 753 |
| Attaccanti (Rapaci, Registi Off., Guerrieri, Funamboli) | 490 · 35 · 241 · 334 | 657 · 166 · 158 · 119 |

### Mediane delle distanze di Rank 1 (i boxplot del cap. 4.3)

| Archetipo | 1-NN originale | WKNN M = 1 |
| --- | --- | --- |
| Terzini d'Assalto | 2,49 | 2,19 |
| Registi Arretrati | 1,57 | 1,57 |
| Mastini della Marcatura | 3,49 | 2,17 |
| Incursori Goleador | 3,07 | 3,07 |
| Cuori Pulsanti | 2,45 | 2,45 |
| Pendolari del Cross | 4,58 | 2,81 |
| Rapaci d'Area | 4,19 | 3,51 |
| Registi Offensivi | 2,84 | 2,73 |
| Guerrieri del Pressing | 2,42 | 2,42 |
| Funamboli Dribblatori | 4,97 | 4,43 |

### Esempi di Shadow Match (Rank 1)

| Top | 1-NN originale | WKNN M = 1 | WKNN M = 5 |
| --- | --- | --- | --- |
| Rodri | Daniel Neil | Daniel Neil | Daniel Neil |
| Jude Bellingham | Jamal Musiala | Jamal Musiala | Jamal Musiala |
| Khvicha Kvaratskhelia | Johan Bakayoko | Johan Bakayoko | Abdul Fatawu Issahaku |
| Alessandro Bastoni | Tomás Avilés | James Hill | Anthony Rouault |
| Erling Haaland | Ali Al Hamadi | Mohamed Sankoh | Mohamed Sankoh |

### Come leggere questi numeri

- **Il metodo sposta i confini tra archetipi**, non le distanze. Dove un Top resta con lo stesso gruppo di candidati (Cuori Pulsanti, Registi Arretrati) il Rank 1 non cambia.
- **Gli archetipi «larghi» cambiano di più**: Mastini della Marcatura (mediana da 3,49 a 2,17) e Pendolari del Cross (da 4,58 a 2,81). Con il WKNN questi due archetipi ricevono giovani diversi e più vicini ai Top, e le distanze si accorciano (è l'effetto osservato; la causa precisa non è stata isolata). Questo cambia anche la lettura del capitolo 4.3: le conclusioni su «difficile da replicare» dipendono dal metodo di assegnazione.
- **Gli archetipi piccoli** (Registi Arretrati, solo 3 Top) hanno pochi candidati in entrambi i casi: i risultati sono fragili con qualunque metodo.
- **I giovani che compaiono in cima a una classifica** possono cambiare del tutto: un giovane assegnato a un altro archetipo non è più candidato per quel Top.

## Quale usare

- **`original` (predefinito):** riproduce esattamente i report originali (top 5 identici per difensori e centrocampisti) e i numeri della tesi. Va bene se si vuole che l'app mostri gli stessi risultati della tesi. Il testo del par. 2.2.2 va però aggiornato: il metodo realmente usato è un 1-NN con selezione Lasso (pesi 0/1) sui rappresentanti dei cluster.
- **`wknn`:** coerente con la descrizione della tesi (pesi Lasso, tutti i Top, voto 1/d). Se lo si adotta, **vanno rifatti** Tabella 4.6, boxplot e conclusioni dei capitoli 4.3 e 5.1, perché i numeri cambiano come sopra.

Per passare da un metodo all'altro:

```bash
python -m scripts.build_artifacts                                # 1-NN originale (predefinito)
python -m scripts.build_artifacts --assign-method wknn --m 1     # WKNN con M = 1
python -m scripts.build_artifacts --assign-method wknn --m 5     # WKNN con M = 5
```

Poi si riavvia l'app: legge i nuovi file in `artifacts/`.

## Un limite da dichiarare, qualunque metodo si scelga

Le etichette dei cluster dei Top (Fase 1) vengono dai pesi e dalle etichette originali di tesi. Non sono state ricalcolate con una nuova esecuzione di Sparse K-Means, quindi ogni confronto tra metodi riguarda **solo la Fase 2** (assegnazione) e la Fase 3 (matching).
