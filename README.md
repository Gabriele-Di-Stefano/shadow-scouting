# Shadow Scouting nel Calcio Moderno

**🔗 App online:** [shadow-scouting.streamlit.app](https://shadow-scouting-abzfjn2upcxs7tzwtxexsc.streamlit.app/)

App Streamlit che automatizza la tesi di laurea (Università di Milano-Bicocca) sul clustering di calciatori
Top e Under-22. Scegli uno dei 75 Top Player e ottieni i giovani più simili nello stesso archetipo tattico.

## Pipeline
1. **Fase 1** – archetipi d'élite e pesi Lasso (Sparse K-Means, Gap Statistic) sui 75 Top Player.
2. **Fase 2** – assegnazione degli Under-22 agli archetipi. Metodo predefinito `original`: 1-NN con selezione Lasso (pesi 0/1) sui Top
   rappresentativi di ciascun cluster, come nel notebook (riproduce i risultati della tesi). Alternativa `wknn`: Weighted
   KNN del par. 2.2.2 (pesi Lasso su tutti i Top).
3. **Fase 3** – matching intra-cluster con distanza euclidea ponderata; Rank 1 = Shadow Match; "Unicorni" = outlier dei boxplot.

## Uso
```bash
pip install -r requirements.txt
streamlit run app.py                      # usa gli artefatti già presenti in artifacts/

# per rigenerare gli artefatti servono i CSV Kaggle in data/raw/ (normalized/*.csv e Shooting.csv)
python -m scripts.build_artifacts [--assign-method original|wknn] [--m 1] [--match-scaling combined|top]
```
Dati: [football-leagues-data-2023-2024](https://www.kaggle.com/datasets/anisguechtouli/football-leagues-data-2023-2024) (Fbref/Opta).

## Nota metodologica
**Fase 2 (assegnazione degli Under-22).** I cluster dei 75 Top (Sparse K-Means) forniscono le *pseudo-etichette*.
Per ogni archetipo si sceglie il **Top reale più vicino al centroide** (il prototipo) e ogni giovane riceve l'archetipo
del prototipo più vicino: 1-NN con distanza euclidea sulle variabili selezionate dal Lasso (peso 1 se scelta, 0 se no).
È un approccio pseudo-supervisionato di tipo *cluster-then-classify*. Il par. 2.2.2 della tesi descrive una formulazione
più generale (WKNN con pesi Lasso continui su tutti i Top); il codice che ha prodotto i risultati usa la variante a
prototipi. Il repository implementa entrambe (`--assign-method original|wknn`).

- Sui 75 Top, in un test leave-one-out, i due approcci ritrovano l'archetipo nel 93% e nel 95% dei casi.
- Sui giovani, il WKNN con M = 1 assegna un archetipo diverso a circa il 25% dei giocatori. La differenza viene quasi
  tutta dal confronto con 3-4 prototipi invece che con tutti i Top (23%), non dai pesi (4%).
- Coorte attaccanti: nel notebook due Top con età ≤ 22 restavano tra i giovani e venivano abbinati a sé stessi. Qui sono
  esclusi, e la mediana di Rank 1 dei «Guerrieri del Pressing» passa da 0,00 a 2,42.

Dettagli e numeri in [docs/Confronto_1NN_vs_WKNN.md](docs/Confronto_1NN_vs_WKNN.md).

## Guide al codice (riga per riga, per non programmatori)
- [Guida 1 – dati e coorti](docs/Guida_1_dati_e_coorti.md): `config.py`, `data_prep.py`, `cohorts.py`
- [Guida 2 – algoritmi](docs/Guida_2_algoritmi.md): `archetypes.py`, `projection.py`, `matching.py`
- [Guida 3 – pipeline e app](docs/Guida_3_pipeline_e_app.md): `build_artifacts.py`, `plots.py`, `app.py`
- [Confronto 1-NN e WKNN](docs/Confronto_1NN_vs_WKNN.md): come cambia la Fase 2 con i due metodi

## Struttura
`src/data_prep.py` pulizia e merge · `src/cohorts.py` Top e Under-22 · `src/archetypes.py` pesi, scaler, cluster ·
`src/projection.py` WKNN · `src/matching.py` distanza ponderata e unicorni · `scripts/build_artifacts.py` pipeline ·
`app.py` interfaccia · `experiments/` confronti del cap. 4.

## Riproducibilità
Con le etichette originali dei cluster (`artifacts/top_clusters_original.csv`) i top 5 per difensori e centrocampisti
coincidono al 100% con i report originali; per gli attaccanti il confronto differisce perché qui i Top con età ≤ 22
sono esclusi dalla coorte giovanile (nel notebook ne restavano due). Le distanze di matching usano media e deviazione
standard di Top + Under-22 insieme (`--match-scaling combined`), come nel notebook.
