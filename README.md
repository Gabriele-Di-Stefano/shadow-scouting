# Shadow Scouting nel Calcio Moderno

App Streamlit che automatizza la tesi di laurea (Università di Milano-Bicocca) sul clustering di calciatori
Top e Under-22. Scegli uno dei 75 Top Player e ottieni i giovani più simili nello stesso archetipo tattico.

## Pipeline
1. **Fase 1** – archetipi d'élite e pesi Lasso (Sparse K-Means, Gap Statistic) sui 75 Top Player.
2. **Fase 2** – assegnazione degli Under-22 agli archetipi. Metodo predefinito `original`: 1-NN non pesato sui Top
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

## Struttura
`src/data_prep.py` pulizia e merge · `src/cohorts.py` Top e Under-22 · `src/archetypes.py` pesi, scaler, cluster ·
`src/projection.py` WKNN · `src/matching.py` distanza ponderata e unicorni · `scripts/build_artifacts.py` pipeline ·
`app.py` interfaccia · `experiments/` confronti del cap. 4.

## Riproducibilità
Con le etichette originali dei cluster (`artifacts/top_clusters_original.csv`) i top 5 per difensori e centrocampisti
coincidono al 100% con i report originali; per gli attaccanti il confronto differisce perché qui i Top con età ≤ 22
sono esclusi dalla coorte giovanile (nel notebook ne restavano due). Le distanze di matching usano media e deviazione
standard di Top + Under-22 insieme (`--match-scaling combined`), come nel notebook.
