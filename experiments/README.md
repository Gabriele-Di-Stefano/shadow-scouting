# Esperimenti (confronti del capitolo 4 della tesi)

Qui vanno i confronti che non servono all'app ma sono nella tesi:

- Tabella 4.2: SKM/P, LR+KM/P, SKM/W, LR+KM/W, LR+SKM/P, SKM(NN)/P
- Tabella 4.6: FKM, KNN, WKNN, SOM, WSOM, ClusterGAN
- Portieri (sviluppo futuro)

`sparse_kmeans.py` è una reimplementazione in numpy di Sparse K-Means e Gap Statistic. Non riproduce
esattamente i pesi della tesi (ottenuti con `pysparcl`): l'app usa i pesi originali in `artifacts/`.
