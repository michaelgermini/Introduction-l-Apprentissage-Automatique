# Chapitre 20 : Clustering (Regroupement)

## 📚 Introduction

Le clustering consiste à grouper des données similaires sans labels supervisés.

---

## 20.1 Introduction

**Objectif** : Partitionner n observations en K groupes.

---

## 20.2 Clustering Hiérarchique

### 20.2.1 Dendrogrammes

**Bottom-up** : Fusionner progressivement les clusters les plus proches.

### 20.2.2 Métriques de Liaison

**Single linkage** : min_{x∈C₁,y∈C₂} d(x, y)
**Complete linkage** : max_{x∈C₁,y∈C₂} d(x, y)
**Average linkage** : moyenne des distances

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Clustering hiérarchique
Z = linkage(X, method='ward')

# Dendrogramme
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.show()
```

---

## 20.3 K-means

### Algorithme

1. Initialiser K centres μ₁, ..., μ_K
2. **Assignment** : c_i = argmin_k ‖x_i - μ_k‖²
3. **Update** : μ_k = mean{x_i : c_i = k}
4. Répéter 2-3 jusqu'à convergence

```python
from sklearn.cluster import KMeans

# K-means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Centres
centers = kmeans.cluster_centers_
```

### K-means++

**Initialisation améliorée** : Choisir centres éloignés.

---

## 20.4 Clustering Spectral

### Principe

1. Construire matrice de similarité W
2. Calculer Laplacien L = D - W
3. Vecteurs propres de L
4. K-means sur les vecteurs propres

```python
from sklearn.cluster import SpectralClustering

spectral = SpectralClustering(n_clusters=3, affinity='rbf')
labels = spectral.fit_predict(X)
```

---

## 20.5 DBSCAN

**Density-Based** : Trouve clusters de forme arbitraire.

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)
```

---

## 20.6 Choix du Nombre de Clusters

### Méthode du Coude (Elbow)

Tracer l'inertie vs K.

### Silhouette Score

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

```python
from sklearn.metrics import silhouette_score

scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(X)
    scores.append(silhouette_score(X, labels))
```

---

## 20.7 Clustering Bayésien

**Modèles de mélange** : Approche probabiliste.

---

[⬅️ Partie 5](../partie-5-methodes-generatives/chapitre-19-generatives-profondes.md) | [Retour](../README.md) | [Suite ➡️](./chapitre-21-reduction-dimension.md)

