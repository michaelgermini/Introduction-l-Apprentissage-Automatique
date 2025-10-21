# Chapitre 22 : Visualisation de Données et Apprentissage de Variétés

## 📚 Introduction

Les techniques de visualisation permettent de représenter des données haute dimension en 2D ou 3D.

---

## 22.1 Multidimensional Scaling (MDS)

### Principe

Préserver les distances entre points.

**Objectif** :
```
minimize Σᵢⱼ (d_ij - ‖y_i - y_j‖)²
```

```python
from sklearn.manifold import MDS

mds = MDS(n_components=2)
X_embedded = mds.fit_transform(X)
```

---

## 22.2 Apprentissage de Variétés

### 22.2.1 Isomap

**Principe** : Utiliser distances géodésiques (plus court chemin sur le graphe de voisinage).

```python
from sklearn.manifold import Isomap

isomap = Isomap(n_components=2, n_neighbors=10)
X_embedded = isomap.fit_transform(X)
```

### 22.2.2 Locally Linear Embedding (LLE)

**Principe** : Préserver reconstructions locales.

```python
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_embedded = lle.fit_transform(X)
```

---

## 22.3 t-SNE

**t-Distributed Stochastic Neighbor Embedding**

**Objectif** : Préserver similarités locales.

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
X_embedded = tsne.fit_transform(X)

plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y)
plt.show()
```

**Paramètres importants** :
- perplexity : taille de voisinage (5-50)
- learning_rate : pas d'apprentissage

---

## 22.4 UMAP

**Uniform Manifold Approximation and Projection**

**Avantages** :
- Plus rapide que t-SNE
- Préserve structure globale

```python
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
X_embedded = reducer.fit_transform(X)
```

---

## 💡 Comparaison

| Méthode | Temps | Préserve | Usage |
|---------|-------|----------|--------|
| PCA | Rapide | Global | Linéaire |
| t-SNE | Lent | Local | Visualisation |
| UMAP | Moyen | Local+Global | Visualisation |
| Isomap | Moyen | Distances | Variétés |

---

[⬅️ Chapitre précédent](./chapitre-21-reduction-dimension.md) | [Retour](../README.md) | [Suite ➡️](../partie-7-theorie/chapitre-23-generalisation.md)

