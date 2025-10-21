# Chapitre 22 : Visualisation de Données et Apprentissage de Variétés

## 📚 Introduction

Les techniques de visualisation permettent de représenter des données haute dimension en 2D ou 3D.

## 🗺️ Carte Mentale : Visualisation

```
              VISUALISATION & MANIFOLDS
                        │
        ┌───────────────┼───────────────┐
        │               │               │
    LINÉAIRE       MANIFOLD         MODERNE
        │               │               │
    ┌───┴───┐       ┌───┴───┐       ┌───┴───┐
    │       │       │       │       │       │
  PCA    MDS    Isomap  LLE     t-SNE  UMAP
    │       │       │       │       │       │
Variance Distance Géodésique Local  KL-div Topologique
Global  Global    Graph    Weights  Probas  Fuzzy Graph
```

## 📊 Tableau Comparatif : t-SNE vs UMAP vs PCA

| **Critère** | **PCA** | **t-SNE** | **UMAP** |
|------------|---------|-----------|----------|
| **Linéaire** | ✓ Oui | ✗ Non | ✗ Non |
| **Scalabilité** | ✓✓✓ O(np²) | ⚠️ O(n²) | ✓✓ O(n^1.14) |
| **Vitesse** | Très rapide | Lent | Rapide |
| **Structure globale** | ✓✓ Bonne | ✗ Perdue | ✓ Préservée |
| **Structure locale** | ⚠️ Moyenne | ✓✓✓ Excellente | ✓✓✓ Excellente |
| **Déterministe** | ✓ Oui | ✗ Non | ⚠️ Quasi |
| **Interprétabilité** | ✓✓ Axes = composantes | ✗ Axes arbitraires | ⚠️ Difficile |
| **Usage** | Prétraitement | Visualisation finale | Moderne, polyvalent |

## 📐 Comparaison Visuelle : PCA vs t-SNE vs UMAP

```
Données Originales (haute dim) :
  Clusters + Topologie complexe

     PCA (2D) :              t-SNE (2D) :            UMAP (2D) :

    ●●●    ○○○              ●●●      ○○○          ●●●      ○○○
     ●      ○                 ●        ○             ●        ○
      ●    ○                  ●        ○             ●        ○
       ●●○○                    ●●    ○○               ●●    ○○
       ■■■■                       ■■■■                 ■■■■
        
✓ Rapide                    ✓ Clusters nets        ✓ Structure préservée
✗ Overlap clusters          ✗ Distances globales   ✓ Rapide
✓ Interprétable             ✗ Non déterministe     ✓ Équilibré

Observations :
  • PCA : Projete linéairement, peut mélanger clusters
  • t-SNE : Sépare bien les clusters mais perd structure globale
  • UMAP : Meilleur compromis local/global
```

## 🔍 t-SNE : Intuition

```
┌──────────────────────────────────────────────────────┐
│                 ALGORITHME t-SNE                      │
└──────────────────────────────────────────────────────┘

ÉTAPE 1 : Probabilités dans l'espace haute dimension
  Pour chaque paire (i,j), calculer similarité :
    p_{j|i} ∝ exp(-‖xᵢ - xⱼ‖²/2σᵢ²)
    
  Symétrisé : p_{ij} = (p_{j|i} + p_{i|j})/(2n)

ÉTAPE 2 : Probabilités dans l'espace 2D
  q_{ij} ∝ (1 + ‖yᵢ - yⱼ‖²)⁻¹  (distribution t)

ÉTAPE 3 : Minimiser KL-divergence
  KL(P||Q) = Σᵢⱼ p_{ij} log(p_{ij}/q_{ij})
  
  Gradient descent pour ajuster les yᵢ

Résultat :
  • Points proches en haute dim → proches en 2D
  • Points éloignés en haute dim → éloignés en 2D
  • Préserve structure locale (voisinage)
```

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

