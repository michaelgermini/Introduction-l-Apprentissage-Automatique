# Chapitre 21 : Réduction de Dimension et Analyse Factorielle

## 📚 Introduction

La réduction de dimension projette les données dans un espace de dimension réduite tout en préservant l'information.

## 🗺️ Carte Mentale : Réduction de Dimension

```
                RÉDUCTION DE DIMENSION
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
    LINÉAIRE         NON-LINÉAIRE       FACTORISATION
        │                 │                 │
    ┌───┴───┐         ┌───┴───┐         ┌───┴───┐
    │       │         │       │         │       │
  PCA    LDA      t-SNE  UMAP        NMF    ICA
    │       │         │       │         │       │
 SVD    Fisher   Manifold Graphe  Positive Independent
Variance  Discr.  Learning          Components
```

## 📊 Tableau Comparatif : Méthodes

| **Méthode** | **Type** | **Linéaire** | **Préserve** | **Scalabilité** | **Usage** |
|------------|---------|-------------|-------------|----------------|-----------|
| **PCA** | Projection | ✓ Oui | Variance | ✓✓✓ Excellente | Standard, prétraitement |
| **t-SNE** | Manifold | ✗ Non | Distances locales | ⚠️ Faible | Visualisation 2D/3D |
| **UMAP** | Graphe | ✗ Non | Structure topologique | ✓✓ Bonne | Visualisation moderne |
| **LDA** | Discriminant | ✓ Oui | Séparabilité classes | ✓✓ Bonne | Classification |
| **Autoencoder** | Deep Learning | ✗ Non | Reconstruction | ✓ Moyenne | Représentations |

---

## 21.1 Analyse en Composantes Principales (PCA)

### Principe

**Maximiser la variance** ou **minimiser l'erreur de reconstruction**.

**Solution** : Vecteurs propres de la matrice de covariance.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Variance expliquée
print(pca.explained_variance_ratio_)

# Composantes principales
print(pca.components_)
```

---

## 21.2 Kernel PCA

**PCA non linéaire** avec le kernel trick.

```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_reduced = kpca.fit_transform(X)
```

---

## 21.3 PCA Probabiliste

**Modèle** : x = Wz + μ + ε où z ~ N(0, I)

**EM Algorithm** pour estimer W.

---

## 21.4 Robust PCA

**Décomposition** : X = L + S
- L : matrice de rang faible
- S : matrice creuse (outliers)

---

## 21.5 Analyse en Composantes Indépendantes (ICA)

### Objectif

Trouver sources indépendantes : x = As où s sont indépendants.

```python
from sklearn.decomposition import FastICA

ica = FastICA(n_components=3)
S = ica.fit_transform(X)  # Sources
A = ica.mixing_  # Matrice de mélange
```

**Applications** : Séparation de sources (audio, EEG).

---

## 21.6 Factorisation Matricielle Non Négative (NMF)

**Contrainte** : X ≈ WH avec W, H ≥ 0

```python
from sklearn.decomposition import NMF

nmf = NMF(n_components=10, init='random')
W = nmf.fit_transform(X)
H = nmf.components_
```

**Applications** : Traitement d'images, analyse de texte.

---

[⬅️ Chapitre précédent](./chapitre-20-clustering.md) | [Retour](../README.md) | [Suite ➡️](./chapitre-22-visualisation.md)

