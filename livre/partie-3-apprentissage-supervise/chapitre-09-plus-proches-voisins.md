# Chapitre 9 : Méthodes des Plus Proches Voisins

## 📚 Introduction

Les méthodes des k plus proches voisins (k-NN) sont des algorithmes simples mais puissants basés sur la proximité locale.

## 🗺️ Carte Mentale : k-NN

```
                    k-NEAREST NEIGHBORS
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
    RÉGRESSION        CLASSIFICATION        PARAMÈTRES
        │                   │                   │
    ┌───┴───┐           ┌───┴───┐           ┌───┴───┐
    │       │           │       │           │       │
 Moyenne  Poids      Vote     Poids       k    Distance
 simple  distance  majoritaire probabiliste │        │
    │       │           │       │        CV    Euclidienne
  ŷ=1/k Σy  Inverse    argmax   Σ1_c/k       Manhattan
         distance²              │         Minkowski
                            Probabilités
```

## 📐 Visualisation : Fonctionnement de k-NN

### Classification avec k = 1, k = 3, k = 7

```
k = 1 (1-NN)              k = 3 (3-NN)              k = 7 (7-NN)

  ● ●                      ● ●                      ● ●
 ● ? ●                    ● ? ●                    ● ? ●
  ● ●                      ● ●                      ● ●
   ●                        ●                        ●
                           ●                        ● ●

Plus proche:             3 voisins:               7 voisins:
  1 bleu                   2 bleus, 1 rouge         4 bleus, 3 rouges
→ Classe: Bleu          → Classe: Bleu           → Classe: Bleu

Frontière:              Frontière:               Frontière:
Très irrégulière        Modérée                  Lisse
(overfit)               (équilibrée)             (peut underfit)
```

### Effet de k sur la Frontière de Décision

```
    k = 1                    k = 5                    k = 15

   ╱╲╱╲╱╲                   ╱──╲                     ╱────╲
  ╱  ●  ╲                  ╱ ●  ╲                   ╱  ●   ╲
 ╱   ●   ╲                ╱   ●  ╲                 ╱    ●   ╲
●─────────●              ●────────●               ●──────────●
╲    ○    ╱              ╲   ○   ╱               ╲     ○     ╱
 ╲   ○   ╱                ╲  ○  ╱                 ╲    ○    ╱
  ╲_○__○_╱                  ╲__○_╱                   ╲___○___╱

Complexe                 Équilibré                Lisse
Variance ↑               Compromis                Biais ↑
Overfitting              Optimal                  Underfitting
```

## 📊 Tableau : Effet de k

| **k** | **Biais** | **Variance** | **Frontière** | **Overfitting** | **Usage** |
|-------|----------|-------------|--------------|----------------|-----------|
| **k = 1** | ✓ Faible | ⬆️ Très élevée | Très irrégulière | ✗ Oui | Test rapide |
| **k = 3-5** | Moyen | Moyen | Modérée | ⚠️ Possible | Standard |
| **k = 10-20** | ⬆️ Élevé | ✓ Faible | Lisse | ✓ Non | Datasets bruités |
| **k = √n** | Élevé | Très faible | Très lisse | ✓ Non | Heuristique |

## 🎯 Algorithme k-NN : Étapes Détaillées

```
┌────────────────────────────────────────────────────────┐
│              ALGORITHME k-NN (Classification)           │
└────────────────────────────────────────────────────────┘

Entrée : 
  • Training set : {(x₁,y₁), ..., (xₙ,yₙ)}
  • Point de requête : x_query
  • Nombre de voisins : k

Étape 1 : CALCULER DISTANCES
  Pour chaque point d'entraînement xᵢ :
    dᵢ = distance(x_query, xᵢ)
    
  [x_query] ──d₁──→ [x₁, y₁]
     │
     ├────d₂──→ [x₂, y₂]
     │
     ├────d₃──→ [x₃, y₃]
     ⋮

Étape 2 : TRIER & SÉLECTIONNER k plus proches
  Trier : d₁ ≤ d₂ ≤ ... ≤ dₙ
  Garder les k premiers
  
  Exemple (k=3) :
    d₃ = 0.5  [x₃, y₃=○]  ✓ Voisin 1
    d₇ = 0.8  [x₇, y₇=●]  ✓ Voisin 2
    d₁ = 0.9  [x₁, y₁=○]  ✓ Voisin 3
    d₅ = 1.2  [x₅, y₅=●]  ✗ Trop loin

Étape 3 : VOTE MAJORITAIRE
  Compter les classes :
    Classe ○ : 2 votes
    Classe ● : 1 vote
    
  ŷ = argmax (votes) = ○

Sortie : ŷ = Classe prédite
```

## 📏 Tableau : Métriques de Distance

| **Distance** | **Formule** | **p** | **Propriétés** | **Usage** |
|-------------|-----------|------|---------------|-----------|
| **Euclidienne** | √(Σ(xᵢ-x'ᵢ)²) | p=2 | Isotrope, sensible échelle | Standard |
| **Manhattan** | Σ\|xᵢ-x'ᵢ\| | p=1 | Robuste outliers | Grilles, City-block |
| **Minkowski** | (Σ\|xᵢ-x'ᵢ\|ᵖ)^(1/p) | p≥1 | Généralise L₁, L₂ | Flexible |
| **Chebyshev** | max\|xᵢ-x'ᵢ\| | p=∞ | Dimension dominante | Jeux (échecs) |
| **Mahalanobis** | √((x-x')ᵀΣ⁻¹(x-x')) | - | Corrélations prises en compte | Features corrélées |

## 🔍 Visualisation : Distances

```
Point de référence : ●

     Manhattan (L₁)           Euclidienne (L₂)         Chebyshev (L∞)
         
    │     │     │                 ╱│╲                     ┌─────┐
  ──┼─────●─────┼──            ╱  │  ╲                   │     │
    │     │     │            ╱    ●    ╲                 │  ●  │
                           ╱      │      ╲               │     │
  Distance = 4          Distance = 2√2            Distance = 2
  (Losange)             (Cercle)                  (Carré)
```

## ⚠️ Malédiction de la Dimensionalité

```
┌──────────────────────────────────────────────────────┐
│         CURSE OF DIMENSIONALITY pour k-NN            │
└──────────────────────────────────────────────────────┘

Dimension d        Volume requis        Données pour
                  pour 1% voisins       densité uniforme
                  
d = 2              10% du rayon         n = 100
    ●●●                                  
    ●x●            ╭──╮                 Faisable
    ●●●            │  │
                   
d = 10             80% du rayon         n = 10¹⁰
    ●              ╭────╮               
    x●●            │    │               Impossible !
    ●              ╰────╯
                   
d = 100            99.5% du rayon       n = 10¹⁰⁰
    x●                                   
    ●              ╭──────╮             Astronomique
                   │      │

Problème : En haute dimension
  • Tous les points sont "loin"
  • Notion de "proximité" perd son sens
  • k-NN devient inefficace

Solution :
  • Réduction de dimension (PCA)
  • Sélection de features
  • Méthodes paramétriques
```

## 💡 Conseils Pratiques

```
┌──────────────────────────────────────────────┐
│         GUIDE D'UTILISATION k-NN             │
└──────────────────────────────────────────────┘

✓ QUAND UTILISER k-NN :
  • Dataset petit/moyen (n < 10,000)
  • Faible dimension (d < 20)
  • Frontière de décision complexe
  • Baseline rapide
  • Pas besoin d'interprétabilité

✗ ÉVITER k-NN SI :
  • Big Data (n > 100,000) → trop lent
  • Haute dimension (d > 50) → curse
  • Données déséquilibrées → biais
  • Temps réel requis → latence élevée

⚙️ OPTIMISATIONS :
  • KD-Tree, Ball-Tree pour recherche rapide
  • Normalisation des features (StandardScaler)
  • Poids par distance inverse
  • Cross-validation pour k optimal
```

---

## 9.1 Plus Proches Voisins pour la Régression

### Algorithme k-NN

**Prédiction** :
```
f̂(x) = (1/k) Σ_{i ∈ N_k(x)} yᵢ
```

où N_k(x) sont les k plus proches voisins de x.

```python
from sklearn.neighbors import KNeighborsRegressor

# Régression k-NN
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

### 9.1.1 Consistance

**Théorème** : Si k → ∞ et k/n → 0, alors :
```
f̂(x) → 𝔼[Y|X = x]  en probabilité
```

**Condition** : k croît, mais plus lentement que n.

### 9.1.2 Optimalité

**Taux de convergence** : O(n^{-4/(4+d)}) où d est la dimension.

**Malédiction de la dimensionalité** : Performance se dégrade rapidement avec d.

---

## 9.2 Classification k-NN

**Règle de décision** :
```
ŷ = argmax_c Σ_{i ∈ N_k(x)} 1_{yᵢ = c}
```

(vote majoritaire parmi les k voisins)

```python
from sklearn.neighbors import KNeighborsClassifier

# Classification k-NN
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)

# Prédiction
y_pred = knn_clf.predict(X_test)

# Probabilités
y_proba = knn_clf.predict_proba(X_test)
```

### Choix de k

**Validation croisée** :
```python
from sklearn.model_selection import cross_val_score

scores = []
k_values = range(1, 31)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train, y_train, cv=5).mean()
    scores.append(score)

best_k = k_values[np.argmax(scores)]
print(f"Meilleur k: {best_k}")
```

---

## 9.3 Conception de la Distance

### Distances Courantes

**Euclidienne** :
```
d(x, x') = ‖x - x'‖₂ = √(Σᵢ (xᵢ - x'ᵢ)²)
```

**Manhattan** :
```
d(x, x') = ‖x - x'‖₁ = Σᵢ |xᵢ - x'ᵢ|
```

**Minkowski** :
```
d(x, x') = (Σᵢ |xᵢ - x'ᵢ|ᵖ)^{1/p}
```

### Distance de Mahalanobis

```
d(x, x') = √((x - x')ᵀΣ⁻¹(x - x'))
```

Prend en compte la corrélation entre variables.

```python
from scipy.spatial.distance import mahalanobis

# Matrice de covariance
cov = np.cov(X_train.T)
cov_inv = np.linalg.inv(cov)

# Distance
dist = mahalanobis(x1, x2, cov_inv)
```

### Apprentissage de Métriques

**Objectif** : Apprendre une matrice M telle que :
```
d(x, x') = √((x - x')ᵀM(x - x'))
```

**LMNN** (Large Margin Nearest Neighbor)

---

## 💡 Avantages et Inconvénients

**Avantages** :
- Simple à comprendre et implémenter
- Pas de phase d'entraînement
- Non paramétrique

**Inconvénients** :
- Coût de prédiction élevé (O(n))
- Sensible à la dimension
- Nécessite beaucoup de mémoire

**Optimisations** :
- KD-Tree pour recherche rapide
- Ball Tree
- LSH (Locality Sensitive Hashing)

```python
# Avec KD-Tree (plus rapide pour d < 20)
knn_tree = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
knn_tree.fit(X_train, y_train)
```

---

[⬅️ Chapitre précédent](./chapitre-08-classification-lineaire.md) | [Retour](../README.md) | [Suite ➡️](./chapitre-10-arbres.md)

