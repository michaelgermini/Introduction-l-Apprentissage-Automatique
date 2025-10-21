# Chapitre 9 : Méthodes des Plus Proches Voisins

## 📚 Introduction

Les méthodes des k plus proches voisins (k-NN) sont des algorithmes simples mais puissants basés sur la proximité locale.

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

