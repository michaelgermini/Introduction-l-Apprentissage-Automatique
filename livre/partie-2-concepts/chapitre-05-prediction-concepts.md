# Chapitre 5 : Prédiction - Concepts de Base

## 📚 Introduction

Ce chapitre établit les fondements théoriques de la prédiction statistique, qui est au cœur du machine learning supervisé.

---

## 5.1 Cadre Général

### Problème de Prédiction

Nous avons :
- **X** : variable d'entrée (features)
- **Y** : variable de sortie (target)
- **Distribution conjointe** P(X, Y)

**Objectif** : Construire un prédicteur f : X → Y qui minimise une fonction de perte L.

**Risque** :
```
R(f) = 𝔼[L(Y, f(X))]
```

### Fonctions de Perte Courantes

**Régression** :
- Erreur quadratique : L(y, ŷ) = (y - ŷ)²
- Erreur absolue : L(y, ŷ) = |y - ŷ|

**Classification** :
- Perte 0-1 : L(y, ŷ) = 1_{y ≠ ŷ}
- Log-loss : L(y, ŷ) = -log P(Y = y|X)

---

## 5.2 Prédicteur de Bayes

### Théorème

Le **prédicteur de Bayes** minimise le risque :

**Régression** (L quadratique) :
```
f*(x) = 𝔼[Y|X = x]
```

**Classification** (perte 0-1) :
```
f*(x) = argmax_y P(Y = y|X = x)
```

### Risque de Bayes

Le **risque de Bayes** R* = R(f*) est le risque minimal atteignable.

**Erreur irréductible** : Même le meilleur modèle ne peut pas faire mieux que R*.

---

## 5.3 Exemples : Approche Basée sur des Modèles

### 5.3.1 Modèles Gaussiens et Naïve Bayes

#### Classification avec Loi Normale

Si X|Y = k ~ N(μₖ, Σ) :
```
P(Y = k|X = x) ∝ πₖ · exp(-½(x - μₖ)ᵀΣ⁻¹(x - μₖ))
```

**Naïve Bayes** : Suppose l'indépendance conditionnelle :
```
P(X|Y = k) = ∏ⱼ P(Xⱼ|Y = k)
```

```python
from sklearn.naive_bayes import GaussianNB

# Données
X_train, y_train = ...  # Features et labels

# Modèle
model = GaussianNB()
model.fit(X_train, y_train)

# Prédiction
y_pred = model.predict(X_test)
```

### 5.3.2 Régression par Noyaux

```
f̂(x) = Σᵢ wᵢ(x) yᵢ
```

où les poids dépendent de la distance :
```
wᵢ(x) = K((x - xᵢ)/h) / Σⱼ K((x - xⱼ)/h)
```

---

## 5.4 Minimisation du Risque Empirique

### 5.4.1 Principes Généraux

On n'a pas accès à P(X, Y), mais aux données (x₁, y₁), ..., (xₙ, yₙ).

**Risque empirique** :
```
R̂(f) = (1/n) Σᵢ L(yᵢ, f(xᵢ))
```

**ERM** (Empirical Risk Minimization) :
```
f̂ = argmin_{f ∈ ℱ} R̂(f)
```

### 5.4.2 Biais et Variance

**Décomposition de l'erreur** :
```
𝔼[L(Y, f̂(X))] = R* + Approximation + Estimation
```

- **R*** : Risque de Bayes (irréductible)
- **Approximation** : inf_{f ∈ ℱ} R(f) - R*
- **Estimation** : R(f̂) - inf_{f ∈ ℱ} R(f)

---

## 5.5 Évaluation de l'Erreur

### 5.5.1 Erreur de Généralisation

**Problème** : R̂(f̂) est optimiste car f̂ est ajusté sur les mêmes données.

**Erreur de généralisation** : Évaluer sur de nouvelles données.

### 5.5.2 Validation Croisée

#### Hold-out

```
Training set (70%) → Ajuster le modèle
Test set (30%) → Évaluer
```

#### K-Fold Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"CV Score: {scores.mean():.3f} ± {scores.std():.3f}")
```

#### Leave-One-Out (LOO)

```
Pour i = 1 à n:
    Entraîner sur {1, ..., n} \ {i}
    Tester sur i
```

**Erreur LOOCV** :
```
CV = (1/n) Σᵢ L(yᵢ, f̂^{(-i)}(xᵢ))
```

---

## 💡 Points Clés

1. **Prédicteur de Bayes** : Optimal en théorie
2. **ERM** : Minimiser l'erreur empirique
3. **Compromis biais-variance** : Choix de la classe de fonctions ℱ
4. **Validation croisée** : Estimation non biaisée de l'erreur

---

[⬅️ Chapitre précédent](./chapitre-04-biais-variance.md) | [Retour](../README.md) | [Suite ➡️](./chapitre-06-noyaux.md)

