# Chapitre 7 : Régression Linéaire

## 📚 Introduction

La régression linéaire est l'un des algorithmes les plus fondamentaux en machine learning. Ce chapitre couvre les moindres carrés, la régularisation (Ridge, Lasso) et les SVM pour la régression.

## 🗺️ Carte Mentale : Régression Linéaire

```
                    RÉGRESSION LINÉAIRE
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
    MOINDRES            RÉGULARISÉE          ROBUSTE
     CARRÉS                 │                   │
    (OLS)           ┌───────┴───────┐          SVM
        │           │               │        Régression
   Non biaisé    RIDGE           LASSO          │
   Variance ++     │               │        ε-insensitive
                L₂ penalty     L₁ penalty
                Rétrécit       Sélectionne
                              (Sparsité)
```

## 📊 Tableau Comparatif : OLS vs Ridge vs Lasso

| **Méthode** | **Formulation** | **Solution** | **Propriétés** | **Usage** |
|------------|----------------|-------------|---------------|-----------|
| **OLS** | min ‖Y-Xβ‖² | β̂ = (XᵀX)⁻¹XᵀY | Non biaisé, variance élevée | p << n, faible collinéarité |
| **Ridge** | min ‖Y-Xβ‖² + λ‖β‖² | β̂ = (XᵀX+λI)⁻¹XᵀY | Biaisé, variance réduite | p ≈ n, forte collinéarité |
| **Lasso** | min ‖Y-Xβ‖² + λ‖β‖₁ | Pas de formule fermée | Sélection de variables | p >> n, features redondantes |
| **Elastic Net** | min ‖Y-Xβ‖² + λ₁‖β‖₁ + λ₂‖β‖² | Itératif | Combine Ridge + Lasso | p >> n, groupes de features |

## 📐 Visualisation Géométrique

### Contraintes de Régularisation

```
Espace des paramètres (β₁, β₂) :

    β₂                          RIDGE (L₂)
     │                             ╭───╮
     │    ╭───────╮              ╱       ╲
     │  ╱           ╲           │    ●β* │  Région convexe
     │ │   Ellipses  │          │ solution│  Cercle
     │  ╲  de RSS   ╱            ╲       ╱
     ─●───────●────────→ β₁       ╰───╯
      │  β̂ (OLS)
      │
    
    β₂                          LASSO (L₁)
     │                             ╱╲
     │    ╭───────╮              ╱  ╲
     │  ╱           ╲           ◆    │     Région en losange
     │ │   Ellipses  │          │  ●β*│   Coins → sparsité
     │  ╲  de RSS   ╱            ╲  ╱ 
     ─●───────────────────→ β₁    ╲╱
      │
      │  Solution souvent sur un coin → β₁ ou β₂ = 0
```

### 🎯 Effet de λ sur les Coefficients

```
    Coefficient β
         │
    β̂OLS ●─────────────────────  OLS (λ=0)
         │╲
         │ ╲                      Ridge : Rétrécissement progressif
         │  ╲─────────           (jamais exactement 0)
         │   ╲      ╲
         │    ╲──────╲───        Lasso : Atteint 0
    0    │─────●──────●────→ λ   (sélection de variables)
         │    λ₁     λ₂
         │     
    Optimal via CV
```

---

## 7.1 Régression par Moindres Carrés

### 7.1.1 Notation et Estimateur de Base

**Modèle** :
```
Y = Xβ + ε
```

où X ∈ ℝⁿˣᵖ, β ∈ ℝᵖ, ε ~ N(0, σ²I)

**Estimateur des moindres carrés** :
```
β̂ = argmin_β ‖Y - Xβ‖²
```

**Solution** :
```
β̂ = (XᵀX)⁻¹XᵀY
```

```python
import numpy as np

def least_squares(X, y):
    """Régression par moindres carrés"""
    return np.linalg.solve(X.T @ X, X.T @ y)

# Avec sklearn
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### 7.1.2 Comportement Limite

**Théorème** : Si XᵀX/n → Σ et XᵀY/n → γ, alors :
```
β̂ → Σ⁻¹γ  quand n → ∞
```

### 7.1.3 Théorème de Gauss-Markov

Parmi tous les estimateurs **linéaires non biaisés**, β̂ a la variance minimale.

### 7.1.4 Version Noyau

```
f̂(x) = Σᵢ αᵢ K(x, xᵢ)
```

avec α = (K + λI)⁻¹y

---

## 7.2 Ridge Regression et Lasso

### 7.2.1 Ridge Regression

**Problème** :
```
minimize  ‖Y - Xβ‖² + λ‖β‖²
```

**Solution** :
```
β̂_ridge = (XᵀX + λI)⁻¹XᵀY
```

**Avantages** :
- Stabilité numérique
- Réduit la variance
- Toujours une solution unique

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
```

**Choix de λ** : Validation croisée

```python
from sklearn.linear_model import RidgeCV

ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])
ridge_cv.fit(X_train, y_train)
print(f"Best alpha: {ridge_cv.alpha_}")
```

### 7.2.2 Équivalence Formulations Contrainte/Pénalisée

Les deux problèmes sont équivalents :

**Forme contrainte** :
```
minimize  ‖Y - Xβ‖²
subject to  ‖β‖² ≤ t
```

**Forme pénalisée** :
```
minimize  ‖Y - Xβ‖² + λ‖β‖²
```

Pour chaque t, il existe λ tel que les solutions coïncident.

### 7.2.3 Régression Lasso

**Problème** :
```
minimize  ‖Y - Xβ‖² + λ‖β‖₁
```

**Propriété clé** : Sélection de variables (certains βᵢ = 0)

```python
from sklearn.linear_model import Lasso, LassoCV

# Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Nombre de coefficients non nuls
n_nonzero = np.sum(lasso.coef_ != 0)
print(f"Variables sélectionnées: {n_nonzero}/{len(lasso.coef_)}")

# Cross-validation
lasso_cv = LassoCV(cv=5)
lasso_cv.fit(X_train, y_train)
```

**Algorithme ISTA** (Iterative Soft Thresholding) :
```python
def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def ista(X, y, lambda_param, max_iter=1000, tol=1e-4):
    n, p = X.shape
    beta = np.zeros(p)
    L = np.linalg.norm(X.T @ X, 2)  # Constante de Lipschitz
    alpha = 1 / L
    
    for k in range(max_iter):
        grad = X.T @ (X @ beta - y)
        beta_new = soft_threshold(beta - alpha * grad, alpha * lambda_param)
        
        if np.linalg.norm(beta_new - beta) < tol:
            break
        
        beta = beta_new
    
    return beta
```

---

## 7.3 Autres Estimateurs de Parcimonie

### 7.3.1 LARS (Least Angle Regression)

Algorithme qui trouve tout le chemin de régularisation du Lasso efficacement.

```python
from sklearn.linear_model import Lars

lars = Lars(n_nonzero_coefs=10)
lars.fit(X_train, y_train)
```

### 7.3.2 Sélecteur de Dantzig

```
minimize  ‖β‖₁
subject to  ‖Xᵀ(Y - Xβ)‖_∞ ≤ λ
```

---

## 7.4 SVM pour la Régression

### 7.4.1 SVM Linéaire

**Fonction de perte ε-insensible** :
```
L_ε(y, f(x)) = max(0, |y - f(x)| - ε)
```

**Problème** :
```
minimize  (1/2)‖w‖² + C Σᵢ (ξᵢ + ξᵢ*)
subject to  yᵢ - (wᵀxᵢ + b) ≤ ε + ξᵢ
           (wᵀxᵢ + b) - yᵢ ≤ ε + ξᵢ*
           ξᵢ, ξᵢ* ≥ 0
```

```python
from sklearn.svm import SVR

svr = SVR(kernel='linear', epsilon=0.1, C=1.0)
svr.fit(X_train, y_train)
```

### 7.4.2 Kernel Trick pour SVM

```python
svr_rbf = SVR(kernel='rbf', gamma=0.1, epsilon=0.1)
svr_rbf.fit(X_train, y_train)
```

---

## 💡 Comparaison des Méthodes

| Méthode | Régularisation | Sélection | Complexité |
|---------|---------------|-----------|------------|
| OLS | Aucune | Non | O(p³) |
| Ridge | L2 | Non | O(p³) |
| Lasso | L1 | Oui | Itératif |
| Elastic Net | L1 + L2 | Oui | Itératif |

---

[⬅️ Partie 2](../partie-2-concepts/chapitre-06-noyaux.md) | [Retour](../README.md) | [Suite ➡️](./chapitre-08-classification-lineaire.md)

