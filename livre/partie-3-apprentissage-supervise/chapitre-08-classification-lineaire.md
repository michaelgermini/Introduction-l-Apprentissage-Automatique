# Chapitre 8 : Modèles de Classification Linéaire

## 📚 Introduction

Ce chapitre couvre les principales méthodes de classification linéaire : régression logistique, analyse discriminante linéaire, et machines à vecteurs de support.

## 🗺️ Carte Mentale : Classification Linéaire

```
                CLASSIFICATION LINÉAIRE
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   DISCRIMINATIVE    GÉNÉRATIVE       MARGIN-BASED
        │                │                │
    ┌───┴───┐        ┌───┴───┐          SVM
    │       │        │       │           │
 Logistic  Perceptron LDA    QDA    Max Margin
 Regression    │        │     │          │
    │      One-layer Fisher's Bayes  Hard/Soft
  Softmax              │    Optimal   Margin
                   Dimension         Kernel
                   Reduction         Trick
```

## 📊 Tableau Comparatif des Méthodes

| **Méthode** | **Approche** | **Hypothèse** | **Frontière** | **Probabilités** | **Avantages** | **Inconvénients** |
|------------|-------------|--------------|--------------|-----------------|--------------|------------------|
| **Logistic Regression** | Discriminative | Aucune | Linéaire | ✓ Oui | Simple, interprétable | Linéaire seulement |
| **LDA** | Générative | Gaussien, Σ commune | Linéaire | ✓ Oui | Efficace, réduction dim. | Hypothèse forte |
| **QDA** | Générative | Gaussien, Σₖ différentes | Quadratique | ✓ Oui | Plus flexible | Plus de paramètres |
| **SVM** | Margin-based | Aucune | Linéaire/Non-linéaire | ✗ Non | Robuste, kernel trick | Pas de probabilités |
| **Perceptron** | Discriminative | Séparable | Linéaire | ✗ Non | Simple, en ligne | Pas de convergence si non séparable |

## 📐 Visualisation des Frontières de Décision

### Classification Binaire : Espace 2D

```
    x₂
     │                    RÉGRESSION LOGISTIQUE
     │    Classe 1        ╱ Frontière linéaire
     │  ●  ●  ●  ●      ╱  P(Y=1|x) = 0.5
     │   ●  ●  ●       ╱
     │  ●  ●  ●  ●    ╱
     │ ────────────  ╱  ───────────
     │          ○  ╱○  ○
     │        ○  ╱  ○  ○  ○
     │      ○  ╱  ○  ○  ○
     │    ○  ╱  ○  ○        Classe 0
     └──────────────────────────→ x₁

    x₂
     │                         LDA
     │    Classe 1        
     │  ●  ●  ●  ●      μ₁●  Centroïde
     │   ●  ●  ●          │
     │  ●  ●  ●  ●        │
     │ ──────────────────────────
     │        ○  ○  ○    │
     │      ○  ○  ○  ○   │
     │    ○  ○  ○  ○     │
     │  ○  ○  ○      μ₀○  Centroïde
     └──────────────────────────→ x₁
     
    Frontière = {x : wᵀx + b = 0}
    w ∝ Σ⁻¹(μ₁ - μ₀)

    x₂
     │                        SVM
     │    Classe 1        
     │  ●  ●  ●  ●      ✱ Support vector
     │   ●  ✱  ●          
     │  ●  ●  ●  ●    ╱───╲  Marges
     │ ────────────  ╱  │  ╲ ────────
     │        ✱  ○ ╱   │   ╲○  ○
     │      ○  ○  ╱    │    ╲  ○  ○
     │    ○  ○  ╱     │     ╲  ○
     │  ○  ○       ✱           ○
     └──────────────────────────→ x₁
     
    Maximise la marge : 2/‖w‖
```

## 🎯 Comparaison Géométrique

```
┌─────────────────────────────────────────────────────────┐
│  LOGISTIC REGRESSION                                     │
│  • Minimise la log-loss (cross-entropy)                 │
│  • Frontière probabiliste douce                         │
│  • Tous les points contribuent                          │
│    Loss = -Σᵢ [yᵢlog(p) + (1-yᵢ)log(1-p)]              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  LDA (Linear Discriminant Analysis)                      │
│  • Suppose distributions gaussiennes                     │
│  • Maximise séparation inter-classes / intra-classe     │
│  • Frontière = équiprobabilité bayésienne               │
│    Frontière : P(Y=1|x) = P(Y=0|x)                      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  SVM (Support Vector Machine)                            │
│  • Maximise la marge (distance minimale)                │
│  • Seuls les support vectors comptent                    │
│  • Robuste aux outliers                                  │
│    Marge = 2/‖w‖,  min ‖w‖²  s.t.  yᵢ(wᵀxᵢ+b) ≥ 1      │
└─────────────────────────────────────────────────────────┘
```

## 📈 Fonctions de Décision

```
    f(x) = wᵀx + b
     │
     │     LOGISTIC              LDA               SVM
     │      
  1  │      ╱────              ╱│╲              ╱│╲
     │     ╱                  ╱ │ ╲            ╱ │ ╲
  0.5│────●────              ──●──            ──●──
     │   ╱   σ(f)             Bayes         Hard margin
  0  │  ╱                       │                │
     │                          │                │
     └──────────────────────────────────────────→ x
             │                  │                │
         Seuil 0            μ₀ = μ₁         Marge max

P(Y=1|x) = σ(f(x))      P(Y=k|x) ∝ πₖφₖ(x)    ŷ = sign(f(x))
```

---

## 8.1 Régression Logistique

### 8.1.1 Cadre Général

**Modèle** : Pour classification binaire (Y ∈ {0, 1}) :
```
P(Y = 1|X = x) = σ(wᵀx + b)
```

où σ(z) = 1/(1 + e⁻ᶻ) est la fonction sigmoïde.

### 8.1.2 Log-Vraisemblance Conditionnelle

**Fonction objectif** :
```
ℓ(w, b) = Σᵢ [yᵢ log p(xᵢ) + (1-yᵢ) log(1-p(xᵢ))]
```

où p(x) = σ(wᵀx + b)

### 8.1.3 Algorithme d'Entraînement

**Gradient** :
```
∇ℓ = Σᵢ (yᵢ - p(xᵢ)) xᵢ
```

```python
from sklearn.linear_model import LogisticRegression

# Classification binaire
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Prédiction
y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)
```

### 8.1.4 Régression Logistique Pénalisée

```
minimize  -ℓ(w, b) + λ‖w‖²   (Ridge)
minimize  -ℓ(w, b) + λ‖w‖₁   (Lasso)
```

```python
# L2 penalty
lr_ridge = LogisticRegression(penalty='l2', C=1.0)

# L1 penalty (sélection de variables)
lr_lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
```

### 8.1.5 Régression Logistique à Noyaux

```
f(x) = Σᵢ αᵢ K(x, xᵢ)
P(Y = 1|X = x) = σ(f(x))
```

---

## 8.2 Analyse Discriminante Linéaire (LDA)

### 8.2.1 Modèle Génératif

**Hypothèses** :
- X|Y = k ~ N(μₖ, Σ)  (même covariance)
- P(Y = k) = πₖ

**Règle de classification** :
```
ŷ = argmax_k [log πₖ - ½(x - μₖ)ᵀΣ⁻¹(x - μₖ)]
```

**Frontière de décision** : Linéaire !

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
```

### 8.2.2 Réduction de Dimension

LDA projette sur l'espace de dimension K-1 (K classes) qui maximise :
```
J(W) = |WᵀS_B W| / |WᵀS_W W|
```

où :
- S_B : matrice de covariance between-class
- S_W : matrice de covariance within-class

```python
# LDA pour réduction de dimension
lda = LinearDiscriminantAnalysis(n_components=2)
X_reduced = lda.fit_transform(X_train, y_train)
```

### 8.2.3 LDA de Fisher

Maximiser :
```
J(w) = (wᵀ(μ₁ - μ₂))² / (wᵀ(Σ₁ + Σ₂)w)
```

**Solution** :
```
w = (Σ₁ + Σ₂)⁻¹(μ₁ - μ₂)
```

---

## 8.3 Notation Optimale

Trouver une transformation Y → Z qui rend la classification plus facile.

**Objectif** : Maximiser la corrélation entre Z et les indicatrices de classes.

---

## 8.4 SVM (Support Vector Machines)

### 8.4.1 Perceptron et Marge

**Perceptron** : Trouver w tel que yᵢ(wᵀxᵢ) > 0

**Marge** : Distance du point le plus proche à l'hyperplan :
```
m = min_i |wᵀxᵢ| / ‖w‖
```

### 8.4.2 Maximisation de la Marge

**Problème primal** :
```
minimize    ½‖w‖²
subject to  yᵢ(wᵀxᵢ + b) ≥ 1  pour tout i
```

**Version soft-margin** :
```
minimize    ½‖w‖² + C Σᵢ ξᵢ
subject to  yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ
            ξᵢ ≥ 0
```

### 8.4.3 Problème Dual et Conditions KKT

**Dual** :
```
maximize    Σᵢ αᵢ - ½Σᵢⱼ αᵢαⱼyᵢyⱼ(xᵢᵀxⱼ)
subject to  0 ≤ αᵢ ≤ C
            Σᵢ αᵢyᵢ = 0
```

**Solution** :
```
w = Σᵢ αᵢyᵢxᵢ
```

**Vecteurs de support** : Points avec αᵢ > 0

### 8.4.4 Version à Noyaux

Remplacer xᵢᵀxⱼ par K(xᵢ, xⱼ) :

```
maximize    Σᵢ αᵢ - ½Σᵢⱼ αᵢαⱼyᵢyⱼK(xᵢ, xⱼ)
```

```python
from sklearn.svm import SVC

# SVM linéaire
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)

# SVM avec noyau RBF
svm_rbf = SVC(kernel='rbf', gamma=0.1, C=1.0)
svm_rbf.fit(X_train, y_train)

# Vecteurs de support
n_support = svm_rbf.n_support_
print(f"Nombre de vecteurs de support: {n_support}")
```

---

## 💡 Comparaison des Méthodes

| Méthode | Type | Frontière | Probabilités |
|---------|------|-----------|--------------|
| Logistic | Discriminatif | Linéaire | Oui |
| LDA | Génératif | Linéaire | Oui |
| SVM | Discriminatif | Non linéaire (kernel) | Non (direct) |

---

[⬅️ Chapitre précédent](./chapitre-07-regression-lineaire.md) | [Retour](../README.md) | [Suite ➡️](./chapitre-09-plus-proches-voisins.md)

