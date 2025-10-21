# Chapitre 10 : Algorithmes Basés sur les Arbres

## 📚 Introduction

Les arbres de décision et leurs extensions (forêts aléatoires, boosting) sont parmi les algorithmes les plus populaires en ML.

## 🗺️ Carte Mentale : Méthodes à Base d'Arbres

```
                ALGORITHMES BASÉS SUR LES ARBRES
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
    ARBRE SIMPLE        BAGGING            BOOSTING
        │                   │                   │
    CART                Random           ┌───────┴───────┐
        │               Forest           │               │
    ┌───┴───┐              │          AdaBoost      Gradient
    │       │          Ensemble        │            Boosting
Classif. Régres.      parallèle    Séquentiel         │
    │       │         (variance↓)    reweight      XGBoost
  Gini    MSE            │          samples       LightGBM
Entropy  MAE         Bootstrap         │           CatBoost
                     + Random       Focus on
                      Features      errors
```

## 📊 Tableau Comparatif : Arbres et Ensembles

| **Méthode** | **Type** | **Biais** | **Variance** | **Vitesse** | **Interprétabilité** | **Usage** |
|------------|---------|----------|-------------|-----------|-------------------|-----------|
| **Arbre Simple** | Base | Moyen | ⬆️ Élevée | ⚡⚡⚡ Rapide | ✓✓✓ Excellente | Exploration |
| **Random Forest** | Bagging | Moyen | ✓ Faible | ⚡⚡ Moyen | ⚠️ Moyenne | Production standard |
| **AdaBoost** | Boosting | ✓ Faible | Moyen | ⚡⚡ Moyen | ⚠️ Difficile | Perf. élevée |
| **XGBoost** | Gradient Boosting | ✓✓ Très faible | ✓ Faible | ⚡⚡ Moyen | ⚠️ Difficile | Compétitions |
| **LightGBM** | Gradient Boosting | ✓✓ Très faible | ✓ Faible | ⚡⚡⚡ Rapide | ⚠️ Difficile | Big Data |

## 📐 Visualisation : Arbre de Décision

### Structure d'un Arbre Binaire

```
                    [Racine]
                   Feature_3 ≤ 2.5 ?
                    /          \
                  OUI           NON
                  /              \
          [Nœud gauche]      [Nœud droit]
         Feature_1 ≤ 5.0 ?  Feature_2 ≤ 3.5 ?
            /      \           /        \
          OUI      NON       OUI        NON
          /         \        /           \
    [Feuille]  [Feuille] [Feuille]  [Feuille]
     Classe A   Classe B  Classe A   Classe C
     (50/50)    (30/32)   (45/47)    (80/80)
     Pureté:     Pureté:   Pureté:    Pureté:
      100%        93.8%     95.7%      100%

Notation :
  - Nœud : Condition de split (feature + seuil)
  - Feuille : Prédiction finale (classe majoritaire ou moyenne)
  - (n_correct/n_total) : Échantillons dans la feuille
```

### Partitionnement de l'Espace

```
    Feature_2
        │
    10  │  ┌────────┬─────────┐
        │  │        │         │
     8  │  │   C    │    B    │  Split 1: Feature_2 = 7
        │  │        │         │
     6  │  ├────────┴─────────┤
        │  │                  │  Split 2: Feature_1 = 5
     4  │  │        A         │
        │  │                  │
     2  │  └──────────────────┘
        │
     0  └──────────────────────→ Feature_1
        0    5         10

Chaque région = une feuille de l'arbre
Frontières de décision = orthogonales aux axes
```

## 🎯 Critères de Division

### Pour Classification

```
┌───────────────────────────────────────────────────────┐
│  GINI IMPURITY                                         │
│  Gini = 1 - Σₖ pₖ²                                    │
│                                                        │
│  Exemple : Nœud avec [50 Class A, 30 Class B]         │
│  p_A = 50/80 = 0.625                                  │
│  p_B = 30/80 = 0.375                                  │
│  Gini = 1 - (0.625² + 0.375²) = 0.469                │
│                                                        │
│  Nœud pur (une seule classe) → Gini = 0              │
│  Nœud équilibré (50/50) → Gini = 0.5                 │
└───────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────┐
│  ENTROPY (Information Gain)                            │
│  H = -Σₖ pₖ log₂(pₖ)                                  │
│                                                        │
│  Même exemple :                                        │
│  H = -(0.625 log₂(0.625) + 0.375 log₂(0.375))        │
│    ≈ 0.954                                            │
│                                                        │
│  Information Gain = H_parent - Σ (n_child/n) H_child  │
└───────────────────────────────────────────────────────┘
```

### Pour Régression

```
┌───────────────────────────────────────────────────────┐
│  MEAN SQUARED ERROR (MSE)                              │
│  MSE = (1/n) Σᵢ (yᵢ - ȳ)²                            │
│                                                        │
│  Exemple : Nœud avec valeurs [1, 2, 5, 7, 9]         │
│  ȳ = (1+2+5+7+9)/5 = 4.8                             │
│  MSE = [(1-4.8)² + (2-4.8)² + ... + (9-4.8)²]/5      │
│      = [14.44 + 7.84 + 0.04 + 4.84 + 17.64]/5        │
│      = 8.96                                            │
│                                                        │
│  On cherche le split qui minimise : MSE_total         │
│  MSE_total = (n_L/n)·MSE_L + (n_R/n)·MSE_R           │
└───────────────────────────────────────────────────────┘
```

## 🌳 Comparaison Visuelle : Ensemble Methods

```
    ARBRE SIMPLE                RANDOM FOREST (Bagging)
    
         🌳                     🌳  🌳  🌳  🌳  🌳
      Un arbre                  100 arbres
      profond                   indépendants
         │                            │
    Variance élevée            Moyenne/Vote → Variance réduite
    Overfitting                Meilleure généralisation


    ADABOOST (Boosting)         GRADIENT BOOSTING
    
    🌱 → 🌱 → 🌱 → 🌱           🌱 → 🌱 → 🌱 → 🌱
    t=1  t=2  t=3  t=4         t=1  t=2  t=3  t=4
    │    │    │    │           │    │    │    │
  Focus  sur   les  erreurs    Fit residuals à chaque étape
  Reweight samples             Apprend gradient de la loss
  Séquentiel                   Séquentiel
```

## 📈 Évolution de l'Erreur

```
    Erreur
      │
      │  ──── Arbre simple (test)
      │ ╲
  1.0 │  ╲     ──── Random Forest
      │   ╲──────────────
      │         ──── XGBoost
  0.5 │          ╲╲
      │            ╲╲╲____________
      │               ╲╲╲╲╲╲╲╲╲╲╲
  0.0 └─────────────────────────────→ Nombre d'arbres
       1    10    50   100   500

Random Forest : Converge rapidement, plateau
XGBoost : Continue à améliorer, risque overfit si trop d'arbres
```

---

## 10.1 Partitionnement Récursif

### 10.1.1 Arbres de Décision Binaires

**Structure** : Arbre binaire où :
- Nœuds internes : tests sur les features
- Feuilles : prédictions

### 10.1.2 Algorithme d'Entraînement

**CART** (Classification And Regression Trees) :

1. Choisir la meilleure division (feature + seuil)
2. Diviser les données
3. Répéter récursivement

**Critère de division** :

**Régression** (MSE) :
```
Impurity = Σ_{samples} (y - ȳ)²
```

**Classification** (Gini) :
```
Gini = 1 - Σ_c p_c²
```

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Classification
tree_clf = DecisionTreeClassifier(max_depth=5, min_samples_split=20)
tree_clf.fit(X_train, y_train)

# Visualisation
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(tree_clf, filled=True, feature_names=feature_names)
plt.show()
```

### 10.1.3 Règle d'Arrêt

**Critères** :
- Profondeur maximale
- Nombre minimal d'échantillons par nœud
- Amélioration minimale

### 10.1.4 Élagage (Pruning)

**Cost-complexity pruning** :
```
R_α(T) = R(T) + α|T|
```

où |T| est le nombre de feuilles.

```python
# Avec élagage
tree = DecisionTreeClassifier(ccp_alpha=0.01)
tree.fit(X_train, y_train)
```

---

## 10.2 Forêts Aléatoires

### 10.2.1 Bagging

**Bootstrap Aggregating** :
1. Générer B échantillons bootstrap
2. Entraîner un arbre sur chaque échantillon
3. Prédiction = moyenne/vote majoritaire

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Classification
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    max_features='sqrt',
    random_state=42
)
rf.fit(X_train, y_train)

# Importance des variables
importances = rf.feature_importances_
```

### 10.2.2 Randomisation des Features

À chaque division, considérer seulement √p features aléatoires (classification) ou p/3 (régression).

**Avantages** :
- Réduit la corrélation entre arbres
- Améliore la généralisation

---

## 10.3 Adaboost

### 10.3.1 Principe

**Boosting** : Combiner séquentiellement des apprenants faibles.

### 10.3.2 Algorithme Adaboost

```
Pour m = 1 à M:
    1. Entraîner h_m avec poids w_i
    2. Calculer erreur ε_m = Σ w_i · 1_{y_i ≠ h_m(x_i)} / Σ w_i
    3. α_m = ½ log((1 - ε_m) / ε_m)
    4. Mettre à jour: w_i ← w_i · exp(-α_m y_i h_m(x_i))

Prédiction: sign(Σ_m α_m h_m(x))
```

```python
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0
)
ada.fit(X_train, y_train)
```

---

## 10.4 Gradient Boosting

### Principe

**Idée** : Ajuster séquentiellement la résiduelle.

```
f_0(x) = 0
Pour m = 1 à M:
    r_i = y_i - f_{m-1}(x_i)
    h_m = fit(X, r)
    f_m(x) = f_{m-1}(x) + η · h_m(x)
```

```python
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8
)
gbr.fit(X_train, y_train)
```

### XGBoost

**Optimisations** :
- Régularisation L1/L2
- Gestion des valeurs manquantes
- Parallélisation

```python
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0
)
xgb_model.fit(X_train, y_train)
```

---

## 💡 Comparaison

| Méthode | Type | Force | Faiblesse |
|---------|------|-------|-----------|
| Arbre | Non paramétrique | Interprétable | Instable |
| Random Forest | Ensemble | Robuste | Boîte noire |
| Adaboost | Boosting | Précis | Sensible au bruit |
| Gradient Boosting | Boosting | Très précis | Temps d'entraînement |

---

[⬅️ Chapitre précédent](./chapitre-09-plus-proches-voisins.md) | [Retour](../README.md) | [Suite ➡️](./chapitre-11-reseaux-neurones.md)

