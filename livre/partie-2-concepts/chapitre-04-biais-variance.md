# Chapitre 4 : Biais et Variance

## 📚 Introduction

Le compromis biais-variance est un concept fondamental en machine learning qui explique pourquoi les modèles peuvent sous-apprendre (underfitting) ou sur-apprendre (overfitting). Ce chapitre illustre ce dilemme à travers l'estimation de paramètres et l'estimation de densité.

## 🗺️ Carte Mentale : Compromis Biais-Variance

```
                 COMPROMIS BIAIS-VARIANCE
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
    UNDERFITTING    OPTIMAL FIT      OVERFITTING
        │                 │                 │
   Biais élevé      Équilibré        Variance élevée
   Variance faible   Biais/Var       Biais faible
        │                 │                 │
   Modèle trop      Généralise       Mémorise
   simple           bien             données
        │                                   │
   ↓ Capacité                          ↑ Capacité
```

## 📐 Visualisation du Compromis

### Décomposition de l'Erreur

```
    Erreur
      │
      │     Erreur totale = Biais² + Variance + Bruit
      │
      │         ╱╲
  1.0 │        ╱  ╲
      │       ╱    ╲              Variance
      │      ╱      ╲────────────────
      │     ╱        ╲╲          ╱
      │    ╱          ╲╲        ╱
  0.5 │   ╱            ╲╲      ╱
      │  ╱   Biais²     ╲╲    ╱
      │ ╱                ╲╲  ╱
      │╱__________________╲╲╱_____ Bruit irréductible
  0   └─────────────────────────────→ Complexité
      Simple          Optimal     Complexe
    (underfit)                   (overfit)
```

## 📊 Tableau Comparatif : Underfitting vs Overfitting

| **Aspect** | **Underfitting** | **Sweet Spot** | **Overfitting** |
|-----------|-----------------|---------------|----------------|
| **Complexité** | Trop simple | Appropriée | Trop complexe |
| **Biais** | ⬆️ Élevé | ✓ Faible | ✓ Très faible |
| **Variance** | ✓ Faible | ✓ Faible | ⬆️ Élevée |
| **Erreur Train** | Élevée | Faible | Très faible |
| **Erreur Test** | Élevée | Faible | Élevée |
| **Généralisation** | ✗ Mauvaise | ✓✓ Excellente | ✗ Mauvaise |
| **Exemple** | Ligne droite | Polynôme deg 3 | Polynôme deg 15 |

## 📈 Courbes d'Apprentissage Typiques

```
    Erreur
      │
      │  UNDERFITTING        OPTIMAL          OVERFITTING
      │  
      │  Train ─────         Train ───        Train ────╲
      │  Test  ─────         Test  ───        Test  ────/╲
      │                                                    ╲
      │  ────────────        ─────────        ──────────   ───
      │   Gap faible         Gap faible       Gap LARGE
      │   Err. élevée        Err. faible      Train → 0
      └────────────────────────────────────────────────→ Epochs

Diagnostic :
  • Gap faible + Err. élevée → Augmenter capacité
  • Gap large               → Régulariser / Plus de données
  • Convergence             → Sweet spot !
```

## 🎯 Diagramme de Décision : Que Faire ?

```
         Analyser erreur Train vs Test
                    │
        ┌───────────┼───────────┐
        │                       │
   Err Train       Err Train    Err Test
   élevée?         faible?      élevée?
        │                       │
      OUI                     OUI
        │                       │
    UNDERFITTING            OVERFITTING
        │                       │
    ┌───┴───┐               ┌───┴───┐
    │       │               │       │
  • Modèle  • Features    • Régula- • Plus de
    plus      meilleures    risation  données
    complexe                │         │
  • Plus    • Feature     • Dropout • Data aug-
    layers    engineering   │         mentation
                          • Early   • Simplifier
                            stopping  modèle
```

---

## 4.1 Estimation de Paramètres et Sieves

### Le Dilemme Biais-Variance

Pour un estimateur θ̂ d'un paramètre θ vrai :

**Erreur quadratique moyenne (MSE)** :
```
MSE(θ̂) = 𝔼[(θ̂ - θ)²] = Bias²(θ̂) + Var(θ̂)
```

où :
- **Biais** : Bias(θ̂) = 𝔼[θ̂] - θ
- **Variance** : Var(θ̂) = 𝔼[(θ̂ - 𝔼[θ̂])²]

### Illustration

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n, noise=0.1):
    """Génère des données d'une fonction polynomiale"""
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, noise, n)
    return x, y

# Différents modèles
def fit_polynomial(x, y, degree):
    """Ajuste un polynôme de degré donné"""
    return np.polyfit(x, y, degree)

# Visualisation
x_train, y_train = generate_data(20)
x_test = np.linspace(0, 1, 100)

plt.figure(figsize=(15, 5))
for i, degree in enumerate([1, 5, 15], 1):
    plt.subplot(1, 3, i)
    coef = fit_polynomial(x_train, y_train, degree)
    y_pred = np.polyval(coef, x_test)
    
    plt.scatter(x_train, y_train, label='Données')
    plt.plot(x_test, y_pred, 'r-', label=f'Degré {degree}')
    plt.title(f'Polynôme degré {degree}')
    plt.legend()
```

**Interprétation** :
- **Degré faible** : Biais élevé, variance faible (underfitting)
- **Degré optimal** : Compromis biais-variance
- **Degré élevé** : Biais faible, variance élevée (overfitting)

### Méthode des Sieves

**Principe** : Augmenter la complexité du modèle avec la taille des données.

Pour n observations, utiliser un espace de dimension d_n tel que :
```
d_n → ∞  mais  d_n/n → 0  quand n → ∞
```

---

## 4.2 Estimation de Densité par Noyaux

### Estimateur de Parzen

Pour estimer la densité p(x) à partir d'échantillons x₁, ..., xₙ :

```
p̂_h(x) = (1/nh) Σᵢ K((x - xᵢ)/h)
```

où :
- **K** : fonction noyau (K(u) ≥ 0, ∫ K(u)du = 1)
- **h** : largeur de bande (bandwidth)

### Noyaux Communs

```python
def gaussian_kernel(u):
    """Noyau gaussien"""
    return (1/np.sqrt(2*np.pi)) * np.exp(-u**2/2)

def epanechnikov_kernel(u):
    """Noyau d'Epanechnikov"""
    return 0.75 * (1 - u**2) * (np.abs(u) <= 1)

def kernel_density_estimate(data, x, h, kernel=gaussian_kernel):
    """
    Estimation de densité par noyaux
    """
    n = len(data)
    density = np.zeros_like(x)
    
    for i, xi in enumerate(x):
        u = (xi - data) / h
        density[i] = np.mean(kernel(u)) / h
    
    return density

# Exemple
np.random.seed(42)
data = np.concatenate([
    np.random.normal(0, 1, 100),
    np.random.normal(5, 1.5, 100)
])

x = np.linspace(-5, 10, 1000)

plt.figure(figsize=(15, 5))
for i, h in enumerate([0.1, 0.5, 2.0], 1):
    plt.subplot(1, 3, i)
    density = kernel_density_estimate(data, x, h)
    plt.plot(x, density, label=f'h = {h}')
    plt.hist(data, bins=30, density=True, alpha=0.3)
    plt.title(f'Bandwidth h = {h}')
    plt.legend()
```

### Analyse Biais-Variance

**Biais** :
```
Bias(p̂_h(x)) = O(h²)  si p est deux fois dérivable
```

**Variance** :
```
Var(p̂_h(x)) = O(1/(nh))
```

**MSE** :
```
MSE(p̂_h(x)) = O(h⁴) + O(1/(nh))
```

**Bandwidth optimal** :
```
h_opt ∝ n^{-1/5}
```

donnant MSE = O(n^{-4/5})

### Choix de la Largeur de Bande

#### Règle de Silverman

Pour des données approximativement normales :
```
h = 1.06 · σ̂ · n^{-1/5}
```

où σ̂ est l'écart-type empirique.

#### Validation Croisée

Minimiser :
```
CV(h) = ∫ p̂_h²(x)dx - (2/n) Σᵢ p̂_h^{(-i)}(xᵢ)
```

où p̂_h^{(-i)} est l'estimateur sans la i-ème observation.

```python
def cross_validation_bandwidth(data, h_values):
    """
    Sélection de h par validation croisée
    """
    n = len(data)
    cv_scores = []
    
    for h in h_values:
        score = 0
        for i in range(n):
            # Leave-one-out
            data_loo = np.delete(data, i)
            dens = kernel_density_estimate(data_loo, [data[i]], h)
            score += np.log(dens[0] + 1e-10)
        
        cv_scores.append(-score / n)
    
    best_h = h_values[np.argmin(cv_scores)]
    return best_h, cv_scores
```

---

## 💡 Points Clés

1. **Compromis Biais-Variance** : MSE = Bias² + Variance
2. **Underfitting** : Modèle trop simple (biais élevé)
3. **Overfitting** : Modèle trop complexe (variance élevée)
4. **Régularisation** : Contrôle le compromis
5. **Estimation de densité** : Le paramètre h contrôle le lissage

---

## 📝 Exercices

### Exercice 1
Générez des données et comparez les MSE de polynômes de différents degrés.

### Exercice 2
Implémentez l'estimation de densité avec différents noyaux et comparez les résultats.

### Exercice 3
Analysez théoriquement le biais et la variance de la moyenne empirique.

---

[⬅️ Partie 1](../partie-1-fondements/chapitre-03-optimisation.md) | [Retour](../README.md) | [Suite ➡️](./chapitre-05-prediction-concepts.md)

