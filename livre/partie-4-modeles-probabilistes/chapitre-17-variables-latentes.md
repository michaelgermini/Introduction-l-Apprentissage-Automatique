# Chapitre 17 : Variables Latentes et Méthodes Variationnelles

## 📚 Introduction

Ce chapitre traite des modèles avec variables non observées et des méthodes d'approximation variationnelle.

## 🗺️ Carte Mentale : Variables Latentes

```
              VARIABLES LATENTES
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   MODÈLES       INFÉRENCE      APPLICATIONS
        │             │             │
    ┌───┴───┐     ┌───┴───┐     ┌───┴───┐
    │       │     │       │     │       │
  GMM    HMM     EM    VI/VB   Topic  PCA
 Mixture Hidden  │     Mean   Model Probab.
    │   Markov  ELBO  Field    LDA    │
 Clusters  │    Max  Approx.        Factor
         States                      Analysis
```

## 📊 Tableau : Algorithme EM vs Inférence Variationnelle

| **Critère** | **EM** | **Inférence Variationnelle** |
|------------|--------|----------------------------|
| **Objectif** | Maximiser P(X\|θ) | Maximiser ELBO |
| **Distribution q** | Exacte P(Z\|X,θ) | Approximation q(Z) |
| **E-step** | ✓ Calculer posteriors | ✓ Optimiser q(Z) |
| **M-step** | ✓ Optimiser θ | ✓ Optimiser θ |
| **Convergence** | ✓ Garantie (croissante) | ✓ Garantie (ELBO ↑) |
| **Calcul** | Exact si simple | Approximatif |
| **Usage** | GMM, HMM | Deep Learning (VAE) |

## 🎯 Algorithme EM : Visualisation

```
┌──────────────────────────────────────────────────────────┐
│           EXPECTATION-MAXIMIZATION (EM)                   │
└──────────────────────────────────────────────────────────┘

Exemple : Gaussian Mixture Model (GMM)

Données :     ●●●    ○○○    ■■■
              ●●      ○○     ■■
              ●●●    ○○○    ■■■

INITIALISATION :
  θ⁽⁰⁾ = {μₖ, Σₖ, πₖ}  (3 Gaussiennes)

ITÉRATION t :

  E-STEP : Calculer responsabilités
    γᵢₖ = P(Zᵢ = k | xᵢ, θ⁽ᵗ⁾)
        = πₖ N(xᵢ|μₖ,Σₖ) / Σⱼ πⱼ N(xᵢ|μⱼ,Σⱼ)
    
    ● point xᵢ → γᵢ₁=0.8, γᵢ₂=0.1, γᵢ₃=0.1
    (80% cluster 1)

  M-STEP : Mettre à jour paramètres
    μₖ⁽ᵗ⁺¹⁾ = Σᵢ γᵢₖ xᵢ / Σᵢ γᵢₖ
    (moyenne pondérée)
    
    Σₖ⁽ᵗ⁺¹⁾ = Σᵢ γᵢₖ (xᵢ-μₖ)(xᵢ-μₖ)ᵀ / Σᵢ γᵢₖ
    
    πₖ⁽ᵗ⁺¹⁾ = Σᵢ γᵢₖ / n

CONVERGENCE :
  log P(X|θ) augmente à chaque itération
  Arrêt : Δ log P(X|θ) < ε

Résultat :    ●●●    ○○○    ■■■
              ●●      ○○     ■■
             ╱  ╲   ╱  ╲   ╱  ╲
           N(μ₁) N(μ₂) N(μ₃)
           Clusters identifiés !
```

---

## 17.1 Introduction

**Variables latentes** Z : non observées mais influencent les données X.

**Exemple** : Modèles de mélange.

---

## 17.2 Principe Variationnel

**ELBO** (Evidence Lower Bound) :
```
log P(X) ≥ 𝔼_q[log P(X, Z)] - 𝔼_q[log q(Z)]
         = 𝔼_q[log P(X, Z)/q(Z)]
```

**Objectif** : Maximiser ELBO par rapport à q.

---

## 17.3 Approximations

### 17.3.1 Approximation de Mode

q(Z) = δ(Z - ẑ) où ẑ = argmax P(Z|X)

### 17.3.2 Approximation Gaussienne

q(Z) = N(μ, Σ)

### 17.3.3 Mean-Field

q(Z) = ∏ᵢ q_i(Z_i) (indépendance)

---

## 17.4 Algorithme EM

**E-step** : Calculer Q(θ) = 𝔼_{Z|X,θ_old}[log P(X, Z|θ)]

**M-step** : θ_new = argmax_θ Q(θ)

```python
def em_algorithm(X, n_iter):
    # Initialisation
    theta = initialize_params()
    
    for t in range(n_iter):
        # E-step
        responsibilities = compute_responsibilities(X, theta)
        
        # M-step
        theta = update_parameters(X, responsibilities)
    
    return theta
```

---

## 17.5 Modèles de Mélange Gaussien

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3)
gmm.fit(X)

# Prédiction
labels = gmm.predict(X)
proba = gmm.predict_proba(X)
```

---

[⬅️ Chapitre précédent](./chapitre-16-reseaux-bayesiens.md) | [Retour](../README.md) | [Suite ➡️](./chapitre-18-apprentissage-graphiques.md)

