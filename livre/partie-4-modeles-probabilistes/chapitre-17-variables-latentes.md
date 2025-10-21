# Chapitre 17 : Variables Latentes et Méthodes Variationnelles

## 📚 Introduction

Ce chapitre traite des modèles avec variables non observées et des méthodes d'approximation variationnelle.

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

