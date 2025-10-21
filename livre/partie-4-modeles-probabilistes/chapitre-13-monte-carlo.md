# Chapitre 13 : Échantillonnage Monte-Carlo

## 📚 Introduction

Les méthodes Monte-Carlo permettent d'échantillonner des distributions complexes et d'approximer des intégrales.

## 🗺️ Carte Mentale : MCMC

```
                    MONTE-CARLO
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
    DIRECT           CHAÎNES DE          AVANCÉ
  (Indépendant)      MARKOV (MCMC)          │
        │                 │                 │
    ┌───┴───┐         ┌───┴───┐         ┌───┴───┐
    │       │         │       │         │       │
 Inverse  Rejet   Metropolis Gibbs     HMC    NUTS
Transform  │      Hastings  Sampling    │       │
    │    Importance    │       │     Hamiltonian
  Box-    Sampling  Accept  Conditionals  Gradient
 Muller              Rate
```

## 📊 Tableau Comparatif : Méthodes d'Échantillonnage

| **Méthode** | **Type** | **Dépendance** | **Convergence** | **Efficacité** | **Usage** |
|------------|---------|---------------|----------------|--------------|-----------|
| **Rejet** | Direct | Indépendant | Immédiate | ⚠️ Faible haute-dim | Simple, 1D-2D |
| **Metropolis-Hastings** | MCMC | Chaîne | Asymptotique | ✓ Moyenne | Standard |
| **Gibbs** | MCMC | Chaîne | Asymptotique | ✓✓ Bonne | Conditionnelles simples |
| **HMC** | MCMC | Chaîne | Rapide | ✓✓✓ Excellente | Gradients disponibles |
| **NUTS** | MCMC | Chaîne | Très rapide | ✓✓✓ Excellente | Stan, PyMC |

## 📐 Visualisation : MCMC en Action

```
Distribution Cible p(x) :         Échantillonnage MCMC :

     p(x)                          Trajectoire MCMC
      │                               
  1.0 │   ╱─╲                     ●──→●──→●
      │  ╱   ╲                   ╱      ↓
  0.5 │ ╱     ╲                 ●       ●
      │╱       ╲___             ↓       ↓
  0   └──────────────→ x        ●←──●←──●
     -3  -1  1  3             
                              Burn-in Phase (1000 iter)
                              Sampling Phase (5000 iter)

Histogramme des échantillons :   Autocorrélation :

     ╱╲                           ACF
    ╱  ╲                           1│●
   ╱    ╲                           │  ●
  ╱      ╲___                       │    ●  ●
──────────────→                     │      ●   ●
Converge vers p(x) !                0└────────────→ Lag
```

---

## 13.1 Principes Généraux

### Approximation Monte-Carlo

Pour estimer 𝔼_P[f(X)] :
```
𝔼[f] ≈ (1/n) Σᵢ f(xᵢ)  où xᵢ ~ P
```

**Loi des grands nombres** : Convergence vers 𝔼[f]

---

## 13.2 Échantillonnage par Rejet

**Algorithme** :
1. Échantillonner x ~ q
2. Accepter avec probabilité p(x)/(M·q(x))

```python
def rejection_sampling(target, proposal, M, n_samples):
    samples = []
    while len(samples) < n_samples:
        x = proposal.rvs()
        u = np.random.uniform()
        if u < target(x) / (M * proposal.pdf(x)):
            samples.append(x)
    return np.array(samples)
```

---

## 13.3 Chaînes de Markov

### 13.3.1 Définitions

Une **chaîne de Markov** {X_t} satisfait :
```
P(X_{t+1}|X_t, X_{t-1}, ..., X_0) = P(X_{t+1}|X_t)
```

### 13.3.2 Distribution Stationnaire

π est **stationnaire** si :
```
π(x') = Σ_x π(x) P(x → x')
```

### 13.3.3 Ergodicité

**Théorème** : Si la chaîne est irréductible et apériodique :
```
lim_{t→∞} P^t(x, ·) = π
```

---

## 13.4 Gibbs Sampling

**Principe** : Échantillonner chaque variable conditionnellement aux autres.

```python
def gibbs_sampling(n_iter, x_init):
    x = x_init.copy()
    samples = [x.copy()]
    
    for _ in range(n_iter):
        # Échantillonner X1 | X2, X3, ...
        x[0] = sample_x1_given_others(x)
        
        # Échantillonner X2 | X1, X3, ...
        x[1] = sample_x2_given_others(x)
        
        # ...
        
        samples.append(x.copy())
    
    return np.array(samples)
```

---

## 13.5 Metropolis-Hastings

**Algorithme** :
1. Proposer x' ~ q(·|x)
2. Accepter avec probabilité :
```
α = min(1, (p(x')q(x|x')) / (p(x)q(x'|x)))
```

```python
def metropolis_hastings(target, proposal, x0, n_iter):
    x = x0
    samples = [x]
    
    for _ in range(n_iter):
        # Proposition
        x_prop = proposal(x)
        
        # Taux d'acceptation
        alpha = min(1, target(x_prop) / target(x))
        
        # Acceptation/rejet
        if np.random.uniform() < alpha:
            x = x_prop
        
        samples.append(x)
    
    return np.array(samples)
```

---

[⬅️ Chapitre précédent](./chapitre-12-comparaison-distributions.md) | [Retour](../README.md) | [Suite ➡️](./chapitre-14-champs-markov.md)

