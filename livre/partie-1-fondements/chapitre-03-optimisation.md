# Chapitre 3 : Introduction à l'Optimisation

## 📚 Introduction

L'optimisation est au cœur du machine learning. Entraîner un modèle revient à résoudre un problème d'optimisation : trouver les paramètres qui minimisent une fonction de coût. Ce chapitre couvre les algorithmes d'optimisation essentiels utilisés en ML.

## 🗺️ Carte Mentale : Méthodes d'Optimisation

```
                        OPTIMISATION EN ML
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
      SANS CONTRAINTE    AVEC CONTRAINTES    STOCHASTIQUE
            │                  │                  │
    ┌───────┴───────┐      ┌───┴───┐          ┌───┴───┐
    │               │      │       │          │       │
 Gradient    Newton &  Lagrange KKT         SGD    ADAM
  Descent    Quasi-N.    │                   │
    │           │     Pénalité           Mini-batch
 Line Search  BFGS   Augmentée
```

## 📊 Tableau Comparatif des Algorithmes

| **Algorithme** | **Complexité/iter** | **Convergence** | **Usage ML** | **Avantages** | **Inconvénients** |
|----------------|-------------------|----------------|-------------|--------------|------------------|
| **Gradient Descent** | O(n) | Linéaire | Universel | Simple | Lent |
| **Newton** | O(n³) | Quadratique | Petit n | Rapide | Coût élevé |
| **SGD** | O(1) | Sous-linéaire | Big Data | Scalable | Bruyant |
| **ADAM** | O(1) | Adaptatif | Deep Learning | Robuste | Hyperparamètres |
| **L-BFGS** | O(n×m) | Superlinéaire | ML classique | Efficace | Mémoire |

## 🎯 Diagramme de Flux : Choix d'Algorithme

```
         Début : Problème d'optimisation
                      │
                      ▼
         ┌─────────────────────────┐
         │ Dataset de taille N ?   │
         └─────────────────────────┘
              │              │
           N < 10⁴        N > 10⁶
              │              │
              ▼              ▼
         ┌─────────┐    ┌─────────┐
         │ Newton  │    │   SGD   │
         │ L-BFGS  │    │  ADAM   │
         └─────────┘    └─────────┘
              │              │
              ▼              ▼
       Convergence       Mini-batch
         rapide           adaptatif
```

---

## 3.1 Terminologie de Base

### Problème d'Optimisation

Un problème d'optimisation général s'écrit :

```
minimize    f(x)
subject to  x ∈ C
```

où :
- **f : ℝⁿ → ℝ** est la fonction objectif (fonction de coût)
- **C ⊂ ℝⁿ** est l'ensemble des contraintes
- **x ∈ ℝⁿ** est la variable d'optimisation

### Définitions

**Minimum global** : x* est un minimum global si f(x*) ≤ f(x) pour tout x ∈ C

**Minimum local** : x* est un minimum local s'il existe ε > 0 tel que f(x*) ≤ f(x) pour tout x ∈ C ∩ B(x*, ε)

**Point stationnaire** : x tel que ∇f(x) = 0

---

## 3.2 Optimisation Sans Contrainte

### 3.2.1 Conditions d'Optimalité

#### Condition Nécessaire du Premier Ordre

Si x* est un minimum local de f différentiable, alors :
```
∇f(x*) = 0
```

#### Condition Nécessaire du Second Ordre

Si x* est un minimum local de f deux fois différentiable, alors :
```
∇f(x*) = 0  et  Hf(x*) ⪰ 0
```

#### Condition Suffisante du Second Ordre

Si ∇f(x*) = 0 et Hf(x*) ≻ 0, alors x* est un minimum local strict.

**Exemple** : Pour f(x) = x² - 2x + 1
```
∇f(x) = 2x - 2 = 0  ⟹  x* = 1
H f(x) = 2 > 0  ⟹  x* est un minimum global
```

### 3.2.2 Ensembles et Fonctions Convexes

#### Ensemble Convexe

Un ensemble C est **convexe** si pour tous x, y ∈ C et θ ∈ [0,1] :
```
θx + (1-θ)y ∈ C
```

**Exemples** :
- ℝⁿ est convexe
- {x : Ax ≤ b} est convexe (demi-espaces)
- Boules : {x : ‖x - x₀‖ ≤ r} est convexe

#### Fonction Convexe

Une fonction f est **convexe** si pour tous x, y et θ ∈ [0,1] :
```
f(θx + (1-θ)y) ≤ θf(x) + (1-θ)f(y)
```

**Propriété fondamentale** : Pour f convexe différentiable :
```
f(y) ≥ f(x) + ∇f(x)ᵀ(y - x)  pour tous x, y
```

**Test de convexité** : f est convexe ⟺ Hf(x) ⪰ 0 pour tout x

**Exemples de fonctions convexes** :
- f(x) = xᵀAx avec A ⪰ 0
- f(x) = ‖x‖
- f(x) = exp(x)
- f(x) = -log(x) pour x > 0

### 3.2.3 Intérieur Relatif

L'**intérieur relatif** de C, noté ri(C), est l'intérieur de C dans son affine hull.

**Importance** : Les conditions d'optimalité pour les fonctions convexes s'appliquent sur ri(dom f).

### 3.2.4 Dérivées de Fonctions Convexes et Conditions d'Optimalité

#### Théorème Fondamental

Pour f convexe différentiable, x* est un minimum global si et seulement si :
```
∇f(x*) = 0
```

**Conséquence** : Pour les fonctions convexes, tout minimum local est global !

### 3.2.5 Direction de Descente et Descente la Plus Rapide

#### Direction de Descente

Un vecteur d est une **direction de descente** en x si :
```
∇f(x)ᵀd < 0
```

La **direction de descente la plus rapide** est :
```
d = -∇f(x) / ‖∇f(x)‖
```

### 3.2.6 Descente de Gradient

**Algorithme de descente de gradient** :
```
x_{k+1} = x_k - α_k ∇f(x_k)
```

où α_k > 0 est le pas d'apprentissage (learning rate).

#### 📐 Intuition Géométrique

```
Paysage de la fonction f(x) :

    f(x)
     ↑
     │     ╱╲              Descente de gradient :
     │    ╱  ╲             On suit la pente négative
     │   ╱    ╲    
     │  ╱  x₀  ╲          x₀ ──────→ x₁ ──────→ x₂ ──→ x*
     │ ╱   ↓    ╲         (gradient négatif à chaque étape)
     │╱    x₁    ╲        
     └─────────────→ x
          ↓
         x₂→x*
```

#### 📝 Exemple Complet Pas à Pas

**Problème** : Minimiser f(x, y) = x² + 4y²

```
DONNÉES INITIALES :
    f(x, y) = x² + 4y²
    Point initial : (x₀, y₀) = (4, 2)
    Pas d'apprentissage : α = 0.1

ÉTAPE 0 : Calcul du gradient
    ∇f(x, y) = [∂f/∂x, ∂f/∂y]ᵀ = [2x, 8y]ᵀ

    Au point (4, 2) :
        ∇f(4, 2) = [2×4, 8×2]ᵀ = [8, 16]ᵀ
        f(4, 2) = 16 + 16 = 32

ITÉRATION 1 :
    x₁ = x₀ - α·∇f_x(x₀)
       = 4 - 0.1 × 8
       = 4 - 0.8 = 3.2
    
    y₁ = y₀ - α·∇f_y(y₀)
       = 2 - 0.1 × 16
       = 2 - 1.6 = 0.4
    
    Point : (x₁, y₁) = (3.2, 0.4)
    f(3.2, 0.4) = 10.24 + 0.64 = 10.88
    Réduction : 32 → 10.88 ✓

ITÉRATION 2 :
    ∇f(3.2, 0.4) = [6.4, 3.2]ᵀ
    
    x₂ = 3.2 - 0.1 × 6.4 = 2.56
    y₂ = 0.4 - 0.1 × 3.2 = 0.08
    
    f(2.56, 0.08) = 6.554 + 0.026 = 6.58
    Réduction : 10.88 → 6.58 ✓

ITÉRATION 3 :
    ∇f(2.56, 0.08) = [5.12, 0.64]ᵀ
    
    x₃ = 2.56 - 0.1 × 5.12 = 2.048
    y₃ = 0.08 - 0.1 × 0.64 = 0.016
    
    f(2.048, 0.016) ≈ 4.19
    Réduction : 6.58 → 4.19 ✓

CONVERGENCE :
    Après plusieurs itérations → (x*, y*) = (0, 0)
    Minimum : f(0, 0) = 0

Schéma de convergence :
    
    y
    │  ●(4,2)          Courbes de niveau de f
    │   ╲              (ellipses)
    │    ●(3.2,0.4)
  1 │      ╲           Trajectoire de descente
    │       ●(2.56,0.08)
    │         ╲
  0 │          ●─────●(0,0) ★
    └───────────────────────→ x
    0         2          4
```

#### ⚙️ Choix du Pas d'Apprentissage α

| **α** | **Effet** | **Convergence** | **Recommandation** |
|-------|----------|----------------|-------------------|
| Trop grand (α > 2/L) | Divergence | ✗ Aucune | Éviter |
| Grand (α ≈ 1/L) | Oscillations | ⚠️ Lente | Attention |
| Optimal (α = μ/L²) | Stable | ✓ Rapide | Idéal |
| Petit (α << 1/L) | Très lent | ✓ Garantie | Sûr mais lent |

**Illustration** :

```
α trop grand :                α optimal :              α trop petit :
    ╱╲                           ╱╲                        ╱╲
   ╱  ╲                         ╱  ╲                      ╱  ╲
  ●────●                       ●─→●─→●                   ●─→─→─→─→●
 ●      ●                        ↓  ↓                     (très lent)
●        ● Diverge !             ★ Converge
```

**Implémentation Python Complète** :
```python
def gradient_descent(f, grad_f, x0, alpha=0.01, max_iter=1000, tol=1e-6):
    """
    Descente de gradient
    
    Args:
        f: fonction objectif
        grad_f: gradient de f
        x0: point initial
        alpha: pas d'apprentissage
        max_iter: nombre maximal d'itérations
        tol: tolérance pour la convergence
    """
    x = x0.copy()
    history = [x.copy()]
    
    for k in range(max_iter):
        grad = grad_f(x)
        
        # Test de convergence
        if np.linalg.norm(grad) < tol:
            print(f"Convergence à l'itération {k}")
            break
        
        # Mise à jour
        x = x - alpha * grad
        history.append(x.copy())
    
    return x, np.array(history)

# Exemple : minimiser f(x) = x^T A x - b^T x
A = np.array([[4, 1], [1, 3]])
b = np.array([1, 2])

f = lambda x: 0.5 * x @ A @ x - b @ x
grad_f = lambda x: A @ x - b

x0 = np.array([0.0, 0.0])
x_opt, history = gradient_descent(f, grad_f, x0, alpha=0.1)
```

#### Théorème de Convergence

Pour f fortement convexe avec L-Lipschitz gradient :
```
f(x_k) - f(x*) ≤ (1 - μ/L)^k (f(x_0) - f(x*))
```

où μ est le paramètre de convexité forte.

**Taux de convergence** : Linéaire (exponentiel en nombre d'itérations)

### 3.2.7 Recherche Linéaire (Line Search)

Au lieu de fixer α, on peut le choisir à chaque itération :

#### Règle d'Armijo (Backtracking)

```python
def backtracking_line_search(f, x, d, grad, alpha=1.0, beta=0.5, sigma=0.1):
    """
    Recherche linéaire avec backtracking
    """
    while f(x + alpha * d) > f(x) + sigma * alpha * grad @ d:
        alpha *= beta
    return alpha
```

#### Recherche Exacte

Minimiser f(x + αd) par rapport à α :
```
α* = argmin_α f(x + αd)
```

---

## 3.3 Descente de Gradient Stochastique

### 3.3.1 Méthodes d'Approximation Stochastique

En ML, on minimise souvent :
```
f(θ) = 𝔼[ℓ(θ; Z)] = (1/n) Σᵢ ℓ(θ; zᵢ)
```

**Problème** : Calculer ∇f(θ) nécessite n évaluations.

**Solution** : Approximer le gradient par un échantillon :
```
∇f(θ) ≈ ∇ℓ(θ; z_i)  où i est choisi aléatoirement
```

## 📊 Comparaison Visuelle : GD vs SGD vs ADAM

### Trajectoires de Convergence

```
Paysage d'optimisation (vue de dessus) :

         Gradient Descent (GD)          
              ╭───────╮                 
             ╱         ╲                
        x₀ ●────→●────→● x*            Lisse, déterministe
           ╲         ╱                 Convergence monotone
            ╰───────╯                  

         Stochastic GD (SGD)           
              ╭───────╮                 
             ╱    ●↗   ╲               Bruyant, stochastique
        x₀ ●→●↘→●↗→●→● x*             Oscille autour de x*
           ╲    ●↘   ╱                Convergence en moyenne
            ╰───────╯                  

              ADAM                     
              ╭───────╮                 
             ╱         ╲               Adaptatif
        x₀ ●──→●──→●──→● x*           Convergence rapide
           ╲         ╱                 Peu d'oscillations
            ╰───────╯                  
```

### 📈 Courbes de Convergence

```
    Loss
     │
 10⁴ │●                        
     │ ╲             ──── GD
     │  ╲───────     ──── SGD (bruyant)
 10² │   ╲  ╲ /╲    ──── ADAM
     │    ╲  ╲/  ╲  
 10⁰ │     ────────●
     │          SGD
 10⁻² │           ─────ADAM
     │               ────GD
     └─────────────────────→ Iterations
       0   50  100  150  200
```

### 📋 Tableau Comparatif Détaillé

| **Critère** | **GD** | **SGD** | **Mini-Batch SGD** | **Momentum** | **ADAM** |
|-------------|--------|---------|-------------------|-------------|----------|
| **Gradient par iter** | Full (N samples) | 1 sample | B samples | B samples | B samples |
| **Complexité/iter** | O(N) | O(1) | O(B) | O(B) | O(B) |
| **Convergence** | Monotone | Bruitée | Stable | Rapide | Très rapide |
| **Mémoire** | O(d) | O(d) | O(d) | O(2d) | O(3d) |
| **Hyperparamètres** | α | α, schedule | α, B | α, β | α, β₁, β₂ |
| **Scalabilité** | ✗ Mauvaise | ✓✓ Excellente | ✓✓ Excellente | ✓✓ Excellente | ✓✓ Excellente |
| **Robustesse** | ⚠️ Moyenne | ⚠️ Sensible | ✓ Bonne | ✓ Bonne | ✓✓ Très bonne |
| **Usage ML** | Petit N | Big Data | Universel | Computer Vision | Deep Learning |

### ⚙️ Formules Côte à Côte

```
┌─────────────────────────────────────────────────────────────────┐
│  GRADIENT DESCENT (GD)                                           │
│  θₖ₊₁ = θₖ - α ∇f(θₖ)                                           │
│  ↪ Utilise toutes les données                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  STOCHASTIC GD (SGD)                                             │
│  θₖ₊₁ = θₖ - α ∇ℓ(θₖ; zᵢ)                                       │
│  ↪ Utilise 1 échantillon aléatoire                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  MINI-BATCH SGD                                                  │
│  θₖ₊₁ = θₖ - α (1/B) Σᵢ∈Batch ∇ℓ(θₖ; zᵢ)                       │
│  ↪ Compromis entre GD et SGD                                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  MOMENTUM                                                        │
│  vₖ = β vₖ₋₁ + gₖ                                               │
│  θₖ₊₁ = θₖ - α vₖ                                               │
│  ↪ Accumule la vitesse (inertie)                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  ADAM (Adaptive Moment Estimation)                               │
│  mₖ = β₁ mₖ₋₁ + (1-β₁) gₖ        (1er moment : moyenne)         │
│  vₖ = β₂ vₖ₋₁ + (1-β₂) gₖ²       (2ème moment : variance)       │
│  θₖ₊₁ = θₖ - α m̂ₖ / (√v̂ₖ + ε)                                  │
│  ↪ Adapte le pas pour chaque paramètre                          │
└─────────────────────────────────────────────────────────────────┘
```

### 🎯 Guide de Choix

```
                Quel algorithme choisir ?
                         │
        ┌────────────────┼────────────────┐
        │                                  │
    N < 10,000                        N > 1,000,000
    Dataset petit                     Big Data
        │                                  │
    ┌───┴───┐                          ┌───┴───┐
    │       │                          │       │
  Convexe Non-convexe              Batch Size ?
    │       │                          │
   GD    L-BFGS                  ┌─────┴─────┐
         Newton                  │           │
                              B=1          B>1
                               │            │
                             SGD      Mini-batch SGD
                                           │
                                      Deep Learning ?
                                           │
                                      ┌────┴────┐
                                      │         │
                                    Oui       Non
                                      │         │
                                   ADAM    Momentum
                                            SGD
```

### 💡 Conseils Pratiques

**Taille de Batch Recommandée** :

| **Cas** | **Batch Size** | **Raison** |
|---------|---------------|-----------|
| Petits modèles | 32-64 | Équilibre vitesse/stabilité |
| CNN (images) | 64-256 | Parallélisation GPU |
| Transformers (NLP) | 16-32 | Contrainte mémoire |
| Très grands modèles | 8-16 | Limite GPU |

**Learning Rate** :

| **Algorithme** | **α initial** | **Schedule** |
|---------------|--------------|-------------|
| GD | 0.01-0.1 | Constant ou decay |
| SGD | 0.01-0.1 | 1/√t ou cosine |
| Momentum | 0.01 | Step decay |
| ADAM | 0.001 | Constant ou warmup |

### 3.3.2 Algorithme SGD

```
θ_{k+1} = θ_k - α_k ∇ℓ(θ_k; z_{i_k})
```

**Implémentation** :
```python
def sgd(data, loss_grad, theta0, epochs=10, batch_size=32, alpha=0.01):
    """
    Descente de gradient stochastique
    """
    theta = theta0.copy()
    n = len(data)
    
    for epoch in range(epochs):
        # Mélanger les données
        np.random.shuffle(data)
        
        for i in range(0, n, batch_size):
            batch = data[i:i+batch_size]
            
            # Gradient sur le mini-batch
            grad = sum(loss_grad(theta, x) for x in batch) / len(batch)
            
            # Mise à jour
            theta -= alpha * grad
        
        print(f"Epoch {epoch+1}/{epochs}")
    
    return theta
```

#### Choix du Pas d'Apprentissage

Conditions de Robbins-Monro pour la convergence :
```
Σ α_k = ∞  et  Σ α_k² < ∞
```

**Exemples** :
- α_k = α₀ / (1 + k)
- α_k = α₀ / √k
- Learning rate schedule : réduction par paliers

### 3.3.3 L'Algorithme ADAM

**ADAM** (Adaptive Moment Estimation) combine :
- Momentum : moyenne mobile des gradients
- RMSprop : moyenne mobile des gradients au carré

**Algorithme** :
```
m_k = β₁ m_{k-1} + (1-β₁) g_k
v_k = β₂ v_{k-1} + (1-β₂) g_k²

m̂_k = m_k / (1 - β₁^k)
v̂_k = v_k / (1 - β₂^k)

θ_{k+1} = θ_k - α m̂_k / (√v̂_k + ε)
```

**Paramètres standards** :
- β₁ = 0.9
- β₂ = 0.999
- ε = 10⁻⁸

**Implémentation** :
```python
def adam(data, loss_grad, theta0, epochs=10, alpha=0.001, 
         beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Optimiseur ADAM
    """
    theta = theta0.copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    t = 0
    
    for epoch in range(epochs):
        for x in data:
            t += 1
            grad = loss_grad(theta, x)
            
            # Mise à jour des moments
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            
            # Correction du biais
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            # Mise à jour des paramètres
            theta -= alpha * m_hat / (np.sqrt(v_hat) + eps)
    
    return theta
```

---

## 3.4 Optimisation Contrainte

### 3.4.1 Multiplicateurs de Lagrange

Pour minimiser f(x) sous contrainte h(x) = 0 :

**Lagrangien** :
```
ℒ(x, λ) = f(x) + λᵀh(x)
```

**Conditions KKT** (Karush-Kuhn-Tucker) :
```
∇_x ℒ(x*, λ*) = 0
h(x*) = 0
```

**Exemple** : Minimiser f(x, y) = x² + y² sous contrainte x + y = 1

```python
# Lagrangien: L = x² + y² + λ(x + y - 1)
# ∇L = 0  ⟹  2x + λ = 0, 2y + λ = 0, x + y = 1
# Solution: x = y = 1/2, λ = -1
```

### 3.4.2 Contraintes Convexes

Pour f convexe et contraintes convexes g(x) ≤ 0, les conditions KKT sont :

```
∇f(x*) + Σᵢ λᵢ* ∇gᵢ(x*) = 0
gᵢ(x*) ≤ 0
λᵢ* ≥ 0
λᵢ* gᵢ(x*) = 0  (complémentarité)
```

**Propriété** : Si (x*, λ*) satisfait KKT, alors x* est optimal.

### 3.4.3 Applications

#### SVM (Support Vector Machine)

Problème primal :
```
minimize    (1/2)‖w‖²
subject to  yᵢ(wᵀxᵢ + b) ≥ 1  pour i = 1, ..., n
```

Problème dual :
```
maximize    Σᵢ αᵢ - (1/2)Σᵢⱼ αᵢαⱼyᵢyⱼ(xᵢᵀxⱼ)
subject to  0 ≤ αᵢ  et  Σᵢ αᵢyᵢ = 0
```

### 3.4.4 Descente de Gradient Projetée

Pour minimiser f(x) avec x ∈ C :

```
x_{k+1} = P_C(x_k - α_k ∇f(x_k))
```

où P_C est la projection sur C.

**Projection sur simplexe** (Σᵢ xᵢ = 1, xᵢ ≥ 0) :
```python
def project_simplex(v):
    """
    Projette v sur le simplexe
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)
```

---

## 3.5 Problèmes Convexes Généraux

### 3.5.1 Épigraphes

L'**épigraphe** de f est :
```
epi(f) = {(x, t) : f(x) ≤ t}
```

**Propriété** : f est convexe ⟺ epi(f) est convexe.

### 3.5.2 Sous-Gradients

Pour f convexe (pas nécessairement différentiable), g est un **sous-gradient** de f en x si :
```
f(y) ≥ f(x) + gᵀ(y - x)  pour tout y
```

**Sous-différentiel** :
```
∂f(x) = {g : g est un sous-gradient de f en x}
```

**Exemples** :
- f(x) = |x| : ∂f(0) = [-1, 1]
- f(x) = max(x, 0) : ∂f(0) = [0, 1]

### 3.5.3 Dérivées Directionnelles

La **dérivée directionnelle** de f en x dans la direction v est :
```
f'(x; v) = lim_{t→0⁺} [f(x + tv) - f(x)] / t
```

Pour f convexe :
```
f'(x; v) = max{gᵀv : g ∈ ∂f(x)}
```

### 3.5.4 Descente de Sous-Gradient

Pour f convexe non différentiable :

```
x_{k+1} = x_k - α_k g_k  où g_k ∈ ∂f(x_k)
```

**Note** : La convergence est plus lente que la descente de gradient (taux en O(1/√k)).

### 3.5.5 Méthodes Proximales

L'**opérateur proximal** de f est :
```
prox_f(v) = argmin_x {f(x) + (1/2)‖x - v‖²}
```

**Algorithme du Gradient Proximal** :
Pour minimiser f(x) + g(x) où f est lisse et g convexe :

```
x_{k+1} = prox_{α_k g}(x_k - α_k ∇f(x_k))
```

**Exemple : Lasso** (f(x) = ‖Ax - b‖², g(x) = λ‖x‖₁)

```python
def soft_threshold(x, threshold):
    """
    Opérateur proximal de ‖·‖₁
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def lasso_proximal_gradient(A, b, lambda_param, alpha=0.01, max_iter=1000):
    """
    ISTA pour Lasso
    """
    n = A.shape[1]
    x = np.zeros(n)
    
    for k in range(max_iter):
        # Gradient de f
        grad = A.T @ (A @ x - b)
        
        # Étape de gradient
        z = x - alpha * grad
        
        # Étape proximale
        x = soft_threshold(z, alpha * lambda_param)
    
    return x
```

---

## 3.6 Dualité

### 3.6.1 Conditions KKT Généralisées

Pour le problème :
```
minimize    f(x)
subject to  g(x) ≤ 0, h(x) = 0
```

**Lagrangien** :
```
ℒ(x, λ, ν) = f(x) + λᵀg(x) + νᵀh(x)
```

### 3.6.2 Problème Dual

**Fonction duale** :
```
q(λ, ν) = inf_x ℒ(x, λ, ν)
```

**Problème dual** :
```
maximize    q(λ, ν)
subject to  λ ≥ 0
```

**Dualité faible** : q(λ, ν) ≤ f(x*) pour tout λ ≥ 0

**Dualité forte** : Sous certaines conditions (Slater), q(λ*, ν*) = f(x*)

---

## 💡 Points Clés

1. **Convexité** : Simplifie grandement l'optimisation
2. **Gradient** : Direction de descente la plus rapide
3. **SGD** : Essentiel pour les grands datasets
4. **ADAM** : Optimiseur adaptatif très efficace
5. **Proximal** : Pour les fonctions non différentiables
6. **Dualité** : Fournit des bornes et des algorithmes alternatifs

---

[⬅️ Chapitre précédent](./chapitre-02-analyse-matricielle.md) | [Retour](../README.md) | [Suite ➡️](../partie-2-concepts/chapitre-04-biais-variance.md)

