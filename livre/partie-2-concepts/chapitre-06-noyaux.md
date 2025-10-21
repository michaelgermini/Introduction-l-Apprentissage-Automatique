# Chapitre 6 : Produits Internes et Noyaux Reproduisants

## 📚 Introduction

Les noyaux sont un outil puissant qui permet d'étendre les méthodes linéaires à des espaces de dimension infinie. Ce chapitre introduit la théorie des noyaux reproduisants.

## 🗺️ Carte Mentale : Le Kernel Trick

```
                    NOYAUX (KERNELS)
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
    THÉORIE           NOYAUX             APPLICATIONS
        │              POPULAIRES             │
    ┌───┴───┐            │              ┌─────┴─────┐
    │       │            │              │           │
 Espace  Produit     ┌───┴───┐       SVM      Kernel
 Hilbert Scalaire    │       │        │       Ridge/PCA
    │       │     Linéaire  RBF    Polynomial  │
  φ(x)   K(x,x')      │       │        │    K-Means
         =⟨φ(x),φ(x')⟩ xᵀx'  Gaussien Degré d  Spectral
```

## 🎯 Le Kernel Trick : Concept Clé

```
┌──────────────────────────────────────────────────────────┐
│              LE KERNEL TRICK EN ACTION                    │
└──────────────────────────────────────────────────────────┘

APPROCHE NAÏVE (Coûteuse) :
  1. Transformer : x → φ(x)     [Dimension peut être ∞!]
  2. Calculer : ⟨φ(x), φ(x')⟩   [Très coûteux]

KERNEL TRICK (Efficace) :
  Directement : K(x, x') = ⟨φ(x), φ(x')⟩
  
Schéma :

    Espace Original            Espace Transformé
         (ℝᵈ)                      (ℋ, dim >> d)
    
    x₁ ●  ● x₂                    φ(x₁) ●
      ●  ●        ─────φ──→              ╲
    x₃●  ●x₄                             ●─● Séparable
         ●                              ╱     linéairement!
    Non-linéairement              φ(x₄) ●
    séparable                     
                                  φ(x₃) ●

Exemple : XOR problem
  Original : (x₁,x₂) non-séparable linéairement
  Après φ  : Séparable avec un hyperplan !
```

## 📊 Tableau Comparatif : Noyaux Populaires

| **Noyau** | **Formule** | **Dimension ℋ** | **Paramètres** | **Avantages** | **Inconvénients** |
|-----------|-----------|----------------|---------------|--------------|------------------|
| **Linéaire** | xᵀx' | d | Aucun | Rapide, simple | Linéaire seulement |
| **Polynomial** | (xᵀx' + c)^p | C(d+p,p) | p (degré), c | Flexible | Numérique instable |
| **RBF (Gaussien)** | exp(-γ‖x-x'‖²) | ∞ | γ (ou σ) | Universel, smooth | Peut overfit |
| **Sigmoid** | tanh(α xᵀx' + c) | ∞ | α, c | Comme réseau | Pas toujours PD |
| **Laplacien** | exp(-γ‖x-x'‖₁) | ∞ | γ | Moins smooth | Rare |

## 📐 Visualisation : Effet des Noyaux

### Noyau RBF avec différents γ

```
γ = 0.1 (large σ)         γ = 1.0 (moyen)         γ = 10 (petit σ)

    ╭───────╮                  ╭──╮                     ╭╮
   ╱         ╲                ╱    ╲                   ╱╲
  ╱           ╲              ╱      ╲                 ╱  ╲
 ╱             ╲            ╱        ╲               ╱    ╲
───────●───────────        ────●────────            ──●──────
  Influence large       Influence moyenne       Influence locale
  Smooth, underfit      Équilibré               Peut overfit
```

### Transformation par Noyau Polynomial

```
Espace Original (2D)          Noyau Poly(deg=2)

    x₂                         z₁ = x₁²
     │                              ╱
   1 │ ○ ○ ●                      ╱  ● ●
     │ ○ ● ● ●                   ╱   ●
   0 │ ● ● ●         ───φ──→   z₂= √2x₁x₂
     │   ●                      │    ●
  -1 │                          │   ● ○ ○
     └────────→ x₁               └────────→ z₃ = x₂²
    -1  0  1                         Séparable !

Non-linéairement              Linéairement séparable
séparable                     dans l'espace transformé
```

## 🔢 Calcul Explicite : Noyau Polynomial

### Exemple Détaillé

```
Soit x = [x₁, x₂]ᵀ ∈ ℝ²
Noyau polynomial : K(x, x') = (xᵀx')²

ÉTAPE 1 : Calcul direct du noyau
  K(x, x') = (x₁x₁' + x₂x₂')²

ÉTAPE 2 : Expansion
  K(x, x') = x₁²x₁'² + 2x₁x₂x₁'x₂' + x₂²x₂'²

ÉTAPE 3 : Identification de φ
  φ(x) = [x₁², √2x₁x₂, x₂²]ᵀ ∈ ℝ³
  
  Vérification :
  φ(x)ᵀφ(x') = x₁²x₁'² + √2x₁x₂·√2x₁'x₂' + x₂²x₂'²
              = x₁²x₁'² + 2x₁x₂x₁'x₂' + x₂²x₂'²
              = K(x, x') ✓

GAIN : Au lieu de calculer φ(x) puis ⟨φ(x),φ(x')⟩
       On calcule directement (xᵀx')² !
```

## 🎨 Matrice de Gram

```
Pour n points : X = [x₁, x₂, ..., xₙ]

Matrice de Gram K :
    
    K = ┌                                      ┐
        │ K(x₁,x₁)  K(x₁,x₂)  ...  K(x₁,xₙ)  │
        │ K(x₂,x₁)  K(x₂,x₂)  ...  K(x₂,xₙ)  │
        │    ⋮         ⋮       ⋱       ⋮      │
        │ K(xₙ,x₁)  K(xₙ,x₂)  ...  K(xₙ,xₙ)  │
        └                                      ┘

Propriétés :
  • Symétrique : K = Kᵀ
  • Semi-définie positive : K ⪰ 0
  • Kᵢⱼ = ⟨φ(xᵢ), φ(xⱼ)⟩

Python :
  from sklearn.metrics.pairwise import rbf_kernel
  K = rbf_kernel(X, gamma=0.5)
```

---

## 6.1 Introduction

Les **noyaux** permettent de calculer des produits scalaires dans des espaces de haute dimension sans calculer explicitement la transformation.

---

## 6.2 Définitions de Base

### 6.2.1 Espaces à Produit Interne

Un **espace à produit interne** (H, ⟨·,·⟩) est un espace vectoriel avec un produit scalaire.

**Propriétés** :
- Linéarité : ⟨αu + βv, w⟩ = α⟨u,w⟩ + β⟨v,w⟩
- Symétrie : ⟨u,v⟩ = ⟨v,u⟩
- Positivité : ⟨u,u⟩ ≥ 0

### 6.2.2 Espaces de Caractéristiques et Noyaux

**Fonction noyau** K : 𝒳 × 𝒳 → ℝ est **définie positive** si :
```
Σᵢⱼ αᵢαⱼ K(xᵢ, xⱼ) ≥ 0
```
pour tous x₁, ..., xₙ et α₁, ..., αₙ.

**Théorème de Moore-Aronszajn** : Tout noyau défini positif correspond à un espace de Hilbert ℋ et une carte φ : 𝒳 → ℋ tels que :
```
K(x, x') = ⟨φ(x), φ(x')⟩_ℋ
```

---

## 6.3 Exemples

### 6.3.1 Produit Scalaire

Le noyau linéaire :
```
K(x, x') = xᵀx'
```

### 6.3.2 Noyaux Polynomiaux

```
K(x, x') = (xᵀx' + c)^d
```

**Exemple** : Pour d = 2, c = 0 en ℝ² :
```
K(x, x') = (x₁x₁' + x₂x₂')²
         = x₁²x₁'² + 2x₁x₂x₁'x₂' + x₂²x₂'²
```

Espace de caractéristiques : φ(x) = [x₁², √2x₁x₂, x₂²]

### 6.3.3 Noyau Gaussien (RBF)

```
K(x, x') = exp(-‖x - x'‖²/(2σ²))
```

**Propriété** : Correspond à un espace de dimension infinie !

```python
from sklearn.metrics.pairwise import rbf_kernel

K = rbf_kernel(X, gamma=1/(2*sigma**2))
```

### 6.3.4 Théorèmes de Construction

**Théorème** : Si K₁ et K₂ sont des noyaux, alors :
- αK₁ + βK₂ (α, β ≥ 0)
- K₁ · K₂
- f(K₁) où f est une série entière à coefficients positifs

### 6.3.5 Opérations sur les Noyaux

**Somme** :
```
K(x, x') = K₁(x, x') + K₂(x, x')
```

**Produit** :
```
K(x, x') = K₁(x, x') · K₂(x, x')
```

**Composition avec fonction** :
```
K(x, x') = f(x)ᵀf(x')
```

---

## 6.4 Projection sur Sous-Espace

Dans un espace ℋ, la projection de f sur span{φ(x₁), ..., φ(xₙ)} est :

```
f̂ = Σᵢ αᵢ φ(xᵢ)
```

**Théorème du représentant** : La solution de minimiser
```
‖f‖²_ℋ + λ Σᵢ L(yᵢ, f(xᵢ))
```
est de la forme f = Σᵢ αᵢ K(·, xᵢ).

---

## 💡 Applications

1. **SVM** : Classification dans espace de caractéristiques
2. **Kernel Ridge Regression**
3. **Kernel PCA** : PCA non linéaire
4. **Kernel K-means** : Clustering non linéaire

```python
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA

# SVM avec noyau RBF
svm = SVC(kernel='rbf', gamma=0.1)
svm.fit(X_train, y_train)

# Kernel PCA
kpca = KernelPCA(n_components=2, kernel='rbf')
X_transformed = kpca.fit_transform(X)
```

---

[⬅️ Chapitre précédent](./chapitre-05-prediction-concepts.md) | [Retour](../README.md) | [Suite ➡️](../partie-3-apprentissage-supervise/chapitre-07-regression-lineaire.md)

