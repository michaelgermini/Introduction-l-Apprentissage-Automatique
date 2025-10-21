# Chapitre 1 : Notations Générales et Prérequis Mathématiques

## 📚 Introduction

Ce chapitre établit les fondations mathématiques nécessaires pour comprendre l'apprentissage automatique. Nous couvrons les notations utilisées tout au long du livre et révisons les concepts essentiels d'algèbre linéaire, de topologie, de calcul différentiel et de théorie des probabilités.

## 🗺️ Carte Mentale du Chapitre

```
                    PRÉREQUIS MATHÉMATIQUES ML
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
    ALGÈBRE            TOPOLOGIE           CALCUL           PROBABILITÉS
    LINÉAIRE                              DIFFÉRENTIEL
        │                   │                   │                   │
    ┌───┴───┐           ┌───┴───┐           ┌───┴───┐           ┌───┴───┐
    │       │           │       │           │       │           │       │
 Vecteurs Matrices   Ouverts Compacts   Gradient Hessienne  Espérance Variance
    │       │           │       │           │       │           │       │
 Normes  Valeurs    Boules  Fermés    Jacobienne Taylor    Bayes   σ-algèbre
        Propres                                                    
```

## 📊 Vue d'Ensemble : Classification des Concepts

| **Domaine** | **Concepts Clés** | **Application ML** | **Complexité** |
|-------------|-------------------|-------------------|----------------|
| **Algèbre Linéaire** | Vecteurs, Matrices, Valeurs propres | Représentation des données, PCA | ⭐⭐ |
| **Topologie** | Ouverts, Fermés, Compacts | Convergence, Continuité | ⭐⭐⭐ |
| **Calcul Différentiel** | Gradient, Hessienne, Taylor | Optimisation, Backprop | ⭐⭐⭐⭐ |
| **Probabilités** | Espérance, Variance, Bayes | Modèles génératifs | ⭐⭐⭐⭐⭐ |

---

## 1.1 Algèbre Linéaire

### 1.1.1 Ensembles et Fonctions

#### Notations de Base

- **ℝ** : Ensemble des nombres réels
- **ℝⁿ** : Espace vectoriel euclidien de dimension n
- **ℝᵐˣⁿ** : Ensemble des matrices réelles m×n
- **ℕ** : Ensemble des entiers naturels {0, 1, 2, ...}
- **ℤ** : Ensemble des entiers relatifs

#### Fonctions

Une **fonction** f : A → B associe à chaque élément de A un unique élément de B.

- **Domaine** : L'ensemble A
- **Codomaine** : L'ensemble B
- **Image** : L'ensemble {f(x) | x ∈ A}

**Exemple** :
```
f : ℝ² → ℝ
f(x₁, x₂) = x₁² + x₂²
```

### 1.1.2 Vecteurs

Un **vecteur** dans ℝⁿ est un n-uplet de nombres réels :

```
v = [v₁, v₂, ..., vₙ]ᵀ
```

#### 📐 Représentation Géométrique

```
Vecteur en 2D :          Vecteur en 3D :
    
    │                         │ z
  v₂│   ●(v₁,v₂)              │
    │  /│                     │    ●(x,y,z)
    │ / │                     │   /│\
    │/  │                     │  / │ \
    └───┴──                   └─────────
      v₁                     x/      \y
```

#### Opérations sur les Vecteurs

##### 1. **Addition Vectorielle**

```
u + v = [u₁+v₁, u₂+v₂, ..., uₙ+vₙ]ᵀ
```

**📝 Exemple détaillé pas à pas** :

```
Soit u = [2, 3, 1]ᵀ  et  v = [1, -1, 4]ᵀ

Étape 1 : Additionner composante par composante
    u₁ + v₁ = 2 + 1 = 3
    u₂ + v₂ = 3 + (-1) = 2
    u₃ + v₃ = 1 + 4 = 5

Étape 2 : Former le vecteur résultat
    u + v = [3, 2, 5]ᵀ

Représentation géométrique (règle du parallélogramme) :
         u+v
      ●------●
     /│     /
    / │    /
   /  │   / v
  ●---│--●
     u
```

##### 2. **Multiplication Scalaire**

```
αv = [αv₁, αv₂, ..., αvₙ]ᵀ
```

**📝 Exemple détaillé** :

```
Soit α = 3  et  v = [2, -1, 4]ᵀ

Étape 1 : Multiplier chaque composante par α
    α·v₁ = 3 × 2 = 6
    α·v₂ = 3 × (-1) = -3
    α·v₃ = 3 × 4 = 12

Étape 2 : Résultat
    3v = [6, -3, 12]ᵀ

Interprétation : Le vecteur est étiré par un facteur 3
```

##### 3. **Produit Scalaire** (Dot Product)

```
⟨u, v⟩ = uᵀv = Σᵢ uᵢvᵢ = u₁v₁ + u₂v₂ + ... + uₙvₙ
```

**📝 Calcul détaillé pas à pas** :

```
Soit u = [1, 2, 3]ᵀ  et  v = [4, 5, 6]ᵀ

Étape 1 : Multiplier terme à terme
    u₁ × v₁ = 1 × 4 = 4
    u₂ × v₂ = 2 × 5 = 10
    u₃ × v₃ = 3 × 6 = 18

Étape 2 : Sommer tous les produits
    ⟨u, v⟩ = 4 + 10 + 18 = 32

Interprétation géométrique :
    ⟨u, v⟩ = ‖u‖ · ‖v‖ · cos(θ)
    où θ est l'angle entre u et v
```

**🔍 Cas particuliers** :

| **Cas** | **⟨u, v⟩** | **Interprétation** |
|---------|-----------|-------------------|
| u ⊥ v (orthogonal) | 0 | Vecteurs perpendiculaires |
| u ∥ v (colinéaire) | ±‖u‖·‖v‖ | Même direction (±) |
| θ = 90° | 0 | Orthogonalité |

##### 4. **Norme Euclidienne**

```
‖v‖ = √(vᵀv) = √(Σᵢ vᵢ²) = √(v₁² + v₂² + ... + vₙ²)
```

**📝 Calcul détaillé** :

```
Soit v = [3, 4, 12]ᵀ

Étape 1 : Calculer le carré de chaque composante
    v₁² = 3² = 9
    v₂² = 4² = 16
    v₃² = 12² = 144

Étape 2 : Sommer les carrés
    Σ vᵢ² = 9 + 16 + 144 = 169

Étape 3 : Prendre la racine carrée
    ‖v‖ = √169 = 13

✓ Le vecteur v a une longueur de 13
```

**🎯 Autres normes importantes** :

| **Norme** | **Définition** | **Exemple pour v=[3,4]** |
|-----------|---------------|-------------------------|
| L₁ (Manhattan) | ‖v‖₁ = Σ\|vᵢ\| | \|3\|+\|4\| = 7 |
| L₂ (Euclidienne) | ‖v‖₂ = √(Σvᵢ²) | √(9+16) = 5 |
| L∞ (Maximum) | ‖v‖∞ = max\|vᵢ\| | max(3,4) = 4 |

#### Propriétés Importantes

##### **Inégalité de Cauchy-Schwarz**

```
|⟨u, v⟩| ≤ ‖u‖ · ‖v‖
```

**📝 Démonstration avec exemple** :

```
Soit u = [1, 2]ᵀ  et  v = [3, 4]ᵀ

Calcul du membre de gauche :
    ⟨u, v⟩ = 1×3 + 2×4 = 11
    |⟨u, v⟩| = 11

Calcul du membre de droite :
    ‖u‖ = √(1² + 2²) = √5 ≈ 2.236
    ‖v‖ = √(3² + 4²) = √25 = 5
    ‖u‖·‖v‖ = √5 × 5 = 5√5 ≈ 11.180

Vérification : 11 ≤ 11.180 ✓
```

##### **Inégalité Triangulaire**

```
‖u + v‖ ≤ ‖u‖ + ‖v‖
```

**📝 Exemple numérique** :

```
Soit u = [3, 4]ᵀ  et  v = [1, 2]ᵀ

Étape 1 : Calculer u + v
    u + v = [3+1, 4+2]ᵀ = [4, 6]ᵀ

Étape 2 : Calculer ‖u + v‖
    ‖u + v‖ = √(4² + 6²) = √52 ≈ 7.211

Étape 3 : Calculer ‖u‖ + ‖v‖
    ‖u‖ = √(9 + 16) = 5
    ‖v‖ = √(1 + 4) = √5 ≈ 2.236
    ‖u‖ + ‖v‖ ≈ 7.236

Vérification : 7.211 ≤ 7.236 ✓

Interprétation : Le chemin direct est toujours ≤ à la somme des chemins
```

**Exemple Pratique en Python** :
```python
import numpy as np

# Vecteurs
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# 1. Produit scalaire
dot_product = np.dot(u, v)
print(f"⟨u, v⟩ = {dot_product}")  # 32

# 2. Norme euclidienne
norm_u = np.linalg.norm(u)
norm_v = np.linalg.norm(v)
print(f"‖u‖ = {norm_u:.3f}")  # 3.742
print(f"‖v‖ = {norm_v:.3f}")  # 8.775

# 3. Angle entre les vecteurs
cos_theta = dot_product / (norm_u * norm_v)
theta_rad = np.arccos(cos_theta)
theta_deg = np.degrees(theta_rad)
print(f"Angle θ = {theta_deg:.2f}°")  # 12.93°

# 4. Vérification Cauchy-Schwarz
print(f"|⟨u,v⟩| = {abs(dot_product)}")
print(f"‖u‖·‖v‖ = {norm_u * norm_v:.3f}")
print(f"Cauchy-Schwarz vérifié : {abs(dot_product) <= norm_u * norm_v}")

# 5. Projection orthogonale de u sur v
projection = (dot_product / (norm_v ** 2)) * v
print(f"proj_v(u) = {projection}")
```

### 1.1.3 Matrices

Une **matrice** A ∈ ℝᵐˣⁿ est un tableau rectangulaire de m lignes et n colonnes :

```
        n colonnes
      ←─────────→
    ┌             ┐  ↑
    │ a₁₁  a₁₂ … a₁ₙ│  │
A = │ a₂₁  a₂₂ … a₂ₙ│  │ m lignes
    │  ⋮    ⋮  ⋱  ⋮ │  │
    │ aᵐ₁  aᵐ₂ … aᵐₙ│  ↓
    └             ┘

Notation : A ∈ ℝᵐˣⁿ  (m=lignes, n=colonnes)
```

#### 🔤 Classification des Matrices

| **Type** | **Propriété** | **Exemple 2×2** | **Application ML** |
|----------|--------------|----------------|-------------------|
| **Carrée** | m = n | [1 2; 3 4] | Transformations |
| **Rectangulaire** | m ≠ n | [1 2 3; 4 5 6] | Données (n×p) |
| **Diagonale** | aᵢⱼ=0 si i≠j | [2 0; 0 3] | Covariance |
| **Identité** | I : Iᵢᵢ=1, reste=0 | [1 0; 0 1] | Neutre |
| **Symétrique** | A = Aᵀ | [1 2; 2 3] | Hessienne |
| **Orthogonale** | AᵀA = I | [cos θ -sin θ; sin θ cos θ] | Rotation |
| **Définie positive** | xᵀAx > 0 ∀x≠0 | [2 1; 1 2] | Convexité |

#### Opérations Matricielles

##### 1. **Multiplication Matricielle**

Si A ∈ ℝᵐˣⁿ et B ∈ ℝⁿˣᵖ, alors C = AB ∈ ℝᵐˣᵖ avec :

```
cᵢⱼ = Σₖ₌₁ⁿ aᵢₖ bₖⱼ = aᵢ₁b₁ⱼ + aᵢ₂b₂ⱼ + ... + aᵢₙbₙⱼ
```

**📝 Exemple détaillé complet** :

```
Soit A = [1  2  3]  ∈ ℝ²ˣ³    et    B = [2  1]  ∈ ℝ³ˣ²
         [4  5  6]                        [0  3]
                                          [1  2]

Dimension de C = AB :  (2×3) × (3×2) = (2×2) ✓ Compatible !

Calcul de chaque élément c ᵢⱼ :

┌─ Élément c₁₁ (ligne 1 de A × colonne 1 de B) ─┐
│ c₁₁ = a₁₁b₁₁ + a₁₂b₂₁ + a₁₃b₃₁                │
│     = 1×2 + 2×0 + 3×1                          │
│     = 2 + 0 + 3 = 5                            │
└────────────────────────────────────────────────┘

┌─ Élément c₁₂ (ligne 1 de A × colonne 2 de B) ─┐
│ c₁₂ = a₁₁b₁₂ + a₁₂b₂₂ + a₁₃b₃₂                │
│     = 1×1 + 2×3 + 3×2                          │
│     = 1 + 6 + 6 = 13                           │
└────────────────────────────────────────────────┘

┌─ Élément c₂₁ (ligne 2 de A × colonne 1 de B) ─┐
│ c₂₁ = a₂₁b₁₁ + a₂₂b₂₁ + a₂₃b₃₁                │
│     = 4×2 + 5×0 + 6×1                          │
│     = 8 + 0 + 6 = 14                           │
└────────────────────────────────────────────────┘

┌─ Élément c₂₂ (ligne 2 de A × colonne 2 de B) ─┐
│ c₂₂ = a₂₁b₁₂ + a₂₂b₂₂ + a₂₃b₃₂                │
│     = 4×1 + 5×3 + 6×2                          │
│     = 4 + 15 + 12 = 31                         │
└────────────────────────────────────────────────┘

Résultat :
    C = AB = [5   13]
             [14  31]

Schéma visuel :
    [1 2 3]   [2 1]   [5  13]
    [4 5 6] × [0 3] = [14 31]
              [1 2]
```

**⚠️ Propriétés importantes** :
- AB ≠ BA en général (non-commutativité)
- (AB)C = A(BC) (associativité)
- A(B + C) = AB + AC (distributivité)

##### 2. **Transposée**

```
(Aᵀ)ᵢⱼ = Aⱼᵢ    (échange lignes ↔ colonnes)
```

**📝 Exemple concret** :

```
Soit A = [1  2  3]  ∈ ℝ²ˣ³
         [4  5  6]

Transposée :
         [1  4]
    Aᵀ = [2  5]  ∈ ℝ³ˣ²
         [3  6]

Visualisation :
    Ligne 1 de A  →  Colonne 1 de Aᵀ
    Ligne 2 de A  →  Colonne 2 de Aᵀ

Propriétés :
    (Aᵀ)ᵀ = A
    (AB)ᵀ = BᵀAᵀ  ⚠️ Attention à l'ordre inversé !
    (A + B)ᵀ = Aᵀ + Bᵀ
```

##### 3. **Trace**

```
tr(A) = Σᵢ₌₁ⁿ aᵢᵢ    (somme des éléments diagonaux, A carrée)
```

**📝 Exemple et propriétés** :

```
Soit A = [2  1  0]
         [1  3  2]
         [0  2  1]

Calcul de la trace :
    tr(A) = a₁₁ + a₂₂ + a₃₃
          = 2 + 3 + 1
          = 6

Propriétés essentielles :
    1. tr(A + B) = tr(A) + tr(B)
    2. tr(αA) = α·tr(A)
    3. tr(AB) = tr(BA)  ⚠️ Important !
    4. tr(Aᵀ) = tr(A)
    5. tr(AᵀA) = Σᵢⱼ aᵢⱼ²  (norme de Frobenius au carré)

Application ML : La trace est utilisée dans l'ACP et la régularisation
```

##### 4. **Déterminant**

Pour une matrice carrée A ∈ ℝⁿˣⁿ :

**📝 Calcul détaillé pour 2×2** :

```
A = [a  b]    ⟹    det(A) = ad - bc
    [c  d]

Exemple :
A = [3  2]    det(A) = 3×5 - 2×1 = 15 - 2 = 13
    [1  5]

Interprétation géométrique :
    det(A) = aire du parallélogramme formé par les vecteurs colonnes

    det(A) > 0  →  Orientation préservée
    det(A) < 0  →  Orientation inversée
    det(A) = 0  →  Matrice singulière (non inversible)
```

**📝 Calcul pour 3×3 (règle de Sarrus)** :

```
A = [a  b  c]
    [d  e  f]
    [g  h  i]

det(A) = aei + bfg + cdh - ceg - afh - bdi

Exemple numérique :
A = [1  2  3]
    [0  1  4]
    [5  6  0]

det(A) = 1×1×0 + 2×4×5 + 3×0×6 - 3×1×5 - 1×4×6 - 2×0×0
       = 0 + 40 + 0 - 15 - 24 - 0
       = 1

Propriétés :
    1. det(AB) = det(A) × det(B)
    2. det(Aᵀ) = det(A)
    3. det(A⁻¹) = 1/det(A)  si A inversible
    4. det(αA) = αⁿ det(A)  pour A ∈ ℝⁿˣⁿ
```

#### Matrices Spéciales

##### **Matrice Identité** I

```
    [1  0  0  …  0]
    [0  1  0  …  0]
I = [0  0  1  …  0]
    [⋮  ⋮  ⋮  ⋱  ⋮]
    [0  0  0  …  1]

Propriété : AI = IA = A  (élément neutre)
```

##### **Matrice Diagonale** D

```
    [d₁  0   0  …  0 ]
    [0   d₂  0  …  0 ]
D = [0   0   d₃ …  0 ]
    [⋮   ⋮   ⋮  ⋱  ⋮ ]
    [0   0   0  …  dₙ]

Multiplication rapide : (DA)ᵢⱼ = dᵢ aᵢⱼ
```

##### **Matrice Symétrique** A = Aᵀ

```
    [1  2  3]       Élément clé :
A = [2  5  6]       aᵢⱼ = aⱼᵢ
    [3  6  9]

Application : Matrices de covariance, Hessienne
```

##### **Matrice Orthogonale** Q : QᵀQ = I

```
Exemple (rotation de 90°) :
    [0  -1]
Q = [1   0]

Vérification :
    QᵀQ = [0  1] [0  -1] = [1  0] = I ✓
          [-1 0] [1   0]   [0  1]

Propriétés :
    - det(Q) = ±1
    - Préserve les normes : ‖Qx‖ = ‖x‖
    - Préserve les angles
```

#### Valeurs Propres et Vecteurs Propres

Pour une matrice carrée A ∈ ℝⁿˣⁿ :

```
Av = λv    où λ ∈ ℝ (valeur propre) et v ≠ 0 (vecteur propre)
```

**📝 Calcul détaillé pour une matrice 2×2** :

```
Soit A = [4  2]
         [2  3]

ÉTAPE 1 : Trouver l'équation caractéristique
    det(A - λI) = 0

    A - λI = [4-λ   2  ]
             [2    3-λ]

    det(A - λI) = (4-λ)(3-λ) - 2×2
                = 12 - 4λ - 3λ + λ² - 4
                = λ² - 7λ + 8 = 0

ÉTAPE 2 : Résoudre l'équation caractéristique
    λ = (7 ± √(49-32))/2 = (7 ± √17)/2

    λ₁ ≈ 5.56    λ₂ ≈ 1.44

ÉTAPE 3 : Trouver les vecteurs propres

Pour λ₁ ≈ 5.56 :
    (A - λ₁I)v₁ = 0
    [4-5.56   2    ] [v₁₁] = [0]
    [2      3-5.56] [v₁₂]   [0]

    [-1.56   2   ] [v₁₁] = [0]
    [2     -2.56] [v₁₂]   [0]

    De la 1ère équation : -1.56v₁₁ + 2v₁₂ = 0
                          v₁₂ = 0.78v₁₁

    Normalisation (‖v₁‖ = 1) :
    v₁ ≈ [0.79]
         [0.62]

Pour λ₂ ≈ 1.44 :
    v₂ ≈ [-0.62]
         [0.79]

Vérification :
    Av₁ = [4  2] [0.79]   [4.39]         [0.79]
          [2  3] [0.62] = [3.44] ≈ 5.56  [0.62] = λ₁v₁ ✓

Interprétation géométrique :
    v₁ : direction d'étirement maximal (facteur 5.56)
    v₂ : direction d'étirement minimal (facteur 1.44)
```

**Propriétés Fondamentales** :

| **Propriété** | **Matrice Générale** | **Matrice Symétrique** |
|--------------|---------------------|----------------------|
| Valeurs propres réelles | Non garanti | ✓ Toujours |
| Vecteurs propres orthogonaux | Non garanti | ✓ Toujours |
| Diagonalisable | Non garanti | ✓ Toujours |
| det(A) | = Π λᵢ | = Π λᵢ |
| tr(A) | = Σ λᵢ | = Σ λᵢ |

**Applications en Machine Learning** :

```
1. PCA (Analyse en Composantes Principales)
   → Les vecteurs propres de la covariance = directions principales

2. PageRank (Google)
   → Vecteur propre dominant de la matrice de transition

3. Spectral Clustering
   → Vecteurs propres de la matrice Laplacienne

4. Stabilité des réseaux de neurones
   → Valeurs propres de la Hessienne
```

**Exemple Python Complet** :
```python
import numpy as np

# Matrice symétrique
A = np.array([[4, 2], 
              [2, 3]])

# Calcul des valeurs et vecteurs propres
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Valeurs propres:", eigenvalues)
print("Vecteurs propres:\n", eigenvectors)

# Vérification : Av = λv
for i in range(len(eigenvalues)):
    λ = eigenvalues[i]
    v = eigenvectors[:, i]
    Av = A @ v
    λv = λ * v
    print(f"\nVérification pour λ_{i+1}:")
    print(f"Av = {Av}")
    print(f"λv = {λv}")
    print(f"Égalité : {np.allclose(Av, λv)}")

# Diagonalisation : A = VΛVᵀ
V = eigenvectors
Λ = np.diag(eigenvalues)
A_reconstructed = V @ Λ @ V.T
print(f"\nReconstitution de A :")
print(f"Erreur : {np.linalg.norm(A - A_reconstructed)}")
```

### 1.1.4 Applications Multilinéaires

Une **application bilinéaire** B : ℝⁿ × ℝⁿ → ℝ est linéaire en chaque argument :

```
B(αu + βv, w) = αB(u,w) + βB(v,w)
B(u, αv + βw) = αB(u,v) + βB(u,w)
```

**Exemple** : Le produit scalaire ⟨u, v⟩ = uᵀv est bilinéaire.

---

## 1.2 Topologie

### 1.2.1 Ensembles Ouverts et Fermés dans ℝᵈ

#### Boule Ouverte

La **boule ouverte** de centre x et rayon r > 0 :
```
B(x, r) = {y ∈ ℝᵈ : ‖y - x‖ < r}
```

#### Ensemble Ouvert

Un ensemble U ⊂ ℝᵈ est **ouvert** si pour tout x ∈ U, il existe r > 0 tel que B(x, r) ⊂ U.

#### Ensemble Fermé

Un ensemble F ⊂ ℝᵈ est **fermé** si son complément ℝᵈ \ F est ouvert.

**Exemples** :
- (a, b) est ouvert dans ℝ
- [a, b] est fermé dans ℝ
- ℝⁿ et ∅ sont à la fois ouverts et fermés

### 1.2.2 Ensembles Compacts

Un ensemble K ⊂ ℝᵈ est **compact** s'il est fermé et borné.

**Théorème de Heine-Borel** : Dans ℝᵈ, un ensemble est compact si et seulement si il est fermé et borné.

**Importance en ML** : Les ensembles compacts garantissent l'existence de minima pour les fonctions continues (théorème de Weierstrass).

### 1.2.3 Espaces Métriques

Un **espace métrique** est un ensemble X muni d'une distance d : X × X → ℝ₊ telle que :

1. d(x, y) = 0 ⟺ x = y
2. d(x, y) = d(y, x) (symétrie)
3. d(x, z) ≤ d(x, y) + d(y, z) (inégalité triangulaire)

**Exemples de distances** :
- Distance euclidienne : d(x, y) = ‖x - y‖
- Distance de Manhattan : d(x, y) = Σᵢ |xᵢ - yᵢ|
- Distance de Tchebychev : d(x, y) = maxᵢ |xᵢ - yᵢ|

---

## 1.3 Calcul Différentiel

### 1.3.1 Différentielles

#### Dérivée Directionnelle

La **dérivée directionnelle** de f : ℝⁿ → ℝ en x dans la direction v est :

```
Dᵥf(x) = lim[h→0] [f(x + hv) - f(x)] / h
```

#### Gradient

Le **gradient** de f : ℝⁿ → ℝ en x est le vecteur :

```
∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
```

**Propriétés** :
- Le gradient pointe dans la direction de plus grande augmentation
- ‖∇f(x)‖ donne le taux de variation maximal

**Application en ML** : Le gradient est au cœur de l'algorithme de descente de gradient.

#### Matrice Jacobienne

Pour f : ℝⁿ → ℝᵐ, la **matrice Jacobienne** est :

```
Jf(x) = [∂f₁/∂x₁  ...  ∂f₁/∂xₙ]
        [...       ...  ...]
        [∂fₘ/∂x₁  ...  ∂fₘ/∂xₙ]
```

### 1.3.2 Exemples Importants

#### Fonction Quadratique

Pour f(x) = ½xᵀAx - bᵀx :
```
∇f(x) = Ax - b
```

#### Fonction Exponentielle

Pour f(x) = exp(aᵀx) :
```
∇f(x) = a exp(aᵀx)
```

#### Fonction Logistique

Pour f(x) = log(1 + exp(aᵀx)) :
```
∇f(x) = a / (1 + exp(-aᵀx))
```

### 1.3.3 Dérivées d'Ordre Supérieur

#### Matrice Hessienne

La **matrice Hessienne** de f : ℝⁿ → ℝ est :

```
Hf(x) = [∂²f/∂xᵢ∂xⱼ]
```

**Propriétés** :
- Si f est deux fois continûment différentiable, H est symétrique
- H définit la courbure locale de f

**Convexité** : f est convexe si H est semi-définie positive partout.

### 1.3.4 Théorème de Taylor

Le **développement de Taylor** d'ordre 2 de f autour de x est :

```
f(x + h) ≈ f(x) + ∇f(x)ᵀh + ½hᵀHf(x)h
```

**Application** : Utilisé pour l'analyse de convergence des algorithmes d'optimisation.

---

## 1.4 Théorie des Probabilités

### 1.4.1 Hypothèses Générales et Notations

#### Espace de Probabilité

Un **espace de probabilité** est un triplet (Ω, ℱ, P) où :
- Ω est l'ensemble des résultats possibles (univers)
- ℱ est une σ-algèbre d'événements
- P : ℱ → [0, 1] est une mesure de probabilité

#### Variable Aléatoire

Une **variable aléatoire** X est une fonction mesurable X : Ω → ℝ.

**Notation** : X ~ P signifie que X suit la loi de probabilité P.

#### Espérance

L'**espérance** d'une variable aléatoire X est :

```
𝔼[X] = ∫ x dP(x)
```

**Propriétés** :
- Linéarité : 𝔼[αX + βY] = α𝔼[X] + β𝔼[Y]
- Si X ≥ 0, alors 𝔼[X] ≥ 0

#### Variance

La **variance** de X est :

```
Var(X) = 𝔼[(X - 𝔼[X])²] = 𝔼[X²] - (𝔼[X])²
```

**Écart-type** : σ(X) = √Var(X)

### 1.4.2 Probabilités et Espérances Conditionnelles

#### Probabilité Conditionnelle

La **probabilité conditionnelle** de A sachant B est :

```
P(A|B) = P(A ∩ B) / P(B)
```

**Formule de Bayes** :
```
P(A|B) = P(B|A)P(A) / P(B)
```

**Application en ML** : La classification bayésienne utilise cette formule pour calculer P(classe|données).

#### Espérance Conditionnelle

L'**espérance conditionnelle** de X sachant Y est :

```
𝔼[X|Y] = g(Y) où g(y) = 𝔼[X|Y=y]
```

**Propriété importante** :
```
𝔼[𝔼[X|Y]] = 𝔼[X]  (Loi de l'espérance totale)
```

### 1.4.3 Théorie des Probabilités Mesurables

#### σ-Algèbre

Une **σ-algèbre** ℱ sur Ω est une famille d'ensembles telle que :
1. Ω ∈ ℱ
2. Si A ∈ ℱ, alors Aᶜ ∈ ℱ
3. Si A₁, A₂, ... ∈ ℱ, alors ⋃ᵢ Aᵢ ∈ ℱ

#### Mesure de Probabilité

Une **mesure de probabilité** P sur (Ω, ℱ) satisfait :
1. P(Ω) = 1
2. P(A) ≥ 0 pour tout A ∈ ℱ
3. Pour A₁, A₂, ... disjoints : P(⋃ᵢ Aᵢ) = Σᵢ P(Aᵢ)

### 1.4.4 Produit de Mesures

Le **produit de mesures** P₁ ⊗ P₂ sur Ω₁ × Ω₂ satisfait :

```
(P₁ ⊗ P₂)(A × B) = P₁(A) · P₂(B)
```

**Théorème de Fubini** : Pour une fonction intégrable f :
```
∫∫ f(x,y) d(P₁⊗P₂)(x,y) = ∫[∫ f(x,y) dP₂(y)] dP₁(x)
```

### 1.4.5 Continuité Absolue et Densités

Une mesure Q est **absolument continue** par rapport à P (noté Q << P) si :
```
P(A) = 0 ⟹ Q(A) = 0
```

**Théorème de Radon-Nikodym** : Si Q << P, il existe une fonction f (densité) telle que :
```
Q(A) = ∫ₐ f dP
```

On note : f = dQ/dP

**Exemple** : La loi normale N(μ, σ²) a pour densité par rapport à la mesure de Lebesgue :
```
f(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
```

### 1.4.6 Probabilités Théoriques Mesurables

#### Indépendance

Deux événements A et B sont **indépendants** si :
```
P(A ∩ B) = P(A) · P(B)
```

Deux variables aléatoires X et Y sont **indépendantes** si :
```
P(X ∈ A, Y ∈ B) = P(X ∈ A) · P(Y ∈ B)
```
pour tous ensembles A et B.

### 1.4.7 Espérances Conditionnelles (Cas Général)

L'**espérance conditionnelle** 𝔼[X|𝒢] est la projection de X sur l'espace L²(𝒢) :

**Propriétés** :
1. 𝔼[𝔼[X|𝒢]] = 𝔼[X]
2. Si Y est 𝒢-mesurable : 𝔼[XY|𝒢] = Y𝔼[X|𝒢]
3. Inégalité de Jensen : 𝔼[φ(X)|𝒢] ≥ φ(𝔼[X|𝒢]) pour φ convexe

### 1.4.8 Probabilités Conditionnelles (Cas Général)

La **probabilité conditionnelle** P(A|𝒢) est définie comme :
```
P(A|𝒢) = 𝔼[1ₐ|𝒢]
```

où 1ₐ est la fonction indicatrice de A.

---

## 💡 Applications en Machine Learning

Les concepts de ce chapitre sont utilisés partout en ML :

```
┌────────────────────────────────────────────────────────────────┐
│                  CONCEPTS → APPLICATIONS ML                     │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📐 ALGÈBRE LINÉAIRE                                            │
│     Vecteurs & Matrices  →  Représentation des données          │
│     Valeurs propres      →  PCA, Spectral Clustering           │
│     Normes              →  Distance, Similarité                │
│                                                                 │
│  🎯 CALCUL DIFFÉRENTIEL                                         │
│     Gradient            →  Descente de gradient                │
│     Hessienne           →  Analyse de convergence              │
│     Jacobienne          →  Backpropagation                     │
│                                                                 │
│  🎲 PROBABILITÉS                                                │
│     Espérance/Variance  →  Estimation, Incertitude             │
│     Bayes               →  Classification bayésienne           │
│     Conditionnelles     →  Modèles graphiques                  │
│                                                                 │
│  📊 TOPOLOGIE                                                   │
│     Compacts            →  Existence de minima                 │
│     Ouverts/Fermés      →  Continuité, Convergence             │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 📋 Résumé Visuel du Chapitre

### 🔑 Formules Clés à Retenir

| **Concept** | **Formule** | **Application** |
|------------|------------|----------------|
| **Produit scalaire** | ⟨u, v⟩ = Σ uᵢvᵢ | Similarité |
| **Norme L₂** | ‖v‖ = √(Σ vᵢ²) | Distance |
| **Matrice × Vecteur** | (Av)ᵢ = Σⱼ aᵢⱼvⱼ | Transformation |
| **Valeur propre** | Av = λv | PCA, Spectral |
| **Gradient** | ∇f = [∂f/∂xᵢ] | Optimisation |
| **Hessienne** | H = [∂²f/∂xᵢ∂xⱼ] | Convexité |
| **Bayes** | P(A\|B) = P(B\|A)P(A)/P(B) | Classification |
| **Variance** | Var(X) = 𝔼[X²] - (𝔼[X])² | Incertitude |

### 🎯 Checklist de Compréhension

Après ce chapitre, vous devriez pouvoir :

- ✅ Calculer le produit scalaire et la norme d'un vecteur
- ✅ Multiplier deux matrices (avec compatibilité des dimensions)
- ✅ Trouver les valeurs et vecteurs propres d'une matrice 2×2
- ✅ Calculer le gradient d'une fonction simple
- ✅ Appliquer la formule de Bayes
- ✅ Distinguer les différents types de matrices (symétrique, orthogonale, etc.)
- ✅ Comprendre l'interprétation géométrique des concepts

### 📊 Diagramme de Décision : Quelle Opération Utiliser ?

```
            Dois-je faire un calcul ?
                     │
        ┌────────────┼────────────┐
        │                         │
    Sur VECTEURS            Sur MATRICES
        │                         │
    ┌───┴───┐               ┌─────┴─────┐
    │       │               │           │
 Distance  Angle      Transformation  Analyse
    │       │               │           │
 Norme L₂  arccos      Multiplication Valeurs propres
  ‖v‖     (⟨u,v⟩/      A×v ou A×B    Av = λv
         ‖u‖‖v‖)
```

---

## 📝 Exercices

### Exercice 1 : Algèbre Linéaire
Soit A = [[2, 1], [1, 3]]. Calculez ses valeurs propres et vecteurs propres.

### Exercice 2 : Calcul Différentiel
Pour f(x) = ‖x‖², calculez ∇f(x) et Hf(x).

### Exercice 3 : Probabilités
Démontrez que Var(X) = 𝔼[X²] - (𝔼[X])².

### Exercice 4 : Formule de Bayes
Un test médical détecte une maladie avec 95% de sensibilité et 90% de spécificité. Si 1% de la population a la maladie, quelle est la probabilité qu'une personne testée positive ait réellement la maladie ?

---

## 🔗 Références et Lectures Complémentaires

- **Algèbre linéaire** : Strang, G. "Linear Algebra and Its Applications"
- **Calcul** : Spivak, M. "Calculus on Manifolds"
- **Probabilités** : Billingsley, P. "Probability and Measure"

---

[⬅️ Retour à la table des matières](../README.md) | [Chapitre suivant ➡️](./chapitre-02-analyse-matricielle.md)

