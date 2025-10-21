# Chapitre 2 : Analyse Matricielle

## 📚 Introduction

L'analyse matricielle est fondamentale en machine learning. Les matrices permettent de représenter les données, les transformations linéaires et les modèles. Ce chapitre couvre les outils matriciels essentiels pour comprendre les algorithmes d'apprentissage.

## 🗺️ Carte Mentale : Analyse Matricielle

```
                    ANALYSE MATRICIELLE
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   DÉCOMPOSITIONS        NORMES            APPLICATIONS
        │                   │                   │
    ┌───┴───┐           ┌───┴───┐           ┌───┴───┐
    │       │           │       │           │       │
  SVD    Valeurs     Frobenius  Spectrale  PCA   Compression
         Propres       │         │               d'Images
    │       │          L₂        σ_max          │
  A=UΣVᵀ  A=QΛQᵀ     Matrix     ‖A‖₂       Réduction
                      Norm                  Dimension
```

## 📊 Tableau Comparatif : Décompositions Matricielles

| **Décomposition** | **Formule** | **Conditions** | **Complexité** | **Applications ML** |
|------------------|------------|---------------|---------------|-------------------|
| **Valeurs propres** | A = QΛQᵀ | A symétrique | O(n³) | PCA, Spectral clustering |
| **SVD** | A = UΣVᵀ | Toute matrice | O(mn²) | Recommandation, compression |
| **QR** | A = QR | Toute matrice | O(mn²) | Régression, moindres carrés |
| **Cholesky** | A = LLᵀ | A définie positive | O(n³/3) | Simulation, optimisation |
| **LU** | A = LU | A inversible | O(n³/3) | Systèmes linéaires |

## 📐 Visualisation : SVD (Décomposition en Valeurs Singulières)

```
                DÉCOMPOSITION SVD : A = U Σ Vᵀ
                
    A           =       U          ×      Σ        ×       Vᵀ
  (m×n)               (m×m)            (m×n)             (n×n)
                                                    
┌─────────┐     ┌──────────┐     ┌────────────┐    ┌──────────┐
│         │     │          │     │σ₁  0   0 …│    │          │
│    A    │  =  │    U     │  ×  │0  σ₂  0  │ ×  │    Vᵀ    │
│         │     │          │     │0   0  σ₃ │    │          │
│         │     │          │     │⋮   ⋮   ⋮ │    │          │
└─────────┘     └──────────┘     │0   0   0 │    └──────────┘
                                 └────────────┘
Données         Rotation         Étirement      Rotation
originales      gauche           (scaling)      droite

Interprétation géométrique :
  1. Vᵀ : Rotation dans l'espace d'entrée
  2. Σ  : Étirement selon les directions principales
  3. U  : Rotation dans l'espace de sortie
```

## 🔢 Tableau des Normes Matricielles

| **Norme** | **Définition** | **Formule** | **Interprétation** | **Usage ML** |
|-----------|---------------|------------|-------------------|-------------|
| **Frobenius** | ‖A‖_F | √(Σᵢⱼ aᵢⱼ²) | Norme L₂ des éléments | Régularisation |
| **Spectrale** | ‖A‖₂ | σ_max(A) | Plus grande valeur singulière | Stabilité |
| **Nucléaire** | ‖A‖_* | Σᵢ σᵢ(A) | Somme val. singulières | Rang faible |
| **L₁** | ‖A‖₁ | max_j Σᵢ \|aᵢⱼ\| | Max somme colonnes | Sparsité |
| **L∞** | ‖A‖∞ | max_i Σⱼ \|aᵢⱼ\| | Max somme lignes | Analyse erreur |

---

## 2.1 Notation et Faits de Base

### Matrices et Opérations

#### Notation
- **A ∈ ℝᵐˣⁿ** : matrice m×n
- **Aᵀ** : transposée de A
- **A⁻¹** : inverse de A (si elle existe)
- **tr(A)** : trace de A (somme des éléments diagonaux)
- **det(A)** : déterminant de A
- **rank(A)** : rang de A

#### Matrices Définies Positives

Une matrice symétrique A est **définie positive** (A ≻ 0) si :
```
xᵀAx > 0  pour tout x ≠ 0
```

**Semi-définie positive** (A ⪰ 0) si :
```
xᵀAx ≥ 0  pour tout x
```

**Propriétés importantes** :
- A ≻ 0 ⟺ toutes les valeurs propres de A sont strictement positives
- A ⪰ 0 ⟺ toutes les valeurs propres de A sont non-négatives

### Décomposition en Valeurs Propres

Pour une matrice symétrique A ∈ ℝⁿˣⁿ :
```
A = QΛQᵀ
```
où :
- Q est orthogonale (QᵀQ = I)
- Λ = diag(λ₁, λ₂, ..., λₙ) contient les valeurs propres

**Exemple Python** :
```python
import numpy as np

A = np.array([[4, 2], [2, 3]])
eigenvalues, Q = np.linalg.eigh(A)  # Pour matrices symétriques
Lambda = np.diag(eigenvalues)

# Vérification
reconstructed = Q @ Lambda @ Q.T
print(np.allclose(A, reconstructed))  # True
```

### Décomposition en Valeurs Singulières (SVD)

Pour toute matrice A ∈ ℝᵐˣⁿ :
```
A = UΣVᵀ
```
où :
- U ∈ ℝᵐˣᵐ orthogonale
- V ∈ ℝⁿˣⁿ orthogonale  
- Σ ∈ ℝᵐˣⁿ diagonale avec σ₁ ≥ σ₂ ≥ ... ≥ 0

**Valeurs singulières** : σᵢ = √λᵢ où λᵢ sont les valeurs propres de AᵀA.

**Application en ML** : La SVD est utilisée dans :
- La réduction de dimension (PCA)
- La recommandation (factorisation matricielle)
- Le débruitage de données

---

## 2.2 L'Inégalité de Trace

### Théorème Fondamental

Pour des matrices A, B ∈ ℝⁿˣⁿ symétriques définies positives :

```
tr(AB) ≤ tr(A)tr(B)
```

avec égalité si et seulement si A et B commutent (AB = BA).

### Inégalité de von Neumann

Pour A, B ∈ ℝⁿˣⁿ :

```
tr(AB) ≤ Σᵢ σᵢ(A)σᵢ(B)
```

où σᵢ(A) désignent les valeurs singulières de A ordonnées par ordre décroissant.

### Inégalité d'Araki-Lieb-Thirring

Pour A ⪰ 0 et B ⪰ 0 :

```
tr(AᵖBᵖ) ≥ tr((AB)ᵖ)  pour 0 < p < 1
tr(AᵖBᵖ) ≤ tr((AB)ᵖ)  pour p ≥ 1
```

**Application** : Ces inégalités sont utilisées dans l'analyse de convergence des algorithmes d'optimisation.

### Lemme de Trace

Pour A ∈ ℝᵐˣⁿ et B ∈ ℝⁿˣᵐ :

```
tr(AB) = tr(BA)
```

**Propriétés utiles** :
- tr(ABC) = tr(CAB) = tr(BCA)
- tr(AᵀB) = ⟨A, B⟩_F (produit scalaire de Frobenius)

---

## 2.3 Applications

### Régression Ridge

Dans la régression ridge, on minimise :

```
‖Xβ - y‖² + λ‖β‖²
```

La solution est donnée par :

```
β̂ = (XᵀX + λI)⁻¹Xᵀy
```

**Analyse matricielle** : 
- XᵀX est semi-définie positive
- XᵀX + λI est définie positive pour λ > 0
- L'inverse existe toujours

### Analyse en Composantes Principales (PCA)

Pour des données X ∈ ℝⁿˣᵈ (n observations, d variables) :

1. Centrer les données : X̃ = X - μ où μ est la moyenne
2. Calculer la matrice de covariance : C = (1/n)X̃ᵀX̃
3. Décomposer : C = QΛQᵀ
4. Projeter sur les k premiers vecteurs propres

**Réduction de dimension** :
```
X_réduite = X̃ · Q[:, :k]
```

**Exemple Python** :
```python
from sklearn.decomposition import PCA
import numpy as np

# Données
X = np.random.randn(100, 10)

# PCA à 3 composantes
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X)

print("Variance expliquée:", pca.explained_variance_ratio_)
```

### Décomposition de Cholesky

Pour A ⪰ 0, il existe une matrice triangulaire inférieure L telle que :

```
A = LLᵀ
```

**Utilité** :
- Résolution efficace de systèmes linéaires
- Simulation de variables gaussiennes multivariées
- Optimisation numérique

**Exemple** : Générer X ~ N(μ, Σ)
```python
import numpy as np

mu = np.array([1, 2])
Sigma = np.array([[2, 0.5], [0.5, 1]])

# Décomposition de Cholesky
L = np.linalg.cholesky(Sigma)

# Générer échantillon
z = np.random.randn(2)  # N(0, I)
x = mu + L @ z          # N(μ, Σ)
```

---

## 2.4 Normes Matricielles

### Norme de Frobenius

```
‖A‖_F = √(Σᵢⱼ aᵢⱼ²) = √tr(AᵀA)
```

**Propriétés** :
- ‖A‖_F² = Σᵢ σᵢ² (somme des carrés des valeurs singulières)
- ‖AB‖_F ≤ ‖A‖_F ‖B‖_F

### Norme Spectrale

```
‖A‖₂ = max_{‖x‖=1} ‖Ax‖ = σ₁(A)
```

La norme spectrale est la plus grande valeur singulière.

**Propriété importante** :
```
‖AB‖₂ ≤ ‖A‖₂ ‖B‖₂
```

### Norme Nucléaire

```
‖A‖_* = Σᵢ σᵢ(A)
```

La norme nucléaire est la somme des valeurs singulières.

**Application en ML** : Utilisée pour la régularisation dans la complétion de matrices :

```
minimize  ‖A‖_*  subject to  A_Ω = M_Ω
```

où Ω représente les entrées observées.

### Comparaison des Normes

Pour A ∈ ℝᵐˣⁿ avec r = min(m, n) :

```
‖A‖₂ ≤ ‖A‖_F ≤ √r ‖A‖₂
‖A‖₂ ≤ ‖A‖_* ≤ √r ‖A‖_F
```

**Tableau récapitulatif** :

| Norme | Définition | Valeurs singulières |
|-------|------------|---------------------|
| Frobenius | √Σᵢⱼ aᵢⱼ² | √Σᵢ σᵢ² |
| Spectrale | max ‖Ax‖/‖x‖ | σ₁ |
| Nucléaire | - | Σᵢ σᵢ |

---

## 2.5 Approximation de Rang Faible

### Théorème de Eckart-Young-Mirsky

La meilleure approximation de rang k de A (au sens de Frobenius ou spectral) est :

```
A_k = Σᵢ₌₁ᵏ σᵢ uᵢvᵢᵀ
```

où A = Σᵢ σᵢ uᵢvᵢᵀ est la SVD de A.

**Erreur d'approximation** :

- Norme de Frobenius : ‖A - A_k‖_F = √(Σᵢ₌ₖ₊₁ⁿ σᵢ²)
- Norme spectrale : ‖A - A_k‖₂ = σₖ₊₁

### Implémentation Python

```python
import numpy as np

def low_rank_approximation(A, k):
    """
    Approximation de rang k de la matrice A
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Garder les k premières composantes
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]
    
    # Reconstruction
    A_k = U_k @ np.diag(s_k) @ Vt_k
    
    # Erreur
    error = np.linalg.norm(A - A_k, 'fro')
    
    return A_k, error

# Exemple
A = np.random.randn(100, 50)
A_k, error = low_rank_approximation(A, k=10)
print(f"Erreur de reconstruction: {error:.4f}")
```

### Application : Compression d'Images

Une image en niveaux de gris peut être vue comme une matrice. L'approximation de rang faible permet de la compresser :

```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Charger image
img = Image.open('image.jpg').convert('L')
A = np.array(img)

# Approximations de différents rangs
for k in [5, 10, 20, 50]:
    A_k, _ = low_rank_approximation(A, k)
    
    # Taux de compression
    m, n = A.shape
    compression = (k * (m + n)) / (m * n) * 100
    
    print(f"Rang {k}: compression {compression:.1f}%")
    
    # Afficher
    plt.figure()
    plt.imshow(A_k, cmap='gray')
    plt.title(f'Rang {k}')
    plt.show()
```

### Factorisation Matricielle en ML

De nombreuses méthodes ML utilisent la factorisation matricielle :

**Systèmes de recommandation** :
```
R ≈ UV^T
```
où :
- R : matrice utilisateur×item
- U : facteurs latents utilisateurs
- V : facteurs latents items

**Algorithme gradient descent** :
```python
def matrix_factorization(R, k, steps=1000, lr=0.01, reg=0.01):
    m, n = R.shape
    U = np.random.randn(m, k)
    V = np.random.randn(n, k)
    
    for step in range(steps):
        # Gradient par rapport à U
        grad_U = (U @ V.T - R) @ V + reg * U
        U -= lr * grad_U
        
        # Gradient par rapport à V
        grad_V = (U @ V.T - R).T @ U + reg * V
        V -= lr * grad_V
        
        if step % 100 == 0:
            loss = np.linalg.norm(R - U @ V.T, 'fro')**2
            print(f"Step {step}, Loss: {loss:.4f}")
    
    return U, V
```

---

## 💡 Points Clés à Retenir

1. **SVD** : Outil fondamental pour l'analyse de matrices
2. **Normes matricielles** : Mesures de "taille" et régularisation
3. **Approximation de rang faible** : Compression et réduction de dimension
4. **Matrices définies positives** : Garantissent la convexité
5. **Inégalités de trace** : Outils d'analyse théorique

---

## 📝 Exercices

### Exercice 1
Démontrez que pour une matrice symétrique A, ‖A‖₂ = max |λᵢ(A)|.

### Exercice 2
Implémentez la PCA from scratch en utilisant la SVD.

### Exercice 3
Pour une matrice de covariance Σ, montrez que Σ ⪰ 0.

### Exercice 4
Écrivez un algorithme de complétion de matrice en utilisant la régularisation nucléaire.

---

## 🔗 Applications en Machine Learning

- **PCA** : Réduction de dimension
- **SVD** : Systèmes de recommandation, traitement du langage naturel
- **Factorisation matricielle** : Apprentissage de représentations latentes
- **Régularisation** : Ridge regression, Lasso
- **Compression** : Approximation de rang faible

---

[⬅️ Chapitre précédent](./chapitre-01-notations-prerequis.md) | [Retour à la table des matières](../README.md) | [Chapitre suivant ➡️](./chapitre-03-optimisation.md)

