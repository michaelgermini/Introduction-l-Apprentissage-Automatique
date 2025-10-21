# Chapitre 1 : Notations Générales et Prérequis Mathématiques

## 📚 Introduction

Ce chapitre établit les fondations mathématiques nécessaires pour comprendre l'apprentissage automatique. Nous couvrons les notations utilisées tout au long du livre et révisons les concepts essentiels d'algèbre linéaire, de topologie, de calcul différentiel et de théorie des probabilités.

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

#### Opérations sur les Vecteurs

1. **Addition** : u + v = [u₁+v₁, u₂+v₂, ..., uₙ+vₙ]ᵀ

2. **Multiplication scalaire** : αv = [αv₁, αv₂, ..., αvₙ]ᵀ

3. **Produit scalaire** :
   ```
   ⟨u, v⟩ = uᵀv = Σᵢ uᵢvᵢ
   ```

4. **Norme euclidienne** :
   ```
   ‖v‖ = √(vᵀv) = √(Σᵢ vᵢ²)
   ```

#### Propriétés Importantes

- **Inégalité de Cauchy-Schwarz** : |⟨u, v⟩| ≤ ‖u‖ · ‖v‖
- **Inégalité triangulaire** : ‖u + v‖ ≤ ‖u‖ + ‖v‖

**Exemple Pratique** :
```python
import numpy as np

# Vecteurs
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# Produit scalaire
dot_product = np.dot(u, v)  # 32

# Norme
norm_u = np.linalg.norm(u)  # √14 ≈ 3.74
```

### 1.1.3 Matrices

Une **matrice** A ∈ ℝᵐˣⁿ est un tableau rectangulaire de m lignes et n colonnes :

```
A = [a₁₁  a₁₂  ...  a₁ₙ]
    [a₂₁  a₂₂  ...  a₂ₙ]
    [...  ...  ...  ...]
    [aᵐ₁  aᵐ₂  ...  aᵐₙ]
```

#### Opérations Matricielles

1. **Multiplication matricielle** : Si A ∈ ℝᵐˣⁿ et B ∈ ℝⁿˣᵖ, alors C = AB ∈ ℝᵐˣᵖ avec :
   ```
   cᵢⱼ = Σₖ aᵢₖ bₖⱼ
   ```

2. **Transposée** : (Aᵀ)ᵢⱼ = Aⱼᵢ

3. **Trace** : tr(A) = Σᵢ aᵢᵢ (somme des éléments diagonaux)

4. **Déterminant** : det(A) pour les matrices carrées

#### Matrices Spéciales

- **Matrice identité** I : Iᵢⱼ = 1 si i=j, 0 sinon
- **Matrice diagonale** : Aᵢⱼ = 0 si i≠j
- **Matrice symétrique** : A = Aᵀ
- **Matrice orthogonale** : AᵀA = I

#### Valeurs Propres et Vecteurs Propres

Pour une matrice carrée A ∈ ℝⁿˣⁿ :
- λ est une **valeur propre** de A
- v est un **vecteur propre** associé à λ
- Si Av = λv

**Propriétés** :
- Une matrice symétrique a des valeurs propres réelles
- Les vecteurs propres d'une matrice symétrique sont orthogonaux

**Exemple** :
```python
import numpy as np

A = np.array([[4, 2], [2, 3]])

# Calcul des valeurs et vecteurs propres
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Valeurs propres:", eigenvalues)
print("Vecteurs propres:\n", eigenvectors)
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

1. **Algèbre linéaire** : Représentation des données, transformations
2. **Optimisation** : Descente de gradient, minimisation de fonctions de coût
3. **Probabilités** : Modèles génératifs, classification bayésienne
4. **Calcul différentiel** : Rétropropagation dans les réseaux de neurones

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

