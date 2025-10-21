# Chapitre 23 : Bornes de Généralisation

## 📚 Introduction

Ce chapitre présente les outils théoriques pour analyser la capacité de généralisation des algorithmes d'apprentissage.

---

## 23.1 Notation

**Risque réel** : R(f) = 𝔼[L(Y, f(X))]
**Risque empirique** : R̂(f) = (1/n) Σᵢ L(yᵢ, f(xᵢ))

**Objectif** : Borner |R(f̂) - R̂(f̂)|

---

## 23.2 Critères d'Information

### 23.2.1 AIC (Akaike Information Criterion)

```
AIC = -2 log L + 2k
```

où k est le nombre de paramètres.

### 23.2.2 BIC (Bayesian Information Criterion)

```
BIC = -2 log L + k log n
```

**Propriété** : BIC pénalise plus la complexité pour n grand.

---

## 23.3 Inégalités de Concentration

### 23.3.1 Inégalité de Hoeffding

Pour X₁, ..., Xₙ i.i.d. bornées dans [a, b] :
```
P(|X̄ - 𝔼[X]| ≥ ε) ≤ 2 exp(-2nε²/(b-a)²)
```

### 23.3.2 Variables Sous-Gaussiennes

X est **sous-gaussienne** si :
```
𝔼[exp(λX)] ≤ exp(λ²σ²/2)
```

**Propriété** : Les variables bornées sont sous-gaussiennes.

---

## 23.4 Dimension VC

### Définition

La **dimension VC** d'une classe ℱ est le plus grand n tel que ℱ peut réaliser toutes les 2ⁿ dichotomies sur n points.

**Exemples** :
- Demi-espaces en ℝᵈ : VC = d + 1
- Perceptron : VC = d + 1

### Théorème de Vapnik

Avec probabilité 1 - δ :
```
R(f̂) ≤ R̂(f̂) + O(√(VC log(n/VC) / n))
```

---

## 23.5 Nombres de Couverture

### Définition

Le **nombre de couverture** N(ε, ℱ, d) est le nombre minimal de boules de rayon ε pour couvrir ℱ.

**Entropie métrique** : H(ε) = log N(ε)

### Inégalité

Avec probabilité 1 - δ :
```
|R(f) - R̂(f)| ≤ O(√(H(ε/n) / n))
```

---

## 23.6 Complexité de Rademacher

### Définition

```
R_n(ℱ) = 𝔼_σ[sup_{f∈ℱ} (1/n) Σᵢ σᵢ f(xᵢ)]
```

où σᵢ ∈ {-1, +1} aléatoires.

### Borne de Généralisation

Avec probabilité 1 - δ :
```
R(f) ≤ R̂(f) + 2R_n(ℱ) + O(√(log(1/δ) / n))
```

---

## 23.7 Application à la Sélection de Modèles

### Principe du Risque Structurel

Pour ℱ₁ ⊂ ℱ₂ ⊂ ... :
```
f̂ = argmin_{f∈ℱ_k} [R̂(f) + Pen(k)]
```

**Pénalité** : Pen(k) ∝ Complexity(ℱ_k)

---

## 💡 Points Clés

1. **Trade-off biais-variance** : Formalisé par les bornes
2. **Complexité** : VC-dimension, nombres de couverture, Rademacher
3. **Concentration** : Hoeffding, sous-gaussiennes
4. **Sélection de modèles** : AIC, BIC, validation croisée

---

## 📝 Résumé du Livre

Ce livre a couvert :

### Partie I : Fondements
- Algèbre linéaire et optimisation
- Outils mathématiques essentiels

### Partie II : Concepts
- Prédiction et évaluation
- Noyaux et espaces de Hilbert

### Partie III : Apprentissage Supervisé
- Régression et classification
- Arbres, forests, boosting
- Réseaux de neurones

### Partie IV : Modèles Probabilistes
- Graphes et inférence
- MCMC et variational inference

### Partie V : Méthodes Génératives
- VAE, GAN, Flows

### Partie VI : Non Supervisé
- Clustering et réduction de dimension
- Visualisation

### Partie VII : Théorie
- Bornes de généralisation
- Analyse théorique

---

## 🎓 Conclusion

Le machine learning est un domaine riche combinant théorie et pratique. Ce livre a fourni les fondements mathématiques et les algorithmes pour :

- Comprendre les principes sous-jacents
- Implémenter des algorithmes
- Analyser leur comportement
- Développer de nouvelles méthodes

**Bonne continuation dans votre apprentissage du ML ! 🚀**

---

[⬅️ Chapitre précédent](../partie-6-non-supervise/chapitre-22-visualisation.md) | [🏠 Retour à l'accueil](../README.md)

