# Chapitre 18 : Apprentissage de Modèles Graphiques

## 📚 Introduction

Ce chapitre couvre l'apprentissage des paramètres et de la structure des modèles graphiques.

---

## 18.1 Apprentissage de Réseaux Bayésiens

### 18.1.1 Apprentissage de Paramètres

**Maximum de vraisemblance** :
```
θ̂ = argmax_θ log P(D|θ)
```

### 18.1.2 Apprentissage de Structure

**Score** : BIC, AIC

**Recherche** : Hill climbing, recherche gloutonne

---

## 18.2 Apprentissage de MRF

### Maximum de Vraisemblance

**Log-vraisemblance** :
```
ℓ(θ) = Σᵢ log P(xᵢ|θ)
       = Σᵢ [θᵀφ(xᵢ) - log Z(θ)]
```

**Gradient** :
```
∇ℓ = 𝔼_data[φ(X)] - 𝔼_model[φ(X)]
```

### Descente de Gradient Stochastique

Approximer 𝔼_model par échantillonnage MCMC.

---

## 18.3 Observations Incomplètes

**Algorithme EM** : Traiter variables manquantes comme latentes.

---

[⬅️ Chapitre précédent](./chapitre-17-variables-latentes.md) | [Retour](../README.md) | [Suite ➡️](../partie-5-methodes-generatives/chapitre-19-generatives-profondes.md)

