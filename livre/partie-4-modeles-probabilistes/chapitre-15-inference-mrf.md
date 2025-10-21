# Chapitre 15 : Inférence Probabiliste pour les MRF

## 📚 Introduction

Ce chapitre couvre les algorithmes d'inférence pour les champs aléatoires de Markov.

---

## 15.1 Échantillonnage Monte-Carlo

**Méthode** : Utiliser Gibbs sampling ou Metropolis-Hastings.

---

## 15.2 Inférence avec Graphes Acycliques

### Algorithme Sum-Product (Belief Propagation)

**Messages** :
```
m_{i→j}(x_j) = Σ_{x_i} ψ(x_i, x_j) φ(x_i) ∏_{k∈N(i)\j} m_{k→i}(x_i)
```

**Croyances** :
```
b_i(x_i) ∝ φ(x_i) ∏_{j∈N(i)} m_{j→i}(x_i)
```

---

## 15.3 Propagation de Croyances

Pour les graphes avec cycles : algorithme itératif (peut ne pas converger).

---

## 15.4 Configuration la Plus Probable

**Algorithme Max-Product** : Remplacer Σ par max dans les messages.

---

[⬅️ Chapitre précédent](./chapitre-14-champs-markov.md) | [Retour](../README.md) | [Suite ➡️](./chapitre-16-reseaux-bayesiens.md)

