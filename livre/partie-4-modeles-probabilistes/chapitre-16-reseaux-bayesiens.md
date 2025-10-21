# Chapitre 16 : Réseaux Bayésiens

## 📚 Introduction

Les réseaux bayésiens sont des modèles graphiques orientés acycliques représentant des dépendances causales.

---

## 16.1 Définitions

**Réseau bayésien** : Un DAG où :
```
P(X₁, ..., X_n) = ∏ᵢ P(Xᵢ | Pa(Xᵢ))
```

---

## 16.2 Graphe d'Indépendance Conditionnelle

### 16.2.1 Graphe Moral

Le **graphe moral** m(G) :
1. Ajouter arêtes entre parents de même enfant
2. Rendre non orienté

### 16.2.2 d-séparation

X et Y sont **d-séparés** par Z si tous les chemins entre X et Y sont bloqués.

---

## 16.3 Inférence

### Algorithme Sum-Product

Sur les arbres : complexité linéaire.

Sur les DAGs généraux : triangulation + junction tree.

---

## 16.4 Modèles d'Équations Structurelles

**SEM** : X = f(Pa(X), ε) où ε est du bruit.

---

[⬅️ Chapitre précédent](./chapitre-15-inference-mrf.md) | [Retour](../README.md) | [Suite ➡️](./chapitre-17-variables-latentes.md)

