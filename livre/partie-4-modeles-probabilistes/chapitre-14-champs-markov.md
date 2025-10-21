# Chapitre 14 : Champs Aléatoires de Markov

## 📚 Introduction

Les champs aléatoires de Markov (MRF) modélisent les dépendances entre variables via des graphes non orientés.

---

## 14.1 Indépendance Conditionnelle

**Définition** : X ⊥ Y | Z si :
```
P(X, Y | Z) = P(X | Z) P(Y | Z)
```

**Propriété de Markov** : Un nœud est indépendant des autres sachant ses voisins.

---

## 14.2 Modèles sur Graphes

### Graphe Non Orienté

Un graphe G = (V, E) représente les dépendances :
- V : ensemble de variables
- E : arêtes entre variables dépendantes

### Propriété de Markov Locale

```
X_i ⊥ X_{V\{i,N(i)}} | X_{N(i)}
```

où N(i) sont les voisins de i.

---

## 14.3 Théorème de Hammersley-Clifford

**Théorème** : P > 0 satisfait la propriété de Markov par rapport à G si et seulement si :
```
P(x) = (1/Z) exp(-Σ_C ψ_C(x_C))
```

où :
- C parcourt les cliques de G
- ψ_C sont des potentiels
- Z est la constante de normalisation

**Exemple : Modèle d'Ising**
```
P(x) = (1/Z) exp(Σ_{i~j} θ_{ij} x_i x_j + Σ_i θ_i x_i)
```

---

## 14.4 Modèles sur Graphes Acycliques

**Arbres** : Graphes sans cycles.

**Avantage** : Inférence exacte efficace (algorithme sum-product).

---

[⬅️ Chapitre précédent](./chapitre-13-monte-carlo.md) | [Retour](../README.md) | [Suite ➡️](./chapitre-15-inference-mrf.md)

