# Chapitre 12 : Comparaison de Distributions de Probabilité

## 📚 Introduction

Ce chapitre présente différentes façons de mesurer la similarité ou la distance entre distributions de probabilité.

---

## 12.1 Distance de Variation Totale

**Définition** :
```
TV(P, Q) = sup_{A} |P(A) - Q(A)|
         = (1/2) ∫ |p(x) - q(x)| dx
```

**Propriétés** :
- 0 ≤ TV(P, Q) ≤ 1
- TV(P, Q) = 0 ⟺ P = Q

---

## 12.2 Divergences

### Divergence de Kullback-Leibler

```
KL(P || Q) = ∫ p(x) log(p(x)/q(x)) dx
           = 𝔼_P[log(p(X)/q(X))]
```

**Propriétés** :
- KL ≥ 0 (inégalité de Gibb's)
- KL = 0 ⟺ P = Q
- **Non symétrique** !

**Applications** :
- Maximum de vraisemblance
- VAE (Variational Autoencoders)

### Divergence de Jensen-Shannon

```
JS(P || Q) = (1/2)KL(P || M) + (1/2)KL(Q || M)
```

où M = (P + Q)/2

**Propriétés** :
- Symétrique
- 0 ≤ JS ≤ log 2

---

## 12.3 Distance de Wasserstein

**Définition** (1-Wasserstein) :
```
W_1(P, Q) = inf_{γ ∈ Γ(P,Q)} ∫∫ d(x,y) dγ(x,y)
```

**Interprétation** : Coût minimal pour transformer P en Q.

**Applications** :
- GANs (Wasserstein GAN)
- Optimal transport

---

## 12.4 Distances Duales

**f-divergence** : Famille générale de divergences.

---

[⬅️ Partie 3](../partie-3-apprentissage-supervise/chapitre-11-reseaux-neurones.md) | [Retour](../README.md) | [Suite ➡️](./chapitre-13-monte-carlo.md)

