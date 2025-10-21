# Chapitre 19 : Méthodes Génératives Profondes

## 📚 Introduction

Ce chapitre présente les techniques modernes de modélisation générative basées sur les réseaux de neurones profonds.

---

## 19.1 Flots Normalisants (Normalizing Flows)

### Principe

Transformer une distribution simple z ~ N(0, I) en distribution complexe x = f(z).

**Changement de variable** :
```
p_x(x) = p_z(f⁻¹(x)) |det ∂f⁻¹/∂x|
```

**Architectures** : RealNVP, Glow, MAF

---

## 19.2 Autoencodeurs Variationnels (VAE)

### Architecture

**Encodeur** : q(z|x) = N(μ(x), Σ(x))
**Décodeur** : p(x|z)

### Fonction Objectif

**ELBO** :
```
ℒ = 𝔼_q[log p(x|z)] - KL(q(z|x) || p(z))
```

```python
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Encodeur
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        
        # Décodeur
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, input_dim)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

---

## 19.3 GANs (Generative Adversarial Networks)

### Principe

**Deux réseaux** :
- Générateur G : z → x
- Discriminateur D : x → [0, 1]

### Objectif Min-Max

```
min_G max_D 𝔼_x[log D(x)] + 𝔼_z[log(1 - D(G(z)))]
```

```python
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, np.prod(img_shape)),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(img_shape), 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.model(img)

# Entraînement
for epoch in range(n_epochs):
    for real_imgs in dataloader:
        # Entraîner discriminateur
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = generator(z)
        
        d_loss = -torch.mean(torch.log(discriminator(real_imgs)) + 
                             torch.log(1 - discriminator(fake_imgs)))
        
        # Entraîner générateur
        g_loss = -torch.mean(torch.log(discriminator(generator(z))))
```

### Variantes

**DCGAN** : Convolutions
**WGAN** : Wasserstein distance
**StyleGAN** : Style transfer

---

## 19.4 Modèles de Diffusion

### Principe

**Forward** : Ajouter progressivement du bruit
**Reverse** : Apprendre à débruiter

**Score matching** : Apprendre ∇_x log p(x)

---

[⬅️ Partie 4](../partie-4-modeles-probabilistes/chapitre-18-apprentissage-graphiques.md) | [Retour](../README.md) | [Suite ➡️](../partie-6-non-supervise/chapitre-20-clustering.md)

