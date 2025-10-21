# Chapitre 11 : Réseaux de Neurones

## 📚 Introduction

Les réseaux de neurones sont la base du deep learning moderne. Ce chapitre couvre les fondements architecturaux et algorithmiques.

---

## 11.1 Définitions de Base

### Neurone Artificiel

Un **neurone** calcule :
```
z = w₁x₁ + w₂x₂ + ... + wₚxₚ + b
a = σ(z)
```

où σ est une **fonction d'activation**.

### Fonctions d'Activation

**Sigmoïde** :
```
σ(z) = 1 / (1 + e⁻ᶻ)
```

**Tanh** :
```
tanh(z) = (e^z - e⁻ᶻ) / (e^z + e⁻ᶻ)
```

**ReLU** (Rectified Linear Unit) :
```
ReLU(z) = max(0, z)
```

**Leaky ReLU** :
```
LeakyReLU(z) = max(α·z, z)  où α ≈ 0.01
```

---

## 11.2 Architecture des Réseaux

### 11.2.1 Réseau Multicouche

**Couche cachée l** :
```
z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = σ(z⁽ˡ⁾)
```

### 11.2.2 Couche de Sortie

**Régression** : Sortie linéaire
**Classification binaire** : Sigmoïde
**Classification multi-classe** : Softmax
```
P(Y = k|x) = exp(z_k) / Σⱼ exp(z_j)
```

### 11.2.3 Réseaux Convolutifs (CNN)

**Convolution** :
```
(f * g)[i] = Σⱼ f[j] · g[i - j]
```

**Pooling** : Max pooling, average pooling

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 14 * 14, 10)
    
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = x.view(-1, 32 * 14 * 14)
        x = self.fc(x)
        return x
```

---

## 11.3 Fonction Objectif

### Perte pour la Régression

**MSE** :
```
L = (1/n) Σᵢ (yᵢ - ŷᵢ)²
```

### Perte pour la Classification

**Cross-entropie** :
```
L = -(1/n) Σᵢ Σₖ yᵢₖ log(ŷᵢₖ)
```

**Avec PyTorch** :
```python
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, targets)
```

---

## 11.4 Rétropropagation

### Algorithme

**Forward pass** : Calculer les activations
**Backward pass** : Calculer les gradients

**Règle de la chaîne** :
```
∂L/∂W⁽ˡ⁾ = (∂L/∂z⁽ˡ⁾) · (∂z⁽ˡ⁾/∂W⁽ˡ⁾)
```

**Implémentation** :
```python
import torch

# Réseau simple
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Entraînement
model = SimpleNN(784, 128, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # Forward
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 11.5 Techniques d'Entraînement

### 11.5.1 Mini-Batches

**SGD par mini-batches** :
```python
batch_size = 32
dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=batch_size,
    shuffle=True
)
```

### 11.5.2 Dropout

**Régularisation** : Désactiver aléatoirement des neurones pendant l'entraînement.

```python
class NNWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Désactivé en mode eval()
        x = self.fc2(x)
        return x
```

### Batch Normalization

```python
class NNWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
```

---

## 11.6 Architectures Modernes

### ResNet (Residual Networks)

**Connexions résiduelles** :
```
F(x) = H(x) - x
H(x) = F(x) + x
```

### Transformers

**Attention** :
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

---

## 💡 Best Practices

1. **Initialisation** : Xavier/He initialization
2. **Learning rate** : Learning rate schedule
3. **Régularisation** : Dropout, L2, Batch norm
4. **Architecture** : Commencer simple, complexifier progressivement
5. **Monitoring** : TensorBoard, visualisation des pertes

```python
# Exemple complet avec TensorBoard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = validate(model, val_loader)
    
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)

writer.close()
```

---

[⬅️ Chapitre précédent](./chapitre-10-arbres.md) | [Retour](../README.md) | [Suite ➡️](../partie-4-modeles-probabilistes/chapitre-12-comparaison-distributions.md)

