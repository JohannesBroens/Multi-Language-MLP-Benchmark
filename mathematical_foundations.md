# Mathematical Foundations

This document describes the mathematics behind the MLP implementations in this project. All implementations (C, NumPy, PyTorch) use the same algorithm described here.

## Table of Contents

- [Feature Normalization](#feature-normalization)
- [MLP Architecture](#mlp-architecture)
- [Weight Initialization](#weight-initialization)
- [Forward Propagation](#forward-propagation)
- [Activation Functions](#activation-functions)
- [Numerically Stable Softmax](#numerically-stable-softmax)
- [Loss Function](#loss-function)
- [Backpropagation](#backpropagation)
- [Mini-Batch Gradient Descent](#mini-batch-gradient-descent)
- [Training Algorithm Pseudo-Code](#training-algorithm-pseudo-code)
- [Notation Reference](#notation-reference)

---

## Feature Normalization

Before training, all input features are z-score normalized per feature:

```math
x'_j = \frac{x_j - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}
```

Where:
- $\mu_j = \frac{1}{N} \sum_{i=1}^{N} x_{ij}$ is the feature mean
- $\sigma_j^2 = \frac{1}{N} \sum_{i=1}^{N} (x_{ij} - \mu_j)^2$ is the feature variance
- $\epsilon = 10^{-8}$ prevents division by zero for constant features

This centers each feature around zero with unit variance, which helps gradient descent converge faster by ensuring all features are on a similar scale.

---

## MLP Architecture

A single hidden-layer MLP with:

- **Input layer**: $n$ features
- **Hidden layer**: $m$ neurons with ReLU activation
- **Output layer**: $k$ neurons with softmax activation (one per class)

```math
\begin{align*}
\mathbf{h} &= \text{ReLU}\left(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}\right) \\
\hat{\mathbf{y}} &= \text{softmax}\left(\mathbf{W}^{(2)} \mathbf{h} + \mathbf{b}^{(2)}\right)
\end{align*}
```

Where:
- $\mathbf{x} \in \mathbb{R}^{n}$: input vector
- $\mathbf{W}^{(1)} \in \mathbb{R}^{m \times n}$: input-to-hidden weights
- $\mathbf{b}^{(1)} \in \mathbb{R}^{m}$: hidden biases (initialized to zero)
- $\mathbf{W}^{(2)} \in \mathbb{R}^{k \times m}$: hidden-to-output weights
- $\mathbf{b}^{(2)} \in \mathbb{R}^{k}$: output biases (initialized to zero)

---

## Weight Initialization

Weights are initialized with **Xavier/He uniform** initialization:

```math
W_{ij} \sim \text{Uniform}\left(-\sqrt{\frac{2}{n_{\text{in}}}},\; \sqrt{\frac{2}{n_{\text{in}}}}\right)
```

Where $n_{\text{in}}$ is the fan-in (number of input connections to the layer). This scale factor prevents vanishing or exploding activations in networks with ReLU, by keeping the variance of activations approximately constant across layers.

Biases are initialized to zero.

---

## Forward Propagation

### Hidden Layer

```math
\begin{align*}
\mathbf{z}^{(1)} &= \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)} \\
\mathbf{h} &= \text{ReLU}(\mathbf{z}^{(1)})
\end{align*}
```

### Output Layer

```math
\begin{align*}
\mathbf{z}^{(2)} &= \mathbf{W}^{(2)} \mathbf{h} + \mathbf{b}^{(2)} \\
\hat{\mathbf{y}} &= \text{softmax}(\mathbf{z}^{(2)})
\end{align*}
```

---

## Activation Functions

### ReLU (Rectified Linear Unit)

Used in the hidden layer:

```math
\text{ReLU}(z) = \max(0, z)
```

Derivative:

```math
\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}
```

ReLU avoids the vanishing gradient problem of sigmoid/tanh and is computationally cheap.

### Softmax

Used in the output layer to produce a probability distribution over classes:

```math
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}
```

---

## Numerically Stable Softmax

Direct computation of $e^{z_i}$ can overflow for large logits. The implementation uses the **max subtraction trick**:

```math
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j=1}^{k} e^{z_j - \max(\mathbf{z})}}
```

This is mathematically equivalent (the constant cancels) but ensures all exponents are $\leq 0$, preventing overflow while preserving numerical precision.

---

## Loss Function

**Cross-entropy loss** for multi-class classification:

```math
L = -\sum_{i=1}^{k} y_i \log(\hat{y}_i + \epsilon)
```

For one-hot encoded labels $\mathbf{y}$ where only the target class $c$ is 1, this simplifies to:

```math
L = -\log(\hat{y}_c + \epsilon)
```

The epsilon ($\epsilon = 10^{-7}$) prevents $\log(0)$.

---

## Backpropagation

### Output Layer Gradient (Softmax + Cross-Entropy Shortcut)

When combining softmax activation with cross-entropy loss, the gradient simplifies elegantly:

```math
\delta^{(2)} = \hat{\mathbf{y}} - \mathbf{y}
```

This avoids computing separate softmax and cross-entropy derivatives. **Proof**: By the chain rule, $\frac{\partial L}{\partial z_i^{(2)}} = \sum_j \frac{\partial L}{\partial \hat{y}_j} \frac{\partial \hat{y}_j}{\partial z_i^{(2)}}$. For cross-entropy loss and softmax, using $\frac{\partial \hat{y}_j}{\partial z_i} = \hat{y}_i(\delta_{ij} - \hat{y}_j)$, this reduces to $\hat{y}_i - y_i$.

### Hidden Layer Gradient (Backprop through ReLU)

```math
\delta^{(1)} = \left((\mathbf{W}^{(2)})^\top \delta^{(2)}\right) \odot \text{ReLU}'(\mathbf{z}^{(1)})
```

Since $\text{ReLU}'(z) = \mathbb{1}[z > 0]$, this zeros out gradients for inactive neurons.

### Weight Gradients

```math
\begin{align*}
\nabla_{\mathbf{W}^{(2)}} L &= \delta^{(2)} \mathbf{h}^\top \\
\nabla_{\mathbf{b}^{(2)}} L &= \delta^{(2)} \\
\nabla_{\mathbf{W}^{(1)}} L &= \delta^{(1)} \mathbf{x}^\top \\
\nabla_{\mathbf{b}^{(1)}} L &= \delta^{(1)}
\end{align*}
```

---

## Mini-Batch Gradient Descent

The implementations use **mini-batch SGD**: the training set is divided into batches of size $B$, and weights are updated after processing each batch.

### Full-Batch vs Mini-Batch vs Stochastic GD

| Method | Batch Size | Trade-off |
|--------|-----------|-----------|
| Full-batch GD | $B = N$ | Exact gradient, slow per update, can get stuck |
| Mini-batch SGD | $1 < B < N$ | Good balance of noise and efficiency |
| Stochastic GD | $B = 1$ | Noisy gradient, fast per update, good regularization |

### Gradient Averaging

Per-sample gradients are accumulated over the batch, then averaged by dividing the learning rate:

```math
\eta_{\text{scaled}} = \frac{\eta}{|B|}
```

```math
\theta \leftarrow \theta - \eta_{\text{scaled}} \sum_{i \in B} \nabla_\theta L_i
```

This is equivalent to $\theta \leftarrow \theta - \eta \cdot \frac{1}{|B|}\sum_{i \in B} \nabla_\theta L_i$ but avoids a separate division step.

---

## Training Algorithm Pseudo-Code

```
Input: dataset {(x_i, y_i)}, epochs T, learning rate η, batch size B
Initialize: W1, b1, W2, b2 with Xavier uniform / zeros

For epoch = 1 to T:
    For each batch B ⊂ {1, ..., N}:
        // Forward pass (per sample in batch)
        z1 = X_B @ W1 + b1
        h  = ReLU(z1)
        z2 = h @ W2 + b2
        ŷ  = stable_softmax(z2)

        // Loss
        L = -mean(log(ŷ[targets] + ε))

        // Backward pass
        δ2 = ŷ - one_hot(targets)           // softmax+CE shortcut
        δ1 = (δ2 @ W2.T) * (h > 0)         // backprop through ReLU

        // Accumulate and average gradients
        η_scaled = η / |B|
        W2 -= η_scaled * (h.T @ δ2)
        b2 -= η_scaled * sum(δ2)
        W1 -= η_scaled * (X_B.T @ δ1)
        b1 -= η_scaled * sum(δ1)
```

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x} \in \mathbb{R}^{n}$ | Input vector |
| $\mathbf{y} \in \mathbb{R}^{k}$ | True label (one-hot encoded) |
| $\hat{\mathbf{y}} \in \mathbb{R}^{k}$ | Predicted probabilities (softmax output) |
| $\mathbf{h} \in \mathbb{R}^{m}$ | Hidden layer activations |
| $\mathbf{W}^{(1)}, \mathbf{W}^{(2)}$ | Weight matrices |
| $\mathbf{b}^{(1)}, \mathbf{b}^{(2)}$ | Bias vectors |
| $\delta^{(1)}, \delta^{(2)}$ | Error signals for backpropagation |
| $\eta$ | Learning rate |
| $B$ | Mini-batch |
| $\odot$ | Element-wise (Hadamard) product |
| $N$ | Total number of training samples |
| $n, m, k$ | Input size, hidden size, output size |
