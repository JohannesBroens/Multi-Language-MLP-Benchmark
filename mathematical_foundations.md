# Mathematical Foundations

This document describes the mathematics behind the neural network implementations in this project. All implementations (C, Rust, NumPy, PyTorch) use the same algorithms described here.

## Table of Contents

### Shared Concepts
- [Feature Normalization](#feature-normalization)
- [Activation Functions](#activation-functions)
- [Numerically Stable Softmax](#numerically-stable-softmax)
- [Loss Function](#loss-function)
- [Weight Initialization](#weight-initialization)
- [Mini-Batch Gradient Descent](#mini-batch-gradient-descent)

### Multi-Layer Perceptron (MLP)
- [MLP Architecture](#mlp-architecture)
- [MLP Forward Propagation](#mlp-forward-propagation)
- [MLP Backpropagation](#mlp-backpropagation)
- [MLP Training Algorithm](#mlp-training-algorithm)

### Convolutional Neural Network (CNN — LeNet-5)
- [CNN Architecture](#cnn-architecture)
- [Convolution via im2col](#convolution-via-im2col)
- [Average Pooling](#average-pooling)
- [CNN Forward Propagation](#cnn-forward-propagation)
- [CNN Backpropagation](#cnn-backpropagation)
- [CNN Training Algorithm](#cnn-training-algorithm)

### Reference
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

---

# Multi-Layer Perceptron (MLP)

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

## MLP Forward Propagation

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

## MLP Backpropagation

### Output Layer Gradient (Softmax + Cross-Entropy Shortcut)

When combining softmax activation with cross-entropy loss, the gradient simplifies elegantly:

```math
\delta^{(2)} = \hat{\mathbf{y}} - \mathbf{y}
```

This avoids computing separate softmax and cross-entropy derivatives. 

**Proof**: By the chain rule, 
```math
\frac{\partial L}{\partial z_i^{(2)}} = \sum_j \frac{\partial L}{\partial \hat{y}_j} \frac{\partial \hat{y}_j}{\partial z_i^{(2)}}
```
For cross-entropy loss and softmax, using
```math
\frac{\partial \hat{y}_j}{\partial z_i}=\hat{y}_i(\delta_{ij}-\hat{y}_j),
```
this reduces to $\hat{y}_i - y_i$. 

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

## MLP Training Algorithm

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

# Convolutional Neural Network (CNN — LeNet-5)

## CNN Architecture

A LeNet-5 variant for MNIST digit classification (28×28 grayscale images, 10 classes):

```
Input (1×28×28)
  ↓
Conv1 (6 filters, 5×5, stride 1, no padding) + ReLU → 6×24×24
  ↓
AvgPool (2×2) → 6×12×12
  ↓
Conv2 (16 filters, 5×5, stride 1, no padding) + ReLU → 16×8×8
  ↓
AvgPool (2×2) → 16×4×4
  ↓
Flatten → 256
  ↓
FC1 (256→120) + ReLU
  ↓
FC2 (120→84) + ReLU
  ↓
Output (84→10) + Softmax
```

| Layer | Input Shape | Kernel/Units | Output Shape | Parameters |
|-------|-------------|-------------|--------------|------------|
| Conv1 | $1 \times 28 \times 28$ | $6$ filters of $5 \times 5$ | $6 \times 24 \times 24$ | $6 \cdot (1 \cdot 25 + 1) = 156$ |
| Pool1 | $6 \times 24 \times 24$ | $2 \times 2$ average | $6 \times 12 \times 12$ | 0 |
| Conv2 | $6 \times 12 \times 12$ | $16$ filters of $5 \times 5$ | $16 \times 8 \times 8$ | $16 \cdot (6 \cdot 25 + 1) = 2416$ |
| Pool2 | $16 \times 8 \times 8$ | $2 \times 2$ average | $16 \times 4 \times 4$ | 0 |
| FC1 | 256 | 120 | 120 | $256 \cdot 120 + 120 = 30840$ |
| FC2 | 120 | 84 | 84 | $120 \cdot 84 + 84 = 10164$ |
| Output | 84 | 10 | 10 | $84 \cdot 10 + 10 = 850$ |

**Total: 44,426 trainable parameters.**

All data uses NCHW layout: $\text{batch} \times \text{channels} \times \text{height} \times \text{width}$.

---

## Convolution via im2col

Direct convolution requires six nested loops (batch, output channel, output height, output width, input channel, kernel height, kernel width). The **im2col** trick converts convolution into matrix multiplication, enabling reuse of optimized BLAS routines.

### im2col Transform

Given an input $\mathbf{X} \in \mathbb{R}^{C_{\text{in}} \times H \times W}$, a kernel of size $k_H \times k_W$, stride $s$, and output spatial dimensions $O_H \times O_W$:

```math
O_H = \frac{H - k_H}{s} + 1, \quad O_W = \frac{W - k_W}{s} + 1
```

The im2col operation extracts every $C_{\text{in}} \times k_H \times k_W$ patch from $\mathbf{X}$ and arranges them as columns of a matrix $\mathbf{C} \in \mathbb{R}^{(C_{\text{in}} \cdot k_H \cdot k_W) \times (O_H \cdot O_W)}$:

```math
\mathbf{C}[c \cdot k_H \cdot k_W + p \cdot k_W + q,\; h \cdot O_W + w] = \mathbf{X}[c,\; h \cdot s + p,\; w \cdot s + q]
```

where $c \in [0, C_{\text{in}})$, $p \in [0, k_H)$, $q \in [0, k_W)$, $h \in [0, O_H)$, $w \in [0, O_W)$.

### Convolution as GEMM

With filter weights reshaped as $\mathbf{F} \in \mathbb{R}^{C_{\text{out}} \times (C_{\text{in}} \cdot k_H \cdot k_W)}$, convolution becomes:

```math
\mathbf{Y} = \mathbf{F} \cdot \mathbf{C} + \mathbf{b}
```

where $\mathbf{Y} \in \mathbb{R}^{C_{\text{out}} \times (O_H \cdot O_W)}$ and $\mathbf{b} \in \mathbb{R}^{C_{\text{out}}}$ is broadcast across columns.

**Concrete sizes in this network:**

| Layer | $\mathbf{F}$ shape | $\mathbf{C}$ shape | $\mathbf{Y}$ shape |
|-------|----------|----------|----------|
| Conv1 | $6 \times 25$ | $25 \times 576$ | $6 \times 576$ |
| Conv2 | $16 \times 150$ | $150 \times 64$ | $16 \times 64$ |

### col2im (Inverse Transform)

During backpropagation, gradients w.r.t. the column matrix must be scattered back to the input spatial layout. Because im2col duplicates overlapping pixels, col2im **accumulates** (sums) gradients at each original position:

```math
\frac{\partial L}{\partial \mathbf{X}}[c,\; h \cdot s + p,\; w \cdot s + q] \mathrel{+}= \frac{\partial L}{\partial \mathbf{C}}[c \cdot k_H \cdot k_W + p \cdot k_W + q,\; h \cdot O_W + w]
```

---

## Average Pooling

### Forward

For a $p \times p$ pooling window with stride $p$ (non-overlapping):

```math
\mathbf{Y}[c, i, j] = \frac{1}{p^2} \sum_{m=0}^{p-1} \sum_{n=0}^{p-1} \mathbf{X}[c,\; i \cdot p + m,\; j \cdot p + n]
```

Output spatial size: $H_{\text{out}} = H / p$, $W_{\text{out}} = W / p$.

### Backward

Each output gradient distributes equally to all positions in its pooling window:

```math
\frac{\partial L}{\partial \mathbf{X}}[c,\; i \cdot p + m,\; j \cdot p + n] = \frac{1}{p^2} \frac{\partial L}{\partial \mathbf{Y}}[c, i, j]
```

---

## CNN Forward Propagation

The forward pass processes convolutions per-sample (each sample needs its own im2col workspace), then batches the fully connected layers as matrix multiplications.

### Convolutional Layers (per sample)

```math
\begin{align*}
\mathbf{C}_1 &= \text{im2col}(\mathbf{X},\; k=5,\; s=1) & \in \mathbb{R}^{25 \times 576} \\
\mathbf{Z}_1 &= \mathbf{F}_1 \mathbf{C}_1 + \mathbf{b}_1 & \in \mathbb{R}^{6 \times 576} \\
\mathbf{A}_1 &= \text{ReLU}(\mathbf{Z}_1) \\
\mathbf{P}_1 &= \text{AvgPool}_{2 \times 2}(\text{reshape}(\mathbf{A}_1, 6 \times 24 \times 24)) & \in \mathbb{R}^{6 \times 12 \times 12} \\[6pt]
\mathbf{C}_2 &= \text{im2col}(\mathbf{P}_1,\; k=5,\; s=1) & \in \mathbb{R}^{150 \times 64} \\
\mathbf{Z}_2 &= \mathbf{F}_2 \mathbf{C}_2 + \mathbf{b}_2 & \in \mathbb{R}^{16 \times 64} \\
\mathbf{A}_2 &= \text{ReLU}(\mathbf{Z}_2) \\
\mathbf{P}_2 &= \text{AvgPool}_{2 \times 2}(\text{reshape}(\mathbf{A}_2, 16 \times 8 \times 8)) & \in \mathbb{R}^{16 \times 4 \times 4} \\
\mathbf{f} &= \text{flatten}(\mathbf{P}_2) & \in \mathbb{R}^{256}
\end{align*}
```

### Fully Connected Layers (batched)

With $\mathbf{F}_B \in \mathbb{R}^{B \times 256}$ as the batch of flattened features:

```math
\begin{align*}
\mathbf{H}_1 &= \text{ReLU}(\mathbf{F}_B \mathbf{W}_1 + \mathbf{b}^{(1)}) & \in \mathbb{R}^{B \times 120} \\
\mathbf{H}_2 &= \text{ReLU}(\mathbf{H}_1 \mathbf{W}_2 + \mathbf{b}^{(2)}) & \in \mathbb{R}^{B \times 84} \\
\hat{\mathbf{Y}} &= \text{softmax}(\mathbf{H}_2 \mathbf{W}_3 + \mathbf{b}^{(3)}) & \in \mathbb{R}^{B \times 10}
\end{align*}
```

---

## CNN Backpropagation

### FC Layers (batched, same as MLP)

```math
\begin{align*}
\boldsymbol{\delta}_3 &= \hat{\mathbf{Y}} - \mathbf{Y}_{\text{one-hot}} & \text{(softmax + CE shortcut)} \\
\nabla_{\mathbf{W}_3} &= \mathbf{H}_2^\top \boldsymbol{\delta}_3, \quad \nabla_{\mathbf{b}^{(3)}} = \mathbf{1}^\top \boldsymbol{\delta}_3 \\[4pt]
\boldsymbol{\delta}_2 &= (\boldsymbol{\delta}_3 \mathbf{W}_3^\top) \odot \text{ReLU}'(\mathbf{H}_2) \\
\nabla_{\mathbf{W}_2} &= \mathbf{H}_1^\top \boldsymbol{\delta}_2, \quad \nabla_{\mathbf{b}^{(2)}} = \mathbf{1}^\top \boldsymbol{\delta}_2 \\[4pt]
\boldsymbol{\delta}_1 &= (\boldsymbol{\delta}_2 \mathbf{W}_2^\top) \odot \text{ReLU}'(\mathbf{H}_1) \\
\nabla_{\mathbf{W}_1} &= \mathbf{F}_B^\top \boldsymbol{\delta}_1, \quad \nabla_{\mathbf{b}^{(1)}} = \mathbf{1}^\top \boldsymbol{\delta}_1 \\[4pt]
\mathbf{d}_{\text{flat}} &= \boldsymbol{\delta}_1 \mathbf{W}_1^\top & \in \mathbb{R}^{B \times 256}
\end{align*}
```

### Convolutional Layers (per sample, reverse order)

For each sample, reshape $\mathbf{d}_{\text{flat}} \in \mathbb{R}^{256}$ to $16 \times 4 \times 4$:

```math
\begin{align*}
\mathbf{d}_{A_2} &= \text{AvgPool\_backward}(\mathbf{d}_{P_2}) \odot \text{ReLU}'(\mathbf{Z}_2) \\
\nabla_{\mathbf{F}_2} &\mathrel{+}= \mathbf{d}_{A_2} \mathbf{C}_2^\top \\
\nabla_{\mathbf{b}_2} &\mathrel{+}= \text{col\_sum}(\mathbf{d}_{A_2}) \\
\mathbf{d}_{P_1} &= \text{col2im}(\mathbf{F}_2^\top \mathbf{d}_{A_2}) \\[6pt]
\mathbf{d}_{A_1} &= \text{AvgPool\_backward}(\mathbf{d}_{P_1}) \odot \text{ReLU}'(\mathbf{Z}_1) \\
\nabla_{\mathbf{F}_1} &\mathrel{+}= \mathbf{d}_{A_1} \mathbf{C}_1^\top \\
\nabla_{\mathbf{b}_1} &\mathrel{+}= \text{col\_sum}(\mathbf{d}_{A_1})
\end{align*}
```

Gradients accumulate ($\mathrel{+}=$) across all samples in the batch before the SGD update.

### Weight Update

Same mini-batch SGD as MLP:

```math
\theta \leftarrow \theta - \frac{\eta}{|B|} \sum_{i \in B} \nabla_\theta L_i
```

---

## CNN Training Algorithm

```
Input: MNIST {(X_i, y_i)}, epochs T, learning rate η, batch size B
Initialize: F1, b1, F2, b2, W1..W3, b(1)..b(3) with Xavier uniform / zeros

For epoch = 1 to T:
    For each batch B ⊂ {1, ..., N}:

        // Per-sample convolutional forward
        For each sample in batch:
            C1 = im2col(X, k=5, s=1)               // 25 × 576
            Z1 = F1 @ C1 + b1;  A1 = ReLU(Z1)      // 6 × 576
            P1 = avg_pool(A1, 2)                     // 6 × 12 × 12
            C2 = im2col(P1, k=5, s=1)               // 150 × 64
            Z2 = F2 @ C2 + b2;  A2 = ReLU(Z2)      // 16 × 64
            P2 = avg_pool(A2, 2)                     // 16 × 4 × 4
            flat[sample] = flatten(P2)               // 256

        // Batched FC forward
        H1 = ReLU(flat @ W1 + b(1))                 // B × 120
        H2 = ReLU(H1 @ W2 + b(2))                   // B × 84
        Ŷ  = softmax(H2 @ W3 + b(3))                // B × 10

        // Loss
        L = -mean(log(Ŷ[targets] + ε))

        // FC backward
        δ3 = Ŷ - one_hot(targets)
        ∇W3 = H2.T @ δ3;   ∇b3 = sum(δ3)
        δ2 = (δ3 @ W3.T) ⊙ (H2 > 0)
        ∇W2 = H1.T @ δ2;   ∇b2 = sum(δ2)
        δ1 = (δ2 @ W2.T) ⊙ (H1 > 0)
        ∇W1 = flat.T @ δ1;  ∇b1 = sum(δ1)
        d_flat = δ1 @ W1.T

        // Per-sample conv backward
        For each sample in batch:
            d_P2 = reshape(d_flat[sample], 16×4×4)
            d_A2 = pool_backward(d_P2) ⊙ (Z2 > 0)
            ∇F2 += d_A2 @ C2.T;  ∇b2_conv += col_sum(d_A2)
            d_C2 = F2.T @ d_A2;  d_P1 = col2im(d_C2)
            d_A1 = pool_backward(d_P1) ⊙ (Z1 > 0)
            ∇F1 += d_A1 @ C1.T;  ∇b1_conv += col_sum(d_A1)

        // SGD update
        η_s = η / |B|
        All params -= η_s × gradients
```

### Workspace Memory

Each sample requires temporary buffers for im2col and intermediate activations:

| Buffer | Size (floats) | Purpose |
|--------|---------------|---------|
| col1 | $25 \times 576 = 14{,}400$ | im2col for Conv1 |
| conv1_out | $6 \times 576 = 3{,}456$ | Conv1 pre-activation |
| pool1 | $6 \times 144 = 864$ | Pool1 output |
| col2 | $150 \times 64 = 9{,}600$ | im2col for Conv2 |
| conv2_out | $16 \times 64 = 1{,}024$ | Conv2 pre-activation |
| pool2 | 256 | Pool2 output (= flat) |
| **Total** | **29,600** | **~116 KB per sample** |

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x} \in \mathbb{R}^{n}$ | Input vector (MLP) or image tensor (CNN) |
| $\mathbf{y} \in \mathbb{R}^{k}$ | True label (one-hot encoded) |
| $\hat{\mathbf{y}} \in \mathbb{R}^{k}$ | Predicted probabilities (softmax output) |
| $\mathbf{h} \in \mathbb{R}^{m}$ | Hidden layer activations |
| $\mathbf{W}, \mathbf{F}$ | Weight matrices (FC layers, conv filters) |
| $\mathbf{b}$ | Bias vectors |
| $\boldsymbol{\delta}$ | Error signals for backpropagation |
| $\mathbf{C}$ | im2col column matrix |
| $\eta$ | Learning rate |
| $B$ | Mini-batch size |
| $\odot$ | Element-wise (Hadamard) product |
| $\mathrel{+}=$ | Accumulate (sum into existing value) |
| $N$ | Total number of training samples |
| $n, m, k$ | Input size, hidden size, number of classes |
| $C_{\text{in}}, C_{\text{out}}$ | Input/output channels (CNN) |
| $k_H, k_W$ | Kernel height and width |
| $O_H, O_W$ | Output spatial dimensions |
| $s$ | Convolution stride |
| $p$ | Pooling window size |
