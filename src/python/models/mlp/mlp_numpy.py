#!/usr/bin/env python3
"""NumPy MLP implementation — exact replica of the C algorithm.

Multi-hidden-layer MLP, ReLU activation, softmax output, cross-entropy loss.
Xavier uniform initialization, mini-batch SGD or Adam with gradient averaging.

Usage:
    python src/python/models/mlp/mlp_numpy.py --dataset iris
"""

import argparse
import math
import time
import sys
import os


def _detect_physical_cores():
    """Detect physical core count (not hyperthreaded logical cores).

    Hyperthreaded cores share the FPU, so for compute-bound matmul
    using only physical cores avoids contention and improves throughput.
    """
    try:
        with open("/proc/cpuinfo") as f:
            core_ids = set()
            for line in f:
                if line.startswith("core id"):
                    core_ids.add(line.strip())
            if core_ids:
                return len(core_ids)
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


# Set BLAS thread count to physical cores BEFORE importing NumPy.
# OpenBLAS reads these env vars at import time.
if "OPENBLAS_NUM_THREADS" not in os.environ:
    _cores = str(_detect_physical_cores())
    os.environ["OPENBLAS_NUM_THREADS"] = _cores
    os.environ["MKL_NUM_THREADS"] = _cores

import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from python.utils.data_utils import load_dataset, normalize_features, shuffle_and_split


def cosine_lr(epoch, total, lr_max, warmup, lr_min=1e-6):
    """Cosine annealing with linear warmup."""
    if epoch < warmup:
        return lr_max * (epoch + 1) / warmup
    progress = (epoch - warmup) / (total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def xavier_init(fan_in, fan_out, rng):
    """Xavier uniform init matching C: scale = sqrt(2/fan_in), range [-scale, +scale]."""
    scale = np.sqrt(2.0 / fan_in)
    return rng.uniform(-scale, scale, size=(fan_in, fan_out)).astype(np.float32)


def relu(z):
    return np.maximum(0, z)


def softmax(logits):
    """Numerically stable softmax (subtract max per sample)."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def forward(X, layers):
    """Forward pass storing all activations for backprop.

    layers: list of (W, b) tuples
    Returns: list of hidden activations + final softmax output
    """
    acts = []  # hidden activations (after ReLU)
    current = X
    for i, (W, b) in enumerate(layers):
        z = current @ W + b
        if i < len(layers) - 1:
            current = relu(z)
            acts.append(current)
        else:
            current = softmax(z)
    return acts, current


def forward_eval(X, layers):
    """Forward pass without storing activations (for evaluation)."""
    current = X
    for i, (W, b) in enumerate(layers):
        z = current @ W + b
        if i < len(layers) - 1:
            current = relu(z)
        else:
            current = softmax(z)
    return current


def cross_entropy_loss(output, targets, num_classes):
    """Average cross-entropy loss: -mean(log(p[target] + eps))."""
    eps = 1e-7
    n = len(targets)
    log_probs = -np.log(output[np.arange(n), targets] + eps)
    return log_probs.sum() / n


def train(X_train, y_train, input_size, num_classes,
          hidden_size=64, num_hidden_layers=1, batch_size=32, num_epochs=1000,
          learning_rate=0.01, scheduler="none", optimizer="sgd"):
    """Train multi-layer MLP with mini-batch SGD or Adam."""
    rng = np.random.default_rng()

    # Build layer sizes and initialize weights
    layer_sizes = [input_size] + [hidden_size] * num_hidden_layers + [num_classes]
    layers = []
    for i in range(len(layer_sizes) - 1):
        W = xavier_init(layer_sizes[i], layer_sizes[i + 1], rng)
        b = np.zeros(layer_sizes[i + 1], dtype=np.float32)
        layers.append((W, b))

    nl = len(layers)
    n = len(X_train)
    num_batches = (n + batch_size - 1) // batch_size
    warmup = max(1, int(num_epochs * 0.05)) if scheduler == "cosine" else 0

    # Adam moment buffers
    if optimizer == "adam":
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        adam_m_W = [np.zeros_like(layers[i][0]) for i in range(nl)]
        adam_v_W = [np.zeros_like(layers[i][0]) for i in range(nl)]
        adam_m_b = [np.zeros_like(layers[i][1]) for i in range(nl)]
        adam_v_b = [np.zeros_like(layers[i][1]) for i in range(nl)]
        adam_step = 0

    for epoch in range(num_epochs):
        lr = cosine_lr(epoch, num_epochs, learning_rate, warmup) if scheduler == "cosine" else learning_rate
        epoch_loss = 0.0

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n)
            bs = end - start

            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            # Forward
            acts, output = forward(X_batch, layers)

            # Loss
            eps = 1e-7
            batch_loss = -np.log(output[np.arange(bs), y_batch] + eps).sum()
            epoch_loss += batch_loss

            # Backward: softmax + CE gradient shortcut
            one_hot = np.zeros_like(output)
            one_hot[np.arange(bs), y_batch] = 1.0
            d_current = output - one_hot  # (batch, output)

            # Backprop through all layers
            grad_W_list = [None] * nl
            grad_b_list = [None] * nl

            for layer in range(nl - 1, -1, -1):
                input_act = X_batch if layer == 0 else acts[layer - 1]
                W, b = layers[layer]

                grad_W_list[layer] = input_act.T @ d_current
                grad_b_list[layer] = d_current.sum(axis=0)

                if layer > 0:
                    d_current = (d_current @ W.T) * (acts[layer - 1] > 0)

            # Parameter update
            lr_scaled = lr / bs
            if optimizer == "adam":
                adam_step += 1
                bc1 = 1.0 - beta1 ** adam_step
                bc2 = 1.0 - beta2 ** adam_step
                for i in range(nl):
                    W, b = layers[i]
                    adam_m_W[i] = beta1 * adam_m_W[i] + (1 - beta1) * grad_W_list[i]
                    adam_v_W[i] = beta2 * adam_v_W[i] + (1 - beta2) * grad_W_list[i] ** 2
                    m_hat = adam_m_W[i] / bc1
                    v_hat = adam_v_W[i] / bc2
                    W -= lr_scaled * m_hat / (np.sqrt(v_hat) + eps_adam)

                    adam_m_b[i] = beta1 * adam_m_b[i] + (1 - beta1) * grad_b_list[i]
                    adam_v_b[i] = beta2 * adam_v_b[i] + (1 - beta2) * grad_b_list[i] ** 2
                    m_hat = adam_m_b[i] / bc1
                    v_hat = adam_v_b[i] / bc2
                    b -= lr_scaled * m_hat / (np.sqrt(v_hat) + eps_adam)

                    layers[i] = (W, b)
            else:
                for i in range(nl):
                    W, b = layers[i]
                    W -= lr_scaled * grad_W_list[i]
                    b -= lr_scaled * grad_b_list[i]
                    layers[i] = (W, b)

        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4d}  loss: {epoch_loss / n:.4f}")

    return layers


def evaluate(X, y, layers, num_classes):
    """Evaluate model, returning average loss and accuracy percentage."""
    output = forward_eval(X, layers)
    loss = cross_entropy_loss(output, y, num_classes)
    predictions = output.argmax(axis=1)
    accuracy = (predictions == y).mean() * 100.0
    return loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="NumPy MLP")
    parser.add_argument("--dataset", required=True,
                        choices=["generated", "iris", "wine-red", "wine-white", "breast-cancer"])
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Number of samples for generated dataset (0 = default)")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-hidden-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    args = parser.parse_args()

    num_epochs = args.epochs
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    learning_rate = args.learning_rate

    X, y, num_classes = load_dataset(args.dataset, num_samples=args.num_samples)
    X = normalize_features(X)
    X_train, y_train, X_test, y_test = shuffle_and_split(X, y)

    input_size = X.shape[1]
    print(f"Dataset: {args.dataset}  ({len(X)} samples, {input_size} features, {num_classes} classes)")
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"\nTraining ({num_epochs} epochs, batch_size={batch_size}, hidden={hidden_size}, layers={args.num_hidden_layers}, lr={learning_rate:.4f})...")

    t_start = time.monotonic()
    layers = train(X_train, y_train, input_size, num_classes,
                   hidden_size=hidden_size, num_hidden_layers=args.num_hidden_layers,
                   batch_size=batch_size, num_epochs=num_epochs,
                   learning_rate=learning_rate,
                   scheduler=args.scheduler, optimizer=args.optimizer)
    t_train = time.monotonic() - t_start

    t_eval_start = time.monotonic()
    loss, accuracy = evaluate(X_test, y_test, layers, num_classes)
    t_eval = time.monotonic() - t_eval_start

    throughput = len(X_train) * num_epochs / t_train

    print(f"\n=== Results on {args.dataset} ===")
    print(f"Test Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Train time:    {t_train:.3f} s")
    print(f"Eval time:     {t_eval:.3f} s")
    print(f"Throughput:    {throughput:.0f} samples/s")


if __name__ == "__main__":
    main()
