#!/usr/bin/env python3
"""NumPy MLP implementation — exact replica of the C algorithm.

Single hidden layer, ReLU activation, softmax output, cross-entropy loss.
Xavier uniform initialization, mini-batch SGD with gradient averaging.

Usage:
    python src/python/models/mlp/mlp_numpy.py --dataset iris
"""

import argparse
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


def forward(X, W1, b1, W2, b2):
    """Forward pass for a batch of samples.

    Returns hidden activations and softmax output probabilities.
    """
    z1 = X @ W1 + b1        # (batch, hidden)
    hidden = relu(z1)        # (batch, hidden)
    z2 = hidden @ W2 + b2   # (batch, output)
    output = softmax(z2)     # (batch, output)
    return hidden, output


def cross_entropy_loss(output, targets, num_classes):
    """Average cross-entropy loss: -mean(log(p[target] + eps))."""
    eps = 1e-7
    n = len(targets)
    log_probs = -np.log(output[np.arange(n), targets] + eps)
    return log_probs.sum() / n


def train(X_train, y_train, input_size, num_classes,
          hidden_size=64, batch_size=32, num_epochs=1000, learning_rate=0.01):
    """Train MLP with mini-batch SGD, matching C implementation exactly."""
    rng = np.random.default_rng()

    W1 = xavier_init(input_size, hidden_size, rng)
    b1 = np.zeros(hidden_size, dtype=np.float32)
    W2 = xavier_init(hidden_size, num_classes, rng)
    b2 = np.zeros(num_classes, dtype=np.float32)

    n = len(X_train)
    num_batches = (n + batch_size - 1) // batch_size

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n)
            bs = end - start

            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            # Forward
            hidden, output = forward(X_batch, W1, b1, W2, b2)

            # Loss
            eps = 1e-7
            batch_loss = -np.log(output[np.arange(bs), y_batch] + eps).sum()
            epoch_loss += batch_loss

            # Backward: softmax + CE gradient shortcut
            one_hot = np.zeros_like(output)
            one_hot[np.arange(bs), y_batch] = 1.0
            d_output = output - one_hot  # (batch, output)

            # Hidden gradient: backprop through ReLU
            d_hidden = (d_output @ W2.T) * (hidden > 0)  # (batch, hidden)

            # Accumulate gradients
            grad_W2 = hidden.T @ d_output      # (hidden, output)
            grad_b2 = d_output.sum(axis=0)     # (output,)
            grad_W1 = X_batch.T @ d_hidden     # (input, hidden)
            grad_b1 = d_hidden.sum(axis=0)     # (hidden,)

            # SGD update with gradient averaging (lr / batch_size)
            lr_scaled = learning_rate / bs
            W1 -= lr_scaled * grad_W1
            b1 -= lr_scaled * grad_b1
            W2 -= lr_scaled * grad_W2
            b2 -= lr_scaled * grad_b2

        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4d}  loss: {epoch_loss / n:.4f}")

    return W1, b1, W2, b2


def evaluate(X, y, W1, b1, W2, b2, num_classes):
    """Evaluate model, returning average loss and accuracy percentage."""
    hidden, output = forward(X, W1, b1, W2, b2)
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
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.02)
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
    print(f"\nTraining ({num_epochs} epochs, batch_size={batch_size}, hidden={hidden_size}, lr={learning_rate:.4f})...")

    t_start = time.monotonic()
    W1, b1, W2, b2 = train(X_train, y_train, input_size, num_classes,
                            hidden_size=hidden_size, batch_size=batch_size,
                            num_epochs=num_epochs, learning_rate=learning_rate)
    t_train = time.monotonic() - t_start

    t_eval_start = time.monotonic()
    loss, accuracy = evaluate(X_test, y_test, W1, b1, W2, b2, num_classes)
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
