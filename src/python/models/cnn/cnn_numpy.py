#!/usr/bin/env python3
"""NumPy CNN (LeNet-5) implementation -- exact replica of the C/Rust algorithm.

Conv1(1->6, 5x5) -> ReLU -> AvgPool(2x2)
Conv2(6->16, 5x5) -> ReLU -> AvgPool(2x2)
FC1(256->120) -> ReLU -> FC2(120->84) -> ReLU -> Output(84->10) -> Softmax

im2col approach for convolution, Xavier uniform init, mini-batch SGD.

Usage:
    python src/python/models/cnn/cnn_numpy.py --dataset mnist --epochs 3
"""

import argparse
import time
import sys
import os

def _detect_physical_cores():
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

if "OPENBLAS_NUM_THREADS" not in os.environ:
    _cores = str(_detect_physical_cores())
    os.environ["OPENBLAS_NUM_THREADS"] = _cores
    os.environ["MKL_NUM_THREADS"] = _cores

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from python.utils.data_utils import load_dataset, shuffle_and_split

# LeNet-5 architecture constants
IN_C, IN_H, IN_W = 1, 28, 28
C1_OUT, C1_K, C1_OH, C1_OW = 6, 5, 24, 24
C1_COL_ROWS, C1_COL_COLS = 25, 576  # 1*5*5, 24*24
P1_SIZE, P1_H, P1_W = 2, 12, 12
C2_OUT, C2_IN, C2_K, C2_OH, C2_OW = 16, 6, 5, 8, 8
C2_COL_ROWS, C2_COL_COLS = 150, 64  # 6*5*5, 8*8
P2_SIZE = 2
FLAT_SIZE = 256  # 16*4*4
FC1_OUT = 120
FC2_OUT = 84
NUM_CLASSES = 10
LEARNING_RATE = 0.32


def im2col(input_data, C, H, W, kH, kW, stride, OH, OW):
    """Convert input patches to columns for GEMM-based convolution.

    input_data: (C, H, W)
    Returns: (C*kH*kW, OH*OW)
    """
    col = np.empty((C * kH * kW, OH * OW), dtype=np.float32)
    col_row = 0
    for c in range(C):
        for kh in range(kH):
            for kw in range(kW):
                for oh in range(OH):
                    for ow in range(OW):
                        h = oh * stride + kh
                        w = ow * stride + kw
                        col[col_row, oh * OW + ow] = input_data[c, h, w]
                col_row += 1
    return col


def im2col_fast(input_data, C, H, W, kH, kW, stride, OH, OW):
    """Vectorized im2col using np.lib.stride_tricks."""
    img = input_data.reshape(C, H, W)
    # Build index arrays
    i0 = np.repeat(np.arange(kH), kW)
    i0 = np.tile(i0, C)
    i1 = stride * np.arange(OH)
    j0 = np.tile(np.arange(kW), kH * C)
    j1 = stride * np.arange(OW)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)   # (C*kH*kW, OH)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)    # (C*kH*kW, OW)

    c_idx = np.repeat(np.arange(C), kH * kW)

    # Advanced indexing
    col = img[c_idx.reshape(-1, 1, 1), i.reshape(-1, OH, 1), j.reshape(-1, 1, OW)]
    return col.reshape(C * kH * kW, OH * OW)


def col2im(col, C, H, W, kH, kW, stride, OH, OW):
    """Scatter col matrix back to image (accumulate overlapping patches)."""
    output = np.zeros((C, H, W), dtype=np.float32)
    col_row = 0
    for c in range(C):
        for kh in range(kH):
            for kw in range(kW):
                for oh in range(OH):
                    for ow in range(OW):
                        h = oh * stride + kh
                        w = ow * stride + kw
                        output[c, h, w] += col[col_row, oh * OW + ow]
                col_row += 1
    return output


def col2im_fast(col, C, H, W, kH, kW, stride, OH, OW):
    """Vectorized col2im."""
    output = np.zeros((C, H, W), dtype=np.float32)
    i0 = np.repeat(np.arange(kH), kW)
    i0 = np.tile(i0, C)
    i1 = stride * np.arange(OH)
    j0 = np.tile(np.arange(kW), kH * C)
    j1 = stride * np.arange(OW)

    c_idx = np.repeat(np.arange(C), kH * kW)

    col_reshaped = col.reshape(C * kH * kW, OH, OW)
    for r in range(C * kH * kW):
        for oh in range(OH):
            for ow in range(OW):
                output[c_idx[r], i0[r] + i1[oh], j0[r] + j1[ow]] += col_reshaped[r, oh, ow]
    return output


def avg_pool_forward(x, C, H, W, pool_size):
    """Average pooling forward. x: (C, H, W) -> (C, H//pool, W//pool)."""
    OH, OW = H // pool_size, W // pool_size
    out = np.empty((C, OH, OW), dtype=np.float32)
    scale = 1.0 / (pool_size * pool_size)
    for c in range(C):
        for oh in range(OH):
            for ow in range(OW):
                s = 0.0
                for ph in range(pool_size):
                    for pw in range(pool_size):
                        s += x[c, oh * pool_size + ph, ow * pool_size + pw]
                out[c, oh, ow] = s * scale
    return out


def avg_pool_forward_fast(x, C, H, W, pool_size):
    """Vectorized average pooling."""
    OH, OW = H // pool_size, W // pool_size
    x_r = x.reshape(C, OH, pool_size, OW, pool_size)
    return x_r.mean(axis=(2, 4)).astype(np.float32)


def avg_pool_backward(grad_out, C, H, W, pool_size):
    """Distribute gradient evenly to all pooled elements."""
    OH, OW = H // pool_size, W // pool_size
    grad_in = np.zeros((C, H, W), dtype=np.float32)
    scale = 1.0 / (pool_size * pool_size)
    for c in range(C):
        for oh in range(OH):
            for ow in range(OW):
                for ph in range(pool_size):
                    for pw in range(pool_size):
                        grad_in[c, oh * pool_size + ph, ow * pool_size + pw] = \
                            grad_out[c, oh, ow] * scale
    return grad_in


def avg_pool_backward_fast(grad_out, C, H, W, pool_size):
    """Vectorized average pooling backward."""
    OH, OW = H // pool_size, W // pool_size
    scale = 1.0 / (pool_size * pool_size)
    grad_in = np.repeat(np.repeat(grad_out, pool_size, axis=1), pool_size, axis=2)
    return (grad_in * scale).astype(np.float32)


def xavier_init(fan_in, fan_out, rng):
    scale = np.sqrt(2.0 / fan_in)
    return rng.uniform(-scale, scale, size=(fan_in, fan_out)).astype(np.float32)


def relu(z):
    return np.maximum(0, z)


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


class LeNet5:
    """LeNet-5 CNN with im2col GEMM convolutions."""

    def __init__(self, rng):
        # Conv weights: (out_channels, in_c * kH * kW) for GEMM
        self.conv1_w = xavier_init(C1_COL_ROWS, C1_OUT, rng).T.copy()  # (6, 25)
        self.conv1_b = np.zeros(C1_OUT, dtype=np.float32)
        self.conv2_w = xavier_init(C2_COL_ROWS, C2_OUT, rng).T.copy()  # (16, 150)
        self.conv2_b = np.zeros(C2_OUT, dtype=np.float32)

        # FC weights
        self.fc1_w = xavier_init(FLAT_SIZE, FC1_OUT, rng)     # (256, 120)
        self.fc1_b = np.zeros(FC1_OUT, dtype=np.float32)
        self.fc2_w = xavier_init(FC1_OUT, FC2_OUT, rng)       # (120, 84)
        self.fc2_b = np.zeros(FC2_OUT, dtype=np.float32)
        self.out_w = xavier_init(FC2_OUT, NUM_CLASSES, rng)    # (84, 10)
        self.out_b = np.zeros(NUM_CLASSES, dtype=np.float32)

    def conv_forward_sample(self, img):
        """Forward pass through conv layers for a single sample.

        img: (1, 28, 28) -> returns (pool2_flat, cache)
        cache contains intermediates needed for backward pass.
        """
        # Conv1: im2col -> GEMM -> add bias -> ReLU -> pool
        col1 = im2col_fast(img, IN_C, IN_H, IN_W, C1_K, C1_K, 1, C1_OH, C1_OW)
        conv1_out = self.conv1_w @ col1  # (6, 576)
        conv1_out += self.conv1_b.reshape(-1, 1)
        conv1_pre = conv1_out.copy()  # before ReLU
        conv1_out = np.maximum(0, conv1_out)
        conv1_3d = conv1_out.reshape(C1_OUT, C1_OH, C1_OW)
        pool1 = avg_pool_forward_fast(conv1_3d, C1_OUT, C1_OH, C1_OW, P1_SIZE)

        # Conv2
        col2 = im2col_fast(pool1, C2_IN, P1_H, P1_W, C2_K, C2_K, 1, C2_OH, C2_OW)
        conv2_out = self.conv2_w @ col2  # (16, 64)
        conv2_out += self.conv2_b.reshape(-1, 1)
        conv2_pre = conv2_out.copy()
        conv2_out = np.maximum(0, conv2_out)
        conv2_3d = conv2_out.reshape(C2_OUT, C2_OH, C2_OW)
        pool2 = avg_pool_forward_fast(conv2_3d, C2_OUT, C2_OH, C2_OW, P2_SIZE)

        flat = pool2.ravel()
        cache = (col1, conv1_pre, conv1_3d, pool1, col2, conv2_pre, conv2_3d)
        return flat, cache

    def forward_batch(self, X_batch):
        """Full forward pass for a batch. X_batch: (bs, 784)."""
        bs = X_batch.shape[0]
        flats = np.empty((bs, FLAT_SIZE), dtype=np.float32)
        caches = []

        for b in range(bs):
            img = X_batch[b].reshape(IN_C, IN_H, IN_W)
            flat, cache = self.conv_forward_sample(img)
            flats[b] = flat
            caches.append(cache)

        # FC layers (batched)
        fc1_z = flats @ self.fc1_w + self.fc1_b
        fc1_out = relu(fc1_z)
        fc2_z = fc1_out @ self.fc2_w + self.fc2_b
        fc2_out = relu(fc2_z)
        out_z = fc2_out @ self.out_w + self.out_b
        output = softmax(out_z)

        return output, flats, fc1_out, fc2_out, caches

    def backward_and_update(self, X_batch, y_batch, output, flats, fc1_out, fc2_out, caches, lr):
        """Backward pass and SGD update."""
        bs = X_batch.shape[0]

        # Output gradient
        one_hot = np.zeros_like(output)
        one_hot[np.arange(bs), y_batch] = 1.0
        d_out = output - one_hot  # (bs, 10)

        # FC backward (batched)
        g_out_w = fc2_out.T @ d_out        # (84, 10)
        g_out_b = d_out.sum(axis=0)

        d_fc2 = (d_out @ self.out_w.T) * (fc2_out > 0)
        g_fc2_w = fc1_out.T @ d_fc2        # (120, 84)
        g_fc2_b = d_fc2.sum(axis=0)

        d_fc1 = (d_fc2 @ self.fc2_w.T) * (fc1_out > 0)
        g_fc1_w = flats.T @ d_fc1          # (256, 120)
        g_fc1_b = d_fc1.sum(axis=0)

        d_flat = d_fc1 @ self.fc1_w.T      # (bs, 256)

        # Conv backward (per-sample, accumulate)
        g_conv2_w = np.zeros_like(self.conv2_w)
        g_conv2_b = np.zeros_like(self.conv2_b)
        g_conv1_w = np.zeros_like(self.conv1_w)
        g_conv1_b = np.zeros_like(self.conv1_b)

        for b in range(bs):
            col1, conv1_pre, conv1_3d, pool1, col2, conv2_pre, conv2_3d = caches[b]
            d_pool2 = d_flat[b].reshape(C2_OUT, C2_OH // P2_SIZE, C2_OW // P2_SIZE)

            # Pool2 backward -> conv2 backward
            d_conv2_out = avg_pool_backward_fast(d_pool2, C2_OUT, C2_OH, C2_OW, P2_SIZE)
            d_conv2_out = d_conv2_out.reshape(C2_OUT, C2_COL_COLS)
            d_conv2_out *= (conv2_pre > 0)  # ReLU mask

            g_conv2_w += d_conv2_out @ col2.T   # (16, 150)
            g_conv2_b += d_conv2_out.sum(axis=1)

            # Propagate to pool1
            d_col2 = self.conv2_w.T @ d_conv2_out  # (150, 64)
            d_pool1 = col2im_fast(d_col2, C2_IN, P1_H, P1_W, C2_K, C2_K, 1, C2_OH, C2_OW)

            # Pool1 backward -> conv1 backward
            d_conv1_out = avg_pool_backward_fast(d_pool1, C1_OUT, C1_OH, C1_OW, P1_SIZE)
            d_conv1_out = d_conv1_out.reshape(C1_OUT, C1_COL_COLS)
            d_conv1_out *= (conv1_pre > 0)

            g_conv1_w += d_conv1_out @ col1.T   # (6, 25)
            g_conv1_b += d_conv1_out.sum(axis=1)

        # SGD update
        lr_s = lr / bs
        self.out_w -= lr_s * g_out_w
        self.out_b -= lr_s * g_out_b
        self.fc2_w -= lr_s * g_fc2_w
        self.fc2_b -= lr_s * g_fc2_b
        self.fc1_w -= lr_s * g_fc1_w
        self.fc1_b -= lr_s * g_fc1_b
        self.conv2_w -= lr_s * g_conv2_w
        self.conv2_b -= lr_s * g_conv2_b
        self.conv1_w -= lr_s * g_conv1_w
        self.conv1_b -= lr_s * g_conv1_b


def train(model, X_train, y_train, batch_size, num_epochs, learning_rate):
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

            output, flats, fc1_out, fc2_out, caches = model.forward_batch(X_batch)

            # Cross-entropy loss
            eps = 1e-7
            batch_loss = -np.log(output[np.arange(bs), y_batch] + eps).sum()
            epoch_loss += batch_loss

            model.backward_and_update(X_batch, y_batch, output, flats, fc1_out, fc2_out, caches, learning_rate)

        print(f"  Epoch {epoch:4d}  loss: {epoch_loss / n:.4f}")


def evaluate(model, X_test, y_test):
    output, _, _, _, _ = model.forward_batch(X_test)
    eps = 1e-7
    n = len(y_test)
    loss = -np.log(output[np.arange(n), y_test] + eps).sum() / n
    preds = output.argmax(axis=1)
    accuracy = (preds == y_test).mean() * 100.0
    return loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="NumPy CNN (LeNet-5)")
    parser.add_argument("--dataset", required=True, choices=["mnist"])
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.32)
    args = parser.parse_args()

    X, y, num_classes = load_dataset(args.dataset)
    X_train, y_train, X_test, y_test = shuffle_and_split(X, y)

    learning_rate = args.learning_rate

    print(f"Dataset: {args.dataset}  ({len(X)} samples, {X.shape[1]} features, {num_classes} classes)")
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"\nTraining LeNet-5 ({args.epochs} epochs, batch_size={args.batch_size}, lr={learning_rate:.4f})...")

    rng = np.random.default_rng()
    model = LeNet5(rng)

    t_start = time.monotonic()
    train(model, X_train, y_train, args.batch_size, args.epochs, learning_rate)
    t_train = time.monotonic() - t_start

    t_eval_start = time.monotonic()
    loss, accuracy = evaluate(model, X_test, y_test)
    t_eval = time.monotonic() - t_eval_start

    throughput = len(X_train) * args.epochs / t_train

    print(f"\n=== Results on {args.dataset} ===")
    print(f"Test Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Train time:    {t_train:.3f} s")
    print(f"Eval time:     {t_eval:.3f} s")
    print(f"Throughput:    {throughput:.0f} samples/s")


if __name__ == "__main__":
    main()
