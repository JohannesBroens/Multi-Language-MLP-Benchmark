#!/usr/bin/env python3
"""PyTorch CNN (LeNet-5) implementation -- same architecture as C/Rust.

Conv1(1->6, 5x5) -> ReLU -> AvgPool(2x2)
Conv2(6->16, 5x5) -> ReLU -> AvgPool(2x2)
FC1(256->120) -> ReLU -> FC2(120->84) -> ReLU -> Output(84->10) -> Softmax

Manual Xavier uniform init to match C's sqrt(2/fan_in) scale.

Usage:
    python src/python/models/cnn/cnn_pytorch.py --dataset mnist --epochs 3 --device cpu
    python src/python/models/cnn/cnn_pytorch.py --dataset mnist --epochs 3 --device cuda
"""

import argparse
import time
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from python.utils.data_utils import load_dataset, shuffle_and_split

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("PyTorch not installed. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)


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


_phys_cores = _detect_physical_cores()
torch.set_num_threads(_phys_cores)
torch.set_num_interop_threads(min(_phys_cores, 4))

LEARNING_RATE = 0.01


class LeNet5(nn.Module):
    """LeNet-5 CNN matching C/Rust architecture exactly."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=True)
        self.fc1 = nn.Linear(256, 120)  # 16*4*4 = 256
        self.fc2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

        # Xavier uniform init matching C: scale = sqrt(2/fan_in)
        # Conv fan_in = in_channels * kH * kW
        fan_in_c1 = 1 * 5 * 5   # = 25
        scale_c1 = (2.0 / fan_in_c1) ** 0.5
        nn.init.uniform_(self.conv1.weight, -scale_c1, scale_c1)
        nn.init.zeros_(self.conv1.bias)

        fan_in_c2 = 6 * 5 * 5   # = 150
        scale_c2 = (2.0 / fan_in_c2) ** 0.5
        nn.init.uniform_(self.conv2.weight, -scale_c2, scale_c2)
        nn.init.zeros_(self.conv2.bias)

        scale_fc1 = (2.0 / 256) ** 0.5
        nn.init.uniform_(self.fc1.weight, -scale_fc1, scale_fc1)
        nn.init.zeros_(self.fc1.bias)

        scale_fc2 = (2.0 / 120) ** 0.5
        nn.init.uniform_(self.fc2.weight, -scale_fc2, scale_fc2)
        nn.init.zeros_(self.fc2.bias)

        scale_out = (2.0 / 84) ** 0.5
        nn.init.uniform_(self.out.weight, -scale_out, scale_out)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = self.pool(torch.relu(self.conv1(x)))   # -> (batch, 6, 12, 12)
        x = self.pool(torch.relu(self.conv2(x)))   # -> (batch, 16, 4, 4)
        x = x.view(x.size(0), -1)                  # -> (batch, 256)
        x = torch.relu(self.fc1(x))                # -> (batch, 120)
        x = torch.relu(self.fc2(x))                # -> (batch, 84)
        x = self.out(x)                            # -> (batch, 10) raw logits
        return x


def train_model(model, X_train, y_train, device, batch_size, num_epochs):
    model.train()

    # Reshape to (N, 1, 28, 28) for conv
    X_tensor = torch.from_numpy(X_train.reshape(-1, 1, 28, 28)).to(device, non_blocking=True)
    y_tensor = torch.from_numpy(y_train).long().to(device, non_blocking=True)

    criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    n = len(X_train)
    num_batches = (n + batch_size - 1) // batch_size

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n)

            optimizer.zero_grad(set_to_none=True)
            logits = model(X_tensor[start:end])
            loss = criterion(logits, y_tensor[start:end])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * (end - start)

        print(f"  Epoch {epoch:4d}  loss: {epoch_loss / n:.4f}")


def evaluate_model(model, X, y, device):
    model.eval()
    with torch.inference_mode():
        X_tensor = torch.from_numpy(X.reshape(-1, 1, 28, 28)).to(device)
        y_tensor = torch.from_numpy(y).long().to(device)

        logits = model(X_tensor)

        # Match C: -log(softmax(logit)[target] + eps)
        probs = torch.softmax(logits, dim=1)
        eps = 1e-7
        log_probs = -torch.log(probs[torch.arange(len(y), device=device), y_tensor] + eps)
        loss = log_probs.sum().item() / len(y)

        predictions = logits.argmax(dim=1)
        accuracy = (predictions == y_tensor).float().mean().item() * 100.0

    return loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="PyTorch CNN (LeNet-5)")
    parser.add_argument("--dataset", required=True, choices=["mnist"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU", file=sys.stderr)
        args.device = "cpu"

    device = torch.device(args.device)

    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    X, y, num_classes = load_dataset(args.dataset)
    X_train, y_train, X_test, y_test = shuffle_and_split(X, y)

    print(f"Dataset: {args.dataset}  ({len(X)} samples, {X.shape[1]} features, {num_classes} classes)")
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"\nTraining LeNet-5 ({args.epochs} epochs, batch_size={args.batch_size}, lr={LEARNING_RATE:.4f})...")

    model = LeNet5().to(device)

    # CUDA warmup
    if args.device == "cuda":
        dummy_x = torch.randn(args.batch_size, 1, 28, 28, device=device)
        dummy_y = torch.zeros(args.batch_size, dtype=torch.long, device=device)
        criterion = nn.CrossEntropyLoss(reduction="mean")
        for _ in range(3):
            out = model(dummy_x)
            loss_warmup = criterion(out, dummy_y)
            loss_warmup.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    if args.device == "cuda":
        torch.cuda.synchronize()
    t_start = time.monotonic()
    train_model(model, X_train, y_train, device, args.batch_size, args.epochs)
    if args.device == "cuda":
        torch.cuda.synchronize()
    t_train = time.monotonic() - t_start

    if args.device == "cuda":
        torch.cuda.synchronize()
    t_eval_start = time.monotonic()
    loss, accuracy = evaluate_model(model, X_test, y_test, device)
    if args.device == "cuda":
        torch.cuda.synchronize()
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
