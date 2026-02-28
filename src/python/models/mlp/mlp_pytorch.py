#!/usr/bin/env python3
"""PyTorch MLP implementation — same architecture and hyperparameters as C.

Single hidden layer, ReLU activation, softmax output, cross-entropy loss.
Manual Xavier uniform initialization to match C's sqrt(2/fan_in) scale.

Usage:
    python src/python/models/mlp/mlp_pytorch.py --dataset iris --device cpu
    python src/python/models/mlp/mlp_pytorch.py --dataset iris --device cuda
"""

import argparse
import time
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from python.utils.data_utils import load_dataset, normalize_features, shuffle_and_split

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("PyTorch not installed. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)


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


# Set thread count to physical cores (not logical/HT) for optimal GEMM
_phys_cores = _detect_physical_cores()
torch.set_num_threads(_phys_cores)
torch.set_num_interop_threads(min(_phys_cores, 4))


class MLP(nn.Module):
    """Single hidden-layer MLP matching C architecture."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Manual Xavier init matching C: scale = sqrt(2/fan_in), uniform [-scale, +scale]
        scale_ih = (2.0 / input_size) ** 0.5
        nn.init.uniform_(self.fc1.weight, -scale_ih, scale_ih)
        nn.init.zeros_(self.fc1.bias)

        scale_ho = (2.0 / hidden_size) ** 0.5
        nn.init.uniform_(self.fc2.weight, -scale_ho, scale_ho)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h)  # raw logits — CrossEntropyLoss applies softmax


def train_model(model, X_train, y_train, device,
                batch_size=32, num_epochs=1000, learning_rate=0.01,
                optimizer_name="sgd", scheduler_name="none"):
    """Train with mini-batch SGD matching C batch structure.

    Key optimizations vs naive loop:
    - reduction='mean' avoids extra loss/bs division in backward graph
    - loss.item() only on print epochs to avoid CUDA synchronization
    """
    model.train()

    X_tensor = torch.from_numpy(X_train).to(device, non_blocking=True)
    y_tensor = torch.from_numpy(y_train).long().to(device, non_blocking=True)

    # reduction='mean' gives gradient/batch_size, matching C's lr_scaled = lr/batch_size
    criterion = nn.CrossEntropyLoss(reduction="mean")
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    lr_scheduler = None
    if scheduler_name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_epochs = max(1, int(num_epochs * 0.05))
        warmup_sched = LinearLR(optimizer, start_factor=1.0 / warmup_epochs, total_iters=warmup_epochs)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6)
        lr_scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs])

    n = len(X_train)
    num_batches = (n + batch_size - 1) // batch_size

    for epoch in range(num_epochs):
        should_print = (epoch % 100 == 0)
        epoch_loss = 0.0

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n)

            optimizer.zero_grad(set_to_none=True)
            logits = model(X_tensor[start:end])
            loss = criterion(logits, y_tensor[start:end])
            loss.backward()
            optimizer.step()

            # Only synchronize on print epochs to avoid blocking GPU pipeline
            if should_print:
                epoch_loss += loss.item() * (end - start)

        if lr_scheduler is not None:
            lr_scheduler.step()

        if should_print:
            print(f"  Epoch {epoch:4d}  loss: {epoch_loss / n:.4f}")


def evaluate_model(model, X, y, device):
    """Evaluate model, returning average loss and accuracy percentage."""
    model.eval()
    with torch.inference_mode():
        X_tensor = torch.from_numpy(X).to(device)
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
    parser = argparse.ArgumentParser(description="PyTorch MLP")
    parser.add_argument("--dataset", required=True,
                        choices=["generated", "iris", "wine-red", "wine-white", "breast-cancer"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Number of samples for generated dataset (0 = default)")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU", file=sys.stderr)
        args.device = "cpu"

    device = torch.device(args.device)

    num_epochs = args.epochs
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    learning_rate = args.learning_rate

    # Hardware-adaptive optimizations
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True
        # TF32 on Ampere+ GPUs: ~2x matmul throughput, 10-bit mantissa (vs 23-bit)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    X, y, num_classes = load_dataset(args.dataset, num_samples=args.num_samples)
    X = normalize_features(X)
    X_train, y_train, X_test, y_test = shuffle_and_split(X, y)

    input_size = X.shape[1]
    print(f"Dataset: {args.dataset}  ({len(X)} samples, {input_size} features, {num_classes} classes)")
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"\nTraining ({num_epochs} epochs, batch_size={batch_size}, hidden={hidden_size}, lr={learning_rate:.4f})...")

    model = MLP(input_size, hidden_size, num_classes).to(device)

    # CUDA warmup: trigger JIT compilation before timing
    if args.device == "cuda":
        dummy_x = torch.randn(batch_size, input_size, device=device)
        dummy_y = torch.zeros(batch_size, dtype=torch.long, device=device)
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
    train_model(model, X_train, y_train, device,
                batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate,
                optimizer_name=args.optimizer, scheduler_name=args.scheduler)
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

    throughput = len(X_train) * num_epochs / t_train

    print(f"\n=== Results on {args.dataset} ===")
    print(f"Test Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Train time:    {t_train:.3f} s")
    print(f"Eval time:     {t_eval:.3f} s")
    print(f"Throughput:    {throughput:.0f} samples/s")


if __name__ == "__main__":
    main()
