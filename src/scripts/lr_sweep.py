#!/usr/bin/env python3
"""Cross-language hyperparameter sweep tool.

Sweeps any numeric CLI parameter (learning-rate, batch-size, hidden-size, epochs)
across any implementation by calling binaries via subprocess and parsing the
standardized output format.

Imports implementation definitions from benchmark.py so any new implementation
is automatically available for sweeping.

Usage:
    # Sweep LR for one implementation
    python src/scripts/lr_sweep.py --impl "C (CUDA)" --param learning-rate \
        --values "0.01,0.1,0.32,1.0" --epochs 20 --dataset mnist --model cnn

    # Sweep ALL implementations
    python src/scripts/lr_sweep.py --impl all --param learning-rate \
        --values "0.1,0.32,1.0" --epochs 20 --model cnn

    # Multi-seed for robustness
    python src/scripts/lr_sweep.py --impl "C (CUDA)" --param learning-rate \
        --values "0.1,0.32,0.5" --seeds 3 --epochs 20 --model cnn
"""

import argparse
import sys
import os

import numpy as np

# Import from benchmark.py
sys.path.insert(0, os.path.dirname(__file__))
from benchmark import (
    IMPLEMENTATIONS, CNN_IMPLEMENTATIONS,
    run_implementation, is_available,
)


def find_impl(name, model):
    """Find implementation by label (case-insensitive partial match)."""
    impl_list = CNN_IMPLEMENTATIONS if model == "cnn" else IMPLEMENTATIONS
    name_lower = name.lower()

    # Exact match first
    for label, cmd in impl_list:
        if label.lower() == name_lower:
            return label, cmd

    # Partial match
    for label, cmd in impl_list:
        if name_lower in label.lower():
            return label, cmd

    return None, None


def sweep_single(label, cmd_template, param, values, base_kwargs, seeds=1):
    """Sweep a single parameter for one implementation.

    Returns list of dicts: [{value, loss, accuracy, throughput, train_time, seeds_data}, ...]
    """
    results = []

    for val in values:
        kwargs = dict(base_kwargs)
        kwargs[param.replace("-", "_")] = val
        dataset = kwargs.pop("dataset", "generated")

        seed_results = []
        for _ in range(seeds):
            r = run_implementation(label, cmd_template, dataset, **kwargs)
            if r is not None:
                seed_results.append(r)

        if seed_results:
            accs = [r["accuracy"] for r in seed_results]
            losses = [r["loss"] for r in seed_results]
            times = [r["train_time"] for r in seed_results]
            throughputs = [r.get("throughput", 0) for r in seed_results]

            results.append({
                "value": val,
                "loss": np.mean(losses),
                "loss_std": np.std(losses) if len(losses) > 1 else 0,
                "accuracy": np.mean(accs),
                "acc_std": np.std(accs) if len(accs) > 1 else 0,
                "throughput": np.mean(throughputs),
                "train_time": np.mean(times),
                "n": len(seed_results),
            })
        else:
            results.append({
                "value": val,
                "loss": float("nan"),
                "loss_std": 0,
                "accuracy": float("nan"),
                "acc_std": 0,
                "throughput": 0,
                "train_time": 0,
                "n": 0,
            })

    return results


def print_results(label, param, results, seeds):
    """Print formatted results table."""
    best_idx = -1
    best_acc = -1
    for i, r in enumerate(results):
        if not np.isnan(r["accuracy"]) and r["accuracy"] > best_acc:
            best_acc = r["accuracy"]
            best_idx = i

    # Header
    val_header = param.upper().replace("-", "_")
    if seeds > 1:
        print(f"  {val_header:>12s}  {'Loss':>10s}  {'Accuracy':>12s}  {'Throughput':>12s}  {'Time':>8s}  {'N':>3s}")
        print("-" * 70)
    else:
        print(f"  {val_header:>12s}  {'Loss':>10s}  {'Accuracy':>10s}  {'Throughput':>12s}  {'Time':>8s}")
        print("-" * 65)

    for i, r in enumerate(results):
        marker = "  <-- best" if i == best_idx else ""

        if r["n"] == 0:
            print(f"  {r['value']:12g}  {'FAILED':>10s}  {'---':>10s}  {'---':>12s}  {'---':>8s}")
            continue

        if seeds > 1:
            acc_str = f"{r['accuracy']:6.2f}+/-{r['acc_std']:.2f}%"
            print(f"  {r['value']:12g}  {r['loss']:10.4f}  {acc_str:>12s}  {r['throughput']:>10,.0f}  {r['train_time']:7.1f}s  {r['n']:3d}{marker}")
        else:
            print(f"  {r['value']:12g}  {r['loss']:10.4f}  {r['accuracy']:9.2f}%  {r['throughput']:>10,.0f}  {r['train_time']:7.1f}s{marker}")

    if seeds > 1:
        print("-" * 70)
    else:
        print("-" * 65)

    if best_idx >= 0:
        best = results[best_idx]
        print(f"Best: {param}={best['value']}, Accuracy={best['accuracy']:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-language hyperparameter sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Sweep LR for C (CUDA) CNN
  python src/scripts/lr_sweep.py --impl "C (CUDA)" --param learning-rate \\
      --values "0.1,0.32,1.0" --epochs 20 --model cnn

  # Sweep all implementations
  python src/scripts/lr_sweep.py --impl all --param learning-rate \\
      --values "0.1,0.32,1.0" --epochs 20 --model cnn

  # Sweep batch size for Rust CPU MLP
  python src/scripts/lr_sweep.py --impl "Rust (CPU)" --param batch-size \\
      --values "256,1024,4096,8192" --epochs 200 --model mlp --dataset generated
""")
    parser.add_argument("--impl", required=True,
                        help="Implementation name or 'all' for all available")
    parser.add_argument("--param", required=True,
                        help="Parameter to sweep (learning-rate, batch-size, hidden-size, epochs)")
    parser.add_argument("--values", required=True,
                        help="Comma-separated values to sweep")
    parser.add_argument("--model", default="mlp", choices=["mlp", "cnn"],
                        help="Model type (default: mlp)")
    parser.add_argument("--dataset", default=None,
                        help="Dataset (default: mnist for cnn, generated for mlp)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: 4096 for cnn, 8192 for mlp)")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (default: 20 for cnn, 200 for mlp)")
    parser.add_argument("--num-samples", type=int, default=262144,
                        help="Number of samples for generated dataset")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Base learning rate (default: model-specific)")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of runs per value for robustness (default: 1)")

    args = parser.parse_args()

    # Set model-specific defaults
    is_cnn = args.model == "cnn"
    if args.dataset is None:
        args.dataset = "mnist" if is_cnn else "generated"
    if args.batch_size is None:
        args.batch_size = 4096
    if args.epochs is None:
        args.epochs = 20 if is_cnn else 200
    if args.learning_rate is None:
        args.learning_rate = 0.32 if is_cnn else 0.02

    # Parse sweep values
    values = [float(v) for v in args.values.split(",")]

    # Base kwargs for run_implementation (dataset handled separately)
    base_kwargs = {
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "hidden_size": args.hidden_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
    }

    # Find implementations to sweep
    impl_list = CNN_IMPLEMENTATIONS if is_cnn else IMPLEMENTATIONS
    if args.impl.lower() == "all":
        targets = [(l, c) for l, c in impl_list if is_available(l)]
    else:
        label, cmd = find_impl(args.impl, args.model)
        if label is None:
            available = [l for l, _ in impl_list]
            print(f"Implementation '{args.impl}' not found.", file=sys.stderr)
            print(f"Available: {available}", file=sys.stderr)
            sys.exit(1)
        if not is_available(label):
            print(f"Implementation '{label}' is not available (binary not built?).", file=sys.stderr)
            sys.exit(1)
        targets = [(label, cmd)]

    # Print sweep header
    seeds_str = f", seeds={args.seeds}" if args.seeds > 1 else ""
    for label, cmd_template in targets:
        print(f"\nSweeping {args.param} for {label} on {args.dataset} "
              f"(bs={args.batch_size}, epochs={args.epochs}{seeds_str})")
        print("=" * 65)

        results = sweep_single(label, cmd_template, args.param, values,
                               base_kwargs, seeds=args.seeds)
        print_results(label, args.param, results, args.seeds)
        print()


if __name__ == "__main__":
    main()
