#!/usr/bin/env python3
"""Grid-search tuning script for MLP benchmark implementations.

Runs a configurable grid of (num_samples, batch_size, hidden_size, epochs)
across all available implementations, logs results to CSV, and prints a
summary when done.

Usage:
    # Default grid (takes ~30-60 min depending on hardware)
    python src/scripts/tune_benchmark.py

    # Quick test
    python src/scripts/tune_benchmark.py --epochs 100 --num-samples 10000,100000 --batch-sizes 512,2048

    # Full grid with multiple runs
    python src/scripts/tune_benchmark.py --runs 3

    # Only specific implementations
    python src/scripts/tune_benchmark.py --impls "C (CPU),NumPy"

Results are saved to figs/tune_results.csv and a summary is printed at the end.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

IMPLEMENTATIONS = [
    ("C (CPU)",
     "./src/c/build_cpu/main --dataset {dataset} --batch-size {batch_size} --num-samples {num_samples} --hidden-size {hidden_size} --epochs {epochs}"),
    ("C (CUDA)",
     "./src/c/build_cuda/main --dataset {dataset} --batch-size {batch_size} --num-samples {num_samples} --hidden-size {hidden_size} --epochs {epochs}"),
    ("NumPy",
     "{python} src/python/models/mlp/mlp_numpy.py --dataset {dataset} --batch-size {batch_size} --num-samples {num_samples} --hidden-size {hidden_size} --epochs {epochs}"),
    ("PyTorch (CPU)",
     "{python} src/python/models/mlp/mlp_pytorch.py --dataset {dataset} --device cpu --batch-size {batch_size} --num-samples {num_samples} --hidden-size {hidden_size} --epochs {epochs}"),
    ("PyTorch (CUDA)",
     "{python} src/python/models/mlp/mlp_pytorch.py --dataset {dataset} --device cuda --batch-size {batch_size} --num-samples {num_samples} --hidden-size {hidden_size} --epochs {epochs}"),
]

PATTERN_LOSS = re.compile(r"Test Loss:\s+([\d.]+)")
PATTERN_ACC = re.compile(r"Test Accuracy:\s+([\d.]+)%")
PATTERN_TRAIN = re.compile(r"Train time:\s+([\d.]+)\s*s")
PATTERN_EVAL = re.compile(r"Eval time:\s+([\d.]+)\s*s")
PATTERN_THROUGHPUT = re.compile(r"Throughput:\s+([\d.]+)\s*samples/s")


def is_available(label):
    if label == "C (CPU)":
        return os.path.isfile(os.path.join(PROJECT_ROOT, "src", "c", "build_cpu", "main"))
    if label == "C (CUDA)":
        return os.path.isfile(os.path.join(PROJECT_ROOT, "src", "c", "build_cuda", "main"))
    if "PyTorch" in label:
        try:
            import torch
            if "CUDA" in label:
                return torch.cuda.is_available()
            return True
        except ImportError:
            return False
    return True


def run_one(label, cmd_template, dataset, batch_size, num_samples, hidden_size, epochs):
    cmd = (cmd_template
           .replace("{dataset}", dataset)
           .replace("{python}", sys.executable)
           .replace("{batch_size}", str(batch_size))
           .replace("{num_samples}", str(num_samples))
           .replace("{hidden_size}", str(hidden_size))
           .replace("{epochs}", str(epochs)))

    env = os.environ.copy()
    if label == "C (CPU)":
        lib_dir = os.path.join(PROJECT_ROOT, "src", "c", "build_cpu")
        env["LD_LIBRARY_PATH"] = lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")
    elif label == "C (CUDA)":
        lib_dir = os.path.join(PROJECT_ROOT, "src", "c", "build_cuda")
        env["LD_LIBRARY_PATH"] = lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")

    try:
        result = subprocess.run(
            cmd.split(), capture_output=True, text=True,
            timeout=600, cwd=PROJECT_ROOT, env=env)
        output = result.stdout + result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return {"error": str(e)}

    if result.returncode != 0:
        return {"error": f"exit code {result.returncode}"}

    m_loss = PATTERN_LOSS.search(output)
    m_acc = PATTERN_ACC.search(output)
    m_train = PATTERN_TRAIN.search(output)
    m_eval = PATTERN_EVAL.search(output)
    m_tp = PATTERN_THROUGHPUT.search(output)

    if not all([m_loss, m_acc, m_train, m_eval]):
        return {"error": "parse failure"}

    return {
        "loss": float(m_loss.group(1)),
        "accuracy": float(m_acc.group(1)),
        "train_time": float(m_train.group(1)),
        "eval_time": float(m_eval.group(1)),
        "throughput": float(m_tp.group(1)) if m_tp else None,
    }


def main():
    parser = argparse.ArgumentParser(description="MLP Tuning Grid Search")
    parser.add_argument("--num-samples", type=str, default="10000,100000,500000",
                        help="Comma-separated num_samples values")
    parser.add_argument("--batch-sizes", type=str, default="32,128,512,2048,8192",
                        help="Comma-separated batch sizes")
    parser.add_argument("--hidden-sizes", type=str, default="64",
                        help="Comma-separated hidden layer sizes")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of training epochs")
    parser.add_argument("--runs", type=int, default=1,
                        help="Runs per config (for averaging)")
    parser.add_argument("--impls", type=str, default=None,
                        help="Comma-separated implementation names to test (default: all available)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: figs/tune_results.csv)")
    args = parser.parse_args()

    num_samples_list = [int(x) for x in args.num_samples.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    hidden_sizes = [int(x) for x in args.hidden_sizes.split(",")]

    # Discover implementations
    all_avail = [(l, c) for l, c in IMPLEMENTATIONS if is_available(l)]
    if args.impls:
        requested = [x.strip() for x in args.impls.split(",")]
        available = [(l, c) for l, c in all_avail if l in requested]
    else:
        available = all_avail

    impl_names = [l for l, _ in available]

    # Calculate total runs
    total_configs = len(num_samples_list) * len(batch_sizes) * len(hidden_sizes) * len(available)
    total_runs = total_configs * args.runs

    # Output file
    out_dir = os.path.join(PROJECT_ROOT, "figs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.output or os.path.join(out_dir, "tune_results.csv")

    print(f"Tuning Grid Search")
    print(f"  Implementations: {impl_names}")
    print(f"  num_samples:     {num_samples_list}")
    print(f"  batch_sizes:     {batch_sizes}")
    print(f"  hidden_sizes:    {hidden_sizes}")
    print(f"  epochs:          {args.epochs}")
    print(f"  runs/config:     {args.runs}")
    print(f"  Total runs:      {total_runs}")
    print(f"  Output:          {out_path}")
    print()

    fieldnames = ["timestamp", "impl", "num_samples", "batch_size", "hidden_size",
                  "epochs", "run", "throughput", "train_time", "accuracy", "loss",
                  "eval_time", "error"]

    # Open CSV (append if exists, write header if new)
    file_exists = os.path.isfile(out_path) and os.path.getsize(out_path) > 0
    csvfile = open(out_path, "a", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    completed = 0
    t_start = time.monotonic()

    try:
        for ns in num_samples_list:
            for bs in batch_sizes:
                for hs in hidden_sizes:
                    for label, cmd_template in available:
                        for run_idx in range(args.runs):
                            completed += 1
                            elapsed = time.monotonic() - t_start
                            eta = (elapsed / completed) * (total_runs - completed) if completed > 0 else 0
                            tag = f" run {run_idx+1}/{args.runs}" if args.runs > 1 else ""
                            print(f"[{completed}/{total_runs}] {label} ns={ns} bs={bs} hs={hs}{tag}  "
                                  f"(ETA: {eta/60:.0f}m)", flush=True)

                            r = run_one(label, cmd_template, "generated",
                                        bs, ns, hs, args.epochs)

                            row = {
                                "timestamp": datetime.now().isoformat(),
                                "impl": label,
                                "num_samples": ns,
                                "batch_size": bs,
                                "hidden_size": hs,
                                "epochs": args.epochs,
                                "run": run_idx + 1,
                                "throughput": r.get("throughput"),
                                "train_time": r.get("train_time"),
                                "accuracy": r.get("accuracy"),
                                "loss": r.get("loss"),
                                "eval_time": r.get("eval_time"),
                                "error": r.get("error"),
                            }
                            writer.writerow(row)
                            csvfile.flush()

                            if "error" in r:
                                print(f"    ERROR: {r['error']}")
                            else:
                                tp = r.get("throughput", 0)
                                print(f"    {tp:,.0f} samples/s  acc={r['accuracy']:.1f}%  "
                                      f"time={r['train_time']:.2f}s")
    except KeyboardInterrupt:
        print("\nInterrupted — partial results saved.")
    finally:
        csvfile.close()

    total_time = time.monotonic() - t_start
    print(f"\nDone in {total_time/60:.1f} minutes. Results: {out_path}")

    # Print summary table
    print_summary(out_path, impl_names)


def print_summary(csv_path, impl_names):
    """Print a summary table from the CSV results."""
    if not os.path.isfile(csv_path):
        return

    # Read CSV
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("error"):
                continue
            rows.append(row)

    if not rows:
        print("No successful results to summarize.")
        return

    # Group by (num_samples, batch_size, hidden_size, impl) -> list of throughputs
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        key = (int(row["num_samples"]), int(row["batch_size"]),
               int(row["hidden_size"]), row["impl"])
        if row["throughput"]:
            groups[key].append(float(row["throughput"]))

    # Print grouped by num_samples
    print("\n" + "=" * 80)
    print("THROUGHPUT SUMMARY (samples/s)")
    print("=" * 80)

    seen_ns = sorted(set(k[0] for k in groups.keys()))
    seen_bs = sorted(set(k[1] for k in groups.keys()))
    seen_hs = sorted(set(k[2] for k in groups.keys()))

    for hs in seen_hs:
        if len(seen_hs) > 1:
            print(f"\n--- hidden_size={hs} ---")
        for ns in seen_ns:
            print(f"\nnum_samples={ns:,}")
            # Header
            header = f"  {'batch_size':>10}"
            for impl in impl_names:
                header += f"  {impl:>16}"
            print(header)
            print("  " + "-" * (len(header) - 2))

            for bs in seen_bs:
                line = f"  {bs:>10}"
                for impl in impl_names:
                    key = (ns, bs, hs, impl)
                    if key in groups and groups[key]:
                        mean_tp = np.mean(groups[key])
                        if mean_tp >= 1e6:
                            line += f"  {mean_tp/1e6:>13.2f}M/s"
                        else:
                            line += f"  {mean_tp/1e3:>13.0f}K/s"
                    else:
                        line += f"  {'---':>16}"
                print(line)


if __name__ == "__main__":
    main()
