#!/usr/bin/env python3
"""Benchmark runner for MLP implementations across languages.

Runs C (CPU), NumPy, and PyTorch (CPU/CUDA) implementations on specified
datasets, parses standardized output, and generates comparison tables and charts.

Modes:
    standard — bar charts for real datasets
    scaling  — throughput scaling across dataset size, batch size, hidden size
               with GPU speedup plots and ranked legends

Usage:
    python src/scripts/benchmark.py --mode standard --datasets generated,iris,breast-cancer --runs 3
    python src/scripts/benchmark.py --mode scaling --runs 1
"""

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Project root (two levels up from src/scripts/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Implementation definitions with configurable placeholders.
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
    ("Rust (CPU)",
     "./src/rust/target/release/mlp-cpu --dataset {dataset} --batch-size {batch_size} --num-samples {num_samples} --hidden-size {hidden_size} --epochs {epochs}"),
    ("Rust (cuBLAS)",
     "./src/rust/target/release/mlp-cuda-cublas --dataset {dataset} --batch-size {batch_size} --num-samples {num_samples} --hidden-size {hidden_size} --epochs {epochs}"),
    ("Rust (CUDA Kernels)",
     "./src/rust/target/release/mlp-cuda-kernels --dataset {dataset} --batch-size {batch_size} --num-samples {num_samples} --hidden-size {hidden_size} --epochs {epochs}"),
]

# Regex patterns for parsing standardized output
PATTERN_LOSS = re.compile(r"Test Loss:\s+([\d.]+)")
PATTERN_ACC = re.compile(r"Test Accuracy:\s+([\d.]+)%")
PATTERN_TRAIN = re.compile(r"Train time:\s+([\d.]+)\s*s")
PATTERN_EVAL = re.compile(r"Eval time:\s+([\d.]+)\s*s")
PATTERN_THROUGHPUT = re.compile(r"Throughput:\s+([\d.]+)\s*samples/s")

# Plot styling
IMPL_STYLES = {
    "C (CUDA)":       {"color": "#2ecc71", "marker": "D", "linewidth": 3.0, "markersize": 10, "zorder": 10},
    "C (CPU)":        {"color": "#3498db", "marker": "o", "linewidth": 2.2, "markersize": 8,  "zorder": 5},
    "NumPy":          {"color": "#e67e22", "marker": "s", "linewidth": 2.0, "markersize": 7,  "zorder": 4},
    "PyTorch (CUDA)": {"color": "#9b59b6", "marker": "^", "linewidth": 2.2, "markersize": 8,  "zorder": 6},
    "PyTorch (CPU)":  {"color": "#e74c3c", "marker": "v", "linewidth": 1.8, "markersize": 7,  "zorder": 3},
    "Rust (CPU)":          {"color": "#f39c12", "marker": "P", "linewidth": 2.2, "markersize": 8,  "zorder": 5},
    "Rust (cuBLAS)":       {"color": "#1abc9c", "marker": "H", "linewidth": 2.8, "markersize": 9,  "zorder": 9},
    "Rust (CUDA Kernels)": {"color": "#e91e63", "marker": "X", "linewidth": 2.8, "markersize": 9,  "zorder": 8},
}

SPEEDUP_PAIRS = [
    ("C (CUDA)", "C (CPU)", "#2ecc71", "C CUDA / C CPU"),
    ("Rust (cuBLAS)", "Rust (CPU)", "#1abc9c", "Rust cuBLAS / Rust CPU"),
    ("Rust (CUDA Kernels)", "Rust (CPU)", "#e91e63", "Rust Kernels / Rust CPU"),
    ("PyTorch (CUDA)", "PyTorch (CPU)", "#9b59b6", "PyTorch CUDA / CPU"),
    ("C (CUDA)", "NumPy", "#e67e22", "C CUDA / NumPy"),
    ("Rust (cuBLAS)", "C (CUDA)", "#f39c12", "Rust cuBLAS / C CUDA"),
]


def is_available(label):
    """Check if an implementation is available."""
    if label == "C (CPU)":
        return os.path.isfile(os.path.join(PROJECT_ROOT, "src", "c", "build_cpu", "main"))
    if label == "C (CUDA)":
        return os.path.isfile(os.path.join(PROJECT_ROOT, "src", "c", "build_cuda", "main"))
    if label == "Rust (CPU)":
        return os.path.isfile(os.path.join(PROJECT_ROOT, "src", "rust", "target", "release", "mlp-cpu"))
    if label == "Rust (cuBLAS)":
        return os.path.isfile(os.path.join(PROJECT_ROOT, "src", "rust", "target", "release", "mlp-cuda-cublas"))
    if label == "Rust (CUDA Kernels)":
        return os.path.isfile(os.path.join(PROJECT_ROOT, "src", "rust", "target", "release", "mlp-cuda-kernels"))
    if "PyTorch" in label:
        try:
            import torch
            if "CUDA" in label:
                return torch.cuda.is_available()
            return True
        except ImportError:
            return False
    return True


def run_implementation(label, cmd_template, dataset, batch_size=32,
                       num_samples=0, hidden_size=64, epochs=1000):
    """Run a single implementation and parse its output."""
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
        print(f"  SKIP {label}: {e}")
        return None

    if result.returncode != 0:
        print(f"  FAIL {label}: exit code {result.returncode}")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[:5]:
                print(f"    {line}")
        return None

    m_loss = PATTERN_LOSS.search(output)
    m_acc = PATTERN_ACC.search(output)
    m_train = PATTERN_TRAIN.search(output)
    m_eval = PATTERN_EVAL.search(output)
    m_throughput = PATTERN_THROUGHPUT.search(output)

    if not all([m_loss, m_acc, m_train, m_eval]):
        print(f"  FAIL {label}: could not parse output")
        return None

    parsed = {
        "loss": float(m_loss.group(1)),
        "accuracy": float(m_acc.group(1)),
        "train_time": float(m_train.group(1)),
        "eval_time": float(m_eval.group(1)),
    }
    if m_throughput:
        parsed["throughput"] = float(m_throughput.group(1))
    return parsed


# ---------------------------------------------------------------------------
#  Plot helpers
# ---------------------------------------------------------------------------

def _setup_style():
    """Dark-ish professional style."""
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "text.color": "#e0e0e0",
        "xtick.color": "#c0c0c0",
        "ytick.color": "#c0c0c0",
        "grid.color": "#334466",
        "grid.alpha": 0.5,
        "legend.facecolor": "#1a1a2e",
        "legend.edgecolor": "#555555",
        "legend.fontsize": 10,
        "font.family": "sans-serif",
        "figure.dpi": 150,
    })


def _ranked_legend(ax, throughput_data, labels):
    """Create legend ordered by peak throughput (best first)."""
    handles, leg_labels = ax.get_legend_handles_labels()
    # Build label->peak map
    peak = {}
    for label in labels:
        vals = [v for v in throughput_data.get(label, []) if v is not None]
        peak[label] = max(vals) if vals else 0

    # Sort handles by peak throughput descending
    paired = sorted(zip(handles, leg_labels), key=lambda x: peak.get(x[1], 0), reverse=True)
    if paired:
        handles, leg_labels = zip(*paired)
        ax.legend(handles, leg_labels, loc="best")


def _format_y_throughput(ax):
    """Format y-axis as readable throughput (K/s, M/s)."""
    def fmt(x, _):
        if x >= 1e6:
            return f"{x/1e6:.0f}M"
        if x >= 1e3:
            return f"{x/1e3:.0f}K"
        return f"{x:.0f}"
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))


def _format_x_samples(ax):
    """Format x-axis sample counts as readable (K, M)."""
    def fmt(x, _):
        if x >= 1e6:
            return f"{x/1e6:.0f}M"
        if x >= 1e3:
            return f"{x/1e3:.0f}K"
        return f"{x:.0f}"
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt))


def _plot_throughput(ax, xs, throughput_data, available_labels, title, xlabel):
    """Plot throughput lines with implementation-specific styling."""
    for label in available_labels:
        vals = throughput_data.get(label, [])
        xf = [xs[i] for i, v in enumerate(vals) if v is not None]
        yf = [v for v in vals if v is not None]
        if yf:
            s = IMPL_STYLES.get(label, {"color": "gray", "marker": ".", "linewidth": 1.5, "markersize": 6, "zorder": 1})
            ax.plot(xf, yf, marker=s["marker"], color=s["color"],
                    linewidth=s["linewidth"], markersize=s["markersize"],
                    zorder=s["zorder"], label=label)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Throughput (samples/s)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    _format_y_throughput(ax)
    _ranked_legend(ax, throughput_data, available_labels)


# ---------------------------------------------------------------------------
#  Data collection helper
# ---------------------------------------------------------------------------

def _collect_throughput(available, runs, vary_param, vary_values, fixed_params):
    """Run benchmarks varying one parameter, return {label: [throughputs]}."""
    labels = [l for l, _ in available]
    throughput = {l: [] for l in labels}

    for val in vary_values:
        print(f"\n--- {vary_param}={val} ---")
        params = dict(fixed_params)
        params[vary_param] = val

        for label, cmd_template in available:
            runs_tp = []
            for run_idx in range(runs):
                tag = f" (run {run_idx+1}/{runs})" if runs > 1 else ""
                print(f"  {label}{tag}...")
                r = run_implementation(
                    label, cmd_template, "generated",
                    batch_size=params.get("batch_size", 2048),
                    num_samples=params.get("num_samples", 100000),
                    hidden_size=params.get("hidden_size", 64),
                    epochs=params.get("epochs", 500),
                )
                if r and "throughput" in r:
                    runs_tp.append(r["throughput"])
            throughput[label].append(np.mean(runs_tp) if runs_tp else None)

    return throughput


# ---------------------------------------------------------------------------
#  Scaling mode
# ---------------------------------------------------------------------------

def run_scaling_experiment(available, runs, figs_dir, cfg):
    """Run GPU-focused scaling experiments with multiple plot types."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available -- scaling mode requires matplotlib")
        return

    _setup_style()
    os.makedirs(figs_dir, exist_ok=True)

    epochs = cfg["epochs"]
    labels = [l for l, _ in available]

    # ================================================================
    #  Experiment 1: Throughput vs Dataset Size
    # ================================================================
    dataset_sizes = cfg["dataset_sizes"]
    fixed_bs = cfg["fixed_batch_size"]

    print("=" * 70)
    print(f"  Experiment 1: Dataset Size Scaling  (batch={fixed_bs}, epochs={epochs})")
    print("=" * 70)

    tp_ds = _collect_throughput(available, runs, "num_samples", dataset_sizes,
                                {"batch_size": fixed_bs, "epochs": epochs})

    fig, ax = plt.subplots(figsize=(12, 7))
    _plot_throughput(ax, dataset_sizes, tp_ds, labels,
                     f"Throughput vs Dataset Size\nbatch_size={fixed_bs}, epochs={epochs}",
                     "Dataset Size (samples)")
    _format_x_samples(ax)
    plt.tight_layout()
    p = os.path.join(figs_dir, "scaling_dataset_size.png")
    plt.savefig(p)
    plt.close()
    print(f"\nSaved: {p}")

    # ================================================================
    #  Experiment 2: Throughput vs Batch Size
    # ================================================================
    batch_sizes = cfg["batch_sizes"]
    fixed_ns = cfg["fixed_num_samples"]

    print("\n" + "=" * 70)
    print(f"  Experiment 2: Batch Size Scaling  (samples={fixed_ns}, epochs={epochs})")
    print("=" * 70)

    tp_bs = _collect_throughput(available, runs, "batch_size", batch_sizes,
                                {"num_samples": fixed_ns, "epochs": epochs})

    fig, ax = plt.subplots(figsize=(12, 7))
    _plot_throughput(ax, batch_sizes, tp_bs, labels,
                     f"Throughput vs Batch Size\nnum_samples={fixed_ns:,}, epochs={epochs}",
                     "Batch Size")
    plt.tight_layout()
    p = os.path.join(figs_dir, "scaling_batch_size.png")
    plt.savefig(p)
    plt.close()
    print(f"\nSaved: {p}")

    # ================================================================
    #  Experiment 3: Throughput vs Hidden Size  (GPU compute scaling)
    # ================================================================
    hidden_sizes = cfg["hidden_sizes"]

    print("\n" + "=" * 70)
    print(f"  Experiment 3: Hidden Size Scaling  (samples={fixed_ns}, batch={fixed_bs}, epochs={epochs})")
    print("=" * 70)

    tp_hs = _collect_throughput(available, runs, "hidden_size", hidden_sizes,
                                {"num_samples": fixed_ns, "batch_size": fixed_bs, "epochs": epochs})

    fig, ax = plt.subplots(figsize=(12, 7))
    _plot_throughput(ax, hidden_sizes, tp_hs, labels,
                     f"Throughput vs Hidden Layer Size\nsamples={fixed_ns:,}, batch={fixed_bs}, epochs={epochs}",
                     "Hidden Size")
    plt.tight_layout()
    p = os.path.join(figs_dir, "scaling_hidden_size.png")
    plt.savefig(p)
    plt.close()
    print(f"\nSaved: {p}")

    # ================================================================
    #  Experiment 4: GPU Speedup plots
    # ================================================================
    print("\n" + "=" * 70)
    print("  Generating GPU speedup plots...")
    print("=" * 70)

    experiments = [
        ("dataset_size", dataset_sizes, tp_ds, "Dataset Size (samples)", True),
        ("batch_size",   batch_sizes,   tp_bs, "Batch Size", False),
        ("hidden_size",  hidden_sizes,  tp_hs, "Hidden Size", False),
    ]

    for exp_name, xs, tp_data, xlabel, use_sample_fmt in experiments:
        fig, ax = plt.subplots(figsize=(12, 7))

        any_plotted = False
        for gpu_label, cpu_label, color, legend_name in SPEEDUP_PAIRS:
            gpu_vals = tp_data.get(gpu_label, [])
            cpu_vals = tp_data.get(cpu_label, [])
            if not gpu_vals or not cpu_vals:
                continue

            xf, yf = [], []
            for i in range(min(len(gpu_vals), len(cpu_vals))):
                if gpu_vals[i] is not None and cpu_vals[i] is not None and cpu_vals[i] > 0:
                    xf.append(xs[i])
                    yf.append(gpu_vals[i] / cpu_vals[i])
            if yf:
                ax.plot(xf, yf, marker="o", color=color, linewidth=2.5,
                        markersize=8, label=legend_name)
                any_plotted = True

        if not any_plotted:
            plt.close()
            continue

        ax.axhline(y=1.0, color="#e74c3c", linestyle="--", alpha=0.7, linewidth=1.5, label="Break-even (1x)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Speedup (x)")
        ax.set_title(f"GPU Speedup vs {xlabel}\n(higher = GPU wins)", fontsize=14, fontweight="bold")
        ax.grid(True, which="both", alpha=0.3)

        if use_sample_fmt:
            _format_x_samples(ax)

        # Sort legend by peak speedup
        handles, leg_labels = ax.get_legend_handles_labels()
        peak_sp = {}
        for h, ll in zip(handles, leg_labels):
            if ll == "Break-even (1x)":
                peak_sp[ll] = -1
            else:
                # extract from plot data
                ydata = h.get_ydata()
                peak_sp[ll] = max(ydata) if len(ydata) > 0 else 0
        paired = sorted(zip(handles, leg_labels), key=lambda x: peak_sp.get(x[1], 0), reverse=True)
        handles, leg_labels = zip(*paired)
        ax.legend(handles, leg_labels, loc="best")

        plt.tight_layout()
        p = os.path.join(figs_dir, f"gpu_speedup_{exp_name}.png")
        plt.savefig(p)
        plt.close()
        print(f"  Saved: {p}")

    # ================================================================
    #  Experiment 5: Combined 2x2 overview
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Top-left: throughput vs dataset size
    _plot_throughput(axes[0, 0], dataset_sizes, tp_ds, labels,
                     f"Dataset Size Scaling (batch={fixed_bs})", "Dataset Size")
    _format_x_samples(axes[0, 0])

    # Top-right: throughput vs batch size
    _plot_throughput(axes[0, 1], batch_sizes, tp_bs, labels,
                     f"Batch Size Scaling (samples={fixed_ns:,})", "Batch Size")

    # Bottom-left: throughput vs hidden size
    _plot_throughput(axes[1, 0], hidden_sizes, tp_hs, labels,
                     f"Hidden Size Scaling (batch={fixed_bs})", "Hidden Size")

    # Bottom-right: GPU speedup (C CUDA vs all CPU)
    for gpu_label, cpu_label, color, legend_name in SPEEDUP_PAIRS[:3]:
        gpu_vals = tp_hs.get(gpu_label, [])
        cpu_vals = tp_hs.get(cpu_label, [])
        if not gpu_vals or not cpu_vals:
            continue
        xf, yf = [], []
        for i in range(min(len(gpu_vals), len(cpu_vals))):
            if gpu_vals[i] and cpu_vals[i] and cpu_vals[i] > 0:
                xf.append(hidden_sizes[i])
                yf.append(gpu_vals[i] / cpu_vals[i])
        if yf:
            axes[1, 1].plot(xf, yf, marker="o", color=color, linewidth=2.5, markersize=8, label=legend_name)

    axes[1, 1].axhline(y=1.0, color="#e74c3c", linestyle="--", alpha=0.7, linewidth=1.5)
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_xlabel("Hidden Size")
    axes[1, 1].set_ylabel("Speedup (x)")
    axes[1, 1].set_title("GPU Speedup vs Hidden Size", fontweight="bold")
    axes[1, 1].grid(True, which="both", alpha=0.3)
    axes[1, 1].legend(loc="best")

    fig.suptitle("MLP Benchmark: GPU Scaling Analysis", fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(figs_dir, "scaling_overview.png")
    plt.savefig(p)
    plt.close()
    print(f"\n  Saved: {p}")

    # Print summary table
    print("\n" + "=" * 70)
    print("  PEAK THROUGHPUT SUMMARY")
    print("=" * 70)
    for exp_name, tp_data in [("Dataset Size", tp_ds), ("Batch Size", tp_bs), ("Hidden Size", tp_hs)]:
        print(f"\n  {exp_name} experiment:")
        ranked = []
        for label in labels:
            vals = [v for v in tp_data.get(label, []) if v is not None]
            if vals:
                ranked.append((label, max(vals)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        for i, (label, peak) in enumerate(ranked):
            bar = "#" * int(peak / ranked[0][1] * 40) if ranked else ""
            if peak >= 1e6:
                print(f"    {i+1}. {label:<18} {peak/1e6:>8.2f}M/s  {bar}")
            else:
                print(f"    {i+1}. {label:<18} {peak/1e3:>8.0f}K/s  {bar}")


# ---------------------------------------------------------------------------
#  Standard mode
# ---------------------------------------------------------------------------

def print_markdown_table(results):
    print("\n## Benchmark Results\n")
    print("| Dataset | Implementation | Accuracy (%) | Loss | Train Time (s) | Eval Time (s) |")
    print("|---------|---------------|-------------|------|----------------|---------------|")
    for dataset in sorted(results.keys()):
        for label, metrics in sorted(results[dataset].items()):
            if metrics["std_acc"] is not None and metrics["std_acc"] > 0:
                acc_str = f"{metrics['mean_acc']:.2f} +/- {metrics['std_acc']:.2f}"
                time_str = f"{metrics['mean_train']:.3f} +/- {metrics['std_train']:.3f}"
            else:
                acc_str = f"{metrics['mean_acc']:.2f}"
                time_str = f"{metrics['mean_train']:.3f}"
            print(f"| {dataset} | {label} | {acc_str} | {metrics['mean_loss']:.4f} | {time_str} | {metrics['mean_eval']:.3f} |")


def generate_charts(results, figs_dir):
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping chart generation")
        return

    os.makedirs(figs_dir, exist_ok=True)
    datasets = sorted(results.keys())
    all_labels = []
    for d in datasets:
        for label in results[d]:
            if label not in all_labels:
                all_labels.append(label)

    x = np.arange(len(datasets))
    width = 0.8 / max(len(all_labels), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(all_labels):
        vals = [results[d][label]["mean_train"] if label in results[d] else 0 for d in datasets]
        errs = [results[d][label]["std_train"] or 0 if label in results[d] else 0 for d in datasets]
        ax.bar(x + i * width, vals, width, yerr=errs, label=label, capsize=3)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Train Time (s)")
    ax.set_title("MLP Training Time by Implementation")
    ax.set_xticks(x + width * (len(all_labels) - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "benchmark_train_time.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(all_labels):
        vals = [results[d][label]["mean_acc"] if label in results[d] else 0 for d in datasets]
        errs = [results[d][label]["std_acc"] or 0 if label in results[d] else 0 for d in datasets]
        ax.bar(x + i * width, vals, width, yerr=errs, label=label, capsize=3)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("MLP Test Accuracy by Implementation")
    ax.set_xticks(x + width * (len(all_labels) - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "benchmark_accuracy.png"), dpi=150)
    plt.close()
    print(f"\nCharts saved to {figs_dir}/")


def run_standard(datasets, available, runs, figs_dir):
    results = defaultdict(dict)
    for dataset in datasets:
        print(f"--- {dataset} ---")
        for label, cmd_template in available:
            run_results = []
            for run in range(runs):
                tag = f" (run {run + 1}/{runs})" if runs > 1 else ""
                print(f"  {label}{tag}...")
                r = run_implementation(label, cmd_template, dataset)
                if r is not None:
                    run_results.append(r)
            if run_results:
                accs = [r["accuracy"] for r in run_results]
                losses = [r["loss"] for r in run_results]
                trains = [r["train_time"] for r in run_results]
                evals = [r["eval_time"] for r in run_results]
                results[dataset][label] = {
                    "mean_acc": np.mean(accs),
                    "std_acc": np.std(accs) if len(accs) > 1 else None,
                    "mean_loss": np.mean(losses),
                    "mean_train": np.mean(trains),
                    "std_train": np.std(trains) if len(trains) > 1 else None,
                    "mean_eval": np.mean(evals),
                }
        print()
    print_markdown_table(results)
    generate_charts(results, figs_dir)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MLP Benchmark Runner")
    parser.add_argument("--mode", default="standard", choices=["standard", "scaling"])
    parser.add_argument("--datasets", default="generated,iris,breast-cancer",
                        help="Comma-separated dataset names (standard mode)")
    parser.add_argument("--runs", type=int, default=1)

    # Scaling parameters -- all power-of-2 by default
    parser.add_argument("--scaling-epochs", type=int, default=500)
    parser.add_argument("--scaling-fixed-bs", type=int, default=2048)
    parser.add_argument("--scaling-fixed-ns", type=int, default=131072,
                        help="Fixed num_samples (default: 2^17 = 131072)")
    parser.add_argument("--scaling-dataset-sizes", type=str, default=None,
                        help="Comma-separated (default: 2^12..2^20)")
    parser.add_argument("--scaling-batch-sizes", type=str, default=None,
                        help="Comma-separated (default: 2^7..2^14)")
    parser.add_argument("--scaling-hidden-sizes", type=str, default=None,
                        help="Comma-separated (default: 2^5..2^10)")
    args = parser.parse_args()

    figs_dir = os.path.join(PROJECT_ROOT, "figs")
    available = [(label, cmd) for label, cmd in IMPLEMENTATIONS if is_available(label)]
    print(f"Available implementations: {[l for l, _ in available]}")
    print(f"Mode: {args.mode}")
    print(f"Runs per config: {args.runs}\n")

    if args.mode == "scaling":
        cfg = {
            "epochs": args.scaling_epochs,
            "fixed_batch_size": args.scaling_fixed_bs,
            "fixed_num_samples": args.scaling_fixed_ns,
            "dataset_sizes": ([int(x) for x in args.scaling_dataset_sizes.split(",")]
                              if args.scaling_dataset_sizes
                              else [2**n for n in range(12, 21)]),     # 4K .. 1M
            "batch_sizes": ([int(x) for x in args.scaling_batch_sizes.split(",")]
                            if args.scaling_batch_sizes
                            else [2**n for n in range(7, 15)]),        # 128 .. 16384
            "hidden_sizes": ([int(x) for x in args.scaling_hidden_sizes.split(",")]
                             if args.scaling_hidden_sizes
                             else [2**n for n in range(5, 11)]),       # 32 .. 1024
        }
        run_scaling_experiment(available, args.runs, figs_dir, cfg)
    else:
        datasets = [d.strip() for d in args.datasets.split(",")]
        print(f"Datasets: {datasets}\n")
        run_standard(datasets, available, args.runs, figs_dir)


if __name__ == "__main__":
    main()
