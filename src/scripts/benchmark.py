#!/usr/bin/env python3
"""Benchmark runner for MLP and CNN implementations across languages.

Supports MLP (8 implementations) and CNN/LeNet-5 (8 implementations) with
standardized output parsing, comparison tables, scaling charts, and result caching.

Modes:
    standard — bar charts for real datasets (accuracy + train time)
    scaling  — throughput scaling with GPU speedup plots and ranked legends
               MLP: dataset size, batch size, hidden size
               CNN: batch size only (fixed MNIST dataset, fixed architecture)

Usage:
    python src/scripts/benchmark.py --mode standard --datasets generated,iris,breast-cancer --runs 3
    python src/scripts/benchmark.py --mode scaling --runs 1
    python src/scripts/benchmark.py --mode scaling --model cnn --runs 1
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import warnings
from collections import defaultdict

# Suppress sklearn GP convergence warnings (noisy with few data points)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

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

# CNN implementations — fixed architecture (LeNet-5), MNIST only.
# No --num-samples or --hidden-size flags; dataset is always mnist.
CNN_IMPLEMENTATIONS = [
    ("CNN C (CPU)",
     "./src/c/build_cpu/cnn_main --dataset mnist --batch-size {batch_size} --epochs {epochs}"),
    ("CNN C (CUDA)",
     "./src/c/build_cuda/cnn_main --dataset mnist --batch-size {batch_size} --epochs {epochs}"),
    ("CNN NumPy",
     "{python} src/python/models/cnn/cnn_numpy.py --dataset mnist --batch-size {batch_size} --epochs {epochs}"),
    ("CNN PyTorch (CPU)",
     "{python} src/python/models/cnn/cnn_pytorch.py --dataset mnist --device cpu --batch-size {batch_size} --epochs {epochs}"),
    ("CNN PyTorch (CUDA)",
     "{python} src/python/models/cnn/cnn_pytorch.py --dataset mnist --device cuda --batch-size {batch_size} --epochs {epochs}"),
    ("CNN Rust (CPU)",
     "./src/rust/target/release/cnn-cpu --dataset mnist --batch-size {batch_size} --epochs {epochs}"),
    ("CNN Rust (cuBLAS)",
     "./src/rust/target/release/cnn-cuda-cublas --dataset mnist --batch-size {batch_size} --epochs {epochs}"),
    ("CNN Rust (CUDA Kernels)",
     "./src/rust/target/release/cnn-cuda-kernels --dataset mnist --batch-size {batch_size} --epochs {epochs}"),
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

# CNN plot styles — same colors, prefixed labels
CNN_IMPL_STYLES = {
    "CNN C (CUDA)":            {"color": "#2ecc71", "marker": "D", "linewidth": 3.0, "markersize": 10, "zorder": 10},
    "CNN C (CPU)":             {"color": "#3498db", "marker": "o", "linewidth": 2.2, "markersize": 8,  "zorder": 5},
    "CNN NumPy":               {"color": "#e67e22", "marker": "s", "linewidth": 2.0, "markersize": 7,  "zorder": 4},
    "CNN PyTorch (CUDA)":      {"color": "#9b59b6", "marker": "^", "linewidth": 2.2, "markersize": 8,  "zorder": 6},
    "CNN PyTorch (CPU)":       {"color": "#e74c3c", "marker": "v", "linewidth": 1.8, "markersize": 7,  "zorder": 3},
    "CNN Rust (CPU)":          {"color": "#f39c12", "marker": "P", "linewidth": 2.2, "markersize": 8,  "zorder": 5},
    "CNN Rust (cuBLAS)":       {"color": "#1abc9c", "marker": "H", "linewidth": 2.8, "markersize": 9,  "zorder": 9},
    "CNN Rust (CUDA Kernels)": {"color": "#e91e63", "marker": "X", "linewidth": 2.8, "markersize": 9,  "zorder": 8},
}

CNN_SPEEDUP_PAIRS = [
    ("CNN C (CUDA)", "CNN C (CPU)", "#2ecc71", "C CUDA / C CPU"),
    ("CNN Rust (cuBLAS)", "CNN Rust (CPU)", "#1abc9c", "Rust cuBLAS / Rust CPU"),
    ("CNN Rust (CUDA Kernels)", "CNN Rust (CPU)", "#e91e63", "Rust Kernels / Rust CPU"),
    ("CNN PyTorch (CUDA)", "CNN PyTorch (CPU)", "#9b59b6", "PyTorch CUDA / CPU"),
    ("CNN Rust (cuBLAS)", "CNN C (CUDA)", "#f39c12", "Rust cuBLAS / C CUDA"),
]


# ---------------------------------------------------------------------------
#  Benchmark result cache — avoids re-running expensive experiments
# ---------------------------------------------------------------------------

CACHE_FILE = os.path.join(PROJECT_ROOT, "figs", "benchmark_cache.json")


def _migrate_cache_entry(value):
    """Migrate a legacy cache value (bare float or None) to the new format.

    New format: {"samples": [val], "wall_times": []}
    If already in new format, return as-is.
    """
    if isinstance(value, dict) and "samples" in value:
        return value
    # Legacy format: bare float or None
    if value is None:
        return {"samples": [], "wall_times": []}
    return {"samples": [value], "wall_times": []}


def _load_cache():
    """Load cached benchmark results from JSON, migrating legacy entries."""
    if os.path.isfile(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
        # Walk the cache and migrate any legacy float entries
        for exp_key, exp_data in raw.items():
            if not isinstance(exp_data, dict):
                continue
            for label, label_data in exp_data.items():
                if not isinstance(label_data, dict):
                    continue
                for val_str, entry in label_data.items():
                    label_data[val_str] = _migrate_cache_entry(entry)
        return raw
    return {}


def _save_cache(cache):
    """Save benchmark results to JSON cache."""
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    # Approximate size for user info
    size_kb = os.path.getsize(CACHE_FILE) / 1024
    print(f"  Cache saved: {CACHE_FILE} ({size_kb:.1f} KB)")


def _cache_key(model, experiment):
    """Build a cache key like 'mlp/dataset_size'."""
    return f"{model}/{experiment}"


def is_available(label):
    """Check if an implementation binary/module is available."""
    # Map labels to binary paths for compiled implementations
    binary_map = {
        "C (CPU)":              "src/c/build_cpu/main",
        "C (CUDA)":             "src/c/build_cuda/main",
        "Rust (CPU)":           "src/rust/target/release/mlp-cpu",
        "Rust (cuBLAS)":        "src/rust/target/release/mlp-cuda-cublas",
        "Rust (CUDA Kernels)":  "src/rust/target/release/mlp-cuda-kernels",
        "CNN C (CPU)":          "src/c/build_cpu/cnn_main",
        "CNN C (CUDA)":         "src/c/build_cuda/cnn_main",
        "CNN Rust (CPU)":       "src/rust/target/release/cnn-cpu",
        "CNN Rust (cuBLAS)":    "src/rust/target/release/cnn-cuda-cublas",
        "CNN Rust (CUDA Kernels)": "src/rust/target/release/cnn-cuda-kernels",
    }
    if label in binary_map:
        return os.path.isfile(os.path.join(PROJECT_ROOT, binary_map[label]))
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
                       num_samples=0, hidden_size=512, epochs=1000):
    """Run a single implementation and parse its output."""
    cmd = (cmd_template
           .replace("{dataset}", dataset)
           .replace("{python}", sys.executable)
           .replace("{batch_size}", str(batch_size))
           .replace("{num_samples}", str(num_samples))
           .replace("{hidden_size}", str(hidden_size))
           .replace("{epochs}", str(epochs)))

    env = os.environ.copy()
    if label in ("C (CPU)", "CNN C (CPU)"):
        lib_dir = os.path.join(PROJECT_ROOT, "src", "c", "build_cpu")
        env["LD_LIBRARY_PATH"] = lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")
    elif label in ("C (CUDA)", "CNN C (CUDA)"):
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


def _fit_gp(cache, cache_key, label, x_values=None):
    """Fit a Gaussian Process in log-log space for a single implementation.

    Returns (x_dense, y_mean, y_lower, y_upper, x_obs, y_obs_mean) or None
    if fewer than 2 data points are available.

    Works in log2 for x, log10 for y.
    """
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    except ImportError:
        return None

    if cache is None or cache_key is None:
        return None

    exp_data = cache.get(cache_key, {}).get(label, {})
    if not exp_data:
        return None

    # Collect all raw samples per x value
    x_obs_list = []
    y_obs_means = []
    x_all = []
    y_all = []
    for val_str, entry in sorted(exp_data.items(), key=lambda kv: float(kv[0])):
        samples = entry.get("samples", [])
        if not samples:
            continue
        x_val = float(val_str)
        mean_val = float(np.mean(samples))
        if mean_val <= 0 or x_val <= 0:
            continue
        x_obs_list.append(x_val)
        y_obs_means.append(mean_val)
        for s in samples:
            if s > 0:
                x_all.append(np.log2(x_val))
                y_all.append(np.log10(s))

    if len(x_obs_list) < 2:
        return None

    X_train = np.array(x_all).reshape(-1, 1)
    y_train = np.array(y_all)

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-6)
    try:
        gp.fit(X_train, y_train)
    except Exception:
        return None

    # Dense prediction grid
    x_log_min = np.log2(min(x_obs_list))
    x_log_max = np.log2(max(x_obs_list))
    X_dense_log = np.linspace(x_log_min, x_log_max, 200).reshape(-1, 1)
    y_pred, y_std = gp.predict(X_dense_log, return_std=True)

    x_dense = 2 ** X_dense_log.ravel()
    y_mean = 10 ** y_pred
    y_lower = 10 ** (y_pred - 1.96 * y_std)
    y_upper = 10 ** (y_pred + 1.96 * y_std)

    return (x_dense, y_mean, y_lower, y_upper,
            np.array(x_obs_list), np.array(y_obs_means))


def _plot_throughput(ax, xs, throughput_data, available_labels, title, xlabel,
                     cache=None, cache_key=None):
    """Plot throughput lines with implementation-specific styling.

    When cache/cache_key are provided, attempts GP regression for smooth curves
    with confidence bands. Falls back to direct line plot if GP fails.
    """
    for label in available_labels:
        vals = throughput_data.get(label, [])
        xf = [xs[i] for i, v in enumerate(vals) if v is not None]
        yf = [v for v in vals if v is not None]
        if not yf:
            continue

        s = IMPL_STYLES.get(label, {"color": "gray", "marker": ".", "linewidth": 1.5, "markersize": 6, "zorder": 1})

        gp_result = _fit_gp(cache, cache_key, label) if cache is not None else None

        if gp_result is not None:
            x_dense, y_mean, y_lower, y_upper, x_obs, y_obs_mean = gp_result
            # Smooth GP curve
            ax.plot(x_dense, y_mean, color=s["color"],
                    linewidth=s["linewidth"], zorder=s["zorder"], label=label)
            # Confidence band
            ax.fill_between(x_dense, y_lower, y_upper,
                            color=s["color"], alpha=0.15, zorder=s["zorder"] - 1)
            # Scatter for observed means
            ax.scatter(x_obs, y_obs_mean, marker=s["marker"], color=s["color"],
                       s=s["markersize"] ** 2, edgecolors="white", linewidths=0.8,
                       zorder=s["zorder"] + 1)
        else:
            # Fallback: direct line plot
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


def _speedup_with_ci(cache, cache_key, gpu_label, cpu_label):
    """Compute speedup with confidence interval using separate GP fits.

    Returns (x_points, speedup_mean, speedup_lower, speedup_upper) or None.
    Speedup = 10^(gpu_log_mean - cpu_log_mean).
    CI via diff_std = sqrt(gpu_std^2 + cpu_std^2).
    """
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    except ImportError:
        return None

    if cache is None or cache_key is None:
        return None

    gpu_data = cache.get(cache_key, {}).get(gpu_label, {})
    cpu_data = cache.get(cache_key, {}).get(cpu_label, {})
    if not gpu_data or not cpu_data:
        return None

    def _build_training_data(impl_data):
        x_all, y_all = [], []
        for val_str, entry in impl_data.items():
            samples = entry.get("samples", [])
            x_val = float(val_str)
            if x_val <= 0:
                continue
            for s in samples:
                if s > 0:
                    x_all.append(np.log2(x_val))
                    y_all.append(np.log10(s))
        return np.array(x_all), np.array(y_all)

    X_gpu, y_gpu = _build_training_data(gpu_data)
    X_cpu, y_cpu = _build_training_data(cpu_data)
    if len(X_gpu) < 2 or len(X_cpu) < 2:
        return None

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

    gp_gpu = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-6)
    gp_cpu = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-6)
    try:
        gp_gpu.fit(X_gpu.reshape(-1, 1), y_gpu)
        gp_cpu.fit(X_cpu.reshape(-1, 1), y_cpu)
    except Exception:
        return None

    # Common x range: intersection of both domains
    x_min = max(X_gpu.min(), X_cpu.min())
    x_max = min(X_gpu.max(), X_cpu.max())
    if x_min >= x_max:
        return None

    X_dense = np.linspace(x_min, x_max, 200).reshape(-1, 1)
    gpu_mean, gpu_std = gp_gpu.predict(X_dense, return_std=True)
    cpu_mean, cpu_std = gp_cpu.predict(X_dense, return_std=True)

    # Speedup in log10 space: log10(speedup) = gpu_log - cpu_log
    diff_mean = gpu_mean - cpu_mean
    diff_std = np.sqrt(gpu_std ** 2 + cpu_std ** 2)

    x_points = 2 ** X_dense.ravel()
    speedup_mean = 10 ** diff_mean
    speedup_lower = 10 ** (diff_mean - 1.96 * diff_std)
    speedup_upper = 10 ** (diff_mean + 1.96 * diff_std)

    return x_points, speedup_mean, speedup_lower, speedup_upper


# ---------------------------------------------------------------------------
#  Cache-to-throughput reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_throughput(cache, cache_key, x_values, labels):
    """Rebuild {label: [mean_or_None]} from cache for plotting.

    For each label and x_value, computes the mean of cached samples,
    or None if no samples exist.
    """
    throughput = {label: [] for label in labels}
    exp_data = cache.get(cache_key, {}) if cache else {}

    for val in x_values:
        val_str = str(val)
        for label in labels:
            entry = exp_data.get(label, {}).get(val_str, {})
            samples = entry.get("samples", []) if isinstance(entry, dict) else []
            if samples:
                throughput[label].append(float(np.mean(samples)))
            else:
                throughput[label].append(None)

    return throughput


# ---------------------------------------------------------------------------
#  Data collection helper
# ---------------------------------------------------------------------------

def _collect_throughput(available, runs, vary_param, vary_values, fixed_params,
                        cache=None, cache_key=None):
    """Run benchmarks varying one parameter, return {label: [throughputs]}.

    Cache format per entry: {"samples": [float, ...], "wall_times": [float, ...]}
    If cache is provided, cached entries are used (mean of samples). New runs
    append to the cache entry rather than overwriting.
    """
    labels = [l for l, _ in available]
    throughput = {l: [] for l in labels}

    # Load existing cached data for this experiment
    cached_exp = {}
    if cache is not None and cache_key is not None:
        cached_exp = cache.get(cache_key, {})

    for val in vary_values:
        val_str = str(val)
        print(f"\n--- {vary_param}={val} ---")
        params = dict(fixed_params)
        params[vary_param] = val

        for label, cmd_template in available:
            # Check cache first
            if label in cached_exp and val_str in cached_exp[label]:
                entry = cached_exp[label][val_str]
                samples = entry.get("samples", [])
                if samples:
                    mean_tp = float(np.mean(samples))
                    n = len(samples)
                    cv = float(np.std(samples) / mean_tp * 100) if n > 1 and mean_tp > 0 else 0.0
                    # Converged: CV < 3% with at least 3 samples — skip
                    if n >= 3 and cv < 3.0:
                        print(f"  {label}... converged ({mean_tp:.0f} samples/s, n={n}, CV={cv:.1f}%)")
                        throughput[label].append(mean_tp)
                        continue
                    # Not converged — use cached mean for throughput but fall through to re-run
                    cv_str = f", CV={cv:.1f}%" if n > 1 else ""
                    print(f"  {label}... n={n}{cv_str}, running again...")
                    # Fall through to run more samples
                else:
                    print(f"  {label}... cached (None)")
                    throughput[label].append(None)
                    continue

            runs_tp = []
            wall_times = []
            for run_idx in range(runs):
                tag = f" (run {run_idx+1}/{runs})" if runs > 1 else ""
                print(f"  {label}{tag}...")
                t0 = time.monotonic()
                r = run_implementation(
                    label, cmd_template, "generated",
                    batch_size=params.get("batch_size", 2048),
                    num_samples=params.get("num_samples", 100000),
                    hidden_size=params.get("hidden_size", 512),
                    epochs=params.get("epochs", 500),
                )
                elapsed = time.monotonic() - t0
                if r and "throughput" in r:
                    runs_tp.append(r["throughput"])
                    wall_times.append(elapsed)

            result = float(np.mean(runs_tp)) if runs_tp else None
            throughput[label].append(result)

            # Store in cache (append, not overwrite)
            if cache is not None and cache_key is not None:
                if cache_key not in cache:
                    cache[cache_key] = {}
                if label not in cache[cache_key]:
                    cache[cache_key][label] = {}
                if val_str not in cache[cache_key][label]:
                    cache[cache_key][label][val_str] = {"samples": [], "wall_times": []}
                cache[cache_key][label][val_str]["samples"].extend(runs_tp)
                cache[cache_key][label][val_str]["wall_times"].extend(wall_times)
                # Print CV info
                all_samples = cache[cache_key][label][val_str]["samples"]
                n = len(all_samples)
                if n > 1 and result and result > 0:
                    cv = float(np.std(all_samples) / np.mean(all_samples) * 100)
                    print(f"    -> stored n={n}, CV={cv:.1f}%")
                elif n > 0:
                    print(f"    -> stored n={n}")

        # Save cache after each x_value completes (ctrl-C safe)
        if cache is not None and cache_key is not None:
            _save_cache(cache)

    return throughput


# ---------------------------------------------------------------------------
#  Variance-weighted scheduler
# ---------------------------------------------------------------------------

def _build_schedule(cache, experiments, available, budget_seconds):
    """Build a greedy schedule of runs prioritized by CV / sqrt(n).

    Args:
        cache: the loaded cache dict
        experiments: list of (cache_key, vary_param, vary_values, fixed_params)
        available: list of (label, cmd_template)
        budget_seconds: total wall-time budget

    Returns list of dicts with keys:
        label, cmd_template, params, cache_key, val_str, est_wall
    """
    # Gather all (impl, x_value) configs with priority scores
    candidates = []
    for cache_key, vary_param, vary_values, fixed_params in experiments:
        exp_data = cache.get(cache_key, {})
        for label, cmd_template in available:
            for val in vary_values:
                val_str = str(val)
                entry = exp_data.get(label, {}).get(val_str, {})
                samples = entry.get("samples", []) if isinstance(entry, dict) else []
                wall_times = entry.get("wall_times", []) if isinstance(entry, dict) else []
                n = len(samples)
                mean_tp = float(np.mean(samples)) if samples else 0.0

                # Skip converged configs (CV < 3%)
                if n >= 2 and mean_tp > 0:
                    cv = float(np.std(samples) / mean_tp * 100)
                    if cv < 3.0:
                        continue
                    priority = cv / np.sqrt(n)
                elif n == 0:
                    priority = float("inf")
                else:
                    # n=1, no CV yet — high priority
                    priority = 100.0

                # Estimate wall time from past runs, or default 60s
                est_wall = float(np.mean(wall_times)) if wall_times else 60.0

                params = dict(fixed_params)
                params[vary_param] = val

                candidates.append({
                    "label": label,
                    "cmd_template": cmd_template,
                    "params": params,
                    "cache_key": cache_key,
                    "val_str": val_str,
                    "est_wall": est_wall,
                    "priority": priority,
                })

    # Sort by priority descending (highest priority first)
    candidates.sort(key=lambda c: c["priority"], reverse=True)

    # Greedy fill within budget
    schedule = []
    remaining = budget_seconds
    for c in candidates:
        if c["est_wall"] <= remaining:
            schedule.append(c)
            remaining -= c["est_wall"]
        if remaining <= 0:
            break

    return schedule


def _run_scheduled(schedule, cache):
    """Execute scheduled runs, appending results to cache after each run.

    Args:
        schedule: list of schedule items from _build_schedule
        cache: the cache dict (modified in-place)
    """
    total = len(schedule)
    for idx, item in enumerate(schedule, 1):
        label = item["label"]
        params = item["params"]
        cache_key = item["cache_key"]
        val_str = item["val_str"]

        # Current stats
        entry = cache.get(cache_key, {}).get(label, {}).get(val_str, {})
        n_before = len(entry.get("samples", []) if isinstance(entry, dict) else [])

        print(f"\n  [{idx}/{total}] {label} {val_str} (n={n_before})...")

        t0 = time.monotonic()
        r = run_implementation(
            label, item["cmd_template"], "generated",
            batch_size=params.get("batch_size", 2048),
            num_samples=params.get("num_samples", 100000),
            hidden_size=params.get("hidden_size", 512),
            epochs=params.get("epochs", 500),
        )
        elapsed = time.monotonic() - t0

        if r and "throughput" in r:
            tp = r["throughput"]
            # Ensure cache structure exists
            if cache_key not in cache:
                cache[cache_key] = {}
            if label not in cache[cache_key]:
                cache[cache_key][label] = {}
            if val_str not in cache[cache_key][label]:
                cache[cache_key][label][val_str] = {"samples": [], "wall_times": []}
            cache[cache_key][label][val_str]["samples"].append(tp)
            cache[cache_key][label][val_str]["wall_times"].append(elapsed)

            all_samples = cache[cache_key][label][val_str]["samples"]
            n = len(all_samples)
            mean_tp = float(np.mean(all_samples))
            cv = float(np.std(all_samples) / mean_tp * 100) if n > 1 and mean_tp > 0 else 0.0
            print(f"    -> {tp:.0f} samples/s, n={n}, CV={cv:.1f}%")
        else:
            print(f"    -> FAILED")

        # Save cache after every run (ctrl-C safe)
        _save_cache(cache)


# ---------------------------------------------------------------------------
#  Scaling mode
# ---------------------------------------------------------------------------

def run_scaling_experiment(available, runs, figs_dir, cfg, model="mlp",
                           use_cache=True):
    """Run GPU-focused scaling experiments with multiple plot types."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available -- scaling mode requires matplotlib")
        return

    _setup_style()
    os.makedirs(figs_dir, exist_ok=True)

    # Temporarily override global IMPL_STYLES for _plot_throughput
    global IMPL_STYLES
    orig_styles = IMPL_STYLES

    # Select model-specific styles and speedup pairs
    styles = CNN_IMPL_STYLES if model == "cnn" else orig_styles
    speedup_pairs = CNN_SPEEDUP_PAIRS if model == "cnn" else SPEEDUP_PAIRS
    prefix = f"{model.upper()} " if model != "mlp" else ""
    file_prefix = f"{model}_" if model != "mlp" else ""

    IMPL_STYLES = styles

    # Load or initialize cache
    cache = _load_cache() if use_cache else {}

    epochs = cfg["epochs"]
    labels = [l for l, _ in available]
    budget_seconds = cfg.get("budget_seconds", None)

    # ================================================================
    #  Budget mode: variance-weighted scheduler
    # ================================================================
    if budget_seconds is not None:
        print(f"\n  Budget mode: {budget_seconds}s wall-time budget")
        print("=" * 70)

        # Build experiment descriptors
        budget_experiments = []
        if model == "cnn":
            batch_sizes = cfg.get("batch_sizes", [2**n for n in range(4, 11)])
            budget_experiments.append((
                _cache_key(model, "batch_size"), "batch_size", batch_sizes,
                {"epochs": epochs}))
        else:
            dataset_sizes = cfg["dataset_sizes"]
            batch_sizes = cfg["batch_sizes"]
            hidden_sizes = cfg["hidden_sizes"]
            fixed_bs = cfg["fixed_batch_size"]
            fixed_ns = cfg["fixed_num_samples"]

            budget_experiments.append((
                _cache_key(model, "dataset_size"), "num_samples", dataset_sizes,
                {"batch_size": fixed_bs, "epochs": epochs}))
            budget_experiments.append((
                _cache_key(model, "batch_size"), "batch_size", batch_sizes,
                {"num_samples": fixed_ns, "epochs": epochs}))
            budget_experiments.append((
                _cache_key(model, "hidden_size"), "hidden_size", hidden_sizes,
                {"num_samples": fixed_ns, "batch_size": fixed_bs, "epochs": epochs}))

        schedule = _build_schedule(cache, budget_experiments, available, budget_seconds)
        print(f"  Scheduled {len(schedule)} runs (est. {sum(s['est_wall'] for s in schedule):.0f}s)")
        _run_scheduled(schedule, cache)

        # Reconstruct throughput for plotting
        if model == "cnn":
            tp_ds = None
            tp_hs = None
            tp_bs = _reconstruct_throughput(
                cache, _cache_key(model, "batch_size"), batch_sizes, labels)
            dataset_sizes = batch_sizes
            fixed_bs = batch_sizes[-1]
            fixed_ns = 60000
        else:
            tp_ds = _reconstruct_throughput(
                cache, _cache_key(model, "dataset_size"), dataset_sizes, labels)
            tp_bs = _reconstruct_throughput(
                cache, _cache_key(model, "batch_size"), batch_sizes, labels)
            tp_hs = _reconstruct_throughput(
                cache, _cache_key(model, "hidden_size"), hidden_sizes, labels)

        # Jump to plotting (skip the normal collection flow)
        # -- individual throughput plots --
        if tp_ds is not None:
            fig, ax = plt.subplots(figsize=(12, 7))
            _plot_throughput(ax, dataset_sizes, tp_ds, labels,
                             f"Throughput vs Dataset Size\nbatch_size={fixed_bs}, epochs={epochs}",
                             "Dataset Size (samples)",
                             cache=cache, cache_key=_cache_key(model, "dataset_size"))
            _format_x_samples(ax)
            plt.tight_layout()
            p = os.path.join(figs_dir, f"{file_prefix}scaling_dataset_size.png")
            plt.savefig(p); plt.close()
            print(f"\nSaved: {p}")

        fig, ax = plt.subplots(figsize=(12, 7))
        bs_title = (f"{prefix}Throughput vs Batch Size\nMNIST (60K samples), epochs={epochs}"
                     if model == "cnn"
                     else f"Throughput vs Batch Size\nnum_samples={fixed_ns:,}, epochs={epochs}")
        _plot_throughput(ax, batch_sizes, tp_bs, labels, bs_title, "Batch Size",
                         cache=cache, cache_key=_cache_key(model, "batch_size"))
        plt.tight_layout()
        p = os.path.join(figs_dir, f"{file_prefix}scaling_batch_size.png")
        plt.savefig(p); plt.close()
        print(f"\nSaved: {p}")

        if tp_hs is not None:
            fig, ax = plt.subplots(figsize=(12, 7))
            _plot_throughput(ax, hidden_sizes, tp_hs, labels,
                             f"Throughput vs Hidden Layer Size\nsamples={fixed_ns:,}, batch={fixed_bs}, epochs={epochs}",
                             "Hidden Size",
                             cache=cache, cache_key=_cache_key(model, "hidden_size"))
            plt.tight_layout()
            p = os.path.join(figs_dir, f"{file_prefix}scaling_hidden_size.png")
            plt.savefig(p); plt.close()
            print(f"\nSaved: {p}")

        # Fall through to speedup plots, overview, and summary below
        # (they use tp_ds, tp_bs, tp_hs which are now set)

    # For CNN, only batch size scaling makes sense (fixed MNIST dataset, fixed architecture)
    elif model == "cnn":
        batch_sizes = cfg.get("batch_sizes", [2**n for n in range(4, 11)])  # 16 .. 1024
        tp_ds = None
        tp_hs = None

        print("=" * 70)
        print(f"  CNN Batch Size Scaling  (MNIST 60K, epochs={epochs})")
        print("=" * 70)

        tp_bs = _collect_throughput(
            available, runs, "batch_size", batch_sizes,
            {"epochs": epochs},
            cache=cache, cache_key=_cache_key(model, "batch_size"))

        fig, ax = plt.subplots(figsize=(12, 7))
        _plot_throughput(ax, batch_sizes, tp_bs, labels,
                         f"{prefix}Throughput vs Batch Size\nMNIST (60K samples), epochs={epochs}",
                         "Batch Size",
                         cache=cache, cache_key=_cache_key(model, "batch_size"))
        plt.tight_layout()
        p = os.path.join(figs_dir, f"{file_prefix}scaling_batch_size.png")
        plt.savefig(p)
        plt.close()
        print(f"\nSaved: {p}")

        dataset_sizes = batch_sizes  # reuse for speedup x-axis
        fixed_bs = batch_sizes[-1]
        fixed_ns = 60000

    else:
        # MLP: full three-axis scaling
        # ================================================================
        #  Experiment 1: Throughput vs Dataset Size
        # ================================================================
        dataset_sizes = cfg["dataset_sizes"]
        fixed_bs = cfg["fixed_batch_size"]

        print("=" * 70)
        print(f"  Experiment 1: Dataset Size Scaling  (batch={fixed_bs}, epochs={epochs})")
        print("=" * 70)

        tp_ds = _collect_throughput(
            available, runs, "num_samples", dataset_sizes,
            {"batch_size": fixed_bs, "epochs": epochs},
            cache=cache, cache_key=_cache_key(model, "dataset_size"))

        fig, ax = plt.subplots(figsize=(12, 7))
        _plot_throughput(ax, dataset_sizes, tp_ds, labels,
                         f"Throughput vs Dataset Size\nbatch_size={fixed_bs}, epochs={epochs}",
                         "Dataset Size (samples)",
                         cache=cache, cache_key=_cache_key(model, "dataset_size"))
        _format_x_samples(ax)
        plt.tight_layout()
        p = os.path.join(figs_dir, f"{file_prefix}scaling_dataset_size.png")
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

        tp_bs = _collect_throughput(
            available, runs, "batch_size", batch_sizes,
            {"num_samples": fixed_ns, "epochs": epochs},
            cache=cache, cache_key=_cache_key(model, "batch_size"))

        fig, ax = plt.subplots(figsize=(12, 7))
        _plot_throughput(ax, batch_sizes, tp_bs, labels,
                         f"Throughput vs Batch Size\nnum_samples={fixed_ns:,}, epochs={epochs}",
                         "Batch Size",
                         cache=cache, cache_key=_cache_key(model, "batch_size"))
        plt.tight_layout()
        p = os.path.join(figs_dir, f"{file_prefix}scaling_batch_size.png")
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

        tp_hs = _collect_throughput(
            available, runs, "hidden_size", hidden_sizes,
            {"num_samples": fixed_ns, "batch_size": fixed_bs, "epochs": epochs},
            cache=cache, cache_key=_cache_key(model, "hidden_size"))

        fig, ax = plt.subplots(figsize=(12, 7))
        _plot_throughput(ax, hidden_sizes, tp_hs, labels,
                         f"Throughput vs Hidden Layer Size\nsamples={fixed_ns:,}, batch={fixed_bs}, epochs={epochs}",
                         "Hidden Size",
                         cache=cache, cache_key=_cache_key(model, "hidden_size"))
        plt.tight_layout()
        p = os.path.join(figs_dir, f"{file_prefix}scaling_hidden_size.png")
        plt.savefig(p)
        plt.close()
        print(f"\nSaved: {p}")

    # ================================================================
    #  GPU Speedup plots
    # ================================================================
    print("\n" + "=" * 70)
    print("  Generating GPU speedup plots...")
    print("=" * 70)

    # Build experiments list based on available data
    experiments = []
    if tp_ds is not None:
        experiments.append(("dataset_size", dataset_sizes, tp_ds, "Dataset Size (samples)", True))
    experiments.append(("batch_size", batch_sizes, tp_bs, "Batch Size", False))
    if tp_hs is not None:
        experiments.append(("hidden_size", hidden_sizes, tp_hs, "Hidden Size", False))

    for exp_name, xs, tp_data, xlabel, use_sample_fmt in experiments:
        fig, ax = plt.subplots(figsize=(12, 7))
        sp_cache_key = _cache_key(model, exp_name)

        any_plotted = False
        for gpu_label, cpu_label, color, legend_name in speedup_pairs:
            gpu_vals = tp_data.get(gpu_label, [])
            cpu_vals = tp_data.get(cpu_label, [])
            if not gpu_vals or not cpu_vals:
                continue

            # Raw ratios for scatter overlay
            xf, yf = [], []
            for i in range(min(len(gpu_vals), len(cpu_vals))):
                if gpu_vals[i] is not None and cpu_vals[i] is not None and cpu_vals[i] > 0:
                    xf.append(xs[i])
                    yf.append(gpu_vals[i] / cpu_vals[i])

            # Try GP-based speedup with CI
            sp_result = _speedup_with_ci(cache, sp_cache_key, gpu_label, cpu_label) if cache else None

            if sp_result is not None and yf:
                x_pts, sp_mean, sp_lower, sp_upper = sp_result
                ax.plot(x_pts, sp_mean, color=color, linewidth=2.5, label=legend_name)
                ax.fill_between(x_pts, sp_lower, sp_upper, color=color, alpha=0.15)
                ax.scatter(xf, yf, color=color, marker="o", s=64,
                           edgecolors="white", linewidths=0.8, zorder=10)
                any_plotted = True
            elif yf:
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
        ax.set_title(f"{prefix}GPU Speedup vs {xlabel}\n(higher = GPU wins)", fontsize=14, fontweight="bold")
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
                ydata = h.get_ydata()
                peak_sp[ll] = max(ydata) if len(ydata) > 0 else 0
        paired = sorted(zip(handles, leg_labels), key=lambda x: peak_sp.get(x[1], 0), reverse=True)
        handles, leg_labels = zip(*paired)
        ax.legend(handles, leg_labels, loc="best")

        plt.tight_layout()
        p = os.path.join(figs_dir, f"{file_prefix}gpu_speedup_{exp_name}.png")
        plt.savefig(p)
        plt.close()
        print(f"  Saved: {p}")

    # ================================================================
    #  Combined overview plot
    # ================================================================
    if model == "cnn":
        # CNN: single row — throughput + speedup
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        _plot_throughput(axes[0], batch_sizes, tp_bs, labels,
                         "Batch Size Scaling (MNIST)", "Batch Size",
                         cache=cache, cache_key=_cache_key(model, "batch_size"))

        for gpu_label, cpu_label, color, legend_name in speedup_pairs[:3]:
            gpu_vals = tp_bs.get(gpu_label, [])
            cpu_vals = tp_bs.get(cpu_label, [])
            if not gpu_vals or not cpu_vals:
                continue
            xf, yf = [], []
            for i in range(min(len(gpu_vals), len(cpu_vals))):
                if gpu_vals[i] and cpu_vals[i] and cpu_vals[i] > 0:
                    xf.append(batch_sizes[i])
                    yf.append(gpu_vals[i] / cpu_vals[i])
            if yf:
                axes[1].plot(xf, yf, marker="o", color=color, linewidth=2.5, markersize=8, label=legend_name)

        axes[1].axhline(y=1.0, color="#e74c3c", linestyle="--", alpha=0.7, linewidth=1.5)
        axes[1].set_xscale("log", base=2)
        axes[1].set_xlabel("Batch Size")
        axes[1].set_ylabel("Speedup (x)")
        axes[1].set_title("GPU Speedup vs Batch Size", fontweight="bold")
        axes[1].grid(True, which="both", alpha=0.3)
        axes[1].legend(loc="best")

        fig.suptitle("CNN Benchmark: GPU Scaling Analysis (LeNet-5 / MNIST)",
                      fontsize=18, fontweight="bold", y=0.98)
    else:
        # MLP: 2x2 overview
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        _plot_throughput(axes[0, 0], dataset_sizes, tp_ds, labels,
                         f"Dataset Size Scaling (batch={fixed_bs})", "Dataset Size",
                         cache=cache, cache_key=_cache_key(model, "dataset_size"))
        _format_x_samples(axes[0, 0])

        _plot_throughput(axes[0, 1], batch_sizes, tp_bs, labels,
                         f"Batch Size Scaling (samples={fixed_ns:,})", "Batch Size",
                         cache=cache, cache_key=_cache_key(model, "batch_size"))

        _plot_throughput(axes[1, 0], hidden_sizes, tp_hs, labels,
                         f"Hidden Size Scaling (batch={fixed_bs})", "Hidden Size",
                         cache=cache, cache_key=_cache_key(model, "hidden_size"))

        for gpu_label, cpu_label, color, legend_name in speedup_pairs[:3]:
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
    p = os.path.join(figs_dir, f"{file_prefix}scaling_overview.png")
    plt.savefig(p)
    plt.close()
    print(f"\n  Saved: {p}")

    # Print summary table
    print("\n" + "=" * 70)
    print("  PEAK THROUGHPUT SUMMARY")
    print("=" * 70)
    all_experiments = []
    if tp_ds is not None:
        all_experiments.append(("Dataset Size", tp_ds))
    all_experiments.append(("Batch Size", tp_bs))
    if tp_hs is not None:
        all_experiments.append(("Hidden Size", tp_hs))

    for exp_name, tp_data in all_experiments:
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
                print(f"    {i+1}. {label:<25} {peak/1e6:>8.2f}M/s  {bar}")
            else:
                print(f"    {i+1}. {label:<25} {peak/1e3:>8.0f}K/s  {bar}")

    # Save cache and restore styles
    if use_cache:
        _save_cache(cache)
    IMPL_STYLES = orig_styles


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
    parser = argparse.ArgumentParser(description="Benchmark Runner (MLP / CNN)")
    parser.add_argument("--mode", default="standard", choices=["standard", "scaling"])
    parser.add_argument("--model", default="mlp", choices=["mlp", "cnn"],
                        help="Model family to benchmark (default: mlp)")
    parser.add_argument("--datasets", default="generated,iris,breast-cancer",
                        help="Comma-separated dataset names (standard mode, MLP only)")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--budget", type=int, default=None, metavar="MINUTES",
                        help="Time budget in minutes for variance-weighted scheduling "
                             "(scaling mode only, mutually exclusive with --runs > 1)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable result caching (re-run everything)")

    # Scaling parameters -- all power-of-2 by default
    parser.add_argument("--scaling-epochs", type=int, default=200)
    parser.add_argument("--scaling-fixed-bs", type=int, default=4096)
    parser.add_argument("--scaling-fixed-ns", type=int, default=262144,
                        help="Fixed num_samples (default: 2^18 = 262144)")
    parser.add_argument("--scaling-dataset-sizes", type=str, default=None,
                        help="Comma-separated (default: 2^12..2^20)")
    parser.add_argument("--scaling-batch-sizes", type=str, default=None,
                        help="Comma-separated (default: 2^7..2^14)")
    parser.add_argument("--scaling-hidden-sizes", type=str, default=None,
                        help="Comma-separated (default: 2^5..2^10)")
    args = parser.parse_args()

    if args.budget is not None and args.runs > 1:
        parser.error("--budget and --runs > 1 are mutually exclusive")

    figs_dir = os.path.join(PROJECT_ROOT, "figs")

    # Select implementation list based on model
    impl_list = CNN_IMPLEMENTATIONS if args.model == "cnn" else IMPLEMENTATIONS
    available = [(label, cmd) for label, cmd in impl_list if is_available(label)]
    print(f"Model: {args.model.upper()}")
    print(f"Available implementations: {[l for l, _ in available]}")
    print(f"Mode: {args.mode}")
    if args.budget is not None:
        print(f"Budget: {args.budget}s (variance-weighted scheduling)")
    else:
        print(f"Runs per config: {args.runs}")
    if not args.no_cache:
        print(f"Cache: {CACHE_FILE}")
    print()

    if args.mode == "scaling":
        if args.model == "cnn":
            # CNN: only batch size scaling (MNIST is fixed 60K, architecture is fixed)
            cfg = {
                "epochs": args.scaling_epochs,
                "batch_sizes": ([int(x) for x in args.scaling_batch_sizes.split(",")]
                                if args.scaling_batch_sizes
                                else [2**n for n in range(4, 11)]),    # 16 .. 1024
            }
        else:
            cfg = {
                "epochs": args.scaling_epochs,
                "fixed_batch_size": args.scaling_fixed_bs,
                "fixed_num_samples": args.scaling_fixed_ns,
                "dataset_sizes": ([int(x) for x in args.scaling_dataset_sizes.split(",")]
                                  if args.scaling_dataset_sizes
                                  else [2**n for n in range(13, 23)]),     # 8K .. 4M
                "batch_sizes": ([int(x) for x in args.scaling_batch_sizes.split(",")]
                                if args.scaling_batch_sizes
                                else [2**n for n in range(8, 16)]),        # 256 .. 32K
                "hidden_sizes": ([int(x) for x in args.scaling_hidden_sizes.split(",")]
                                 if args.scaling_hidden_sizes
                                 else [2**n for n in range(6, 13)]),       # 64 .. 4096
            }
        if args.budget is not None:
            cfg["budget_seconds"] = args.budget * 60 if args.budget else None
        run_scaling_experiment(available, args.runs, figs_dir, cfg,
                               model=args.model, use_cache=not args.no_cache)
    else:
        if args.model == "cnn":
            # CNN standard mode: run on MNIST only
            datasets = ["mnist"]
        else:
            datasets = [d.strip() for d in args.datasets.split(",")]
        print(f"Datasets: {datasets}\n")
        run_standard(datasets, available, args.runs, figs_dir)


if __name__ == "__main__":
    main()
