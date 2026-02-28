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

from config import load_config

# Suppress sklearn GP convergence warnings (noisy with few data points)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

import numpy as np

from plotting import (
    HAS_MATPLOTLIB, IMPL_STYLES, CNN_IMPL_STYLES, SPEEDUP_PAIRS, CNN_SPEEDUP_PAIRS,
    _setup_style, _plot_throughput, _format_x_samples, _speedup_with_ci,
    _reconstruct_throughput, _discover_x_values,
    generate_charts, plot_scaling_results, plot_speedup_results, plot_overview,
    print_peak_summary,
)

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


# Project root (two levels up from src/scripts/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Implementation definitions with configurable placeholders.
IMPLEMENTATIONS = [
    ("C (CPU)",
     "./src/c/build_cpu/main --dataset {dataset} --batch-size {batch_size} --num-samples {num_samples} --hidden-size {hidden_size} --epochs {epochs} --learning-rate {learning_rate}"),
    ("C (CUDA)",
     "./src/c/build_cuda/main --dataset {dataset} --batch-size {batch_size} --num-samples {num_samples} --hidden-size {hidden_size} --epochs {epochs} --learning-rate {learning_rate}"),
    ("NumPy",
     "{python} src/python/models/mlp/mlp_numpy.py --dataset {dataset} --batch-size {batch_size} --num-samples {num_samples} --hidden-size {hidden_size} --epochs {epochs} --learning-rate {learning_rate}"),
    ("PyTorch (CPU)",
     "{python} src/python/models/mlp/mlp_pytorch.py --dataset {dataset} --device cpu --batch-size {batch_size} --num-samples {num_samples} --hidden-size {hidden_size} --epochs {epochs} --learning-rate {learning_rate}"),
    ("PyTorch (CUDA)",
     "{python} src/python/models/mlp/mlp_pytorch.py --dataset {dataset} --device cuda --batch-size {batch_size} --num-samples {num_samples} --hidden-size {hidden_size} --epochs {epochs} --learning-rate {learning_rate}"),
    ("Rust (CPU)",
     "./src/rust/target/release/mlp-cpu --dataset {dataset} --batch-size {batch_size} --num-samples {num_samples} --hidden-size {hidden_size} --epochs {epochs} --learning-rate {learning_rate}"),
    ("Rust (cuBLAS)",
     "./src/rust/target/release/mlp-cuda-cublas --dataset {dataset} --batch-size {batch_size} --num-samples {num_samples} --hidden-size {hidden_size} --epochs {epochs} --learning-rate {learning_rate}"),
    ("Rust (CUDA Kernels)",
     "./src/rust/target/release/mlp-cuda-kernels --dataset {dataset} --batch-size {batch_size} --num-samples {num_samples} --hidden-size {hidden_size} --epochs {epochs} --learning-rate {learning_rate}"),
]

# CNN implementations — fixed architecture (LeNet-5), MNIST only.
# No --num-samples or --hidden-size flags; dataset is always mnist.
# NumPy CNN omitted — pure-Python im2col is too slow for meaningful benchmarks
# (times out at 600s even with 5 epochs).  NumPy remains in MLP where its
# BLAS backend makes it a competitive CPU baseline.
CNN_IMPLEMENTATIONS = [
    ("CNN C (CPU)",
     "./src/c/build_cpu/cnn_main --dataset mnist --batch-size {batch_size} --epochs {epochs} --learning-rate {learning_rate}"),
    ("CNN C (CUDA)",
     "./src/c/build_cuda/cnn_main --dataset mnist --batch-size {batch_size} --epochs {epochs} --learning-rate {learning_rate}"),
    ("CNN PyTorch (CPU)",
     "{python} src/python/models/cnn/cnn_pytorch.py --dataset mnist --device cpu --batch-size {batch_size} --epochs {epochs} --learning-rate {learning_rate}"),
    ("CNN PyTorch (CUDA)",
     "{python} src/python/models/cnn/cnn_pytorch.py --dataset mnist --device cuda --batch-size {batch_size} --epochs {epochs} --learning-rate {learning_rate}"),
    ("CNN Rust (CPU)",
     "./src/rust/target/release/cnn-cpu --dataset mnist --batch-size {batch_size} --epochs {epochs} --learning-rate {learning_rate}"),
    ("CNN Rust (cuBLAS)",
     "./src/rust/target/release/cnn-cuda-cublas --dataset mnist --batch-size {batch_size} --epochs {epochs} --learning-rate {learning_rate}"),
    ("CNN Rust (CUDA Kernels)",
     "./src/rust/target/release/cnn-cuda-kernels --dataset mnist --batch-size {batch_size} --epochs {epochs} --learning-rate {learning_rate}"),
]

# Regex patterns for parsing standardized output
PATTERN_LOSS = re.compile(r"Test Loss:\s+([\d.]+)")
PATTERN_ACC = re.compile(r"Test Accuracy:\s+([\d.]+)%")
PATTERN_TRAIN = re.compile(r"Train time:\s+([\d.]+)\s*s")
PATTERN_EVAL = re.compile(r"Eval time:\s+([\d.]+)\s*s")
PATTERN_THROUGHPUT = re.compile(r"Throughput:\s+([\d.]+)\s*samples/s")


# Style constants imported from plotting module


# ---------------------------------------------------------------------------
#  Benchmark result cache — avoids re-running expensive experiments
# ---------------------------------------------------------------------------

CACHE_FILE = os.path.join(PROJECT_ROOT, "results", "cache", "benchmark_cache.json")


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


def run_implementation(label, cmd_template, dataset, batch_size=4096,
                       num_samples=0, hidden_size=512, epochs=1000,
                       learning_rate=0.02):
    """Run a single implementation and parse its output."""
    cmd = (cmd_template
           .replace("{dataset}", dataset)
           .replace("{python}", sys.executable)
           .replace("{batch_size}", str(batch_size))
           .replace("{num_samples}", str(num_samples))
           .replace("{hidden_size}", str(hidden_size))
           .replace("{epochs}", str(epochs))
           .replace("{learning_rate}", str(learning_rate)))

    env = os.environ.copy()
    if label in ("C (CPU)", "CNN C (CPU)"):
        build_dir = os.path.join(PROJECT_ROOT, "src", "c", "build_cpu")
        lib_dirs = [build_dir, os.path.join(build_dir, "utils"), os.path.join(build_dir, "models", "mlp"), os.path.join(build_dir, "models", "cnn")]
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + env.get("LD_LIBRARY_PATH", "")
    elif label in ("C (CUDA)", "CNN C (CUDA)"):
        build_dir = os.path.join(PROJECT_ROOT, "src", "c", "build_cuda")
        lib_dirs = [build_dir, os.path.join(build_dir, "utils"), os.path.join(build_dir, "models", "mlp"), os.path.join(build_dir, "models", "cnn")]
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + env.get("LD_LIBRARY_PATH", "")

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



# Plot helpers, GP regression, cache reconstruction, and chart generation
# are all imported from plotting.py above.


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
                    learning_rate=params.get("learning_rate", 0.02),
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

def _sample_from_gaps(lo_exp, hi_exp, existing_log2_vals, sample_counts=None):
    """Sample a new x-value, balancing spatial coverage with variance reduction.

    Phase 1 (largest gap > 1.0 in log2 space): Gap-fill — sample from the
    widest unexplored region so the GP gets spread-out training data.

    Phase 2 (range well-covered): Resample near existing points that have
    the fewest observations, so the GP can better estimate noise/variance
    at each location.  Adds a small jitter (±0.2 in log2) so the GP sees
    *nearby* points rather than exact duplicates.

    Args:
        lo_exp, hi_exp: log2 boundaries of the x-range
        existing_log2_vals: list of log2(x) for all cached x-values
        sample_counts: dict mapping log2_val -> number of samples at that point.
            If provided, used to prioritize under-sampled points in phase 2.

    Returns: int, the sampled x-value (round(2^x))
    """
    import random

    if not existing_log2_vals:
        return int(round(2 ** random.uniform(lo_exp, hi_exp)))

    # Build sorted unique points including boundaries
    points = sorted(set([lo_exp] + list(existing_log2_vals) + [hi_exp]))
    gaps = [(points[i + 1] - points[i], points[i], points[i + 1])
            for i in range(len(points) - 1)]
    max_gap = max(g[0] for g in gaps)

    if max_gap > 1.0:
        # Phase 1: Gap-fill — largest gap is > 1 octave (factor of 2)
        gaps.sort(reverse=True)
        gap_size, gap_lo, gap_hi = gaps[0]
        inset = min(0.05, gap_size * 0.1)
        x_exp = random.uniform(gap_lo + inset, gap_hi - inset)
    else:
        # Phase 2: Range is well-covered — resample near the point with
        # fewest observations to reduce variance there.
        if sample_counts:
            # Pick the existing point with fewest samples
            min_n = min(sample_counts.values())
            candidates = [v for v, n in sample_counts.items() if n == min_n]
            target = random.choice(candidates)
        else:
            # No count info — pick a random existing point
            target = random.choice(list(existing_log2_vals))
        # Jitter ±0.2 in log2 (factor of ~1.15x) — close enough for GP
        # noise estimation but avoids exact duplicates
        jitter = random.uniform(-0.2, 0.2)
        x_exp = target + jitter

    x_exp = max(lo_exp, min(hi_exp, x_exp))
    return int(round(2 ** x_exp))


def _build_schedule(cache, experiments, available, budget_seconds):
    """Build a round-robin schedule that samples continuous x-values.

    Instead of a fixed grid, each (experiment, implementation) group gets
    x-values sampled from the largest gap in its existing data (gap-filling).
    Groups with fewer cached points are prioritized, ensuring every impl and
    experiment gets coverage.  Running --budget 1 sixty times produces similar
    coverage to --budget 60 once.

    Args:
        cache: the loaded cache dict
        experiments: list of (cache_key, vary_param, (lo_exp, hi_exp), fixed_params)
            where lo_exp/hi_exp define the log2 range, e.g. (13, 22) → 8K..4M
        available: list of (label, cmd_template)
        budget_seconds: total wall-time budget

    Returns list of dicts with keys:
        label, cmd_template, params, cache_key, val_str, est_wall
    """
    # Build group info for each (experiment, implementation) pair
    groups = []
    for cache_key, vary_param, x_range, fixed_params in experiments:
        lo_exp, hi_exp = x_range
        exp_data = cache.get(cache_key, {})
        for label, cmd_template in available:
            impl_data = exp_data.get(label, {})
            # Collect existing x-values in log2 space, sample counts, and wall times
            existing_log2 = []
            sample_counts = {}  # log2_val -> number of samples
            wall_times = []
            n_cached = 0
            for val_str, entry in impl_data.items():
                if isinstance(entry, dict):
                    ns = len(entry.get("samples", []))
                    n_cached += ns
                    wall_times.extend(entry.get("wall_times", []))
                    try:
                        log2_val = np.log2(float(val_str))
                        existing_log2.append(log2_val)
                        sample_counts[log2_val] = ns
                    except (ValueError, ZeroDivisionError):
                        pass
            est_wall = float(np.mean(wall_times)) if wall_times else 10.0
            groups.append({
                "cache_key": cache_key,
                "label": label,
                "cmd_template": cmd_template,
                "vary_param": vary_param,
                "lo_exp": lo_exp,
                "hi_exp": hi_exp,
                "fixed_params": fixed_params,
                "n_cached": n_cached,
                "est_wall": est_wall,
                "existing_log2": existing_log2,
                "sample_counts": sample_counts,
            })

    # Round-robin across groups, least-sampled first, gap-filling x-values
    schedule = []
    remaining = budget_seconds
    while remaining > 0 and groups:
        # Sort by n_cached so least-known groups get picks first
        groups.sort(key=lambda g: g["n_cached"])
        made_progress = False
        for group in groups:
            if group["est_wall"] > remaining:
                continue
            # Sample from gap or resample near under-observed points
            x_val = _sample_from_gaps(
                group["lo_exp"], group["hi_exp"], group["existing_log2"],
                sample_counts=group["sample_counts"])
            params = dict(group["fixed_params"])
            params[group["vary_param"]] = x_val
            schedule.append({
                "label": group["label"],
                "cmd_template": group["cmd_template"],
                "params": params,
                "cache_key": group["cache_key"],
                "val_str": str(x_val),
                "est_wall": group["est_wall"],
            })
            remaining -= group["est_wall"]
            group["n_cached"] += 1
            log2_val = np.log2(float(x_val))
            group["existing_log2"].append(log2_val)
            group["sample_counts"][log2_val] = group["sample_counts"].get(log2_val, 0) + 1
            made_progress = True
        if not made_progress:
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
            learning_rate=params.get("learning_rate", 0.02),
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
    """Run GPU-focused scaling experiments, then generate plots.

    Data collection stays here; all chart generation is delegated to plotting.py.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available -- scaling mode requires matplotlib")
        return

    os.makedirs(figs_dir, exist_ok=True)

    # Load or initialize cache
    cache = _load_cache() if use_cache else {}

    from plotting import _cache_key
    epochs = cfg["epochs"]
    labels = [l for l, _ in available]
    budget_seconds = cfg.get("budget_seconds", None)

    # ================================================================
    #  Budget mode: variance-weighted scheduler
    # ================================================================
    if budget_seconds is not None:
        print(f"\n  Budget mode: {budget_seconds}s wall-time budget")
        print("=" * 70)

        budget_experiments = []
        if model == "cnn":
            budget_experiments.append((
                _cache_key(model, "batch_size"), "batch_size", (4, 10),
                {"epochs": epochs}))
        else:
            fixed_bs = cfg["fixed_batch_size"]
            fixed_ns = cfg["fixed_num_samples"]

            budget_experiments.append((
                _cache_key(model, "dataset_size"), "num_samples",
                cfg.get("dataset_size_range", (13, 22)),
                {"batch_size": fixed_bs, "epochs": epochs}))
            budget_experiments.append((
                _cache_key(model, "batch_size"), "batch_size",
                cfg.get("batch_size_range", (8, 15)),
                {"num_samples": fixed_ns, "epochs": epochs}))
            budget_experiments.append((
                _cache_key(model, "hidden_size"), "hidden_size",
                cfg.get("hidden_size_range", (6, 12)),
                {"num_samples": fixed_ns, "batch_size": fixed_bs, "epochs": epochs}))

        schedule = _build_schedule(cache, budget_experiments, available, budget_seconds)
        print(f"  Scheduled {len(schedule)} runs (est. {sum(s['est_wall'] for s in schedule):.0f}s)")
        _run_scheduled(schedule, cache)

    # For CNN, only batch size scaling (fixed MNIST dataset, fixed architecture)
    elif model == "cnn":
        batch_sizes = cfg.get("batch_sizes", [2**n for n in range(4, 11)])

        print("=" * 70)
        print(f"  CNN Batch Size Scaling  (MNIST 60K, epochs={epochs})")
        print("=" * 70)

        _collect_throughput(
            available, runs, "batch_size", batch_sizes,
            {"epochs": epochs, "learning_rate": 0.32},
            cache=cache, cache_key=_cache_key(model, "batch_size"))

    else:
        # MLP: full three-axis scaling
        dataset_sizes = cfg["dataset_sizes"]
        fixed_bs = cfg["fixed_batch_size"]

        print("=" * 70)
        print(f"  Experiment 1: Dataset Size Scaling  (batch={fixed_bs}, epochs={epochs})")
        print("=" * 70)

        _collect_throughput(
            available, runs, "num_samples", dataset_sizes,
            {"batch_size": fixed_bs, "epochs": epochs},
            cache=cache, cache_key=_cache_key(model, "dataset_size"))

        batch_sizes = cfg["batch_sizes"]
        fixed_ns = cfg["fixed_num_samples"]

        print("\n" + "=" * 70)
        print(f"  Experiment 2: Batch Size Scaling  (samples={fixed_ns}, epochs={epochs})")
        print("=" * 70)

        _collect_throughput(
            available, runs, "batch_size", batch_sizes,
            {"num_samples": fixed_ns, "epochs": epochs},
            cache=cache, cache_key=_cache_key(model, "batch_size"))

        hidden_sizes = cfg["hidden_sizes"]

        print("\n" + "=" * 70)
        print(f"  Experiment 3: Hidden Size Scaling  (samples={fixed_ns}, batch={fixed_bs}, epochs={epochs})")
        print("=" * 70)

        _collect_throughput(
            available, runs, "hidden_size", hidden_sizes,
            {"num_samples": fixed_ns, "batch_size": fixed_bs, "epochs": epochs},
            cache=cache, cache_key=_cache_key(model, "hidden_size"))

    # Save cache
    if use_cache:
        _save_cache(cache)

    # Delegate all plotting to the plotting module
    print("\n" + "=" * 70)
    print("  Generating plots...")
    print("=" * 70)
    plot_scaling_results(cache, cfg, model, labels, figs_dir)
    plot_speedup_results(cache, cfg, model, labels, figs_dir)
    plot_overview(cache, cfg, model, labels, figs_dir)
    print_peak_summary(cache, model, labels)


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


def run_standard(datasets, available, runs, figs_dir, model="mlp"):
    results = defaultdict(dict)
    for dataset in datasets:
        print(f"--- {dataset} ---")
        for label, cmd_template in available:
            run_results = []
            for run in range(runs):
                tag = f" (run {run + 1}/{runs})" if runs > 1 else ""
                print(f"  {label}{tag}...")
                r = run_implementation(label, cmd_template, dataset,
                                       learning_rate=learning_rate)
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
    generate_charts(results, figs_dir, model=model)


# ---------------------------------------------------------------------------
#  Unified cross-model scheduler
# ---------------------------------------------------------------------------

def run_unified_scaling(budget_seconds, bench_cfg, figs_dir_root, cfg_by_model,
                        available_by_model, use_cache=True):
    """Run variance-weighted scheduling across both MLP and CNN models.

    Phase 1: Round-robin fills experiments with < min_samples measurements.
    Phase 2: Remaining budget allocated to highest-variance experiments.
    After all runs, replots each model by calling run_scaling_experiment
    with budget=0 (replot mode).

    Args:
        budget_seconds: total wall-time budget in seconds
        bench_cfg: benchmark config section (has unified, min_samples keys)
        figs_dir_root: root figs dir (e.g. results/plots); model subdir appended
        cfg_by_model: {"mlp": mlp_scaling_cfg, "cnn": cnn_scaling_cfg}
        available_by_model: {"mlp": [(label, cmd)...], "cnn": [(label, cmd)...]}
        use_cache: whether to use the cache
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available -- scaling mode requires matplotlib")
        return

    from plotting import _cache_key
    min_samples = bench_cfg.get("min_samples", 3)

    # Load cache
    cache = _load_cache() if use_cache else {}

    # 1. Enumerate all experiments across both models
    all_experiments = []  # list of (cache_key, vary_param, x_range, fixed_params, model, label, cmd_template)

    for model, available in available_by_model.items():
        cfg = cfg_by_model[model]
        epochs = cfg["epochs"]

        if model == "cnn":
            for label, cmd_template in available:
                all_experiments.append({
                    "cache_key": _cache_key(model, "batch_size"),
                    "vary_param": "batch_size",
                    "x_range": (4, 10),
                    "fixed_params": {"epochs": epochs, "learning_rate": 0.32},
                    "model": model,
                    "label": label,
                    "cmd_template": cmd_template,
                })
        else:  # mlp
            fixed_bs = cfg["fixed_batch_size"]
            fixed_ns = cfg["fixed_num_samples"]

            for label, cmd_template in available:
                all_experiments.append({
                    "cache_key": _cache_key(model, "dataset_size"),
                    "vary_param": "num_samples",
                    "x_range": cfg.get("dataset_size_range", (13, 22)),
                    "fixed_params": {"batch_size": fixed_bs, "epochs": epochs},
                    "model": model,
                    "label": label,
                    "cmd_template": cmd_template,
                })
                all_experiments.append({
                    "cache_key": _cache_key(model, "batch_size"),
                    "vary_param": "batch_size",
                    "x_range": cfg.get("batch_size_range", (8, 15)),
                    "fixed_params": {"num_samples": fixed_ns, "epochs": epochs},
                    "model": model,
                    "label": label,
                    "cmd_template": cmd_template,
                })
                all_experiments.append({
                    "cache_key": _cache_key(model, "hidden_size"),
                    "vary_param": "hidden_size",
                    "x_range": cfg.get("hidden_size_range", (6, 12)),
                    "fixed_params": {"num_samples": fixed_ns, "batch_size": fixed_bs, "epochs": epochs},
                    "model": model,
                    "label": label,
                    "cmd_template": cmd_template,
                })

    # 2. Count samples per experiment from cache and compute stats
    def _experiment_stats(exp):
        """Return (total_samples, mean_variance, est_wall) for an experiment."""
        ck = exp["cache_key"]
        label = exp["label"]
        exp_data = cache.get(ck, {}).get(label, {})
        total_samples = 0
        variances = []
        wall_times = []
        for val_str, entry in exp_data.items():
            if isinstance(entry, dict):
                samples = entry.get("samples", [])
                total_samples += len(samples)
                wall_times.extend(entry.get("wall_times", []))
                if len(samples) >= 2:
                    variances.append(float(np.var(samples)))
        mean_var = float(np.mean(variances)) if variances else float("inf")
        est_wall = float(np.mean(wall_times)) if wall_times else 10.0
        return total_samples, mean_var, est_wall

    print(f"\n  Unified cross-model scheduling: {budget_seconds:.0f}s budget")
    print(f"  min_samples={min_samples}, {len(all_experiments)} experiment slots")
    print("=" * 70)

    remaining = budget_seconds
    total_runs = 0

    # 3. Phase 1: Round-robin fill experiments with < min_samples
    print("\n  Phase 1: Round-robin fill (min_samples={})".format(min_samples))

    phase1_ran = True
    while remaining > 0 and phase1_ran:
        phase1_ran = False
        # Sort by total_samples ascending so least-sampled go first
        scored = [(exp, *_experiment_stats(exp)) for exp in all_experiments]
        scored.sort(key=lambda x: x[1])  # sort by total_samples

        for exp, total_samples, mean_var, est_wall in scored:
            if total_samples >= min_samples:
                continue
            if est_wall > remaining:
                continue

            # Sample an x-value using gap-filling
            lo_exp, hi_exp = exp["x_range"]
            exp_data = cache.get(exp["cache_key"], {}).get(exp["label"], {})
            existing_log2 = []
            sample_counts = {}
            for val_str, entry in exp_data.items():
                if isinstance(entry, dict):
                    try:
                        log2_val = np.log2(float(val_str))
                        existing_log2.append(log2_val)
                        sample_counts[log2_val] = len(entry.get("samples", []))
                    except (ValueError, ZeroDivisionError):
                        pass

            x_val = _sample_from_gaps(lo_exp, hi_exp, existing_log2,
                                       sample_counts=sample_counts)
            params = dict(exp["fixed_params"])
            params[exp["vary_param"]] = x_val
            val_str = str(x_val)

            total_runs += 1
            print(f"\n  [P1 #{total_runs}] {exp['model']}/{exp['label']} "
                  f"{exp['vary_param']}={x_val} (n={total_samples})")

            t0 = time.monotonic()
            r = run_implementation(
                exp["label"], exp["cmd_template"], "generated",
                batch_size=params.get("batch_size", 2048),
                num_samples=params.get("num_samples", 100000),
                hidden_size=params.get("hidden_size", 512),
                epochs=params.get("epochs", 500),
                learning_rate=params.get("learning_rate", 0.02),
            )
            elapsed = time.monotonic() - t0
            remaining -= elapsed

            if r and "throughput" in r:
                tp = r["throughput"]
                ck = exp["cache_key"]
                if ck not in cache:
                    cache[ck] = {}
                if exp["label"] not in cache[ck]:
                    cache[ck][exp["label"]] = {}
                if val_str not in cache[ck][exp["label"]]:
                    cache[ck][exp["label"]][val_str] = {"samples": [], "wall_times": []}
                cache[ck][exp["label"]][val_str]["samples"].append(tp)
                cache[ck][exp["label"]][val_str]["wall_times"].append(elapsed)
                print(f"    -> {tp:.0f} samples/s")
            else:
                print(f"    -> FAILED")

            _save_cache(cache)
            phase1_ran = True

    # 4. Phase 2: Remaining budget to highest-variance experiments
    if remaining > 0:
        print(f"\n  Phase 2: Variance-weighted allocation ({remaining:.0f}s remaining)")

        while remaining > 0:
            # Score all experiments by variance (highest first)
            scored = []
            for exp in all_experiments:
                total_samples, mean_var, est_wall = _experiment_stats(exp)
                if est_wall > remaining:
                    continue
                scored.append((exp, total_samples, mean_var, est_wall))

            if not scored:
                break

            # Sort by variance descending (inf = no data, gets priority too)
            scored.sort(key=lambda x: -x[2])
            exp, total_samples, mean_var, est_wall = scored[0]

            lo_exp, hi_exp = exp["x_range"]
            exp_data = cache.get(exp["cache_key"], {}).get(exp["label"], {})
            existing_log2 = []
            sample_counts = {}
            for val_str, entry in exp_data.items():
                if isinstance(entry, dict):
                    try:
                        log2_val = np.log2(float(val_str))
                        existing_log2.append(log2_val)
                        sample_counts[log2_val] = len(entry.get("samples", []))
                    except (ValueError, ZeroDivisionError):
                        pass

            x_val = _sample_from_gaps(lo_exp, hi_exp, existing_log2,
                                       sample_counts=sample_counts)
            params = dict(exp["fixed_params"])
            params[exp["vary_param"]] = x_val
            val_str = str(x_val)

            total_runs += 1
            var_str = f"var={mean_var:.0f}" if mean_var != float("inf") else "var=inf"
            print(f"\n  [P2 #{total_runs}] {exp['model']}/{exp['label']} "
                  f"{exp['vary_param']}={x_val} ({var_str}, n={total_samples})")

            t0 = time.monotonic()
            r = run_implementation(
                exp["label"], exp["cmd_template"], "generated",
                batch_size=params.get("batch_size", 2048),
                num_samples=params.get("num_samples", 100000),
                hidden_size=params.get("hidden_size", 512),
                epochs=params.get("epochs", 500),
                learning_rate=params.get("learning_rate", 0.02),
            )
            elapsed = time.monotonic() - t0
            remaining -= elapsed

            if r and "throughput" in r:
                tp = r["throughput"]
                ck = exp["cache_key"]
                if ck not in cache:
                    cache[ck] = {}
                if exp["label"] not in cache[ck]:
                    cache[ck][exp["label"]] = {}
                if val_str not in cache[ck][exp["label"]]:
                    cache[ck][exp["label"]][val_str] = {"samples": [], "wall_times": []}
                cache[ck][exp["label"]][val_str]["samples"].append(tp)
                cache[ck][exp["label"]][val_str]["wall_times"].append(elapsed)
                print(f"    -> {tp:.0f} samples/s")
            else:
                print(f"    -> FAILED")

            _save_cache(cache)

    print(f"\n  Unified scheduling complete: {total_runs} runs in "
          f"{budget_seconds - remaining:.0f}s")

    # 5. Replot each model by calling run_scaling_experiment with budget=0
    print("\n" + "=" * 70)
    print("  Regenerating plots for all models...")
    print("=" * 70)

    for model, available in available_by_model.items():
        cfg = dict(cfg_by_model[model])
        cfg["budget_seconds"] = 0  # replot mode: no new runs
        model_figs_dir = os.path.join(figs_dir_root, model)
        os.makedirs(model_figs_dir, exist_ok=True)
        labels = [l for l, _ in available]
        print(f"\n  Plotting {model.upper()}...")
        plot_scaling_results(cache, cfg, model, labels, model_figs_dir)
        plot_speedup_results(cache, cfg, model, labels, model_figs_dir)
        plot_overview(cache, cfg, model, labels, model_figs_dir)
        print_peak_summary(cache, model, labels)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark Runner (MLP / CNN)")
    parser.add_argument("--mode", default=None, choices=["standard", "scaling"])
    parser.add_argument("--model", default="mlp", choices=["mlp", "cnn"],
                        help="Model family to benchmark (default: mlp)")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated dataset names (standard mode, MLP only)")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--budget", type=float, default=None, metavar="MINUTES",
                        help="Time budget in minutes for variance-weighted scheduling "
                             "(scaling mode only; use 0 to just replot from cache)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable result caching (re-run everything)")

    # Scaling parameters -- None means "use config value"
    parser.add_argument("--scaling-epochs", type=int, default=None)
    parser.add_argument("--scaling-fixed-bs", type=int, default=None)
    parser.add_argument("--scaling-fixed-ns", type=int, default=None,
                        help="Fixed num_samples (config default: 2^18 = 262144)")
    parser.add_argument("--scaling-dataset-sizes", type=str, default=None,
                        help="Comma-separated (config default: 8K..4M)")
    parser.add_argument("--scaling-batch-sizes", type=str, default=None,
                        help="Comma-separated (config default: 256..32K)")
    parser.add_argument("--scaling-hidden-sizes", type=str, default=None,
                        help="Comma-separated (config default: 64..4096)")
    args = parser.parse_args()

    # Load layered config: base.yaml <- models/{model}.yaml <- local.yaml
    config = load_config(args.model)
    bench_cfg = config["benchmark"]
    scale_cfg = config["scaling"]

    # Apply config defaults where CLI args were not provided
    mode = args.mode or bench_cfg.get("mode", "standard")
    if args.budget is not None and args.runs is None:
        runs = 1  # budget mode implies single runs (scheduler handles repetition)
    else:
        runs = args.runs if args.runs is not None else bench_cfg.get("runs", 1)

    if args.budget is not None and runs > 1:
        parser.error("--budget and --runs > 1 are mutually exclusive")

    figs_dir = os.path.join(PROJECT_ROOT, "results", "plots", args.model)

    # Select implementation list based on model
    impl_list = CNN_IMPLEMENTATIONS if args.model == "cnn" else IMPLEMENTATIONS
    available = [(label, cmd) for label, cmd in impl_list if is_available(label)]
    print(f"Model: {args.model.upper()}")
    print(f"Available implementations: {[l for l, _ in available]}")
    print(f"Mode: {mode}")
    if args.budget is not None:
        print(f"Budget: {args.budget}s (variance-weighted scheduling)")
    else:
        print(f"Runs per config: {runs}")
    if not args.no_cache:
        print(f"Cache: {CACHE_FILE}")
    print()

    # Detect unified mode: --budget set and --model not explicitly passed
    model_explicit = "--model" in sys.argv
    unified_mode = (args.budget is not None and not model_explicit
                    and mode == "scaling" and bench_cfg.get("unified", True))

    if unified_mode:
        # Unified cross-model scheduling across MLP + CNN
        def _parse_list(arg):
            return [int(x) for x in arg.split(",")] if arg else None

        # Build scaling config for each model
        mlp_config = load_config("mlp")
        cnn_config = load_config("cnn")
        mlp_scale = mlp_config["scaling"]
        cnn_scale = cnn_config["scaling"]

        cfg_by_model = {
            "mlp": {
                "epochs": args.scaling_epochs or mlp_scale["epochs"],
                "fixed_batch_size": args.scaling_fixed_bs or mlp_scale["fixed_batch_size"],
                "fixed_num_samples": args.scaling_fixed_ns or mlp_scale["fixed_num_samples"],
                "dataset_sizes": _parse_list(args.scaling_dataset_sizes) or mlp_scale["dataset_sizes"],
                "batch_sizes": _parse_list(args.scaling_batch_sizes) or mlp_scale["batch_sizes"],
                "hidden_sizes": _parse_list(args.scaling_hidden_sizes) or mlp_scale["hidden_sizes"],
            },
            "cnn": {
                "epochs": args.scaling_epochs or cnn_scale["epochs"],
                "batch_sizes": _parse_list(args.scaling_batch_sizes) or cnn_scale["batch_sizes"],
            },
        }

        # Build available implementations for each model
        mlp_available = [(l, c) for l, c in IMPLEMENTATIONS if is_available(l)]
        cnn_available = [(l, c) for l, c in CNN_IMPLEMENTATIONS if is_available(l)]
        available_by_model = {"mlp": mlp_available, "cnn": cnn_available}

        print(f"Unified mode: scheduling across MLP ({len(mlp_available)} impls) "
              f"and CNN ({len(cnn_available)} impls)")
        print()

        figs_dir_root = os.path.join(PROJECT_ROOT, "results", "plots")
        run_unified_scaling(
            budget_seconds=args.budget * 60,
            bench_cfg=bench_cfg,
            figs_dir_root=figs_dir_root,
            cfg_by_model=cfg_by_model,
            available_by_model=available_by_model,
            use_cache=not args.no_cache,
        )

    elif mode == "scaling":
        # Build scaling cfg from YAML defaults, CLI args override
        def _parse_list(arg):
            return [int(x) for x in arg.split(",")] if arg else None

        if args.model == "cnn":
            # CNN: only batch size scaling (MNIST is fixed 60K, architecture is fixed)
            cfg = {
                "epochs": args.scaling_epochs or scale_cfg["epochs"],
                "batch_sizes": _parse_list(args.scaling_batch_sizes) or scale_cfg["batch_sizes"],
            }
        else:
            cfg = {
                "epochs": args.scaling_epochs or scale_cfg["epochs"],
                "fixed_batch_size": args.scaling_fixed_bs or scale_cfg["fixed_batch_size"],
                "fixed_num_samples": args.scaling_fixed_ns or scale_cfg["fixed_num_samples"],
                "dataset_sizes": _parse_list(args.scaling_dataset_sizes) or scale_cfg["dataset_sizes"],
                "batch_sizes": _parse_list(args.scaling_batch_sizes) or scale_cfg["batch_sizes"],
                "hidden_sizes": _parse_list(args.scaling_hidden_sizes) or scale_cfg["hidden_sizes"],
            }
        if args.budget is not None:
            cfg["budget_seconds"] = args.budget * 60
        run_scaling_experiment(available, runs, figs_dir, cfg,
                               model=args.model, use_cache=not args.no_cache)
    else:
        if args.model == "cnn":
            datasets = bench_cfg.get("datasets", ["mnist"])
        elif args.datasets:
            datasets = [d.strip() for d in args.datasets.split(",")]
        else:
            datasets = bench_cfg.get("datasets", ["generated", "iris", "breast-cancer"])
        print(f"Datasets: {datasets}\n")
        run_standard(datasets, available, runs, figs_dir, model=args.model)


if __name__ == "__main__":
    main()
