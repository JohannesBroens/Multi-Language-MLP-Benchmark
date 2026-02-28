#!/usr/bin/env python3
"""Unified pipeline: build -> tune -> benchmark -> plot.

Usage:
    python src/scripts/pipeline.py all                # all models
    python src/scripts/pipeline.py all --model mlp    # just MLP
    python src/scripts/pipeline.py all --model cnn    # just CNN
    python src/scripts/pipeline.py benchmark --model cnn
    python src/scripts/pipeline.py tune --model mlp
    python src/scripts/pipeline.py build
"""
import argparse
import os
import subprocess
import sys

# Ensure src/scripts is on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config import load_config

PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PYTHON = sys.executable

ALL_MODELS = ["mlp", "cnn"]


def phase_build(config, models):
    """Compile C and Rust implementations based on config."""
    build = config.get("build", {})

    # C builds (shared: both main and cnn_main are built together)
    c_dir = os.path.join(PROJECT_ROOT, "src", "c")
    if build.get("c_cpu", True):
        print("=== Building C (CPU) ===")
        build_dir = os.path.join(c_dir, "build_cpu")
        os.makedirs(build_dir, exist_ok=True)
        subprocess.run(["cmake", "..", "-DUSE_CUDA=OFF"], cwd=build_dir, check=True)
        subprocess.run(["make", "-j"], cwd=build_dir, check=True)

    if build.get("c_cuda", True):
        print("=== Building C (CUDA) ===")
        build_dir = os.path.join(c_dir, "build_cuda")
        os.makedirs(build_dir, exist_ok=True)
        subprocess.run(["cmake", "..", "-DUSE_CUDA=ON"], cwd=build_dir, check=True)
        subprocess.run(["make", "-j"], cwd=build_dir, check=True)

    # Rust builds — per model
    rust_dir = os.path.join(PROJECT_ROOT, "src", "rust")
    rust_targets = []
    for model in models:
        if build.get("rust_cpu", True):
            rust_targets.append(f"{model}-cpu")
        if build.get("rust_cublas", True):
            rust_targets.append(f"{model}-cuda-cublas")
        if build.get("rust_kernels", True):
            rust_targets.append(f"{model}-cuda-kernels")

    for target in rust_targets:
        print(f"=== Building Rust ({target}) ===")
        subprocess.run(
            ["cargo", "build", "--release", "-p", target],
            cwd=rust_dir, check=True,
        )

    # Python: verify venv
    print("=== Checking Python environment ===")
    subprocess.run([PYTHON, "-c", "import numpy; import torch"], check=True)
    print("Python environment OK")


def phase_tune(config, model):
    """Find optimal hyperparameters via grid search."""
    tuning = config.get("tuning", {})
    if not tuning.get("enabled", True):
        print("Tuning disabled in config, skipping.")
        return

    cache_path = os.path.join(PROJECT_ROOT, "results", "cache", f"tuning_{model}.yaml")
    if os.path.exists(cache_path) and not config.get("pipeline", {}).get("force_retune", False):
        print(f"Tuning cache exists at {cache_path}, skipping. Use force_retune: true to override.")
        return

    print(f"=== Tuning {model.upper()} hyperparameters ===")
    # Delegate to tuning module
    from tuning import run_tuning
    run_tuning(config, model, cache_path)


def phase_benchmark(config, model):
    """Run benchmarks using config values (with tuned params if available)."""
    # Load tuned hyperparameters if available
    tuning_path = os.path.join(PROJECT_ROOT, "results", "cache", f"tuning_{model}.yaml")
    if os.path.exists(tuning_path):
        import yaml
        with open(tuning_path) as f:
            tuned = yaml.safe_load(f)
        best = tuned.get("selected", tuned.get("best_params", {}))
        if best:
            config["training"]["batch_size"] = best.get("batch_size", config["training"]["batch_size"])
            config["training"]["learning_rate"] = best.get("learning_rate", config["training"]["learning_rate"])
            config["training"]["optimizer"] = best.get("optimizer", "sgd")
            config["training"]["scheduler"] = best.get("scheduler", "none")
            extra = ""
            if "accuracy" in best and "throughput" in best:
                extra = f" (acc={best['accuracy']:.2f}%, {best['throughput']:.0f} samples/s)"
            elif "accuracy" in best:
                extra = f" (acc={best['accuracy']:.2f}%)"
            print(f"Loaded tuned params: bs={best.get('batch_size')}, "
                  f"lr={best.get('learning_rate')}, opt={best.get('optimizer')}, "
                  f"sched={best.get('scheduler')}{extra}")

    print(f"=== Benchmarking {model.upper()} ===")
    bench = config.get("benchmark", {})
    cmd = [
        PYTHON, os.path.join(SCRIPT_DIR, "benchmark.py"),
        "--mode", bench.get("mode", "standard"),
        "--model", model,
        "--datasets", ",".join(bench.get("datasets", ["generated"])),
        "--runs", str(bench.get("runs", 3)),
    ]
    budget = bench.get("budget_minutes")
    if budget:
        cmd.extend(["--budget", str(budget)])

    subprocess.run(cmd, check=True)


def phase_plot(config, model):
    """Generate plots from cached benchmark data."""
    print(f"=== Generating {model.upper()} plots ===")
    from plotting import regenerate_all_plots
    regenerate_all_plots(config, model)


def run_for_model(model, command, phases):
    """Run a pipeline command for a single model."""
    config = load_config(model)

    if command == "all":
        # Build is handled separately (once for all models)
        for phase_name in config.get("pipeline", {}).get("phases", ["tune", "benchmark"]):
            phases[phase_name](config, model)
        phase_plot(config, model)
    else:
        phases[command](config, model)


def main():
    parser = argparse.ArgumentParser(description="ML-in-C unified pipeline")
    parser.add_argument("command", choices=["all", "build", "tune", "benchmark", "plot"],
                        help="Pipeline phase to run")
    parser.add_argument("--model", default="all", choices=["all", "mlp", "cnn"],
                        help="Model to operate on (default: all)")
    parser.add_argument("--config", default=None, help="Path to base config YAML")
    args = parser.parse_args()

    models = ALL_MODELS if args.model == "all" else [args.model]

    phases = {
        "build": lambda config, model: None,  # handled separately
        "tune": phase_tune,
        "benchmark": phase_benchmark,
        "plot": phase_plot,
    }

    # Build phase: done once (C builds both models, Rust builds per model)
    if args.command in ("all", "build"):
        config = load_config(models[0], config_path=args.config)
        phase_build(config, models)
        if args.command == "build":
            return

    # Per-model phases
    for model in models:
        print(f"\n{'#'*60}")
        print(f"#  {model.upper()}")
        print(f"{'#'*60}\n")
        run_for_model(model, args.command, phases)


if __name__ == "__main__":
    main()
